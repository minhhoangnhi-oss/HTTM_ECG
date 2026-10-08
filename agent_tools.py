import math
import os
from pathlib import Path

import httpx
import numpy as np
import wfdb
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(
    ROOT / ".env",
    encoding="utf-8-sig",
)

CLASSES = ["N", "S", "V", "F", "Q"]
RECORD_ID = "100"
RECORD_PATH = ROOT / "data" / "mitdb" / RECORD_ID
SOURCE = "physionet_mit_bih"

AAMI_SYMBOLS = {
    "N", "L", "R", "e", "j",
    "A", "a", "J", "S",
    "V", "E", "F", "/", "f", "Q",
}


def tool_schema(name, description, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    tool_schema(
        "list_records",
        "Liệt kê bản ghi ECG thật được phép truy cập trên máy.",
        {},
    ),
    tool_schema(
        "classify_record",
        "Gọi API ONNX cho bản ghi ECG thật và trả thống kê dự đoán.",
        {
            "record_id": {
                "type": "string",
                "enum": [RECORD_ID],
            },
        },
    ),
]


def record_metadata():
    for extension in ["dat", "hea", "atr"]:
        path = RECORD_PATH.with_suffix(f".{extension}")
        if not path.is_file():
            raise ValueError("Thiếu tệp bản ghi ECG được phép.")

    header = wfdb.rdheader(str(RECORD_PATH))
    channel = header.sig_name.index("MLII") if "MLII" in header.sig_name else 0

    if header.fs != 360:
        raise ValueError("Bản ghi phải có tần số 360 Hz.")
    if header.units[channel] != "mV":
        raise ValueError("Biên độ ECG phải có đơn vị mV.")

    return {
        "record_id": RECORD_ID,
        "source": SOURCE,
        "fs": 360,
        "samples": int(header.sig_len),
        "duration_seconds": float(header.sig_len / header.fs),
        "channel": header.sig_name[channel],
        "amplitude_unit": "mV",
        "r_peak_source": "provided_annotations",
    }


def execute_tool(name, arguments):
    if not isinstance(arguments, dict):
        raise ValueError("Tham số công cụ phải là một đối tượng.")

    if name == "list_records":
        if arguments:
            raise ValueError("Công cụ này không nhận tham số.")
        return {"records": [record_metadata()]}

    if name != "classify_record":
        raise ValueError("Công cụ ngoài danh sách cho phép.")

    if set(arguments) != {"record_id"}:
        raise ValueError("Tham số không đúng cấu trúc.")

    if arguments["record_id"] != RECORD_ID:
        raise ValueError("Bản ghi không được phép truy cập.")

    api_url = os.getenv(
        "ECG_API_URL", "http://127.0.0.1:8000"
    ).rstrip("/")

    if api_url not in {
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    }:
        raise ValueError("Địa chỉ API ngoài danh sách cho phép.")

    metadata = record_metadata()
    record = wfdb.rdrecord(str(RECORD_PATH))
    annotation = wfdb.rdann(str(RECORD_PATH), "atr")

    channel = record.sig_name.index(metadata["channel"])
    signal = record.p_signal[:, channel]

    if not np.isfinite(signal).all():
        raise ValueError("ECG chứa giá trị không hữu hạn.")

    selected = np.array([
        symbol in AAMI_SYMBOLS
        for symbol in annotation.symbol
    ])
    peaks = annotation.sample[selected]

    if len(peaks) == 0:
        raise ValueError("Bản ghi không có đỉnh R phù hợp.")
    if np.any(peaks < 0) or np.any(peaks >= len(signal)):
        raise ValueError("Chú thích R nằm ngoài tín hiệu.")
    if np.any(peaks[1:] <= peaks[:-1]):
        raise ValueError("Chú thích R phải tăng dần và không trùng.")

    valid_windows = (peaks >= 90) & (peaks + 144 <= len(signal))
    expected_kept = peaks[valid_windows].tolist()
    expected_skipped = peaks[~valid_windows].tolist()

    with httpx.Client(timeout=180.0, trust_env=False) as client:
        response = client.post(
            f"{api_url}/predict",
            json={
                "ecg": signal.tolist(),
                "fs": 360,
                "r_peaks": peaks.tolist(),
            },
        )

    response.raise_for_status()
    data = response.json()

    if data["mode"] != "onnx":
        raise ValueError("API chưa chạy mô hình ONNX thật.")
    if not isinstance(data["model_version"], str) or not data["model_version"]:
        raise ValueError("API thiếu phiên bản mô hình.")
    if data["class_order"] != CLASSES:
        raise ValueError("Thứ tự lớp API không đúng.")

    actual_kept = [beat["r_peak_index"] for beat in data["beats"]]
    actual_skipped = [beat["r_peak_index"] for beat in data["skipped"]]

    if actual_kept != expected_kept or actual_skipped != expected_skipped:
        raise ValueError("Danh sách nhịp API không khớp đầu vào.")
    if data["processed_count"] != len(expected_kept):
        raise ValueError("Số nhịp xử lý API không khớp.")
    if data["skipped_count"] != len(expected_skipped):
        raise ValueError("Số nhịp bỏ qua API không khớp.")

    counts = {label: 0 for label in CLASSES}

    for beat in data["beats"]:
        label = beat["predicted_class"]
        scores = beat["probabilities"]

        if label not in counts or not isinstance(scores, dict):
            raise ValueError("API trả kết quả phân loại không hợp lệ.")
        if set(scores) != set(CLASSES):
            raise ValueError("API trả thiếu xác suất các lớp.")

        values = [scores[item] for item in CLASSES]

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 1
            for value in values
        ):
            raise ValueError("Xác suất API không hợp lệ.")

        if not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Tổng xác suất API không bằng 1.")

        if label != max(CLASSES, key=lambda item: scores[item]):
            raise ValueError("Nhãn API không khớp xác suất lớn nhất.")

        counts[label] += 1

    return {
        **metadata,
        "mode": data["mode"],
        "model_version": data["model_version"],
        "r_peak_count": int(len(peaks)),
        "processed_count": data["processed_count"],
        "skipped_count": data["skipped_count"],
        "class_counts": counts,
        "unclassified_count": 0,
    }