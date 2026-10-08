import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import wfdb

from model_runner import ECGModel
from preprocessing import preprocess_ecg

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "mitdb"
API_URL = "http://127.0.0.1:8001"
RECORD_ID = "100"

AAMI = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
    "/": 4, "f": 4, "Q": 4,
}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = [RAW_DIR / f"{RECORD_ID}.{ext}" for ext in ["dat", "hea", "atr"]]
    if not all(path.is_file() for path in files):
        print("Dang tai ban ghi 100 tu PhysioNet...", flush=True)
        wfdb.dl_database(
            "mitdb",
            str(RAW_DIR),
            records=[RECORD_ID],
            annotators=["atr"],
            overwrite=False,
        )

    record_path = str(RAW_DIR / RECORD_ID)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    if record.fs != 360:
        raise ValueError("Ban ghi khong co tan so 360 Hz.")

    channel = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
    if record.units[channel] != "mV":
        raise ValueError("Don vi ECG phai la mV.")

    signal = record.p_signal[:, channel]
    selected = np.array([symbol in AAMI for symbol in annotation.symbol])
    peaks = annotation.sample[selected]
    labels = np.array(
        [AAMI[symbol] for symbol in annotation.symbol if symbol in AAMI],
        dtype=np.int64,
    )

    waveform, rr, kept_peaks, skipped = preprocess_ecg(signal, peaks, 360)
    valid_windows = (peaks >= 90) & (peaks + 144 <= len(signal))

    np.testing.assert_array_equal(kept_peaks, peaks[valid_windows])

    with np.load(ROOT / "data" / "tv1" / "ds2_data.npz", allow_pickle=False) as data:
        mask = data["record_id"].astype(str) == RECORD_ID
        reference_waveform = data["X_waveform"][mask].copy()
        reference_rr = data["X_rr"][mask].copy()
        reference_peaks = data["r_peak_index"][mask].copy()
        reference_labels = data["y"][mask].copy()

    if len(reference_peaks) == 0:
        raise ValueError("Khong tim thay ban ghi 100 trong du lieu TV1.")

    np.testing.assert_array_equal(kept_peaks, reference_peaks)
    np.testing.assert_array_equal(labels[valid_windows], reference_labels)
    np.testing.assert_allclose(
        waveform, reference_waveform, rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        rr, reference_rr, rtol=1e-5, atol=1e-6,
    )

    waveform_error = float(np.max(np.abs(waveform - reference_waveform)))
    rr_error = float(np.max(np.abs(rr - reference_rr)))

    print(f"OK: Tien xu ly khop TV1, {len(kept_peaks)} nhip.", flush=True)
    print(f"Sai so ECG lon nhat: {waveform_error:.8g}", flush=True)
    print(f"Sai so RR lon nhat: {rr_error:.8g}", flush=True)

    model = ECGModel()
    expected_probabilities = model.predict(reference_waveform, reference_rr)
    expected_labels = np.array(model.classes)[expected_probabilities.argmax(axis=1)]

    print("Dang doi chieu ket qua API...", flush=True)

    with httpx.Client(timeout=180.0, trust_env=False) as client:
        health_response = client.get(f"{API_URL}/health")
        health_response.raise_for_status()
        health = health_response.json()

        if health.get("mode") != "onnx":
            raise ValueError("API chua chay che do ONNX.")
        if health.get("model_version") != model.version:
            raise ValueError("API dang dung phien ban mo hinh khac.")

        response = client.post(
            f"{API_URL}/predict",
            json={
                "ecg": signal.tolist(),
                "fs": 360,
                "r_peaks": peaks.tolist(),
            },
        )
        response.raise_for_status()
        result = response.json()

    if result["mode"] != "onnx" or result["model_version"] != model.version:
        raise ValueError("Thong tin mo hinh trong ket qua API khong dung.")
    if result["class_order"] != model.classes:
        raise ValueError("Thu tu lop API khong dung.")
    if result["processed_count"] != len(kept_peaks):
        raise ValueError("So nhip xu ly cua API khong khop.")
    if result["skipped_count"] != len(skipped) or result["skipped"] != skipped:
        raise ValueError("Danh sach nhip bo qua cua API khong khop.")

    api_peaks = [beat["r_peak_index"] for beat in result["beats"]]
    api_labels = [beat["predicted_class"] for beat in result["beats"]]
    api_probabilities = np.array([
        [beat["probabilities"][label] for label in model.classes]
        for beat in result["beats"]
    ])

    np.testing.assert_array_equal(api_peaks, reference_peaks)
    np.testing.assert_array_equal(api_labels, expected_labels)
    np.testing.assert_allclose(
        api_probabilities,
        expected_probabilities,
        rtol=1e-4,
        atol=1e-5,
    )

    probability_error = float(
        np.max(np.abs(api_probabilities - expected_probabilities))
    )

    evidence = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "record_id": RECORD_ID,
        "source": "PhysioNet MIT-BIH",
        "sampling_rate_hz": 360,
        "channel": record.sig_name[channel],
        "signal_samples": len(signal),
        "model_version": model.version,
        "processed_count": len(kept_peaks),
        "skipped_count": len(skipped),
        "preprocessing_matches_tv1": True,
        "waveform_max_abs_error": waveform_error,
        "rr_max_abs_error": rr_error,
        "api_labels_match_model_runner": True,
        "api_max_abs_probability_error": probability_error,
        "class_counts": {
            label: int(np.count_nonzero(expected_labels == label))
            for label in model.classes
        },
    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "check_real_ecg_100.json"
    output_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"OK: Nhan va xac suat API khop, sai so {probability_error:.8g}")
    print("KIEM TRA BAN GHI ECG THAT THANH CONG.")
    print(f"Da luu: {output_path}")


if __name__ == "__main__":
    main()