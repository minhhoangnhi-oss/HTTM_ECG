import math
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(
    Path(__file__).with_name(".env"),
    encoding="utf-8-sig"
)

CLASSES = ["N", "S", "V", "F", "Q"]
RECORD_ID = "demo_001"


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
                "additionalProperties": False
            }
        }
    }


TOOL_SCHEMAS = [
    tool_schema(
        "list_records",
        "Liệt kê các bản ghi được phép truy cập.",
        {}
    ),
    tool_schema(
        "classify_record",
        "Gọi API ECG cho bản ghi được phép. Có thể trả kết quả mock.",
        {
            "record_id": {
                "type": "string",
                "enum": [RECORD_ID]
            }
        }
    )
]


def execute_tool(name, arguments):
    if not isinstance(arguments, dict):
        raise ValueError("Tham số công cụ phải là một đối tượng.")

    if name == "list_records":
        if arguments:
            raise ValueError("Công cụ này không nhận tham số.")

        return {
            "records": [
                {
                    "record_id": RECORD_ID,
                    "source": "synthetic_test_signal",
                    "fs": 360,
                    "samples": 1000
                }
            ]
        }

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
        "http://localhost:8000"
    }:
        raise ValueError("Địa chỉ API ngoài danh sách cho phép.")

    ecg = [
        math.sin(2 * math.pi * 1.2 * i / 360)
        for i in range(1000)
    ]

    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post(
            f"{api_url}/predict",
            json={
                "ecg": ecg,
                "fs": 360,
                "r_peaks": [20, 400, 700]
            }
        )

    response.raise_for_status()
    data = response.json()

    counts = {label: 0 for label in CLASSES}
    unclassified = 0

    for beat in data["beats"]:
        label = beat["predicted_class"]

        if label is None:
            unclassified += 1
        elif label in counts:
            counts[label] += 1
        else:
            raise ValueError("API trả về lớp không hợp lệ.")

    return {
        "record_id": RECORD_ID,
        "source": "synthetic_test_signal",
        "mode": data["mode"],
        "model_version": data["model_version"],
        "processed_count": data["processed_count"],
        "skipped_count": data["skipped_count"],
        "class_counts": counts,
        "unclassified_count": unclassified
    }