import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(
    Path(__file__).with_name(".env"),
    encoding="utf-8-sig"
)

key = os.getenv("OPENROUTER_API_KEY", "").strip()
model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

if not key:
    raise SystemExit("Chưa đọc được API key từ .env")

payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "Hãy dùng công cụ check_ecg_api để kiểm tra dịch vụ ECG."
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "check_ecg_api",
                "description": "Kiểm tra trạng thái dịch vụ ECG.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False
                }
            }
        }
    ],
    "tool_choice": "required",
    "max_tokens": 1024
}

try:
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=90
    )

    data = response.json()

    if response.is_error:
        message = data.get("error", {}).get(
            "message", "Yêu cầu thất bại"
        )
        print(f"Lỗi HTTP {response.status_code}: {message}")
    else:
        print("Mô hình:", data.get("model"))
        message = data["choices"][0]["message"]
        calls = message.get("tool_calls") or []

        if calls:
            for call in calls:
                print("Công cụ được yêu cầu:", call["function"]["name"])
        else:
            print("Đã kết nối nhưng chưa nhận được tool call.")
            print("Trạng thái:", data["choices"][0].get("finish_reason"))

except httpx.RequestError:
    print("Lỗi kết nối hoặc hết thời gian chờ OpenRouter.")
except (ValueError, KeyError, IndexError, TypeError):
    print("Phản hồi OpenRouter chưa có cấu trúc mong đợi.")