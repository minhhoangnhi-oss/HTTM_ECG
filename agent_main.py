import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent_tools import ALLOWED_RECORD_IDS, TOOL_SCHEMAS, execute_tool

ROOT = Path(__file__).resolve().parent
RUN_ID = uuid.uuid4().hex
KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

SYSTEM_PROMPT = """
Bạn là trợ lý điều phối dịch vụ ECG.
Trước tiên dùng list_records để xem bản ghi được phép.
Sau đó dùng classify_record cho đúng bản ghi người dùng yêu cầu, nếu có trong danh sách và ready=true.
Chỉ tổng hợp thông tin do công cụ trả về.
Không tự tạo nhãn, xác suất hoặc số liệu.
Nêu rõ nguồn dữ liệu, record_id, mode và phiên bản mô hình.
Nêu số nhịp đã xử lý, bỏ qua và số nhịp dự đoán theo từng lớp.
Đây là kết quả dự đoán của mô hình trên ECG đã thu nhận.
Không coi số lượng nhãn dự đoán là độ chính xác của mô hình.
Không kết luận bệnh, chẩn đoán hoặc đề xuất điều trị.
Bạn không có công cụ ghi tệp hay gửi báo cáo.
Việc lưu do chương trình hỏi người dùng sau khi hiển thị kết quả.
Trả lời ngắn gọn bằng tiếng Việt.
"""

def now():
    return datetime.now(timezone.utc).isoformat()


def audit(event, **details):
    folder = ROOT / "logs"
    folder.mkdir(exist_ok=True)

    entry = {
        "time": now(),
        "run_id": RUN_ID,
        "event": event,
        **details
    }

    with (folder / "agent_audit.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_model(client, messages):
    response = client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}"},
        json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "max_tokens": 2048
        }
    )

    if response.is_error:
        audit("llm_error", status=response.status_code)
        raise RuntimeError(
            f"OpenRouter báo lỗi HTTP {response.status_code}."
        )

    data = response.json()
    message = data["choices"][0]["message"]
    actual_model = data.get("model")

    audit(
        "llm_response",
        requested_model=MODEL,
        actual_model=actual_model,
        finish_reason=data["choices"][0].get("finish_reason")
    )

    return message, actual_model


def run_agent(record_id="100"):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Xem bản ghi được phép và xử lý bản ghi {record_id}."
        }
    ]

    last_result = None
    models_used = []
    tool_count = 0

    with httpx.Client(timeout=120) as client:
        for step in range(6):
            message, actual_model = call_model(client, messages)
            models_used.append(actual_model)
            calls = message.get("tool_calls") or []

            if not calls:
                content = message.get("content")

                if not last_result:
                    raise RuntimeError(
                        "Chưa có kết quả API. Không tạo báo cáo."
                    )

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        "LLM chưa trả bản tóm tắt. Không lưu báo cáo."
                    )

                return last_result, content, models_used

            tool_count += len(calls)

            if tool_count > 8:
                raise RuntimeError("Đã đạt giới hạn gọi công cụ.")

            assistant = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls
            }

            if message.get("reasoning_details"):
                assistant["reasoning_details"] = message["reasoning_details"]

            messages.append(assistant)

            for call in calls:
                name = call["function"]["name"]
                audit("tool_requested", tool=name)
                print("Đang gọi:", name)

                try:
                    arguments = json.loads(
                        call["function"]["arguments"]
                    )
                    result = execute_tool(name, arguments)

                    if name == "classify_record":
                        last_result = result

                    audit(
                        "tool_success",
                        tool=name,
                        record_id=result.get("record_id"),
                        mode=result.get("mode")
                    )

                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    httpx.RequestError,
                    httpx.HTTPStatusError
                ) as error:
                    if name == "classify_record":
                        last_result = None

                    audit(
                        "tool_error",
                        tool=name,
                        error_type=type(error).__name__
                    )

                    result = {
                        "error": "Công cụ bị từ chối hoặc API gặp lỗi."
                    }

                    print("Công cụ không thực hiện được:", name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(
                        result, ensure_ascii=False
                    )
                })

    raise RuntimeError("Đã đạt giới hạn vòng lặp agent.")


def main():
    parser = argparse.ArgumentParser(description="ECG Agent cho 21 bản ghi DS2")
    parser.add_argument("--record-id", choices=ALLOWED_RECORD_IDS, default="100")
    args = parser.parse_args()
    if not KEY:
        raise RuntimeError("Chưa đọc được API key từ .env.")

    audit("run_started")
    result, summary, models_used = run_agent(args.record_id)

    print("\nSỐ LIỆU TỪ API")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\nBẢN TÓM TẮT ĐỂ XEM LẠI")
    print(summary)

    audit("confirmation_requested")

    try:
        answer = input("\nLưu báo cáo? Nhập y để lưu, n để bỏ: ")
    except (EOFError, KeyboardInterrupt):
        audit("confirmation_cancelled")
        print("\nChưa xác nhận. Không lưu báo cáo.")
        return

    if answer.strip().lower() != "y":
        audit("save_declined")
        print("Không lưu báo cáo.")
        return

    audit("save_confirmed")

    report = {
        "run_id": RUN_ID,
        "created_at": now(),
        "confirmed": True,
        "requested_llm_model": MODEL,
        "actual_llm_models": models_used,
        "api_result": result,
        "summary": summary
    }

    folder = ROOT / "outputs"
    folder.mkdir(exist_ok=True)
    destination = folder / f"report_{RUN_ID}.json"

    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    audit("report_saved", filename=destination.name)
    print("Đã lưu:", destination)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        audit("run_cancelled")
        print("\nAgent đã dừng.")
    except (
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OSError,
        httpx.RequestError
    ) as error:
        try:
            audit("run_failed", error_type=type(error).__name__)
        except OSError:
            pass

        if isinstance(error, RuntimeError):
            print("Agent dừng:", error)
        else:
            print("Agent dừng do lỗi:", type(error).__name__)
