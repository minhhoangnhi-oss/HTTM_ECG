import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from agent_tools import (
    RECORD_ID, RECORD_PATH, TOOL_SCHEMAS, ToolArgumentError, execute_tool,
    normalize_record_id,
)

ROOT = Path(__file__).resolve().parent
router = APIRouter(prefix="/agent", tags=["Web Agent"])
PENDING = {}
PENDING_LOCK = threading.Lock()
AUDIT_LOCK = threading.Lock()
RUN_LIMIT = threading.BoundedSemaphore(2)

SYSTEM_PROMPT = """
Bạn là trợ lý điều phối dịch vụ ECG MIT-BIH.
Mỗi lượt phải dùng list_records để kiểm tra các bản ghi được phép.
Khi người dùng yêu cầu phân loại hoặc số lượng từng lớp,
phải gọi classify_record cho bản ghi được phép.
Chỉ tổng hợp metadata và số liệu công cụ vừa trả về.
Lịch sử trò chuyện chỉ dùng để hiểu yêu cầu, không thay thế dữ liệu công cụ.
Ưu tiên mã bản ghi trong yêu cầu hiện tại, không lấy mã từ lượt trước.
Khi gọi classify_record, chỉ truyền một trường record_id, dạng chuỗi.
Nếu công cụ báo lỗi tham số có thể sửa, sửa đúng tham số rồi thử lại;
không đổi bản ghi ngoài quyền thành bản ghi 100 để làm thay yêu cầu.
samples là số mẫu tín hiệu ECG.
r_peak_count là số đỉnh R lấy từ chú thích có sẵn.
Không nói hệ thống tự phát hiện đỉnh R.
processed_count, skipped_count và unclassified_count đều đếm nhịp.
class_counts là số nhịp được mô hình dự đoán theo lớp.
Giữ tên lớp N, S, V, F, Q; không tự thêm ý nghĩa lớp hoặc tỷ lệ.
Không tự tạo số liệu, xác suất hoặc độ chính xác.
Không đưa ra chẩn đoán hay đề xuất điều trị.
Từ chối bản ghi ngoài danh sách, lệnh shell và đọc tệp tùy ý.
Bạn không có công cụ ghi tệp, lưu hoặc gửi báo cáo.
Người dùng tự chọn lưu hoặc bỏ qua trên giao diện.
Nếu công cụ lỗi, nói rõ chưa có kết quả và không suy đoán thay thế.
Trả lời ngắn gọn bằng tiếng Việt.
"""


class AgentError(RuntimeError):
    pass


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryMessage(InputModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class ChatInput(InputModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=6,
    )


class ReportDecision(InputModel):
    run_id: str = Field(
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
    )
    token: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    action: Literal["save", "discard"]
    confirmed: StrictBool = False


def now():
    return datetime.now(timezone.utc).isoformat()


def audit(run_id, event, **details):
    folder = ROOT / "logs"
    folder.mkdir(exist_ok=True)
    entry = {
        "time": now(),
        "run_id": run_id,
        "source": "web",
        "event": event,
        **details,
    }
    with AUDIT_LOCK:
        with (folder / "agent_audit.jsonl").open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def check_origin(request):
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        host = request.headers.get("host", "")
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.netloc.lower() != host.lower()
        ):
            raise HTTPException(
                status_code=403,
                detail="Chỉ chấp nhận yêu cầu từ cùng giao diện.",
            )


def clean_pending():
    expired = [
        run_id
        for run_id, entry in PENDING.items()
        if entry["expires_at"] < time.monotonic()
    ]
    for run_id in expired:
        del PENDING[run_id]


def run_conversation(data, run_id, key, model):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[message.model_dump() for message in data.history],
        {"role": "user", "content": data.message.strip()},
    ]

    last_result = None
    models_used = []
    tool_events = []
    tool_count = 0
    successful_tools = 0
    repair_attempts = 0
    argument_error = False
    force_classify = False
    requested_records = set(re.findall(
        r"(?:bản\s*ghi|record(?:_id)?)\s*(?:là\s*)?[:=]?\s*[\"'`]*([0-9]+)\b",
        data.message,
        flags=re.IGNORECASE,
    ))

    with httpx.Client(timeout=60.0, trust_env=False) as client:
        for _ in range(6):
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "tools": TOOL_SCHEMAS,
                    "tool_choice": (
                        {"type": "function", "function": {"name": "classify_record"}}
                        if force_classify
                        else ("required" if successful_tools == 0 else "auto")
                    ),
                    "max_tokens": 2048,
                },
            )

            if response.is_error:
                audit(
                    run_id,
                    "llm_error",
                    status=response.status_code,
                )
                raise AgentError(
                    f"OpenRouter báo lỗi HTTP {response.status_code}."
                )

            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            actual_model = payload.get("model")
            models_used.append(actual_model)

            audit(
                run_id,
                "llm_response",
                requested_model=model,
                actual_model=actual_model,
                finish_reason=choice.get("finish_reason"),
            )

            calls = message.get("tool_calls") or []

            if not calls:
                summary = message.get("content")
                if not isinstance(summary, str) or not summary.strip():
                    raise AgentError("Agent chưa trả lời. Hãy thử lại.")

                if successful_tools == 0:
                    # Some routed models ignore tool_choice="required".
                    # Discard their ungrounded reply and fetch the allowlist.
                    audit(
                        run_id,
                        "tool_requested",
                        tool="list_records",
                        initiated_by="backend_fallback",
                    )
                    try:
                        records = execute_tool("list_records", {})
                    except (
                        ValueError, KeyError, TypeError, OSError, httpx.HTTPError
                    ) as error:
                        audit(
                            run_id,
                            "tool_error",
                            tool="list_records",
                            error_type=type(error).__name__,
                            initiated_by="backend_fallback",
                        )
                        raise AgentError(
                            "Không lấy được danh sách bản ghi được phép. "
                            "Chưa có kết quả phân loại."
                        ) from error

                    successful_tools += 1
                    tool_count += 1
                    tool_events.append({
                        "tool": "list_records",
                        "status": "success",
                    })
                    audit(
                        run_id,
                        "tool_success",
                        tool="list_records",
                        record_id=None,
                        mode=None,
                        initiated_by="backend_fallback",
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            "Chương trình vừa thực sự gọi list_records. "
                            "Kết quả công cụ: "
                            + json.dumps(records, ensure_ascii=False)
                            + "\nHãy trả lời lại yêu cầu hiện tại dựa trên "
                            "danh sách này. Nếu bản ghi ngoài danh sách, "
                            "nói rõ không được phép truy cập, không phân loại "
                            "bản ghi khác thay thế. Nếu cần phân loại bản ghi "
                            "hợp lệ, phải gọi classify_record. Chưa có số liệu "
                            "phân loại cho tới khi công cụ đó thành công."
                        ),
                    })
                    continue

                if (
                    last_result is None
                    and argument_error
                    and requested_records == {RECORD_ID}
                    and repair_attempts < 1
                ):
                    repair_attempts += 1
                    force_classify = True
                    messages.append({
                        "role": "system",
                        "content": (
                            "Lần gọi phân loại bị lỗi tham số. Yêu cầu hiện tại "
                            "chỉ định bản ghi 100. Hãy gọi classify_record với "
                            '{"record_id":"100"}; không thêm trường khác. '
                            "Chỉ tóm tắt sau khi nhận kết quả hoặc lỗi công cụ."
                        ),
                    })
                    audit(run_id, "argument_retry", tool="classify_record")
                    continue

                if last_result is None and argument_error and force_classify:
                    raise AgentError(
                        "Agent chưa gửi được tham số phân loại hợp lệ sau "
                        "một lần thử sửa. Chưa có kết quả dự đoán."
                    )

                return last_result, summary.strip(), models_used, tool_events

            tool_count += len(calls)
            if tool_count > 8:
                raise AgentError("Đã đạt giới hạn gọi công cụ.")

            assistant = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls,
            }
            if message.get("reasoning_details"):
                assistant["reasoning_details"] = message["reasoning_details"]

            messages.append(assistant)

            for call in calls:
                name = call["function"]["name"]
                audit(run_id, "tool_requested", tool=name)

                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if name == "classify_record":
                        raw_id = arguments.get("record_id") if isinstance(arguments, dict) else None
                        audit(
                            run_id,
                            "tool_arguments",
                            tool=name,
                            record_id=(
                                str(raw_id).strip()[:16]
                                if type(raw_id) in (str, int)
                                and str(raw_id).strip().isascii()
                                and str(raw_id).strip().isdigit()
                                else "invalid"
                            ),
                            record_id_type=type(raw_id).__name__,
                        )
                        normalized = normalize_record_id(raw_id)
                        if requested_records and requested_records != {normalized}:
                            raise ToolArgumentError(
                                "Mã bản ghi công cụ không khớp yêu cầu hiện tại. "
                                "Không phân loại bản ghi khác thay thế."
                            )
                    result = execute_tool(name, arguments)
                    successful_tools += 1

                    if name == "classify_record":
                        last_result = result
                        argument_error = False
                        force_classify = False

                    tool_events.append({
                        "tool": name,
                        "status": "success",
                    })
                    audit(
                        run_id,
                        "tool_success",
                        tool=name,
                        record_id=result.get("record_id"),
                        mode=result.get("mode"),
                    )

                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                    httpx.HTTPError,
                ) as error:
                    if name == "classify_record":
                        last_result = None

                    argument_error = name == "classify_record" and isinstance(error, ToolArgumentError)
                    if isinstance(error, ToolArgumentError):
                        error_text = str(error)
                    elif isinstance(error, httpx.HTTPError):
                        error_text = "Không gọi được API phân loại ECG. Chưa có kết quả dự đoán."
                    else:
                        error_text = "Không xử lý được dữ liệu hoặc kết quả công cụ. Chưa có kết quả dự đoán."
                    result = {
                        "error": error_text,
                        "retryable_argument_error": argument_error,
                    }
                    tool_events.append({
                        "tool": name,
                        "status": "error",
                    })
                    audit(
                        run_id,
                        "tool_error",
                        tool=name,
                        error_type=type(error).__name__,
                        error_message=str(error) if isinstance(error, ValueError) else None,
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

    raise AgentError("Đã đạt giới hạn vòng lặp Agent.")


@router.get("/config")
def config():
    return {
        "key_configured": bool(
            os.getenv("OPENROUTER_API_KEY", "").strip()
        ),
        "requested_model": os.getenv(
            "OPENROUTER_MODEL", "openrouter/free"
        ),
        "allowed_records": [RECORD_ID],
        "record_ready": all(
            RECORD_PATH.with_suffix(f".{extension}").is_file()
            for extension in ["dat", "hea", "atr"]
        ),
    }


@router.post("/chat")
def chat(data: ChatInput, request: Request):
    check_origin(request)

    if not data.message.strip():
        raise HTTPException(status_code=422, detail="Hãy nhập yêu cầu.")

    if sum(len(message.content) for message in data.history) > 12000:
        raise HTTPException(
            status_code=422,
            detail="Lịch sử quá dài. Hãy bắt đầu cuộc trò chuyện mới.",
        )

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    if not key:
        raise HTTPException(
            status_code=503,
            detail="Chưa cấu hình API key OpenRouter trong .env.",
        )

    if not RUN_LIMIT.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Agent đang bận. Hãy đợi rồi thử lại.",
        )

    run_id = uuid.uuid4().hex
    started = time.perf_counter()

    try:
        audit(run_id, "run_started")
        result, summary, models_used, tools = run_conversation(
            data, run_id, key, model
        )

        token = None

        if result is not None:
            token = secrets.token_hex(32)
            report = {
                "run_id": run_id,
                "created_at": now(),
                "source": "web",
                "confirmed": False,
                "user_request": data.message.strip(),
                "requested_llm_model": model,
                "actual_llm_models": models_used,
                "api_result": result,
                "summary": summary,
            }

            with PENDING_LOCK:
                clean_pending()
                while len(PENDING) >= 64:
                    del PENDING[next(iter(PENDING))]
                PENDING[run_id] = {
                    "token": token,
                    "report": report,
                    "expires_at": time.monotonic() + 3600,
                    "saved_filename": None,
                }

            audit(run_id, "confirmation_requested")

        audit(run_id, "run_completed")

        return {
            "run_id": run_id,
            "summary": summary,
            "api_result": result,
            "requested_model": model,
            "actual_models": models_used,
            "tools": tools,
            "elapsed_seconds": time.perf_counter() - started,
            "save_token": token,
        }

    except Exception as error:
        try:
            audit(
                run_id,
                "run_failed",
                error_type=type(error).__name__,
            )
        except OSError:
            pass

        detail = (
            str(error)
            if isinstance(error, AgentError)
            else "Agent gặp lỗi. Kiểm tra API, dữ liệu và nhật ký."
        )
        raise HTTPException(status_code=502, detail=detail) from error

    finally:
        RUN_LIMIT.release()


@router.post("/report")
def report_decision(data: ReportDecision, request: Request):
    check_origin(request)

    if data.action == "save" and data.confirmed is not True:
        raise HTTPException(
            status_code=422,
            detail="Cần xác nhận trước khi lưu.",
        )

    with PENDING_LOCK:
        clean_pending()
        entry = PENDING.get(data.run_id)

        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="Bản xem trước đã hết hạn. Hãy gửi lại yêu cầu.",
            )

        if not hmac.compare_digest(entry["token"], data.token):
            raise HTTPException(
                status_code=403,
                detail="Không được phép thao tác với báo cáo này.",
            )

        if data.action == "discard":
            if entry["saved_filename"]:
                raise HTTPException(
                    status_code=409,
                    detail="Báo cáo đã được lưu.",
                )
            audit(data.run_id, "save_declined")
            del PENDING[data.run_id]
            return {"saved": False, "message": "Không lưu báo cáo."}

        if entry["saved_filename"]:
            return {
                "saved": True,
                "filename": entry["saved_filename"],
                "report": entry["report"],
            }

        report = {
            **entry["report"],
            "confirmed": True,
            "confirmed_at": now(),
        }
        folder = ROOT / "outputs"
        folder.mkdir(exist_ok=True)
        destination = folder / f"report_{data.run_id}.json"

        audit(data.run_id, "save_confirmed")
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        entry["report"] = report
        entry["saved_filename"] = destination.name

        audit(
            data.run_id,
            "report_saved",
            filename=destination.name,
        )

        return {
            "saved": True,
            "filename": destination.name,
            "report": report,
        }