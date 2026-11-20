"""Lớp gọi LLM duy nhất của app. Hỗ trợ Structured Output, Function Calling, Streaming, Vision, Retry và Telemetry."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Callable, Iterator, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel

import config
from core import db
from core.contracts import ToolCallTrace

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Lỗi khi gọi LLM, thông báo đã viết sẵn để hiện cho người dùng."""


def friendly_error(e: Exception) -> str:
    """Đổi lỗi của OpenAI thành thông báo tiếng Việt, không lộ chi tiết kỹ thuật."""
    if isinstance(e, AuthenticationError):
        return "API key không hợp lệ hoặc đã hết hạn. Kiểm tra OPENAI_API_KEY trong file .env."
    if isinstance(e, RateLimitError):
        return "Đã vượt giới hạn gọi API hoặc hết ngân sách của key. Chờ một lúc rồi thử lại."
    if isinstance(e, APITimeoutError):
        return "Máy chủ AI phản hồi quá lâu. Thử lại, hoặc giảm số câu và độ dài văn bản."
    if isinstance(e, APIConnectionError):
        return "Không kết nối được tới máy chủ AI. Kiểm tra mạng và OPENAI_BASE_URL trong file .env."
    if isinstance(e, APIStatusError):
        return f"Máy chủ AI báo lỗi (mã {e.status_code}). Thử lại sau, hoặc đổi model."
    return "Gọi máy chủ AI thất bại. Thử lại sau."


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(e, APIStatusError):
        status = getattr(e, "status_code", 0) or 0
        return status >= 500
    return False


def _call_with_retry(fn: Callable[[], object], max_retries: int = 2) -> tuple[object, int]:
    """Thực thi lời gọi API với retry backoff có kiểm soát cho lỗi tạm thời."""
    delays = (0.8, 1.6)
    last_err: Exception | None = None
    start = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            res = fn()
            latency_ms = int((time.perf_counter() - start) * 1000)
            return res, latency_ms
        except OpenAIError as e:
            last_err = e
            if attempt < max_retries and _is_retryable(e):
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            raise LLMError(friendly_error(e)) from e
    raise LLMError(friendly_error(last_err or Exception("Unknown error")))


def _check_budget(db_path: str | Path | None = None) -> None:
    try:
        spent = db.get_session_cost_usd(db_path=db_path)
    except Exception:
        return
    if spent >= config.MAX_SESSION_COST_USD:
        raise LLMError(
            f"Đã chạm ngưỡng ngân sách phiên (${spent:.3f} / ${config.MAX_SESSION_COST_USD:.2f}). "
            "Chuyển sang Chế độ Demo Offline hoặc tăng MAX_SESSION_COST_USD để tiếp tục."
        )


def _record_usage(
    resp: object,
    model_name: str,
    purpose: str,
    latency_ms: int,
    db_path: str | Path | None = None,
) -> tuple[int, int, float]:
    usage = getattr(resp, "usage", None)
    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = config.estimate_cost_usd(model_name, tokens_in, tokens_out)
    try:
        db.log_llm_call(
            model=model_name,
            purpose=purpose,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
            is_demo=False,
            db_path=db_path,
        )
    except Exception:
        pass
    return tokens_in, tokens_out, cost


def get_client() -> OpenAI:
    api_key = config.get_effective_api_key()
    if not api_key:
        raise LLMError(
            "Chưa có OPENAI_API_KEY. Tạo file .env từ .env.example và điền key vào."
        )
    base_url = config.get_effective_base_url()
    return OpenAI(api_key=api_key, base_url=base_url or None)


def complete_structured(
    messages: list[dict],
    schema: type[T],
    model: str | None = None,
    temperature: float = 0.4,
    purpose: str = "structured_output",
    db_path: str | Path | None = None,
) -> T:
    """Gọi OpenAI và ép kết quả theo schema Pydantic (structured outputs)."""
    client = get_client()
    _check_budget(db_path)
    target_model = model or config.DEFAULT_MODEL
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse

    resp, latency_ms = _call_with_retry(
        lambda: parse(
            model=target_model,
            messages=messages,
            response_format=schema,
            temperature=temperature,
        )
    )
    _record_usage(resp, target_model, purpose, latency_ms, db_path=db_path)

    message = resp.choices[0].message
    if getattr(message, "refusal", None):
        raise LLMError("Model từ chối xử lý nội dung này. Thử với văn bản khác.")
    if message.parsed is None:
        raise LLMError("Model không trả về dữ liệu đúng định dạng.")
    return message.parsed


def complete_with_tools(
    messages: list[dict],
    tools_schema: list[dict],
    tool_dispatch: dict[str, Callable[..., str]],
    model: str | None = None,
    temperature: float = 0.3,
    max_rounds: int = 4,
    purpose: str = "tool_calling",
    db_path: str | Path | None = None,
) -> tuple[str, list[ToolCallTrace]]:
    """Thực thi vòng lặp OpenAI Function Calling chuẩn và ghi nhận trace các tool đã gọi."""
    client = get_client()
    _check_budget(db_path)
    target_model = model or config.DEFAULT_MODEL
    history = list(messages)
    traces: list[ToolCallTrace] = []

    for _ in range(max_rounds):
        resp, latency_ms = _call_with_retry(
            lambda: client.chat.completions.create(
                model=target_model,
                messages=history,
                tools=tools_schema,
                tool_choice="auto",
                temperature=temperature,
            )
        )
        _record_usage(resp, target_model, purpose, latency_ms, db_path=db_path)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return (msg.content or "").strip(), traces

        assistant_msg: dict = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        }
        history.append(assistant_msg)

        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            fn = tool_dispatch.get(fn_name)
            if fn is None:
                output = f"Tool '{fn_name}' không tồn tại."
            else:
                try:
                    output = str(fn(**args))
                except Exception as e:  # noqa: BLE001 - lỗi tool trả cho model như output, không làm sập phiên
                    output = f"Lỗi khi chạy tool '{fn_name}': {type(e).__name__}: {e}"
            traces.append(
                ToolCallTrace(
                    tool_name=fn_name,
                    arguments=args,
                    summary=output[:180].replace("\n", " "),
                )
            )
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )

    final_resp, latency_ms = _call_with_retry(
        lambda: client.chat.completions.create(
            model=target_model,
            messages=history,
            temperature=temperature,
        )
    )
    _record_usage(final_resp, target_model, purpose, latency_ms, db_path=db_path)
    return (final_resp.choices[0].message.content or "").strip(), traces


def stream_chat_completion(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.3,
    purpose: str = "chat_stream",
    db_path: str | Path | None = None,
) -> Iterator[str]:
    """Stream từng cụm văn bản phản hồi từ OpenAI để hiển thị mượt trên UI."""
    client = get_client()
    _check_budget(db_path)
    target_model = model or config.DEFAULT_MODEL
    start = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        chunks_len = 0
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta else None
            if text:
                chunks_len += len(text)
                yield text
        latency_ms = int((time.perf_counter() - start) * 1000)
        est_in = sum(len(str(m.get("content", ""))) for m in messages) // 4
        est_out = max(1, chunks_len // 4)
        try:
            db.log_llm_call(
                model=target_model,
                purpose=purpose,
                tokens_in=est_in,
                tokens_out=est_out,
                latency_ms=latency_ms,
                is_demo=False,
                db_path=db_path,
            )
        except Exception:
            pass
    except OpenAIError as e:
        raise LLMError(friendly_error(e)) from e


def analyze_image_structured(
    image_bytes: bytes,
    mime_type: str,
    user_prompt: str,
    system_prompt: str,
    schema: type[T],
    model: str | None = None,
    purpose: str = "vision_critique",
    db_path: str | Path | None = None,
) -> T:
    """Gọi model Vision (GPT-4o-mini / GPT-4.1-mini) phân tích ảnh bài làm theo schema Pydantic."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        },
    ]
    return complete_structured(
        messages=messages,
        schema=schema,
        model=model or config.DEFAULT_MODEL,
        temperature=0.2,
        purpose=purpose,
        db_path=db_path,
    )
