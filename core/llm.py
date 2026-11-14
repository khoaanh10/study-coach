"""Lớp gọi LLM duy nhất của app. Tuần 1 chỉ dùng OpenAI API.

Mọi nơi khác gọi qua đây, để sau này dễ thêm model khác (ví dụ Llama 3 qua API host).
"""
from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

import config

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Lỗi khi gọi LLM, thông báo đã viết sẵn để hiện cho người dùng."""


def get_client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise LLMError(
            "Chưa có OPENAI_API_KEY. Tạo file .env từ .env.example và điền key vào."
        )
    # OPENAI_BASE_URL để trống = dùng api.openai.com; điền nếu khóa học cấp cổng API riêng.
    return OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None)


def complete_structured(
    messages: list[dict],
    schema: type[T],
    model: str | None = None,
    temperature: float = 0.4,
) -> T:
    """Gọi OpenAI và ép kết quả theo schema Pydantic (structured outputs)."""
    client = get_client()
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    try:
        resp = parse(
            model=model or config.DEFAULT_MODEL,
            messages=messages,
            response_format=schema,
            temperature=temperature,
        )
    except OpenAIError as e:
        raise LLMError(f"Gọi OpenAI thất bại: {e}") from e

    message = resp.choices[0].message
    if getattr(message, "refusal", None):
        raise LLMError(f"Model từ chối trả lời: {message.refusal}")
    if message.parsed is None:
        raise LLMError("Model không trả về dữ liệu đúng định dạng.")
    return message.parsed
