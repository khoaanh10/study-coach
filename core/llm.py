"""Lớp gọi LLM duy nhất của app. Tuần 1 chỉ dùng OpenAI API.

Mọi nơi khác gọi qua đây, để sau này dễ thêm model khác (ví dụ Llama 3 qua API host).
"""
from typing import TypeVar

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
        raise LLMError(friendly_error(e)) from e

    message = resp.choices[0].message
    if getattr(message, "refusal", None):
        raise LLMError("Model từ chối xử lý nội dung này. Thử với văn bản khác.")
    if message.parsed is None:
        raise LLMError("Model không trả về dữ liệu đúng định dạng.")
    return message.parsed
