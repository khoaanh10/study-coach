"""Cấu hình chung. Mọi giá trị đọc từ biến môi trường / file .env hoặc runtime UI."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get_setting(name: str, default: str = "") -> str:
    val = os.getenv(name, "").strip()
    if val:
        return val
    try:
        import streamlit as st

        sec = st.secrets.get(name, "")
        if sec is not None and str(sec).strip():
            return str(sec).strip()
    except Exception:
        pass
    return default


OPENAI_API_KEY = _get_setting("OPENAI_API_KEY")
OPENAI_BASE_URL = _get_setting("OPENAI_BASE_URL")
DEFAULT_MODEL = _get_setting("OPENAI_MODEL", "GPT-4o-mini") or "GPT-4o-mini"

# Model mạnh hơn, giá vẫn vừa phải (khoảng gấp 2-3 lần gpt-4o-mini). Chỉ dùng khi cần chất lượng cao.
STRONG_MODEL = "GPT-4.1-mini"
EMBEDDING_MODEL = _get_setting("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"

# Các model cho người dùng chọn trong giao diện.
MODEL_CHOICES = list(dict.fromkeys([DEFAULT_MODEL, STRONG_MODEL]))

# Giới hạn độ dài văn bản đưa vào prompt (ký tự) để kiểm soát chi phí.
MAX_INPUT_CHARS = 12000

DIFFICULTIES = ["Dễ", "Trung bình", "Khó"]
MIN_QUESTIONS, MAX_QUESTIONS = 3, 10

BLOOM_LEVELS = ["Nhớ", "Hiểu", "Vận dụng", "Phân tích"]
LEARNER_LEVELS = ["Mới bắt đầu", "Trung bình", "Nâng cao"]

DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "study_coach.db"))
MAX_SESSION_COST_USD = float(os.getenv("MAX_SESSION_COST_USD", "1.0"))

# Giá ước tính (USD / 1M token) theo bảng giá tham khảo cho việc theo dõi chi phí.
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "GPT-4o-mini": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "GPT-4.1-mini": (0.40, 1.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
}

_RUNTIME_API_KEY: str | None = None


def set_runtime_api_key(key: str | None) -> None:
    """Cho phép người dùng nhập API key tạm thời ngay trên giao diện mà không ghi xuống đĩa."""
    global _RUNTIME_API_KEY
    cleaned = (key or "").strip()
    _RUNTIME_API_KEY = cleaned or None


def get_effective_api_key() -> str:
    """Ưu tiên key người dùng nhập ở phiên UI, nếu không có thì đọc OPENAI_API_KEY."""
    if _RUNTIME_API_KEY is not None:
        return _RUNTIME_API_KEY
    return OPENAI_API_KEY.strip()


def get_effective_base_url() -> str:
    """Base URL chỉ lấy từ môi trường (.env / biến môi trường), không cho đổi từ giao diện:
    cấu hình là biến toàn tiến trình nên đổi ở một phiên sẽ kéo request của mọi phiên (kèm API key) sang URL đó."""
    return OPENAI_BASE_URL.strip()


_SESSION_SECRET_CACHE: dict[str, bytes] = {}


def get_session_secret() -> bytes:
    """Khóa ký cookie phiên đăng nhập.

    Ưu tiên biến môi trường SESSION_SECRET. Nếu chưa đặt, sinh ngẫu nhiên một khóa và lưu vào
    file `.session_secret` cạnh file SQLite (thư mục data/ đã nằm trong .gitignore) để phiên
    đăng nhập vẫn hợp lệ sau khi khởi động lại app.
    """
    env_val = _get_setting("SESSION_SECRET")
    if env_val:
        return env_val.encode("utf-8")
    secret_path = Path(DB_PATH).resolve().parent / ".session_secret"
    cache_key = str(secret_path)
    cached = _SESSION_SECRET_CACHE.get(cache_key)
    if cached:
        return cached
    secret = b""
    try:
        if secret_path.exists():
            secret = secret_path.read_text(encoding="utf-8").strip().encode("utf-8")
        if len(secret) < 32:
            import secrets as _secrets

            secret = _secrets.token_hex(32).encode("utf-8")
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            secret_path.write_text(secret.decode("utf-8"), encoding="utf-8")
    except OSError:
        # Không ghi được đĩa: vẫn dùng khóa ngẫu nhiên trong bộ nhớ (cookie hết hiệu lực khi khởi động lại).
        if len(secret) < 32:
            import secrets as _secrets

            secret = _secrets.token_hex(32).encode("utf-8")
    _SESSION_SECRET_CACHE[cache_key] = secret
    return secret


def has_live_api_key() -> bool:
    return bool(get_effective_api_key())


def estimate_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    in_price, out_price = MODEL_PRICING_PER_1M.get(model, (0.20, 0.80))
    return round((tokens_in * in_price + tokens_out * out_price) / 1_000_000, 6)
