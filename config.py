"""Cấu hình chung. Mọi giá trị đọc từ biến môi trường / file .env."""
import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "GPT-4o-mini").strip() or "GPT-4o-mini"

# Model mạnh hơn, giá vẫn vừa phải (khoảng gấp 2-3 lần gpt-4o-mini). Chỉ dùng khi cần chất lượng cao.
STRONG_MODEL = "GPT-4.1-mini"

# Các model cho người dùng chọn trong giao diện (tuần 2 sẽ so sánh model).
MODEL_CHOICES = list(dict.fromkeys([DEFAULT_MODEL, STRONG_MODEL]))

# Giới hạn độ dài văn bản đưa vào prompt (ký tự) để kiểm soát chi phí.
MAX_INPUT_CHARS = 12000

DIFFICULTIES = ["Dễ", "Trung bình", "Khó"]
MIN_QUESTIONS, MAX_QUESTIONS = 3, 10
