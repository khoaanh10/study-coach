# Study Coach - Tuần 1: Quiz Maker

Dán văn bản hoặc upload PDF/TXT, AI sinh câu trắc nghiệm (structured output), làm bài và chấm điểm.
Công nghệ tuần 1: Python, OpenAI API, Pydantic, Streamlit.

## Chạy

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Mở file `.env`, điền `OPENAI_API_KEY`. Rồi chạy:

```bash
streamlit run app.py
```

Chạy test (không cần API key):

```bash
pytest
```

## Demo nhanh

Trong thư mục `samples/` có 2 file mẫu tiếng Việt (RAG cơ bản, OpenAI API cơ bản). Trên app, vào tab "Upload PDF/TXT" và chọn một file, hoặc mở file rồi dán nội dung vào tab "Dán văn bản".

## Cấu trúc

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit |
| `config.py` | Đọc `.env`, hằng số |
| `core/contracts.py` | Schema dữ liệu dùng chung (Question, Quiz, QuizResult) |
| `core/llm.py` | Gọi OpenAI duy nhất tại đây (sau này dễ thêm model khác) |
| `core/quiz.py` | Sinh quiz và chấm điểm |
| `core/loader.py` | Đọc PDF/TXT |
| `tests/` | Test chấm điểm, không gọi API |

## Lưu ý
- Không commit file `.env`. Đặt hạn mức chi tiêu trên dashboard OpenAI.
- Văn bản dài bị cắt ở `MAX_INPUT_CHARS` (xem `config.py`) để kiểm soát chi phí.
