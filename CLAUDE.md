# Study Coach

Trợ lý học tập trên tài liệu của người dùng, làm cho hackathon khóa AI Application. Bản hiện tại: **Quiz Maker (tuần 1)**. Dán văn bản hoặc upload PDF/TXT, AI sinh câu trắc nghiệm, người dùng làm bài và xem điểm.
Trả lời bằng tiếng Việt.

## Công nghệ
Python 3.12, Streamlit (giao diện), OpenAI API qua cổng AI Portal của khóa học, Pydantic (schema và structured output), pypdf (đọc PDF), python-dotenv, pytest.
Công nghệ các tuần sau (embeddings, LangChain, Pinecone/Chroma, LangGraph) chưa dùng trong code hiện tại.

## Cấu trúc
```
app.py               Giao diện Streamlit: nhập văn bản/upload file, làm bài, hiện điểm
config.py            Đọc .env, hằng số (model, giới hạn)
core/contracts.py    Kiểu dữ liệu dùng chung: Question, Quiz, GradedQuestion, QuizResult
core/llm.py          Gọi OpenAI (structured output), lỗi LLMError
core/quiz.py         generate_quiz (sinh + lọc câu hợp lệ), grade (chấm điểm)
core/loader.py       load_text: PDF (có dấu [Trang n]) và TXT
samples/             Văn bản mẫu tiếng Việt để demo
tests/               Test không cần API thật
docs/                Tài liệu phân công tuần 1
```

## Luồng chính
`app.py` đọc file qua `load_text` → `generate_quiz(text, n, difficulty, model)` cắt văn bản theo `MAX_INPUT_CHARS`, dựng prompt → `complete_structured` gọi OpenAI ép kết quả theo schema `Quiz` → lọc câu hợp lệ (đủ 4 đáp án, `correct_index` 0..3), thử lại tối đa 2 lần nếu thiếu → `grade` chấm điểm trả `QuizResult`.

## Cấu hình (`.env`)
- `OPENAI_API_KEY`: key AI Portal.
- `OPENAI_BASE_URL`: `https://aiportalapi.stu-platform.live/jpe`.
- `OPENAI_MODEL`: model mặc định `GPT-4o-mini`. Model mạnh hơn `GPT-4.1-mini` cố định trong `config.py`. Embeddings có sẵn trên cổng: `text-embedding-3-small`.
- Tên model viết đúng như trên AI Portal (có phân biệt hoa thường).
- Key có hạn mức $5/tháng; `MAX_INPUT_CHARS` = 12000 để giới hạn chi phí mỗi lần gọi.

## Lệnh
```
pip install -r requirements.txt
streamlit run app.py
pytest
```

## Quy ước kỹ thuật
- Mọi lời gọi LLM đi qua `core/llm.py`; giao diện không gọi OpenAI trực tiếp.
- Dữ liệu trao đổi giữa module dùng kiểu trong `core/contracts.py`.
- Test mock LLM, không gọi API thật.
- Không ghi API key vào mã nguồn hay log; `.env` đã nằm trong `.gitignore`.
