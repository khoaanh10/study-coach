# Study Coach 2.0

Gia sư AI thích ứng trên chính tài liệu của người học, phát triển cho hackathon khóa AI Application Engineer.
Trả lời bằng tiếng Việt.

## Công nghệ (đủ 5 lớp giáo trình)
Python 3.12, Streamlit, OpenAI API (AI Portal: `GPT-4o-mini`, `GPT-4.1-mini`, `text-embedding-3-small`), Pydantic, LangChain (`langchain-core`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`), FAISS (`faiss-cpu`), LangGraph (`langgraph`), SQLite (`sqlite3`), Pillow, pypdf, pytest.

## Cấu trúc chính
```
app.py                   Giao diện Streamlit: thanh điều hướng dọc + 7 trang
config.py                Đọc .env / st.secrets, khóa ký phiên, bảng giá token, hằng số
core/contracts.py        Schema Pydantic dùng chung toàn app
core/prompts.py          Toàn bộ system prompt chuẩn hóa
core/db.py               SQLite 11 bảng: tài khoản, Learner DNA, lỗi tư duy, hội thoại, lịch sử mastery, telemetry
core/llm.py              Lớp gọi LLM duy nhất (structured, tools, stream, vision, retry)
core/demo_provider.py    Xử lý Grounded Offline Demo bám chunk thật khi chưa có API key
core/loader.py           Đọc PDF/TXT/MD/DOCX/PPTX giữ metadata [Trang n] / [Mục n], nạp sẵn samples/
core/quiz.py             generate_quiz, generate_adaptive_quiz, grade, chấm câu mở (grade_open_answers)
core/rag.py              Embeddings + FAISS Dual-Memory RAG (tài liệu + lỗi sai)
core/adaptive.py         Concept Map, Placement Test, Feynman Teach-Back, Study Plan, phiên Ôn hôm nay
core/breakthrough.py     Phòng Lab: Cognitive Diff, Incident Simulator, Trojan Bug-Hunt, Devil's Advocate
core/coach_graph.py      LangGraph StateGraph 4 nhánh sư phạm + Tool Calling traces
core/vision.py           Chấm ảnh bài nháp / code đối chiếu tài liệu nguồn
core/export.py           Xuất sổ tay lỗi tư duy, flashcard, ghi chú đã hiệu đính (Markdown/CSV)
core/ui_theme.py         SVG icons + các hàm render HTML (KPI, thẻ khái niệm, chat, Diff/Radar)
core/ui_theme.css        Design tokens (màu, bo góc, bóng) và toàn bộ CSS giao diện
eval/                    questions.json (20 câu) + run_eval.py
samples/                 Văn bản mẫu + 2 ảnh mẫu trong samples/vision/
tests/                   249 test tự động (pytest + AppTest), chạy không cần API key
```

## Lệnh chuẩn (Windows dùng python -m)
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m pytest -v
python eval/run_eval.py
```

## Quy ước kỹ thuật
- Mọi lời gọi LLM đi qua `core/llm.py`; giữ nguyên tương thích ngược cho `generate_quiz` và `grade`.
- Dữ liệu trao đổi giữa module dùng kiểu Pydantic trong `core/contracts.py`.
- Test và CI không gọi API thật (`pytest` mock LLM / dùng `demo_mode=True`).
- Không commit `.env`, `data/`, hay `*.db` (đã khai báo trong `.gitignore`).
- API key và Base URL chỉ lấy từ `.env` / `st.secrets`; không thêm ô nhập key hay Base URL lên giao diện.
- CSS và màu sắc sửa trong `core/ui_theme.css`; card là `st.container(border=True, key="card_<tên>")`.
