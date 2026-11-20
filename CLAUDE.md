# Study Coach 2.0

Gia sư AI thích ứng trên chính tài liệu của người học, phát triển cho hackathon khóa AI Application Engineer.
Trả lời bằng tiếng Việt.

## Công nghệ (đủ 5 lớp giáo trình)
Python 3.12, Streamlit, OpenAI API (AI Portal: `GPT-4o-mini`, `GPT-4.1-mini`, `text-embedding-3-small`), Pydantic, LangChain (`langchain-core`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`), FAISS (`faiss-cpu`), LangGraph (`langgraph`), SQLite (`sqlite3`), Pillow, pypdf, pytest.

## Cấu trúc chính
```
app.py                   Giao diện Streamlit 4 không gian làm việc
config.py                Đọc .env, runtime key, bảng giá token, hằng số
core/contracts.py        Schema Pydantic dùng chung toàn app
core/prompts.py          Toàn bộ system prompt chuẩn hóa
core/db.py               SQLite: Learner DNA, lỗi tư duy, lịch sử quiz, telemetry llm_calls
core/llm.py              Lớp gọi LLM duy nhất (structured, tools, stream, vision, retry)
core/demo_provider.py    Xử lý Grounded Offline Demo bám chunk thật khi chưa có API key
core/loader.py           Đọc PDF/TXT giữ metadata [Trang n] / [Mục n], nạp sẵn samples/
core/quiz.py             generate_quiz, generate_adaptive_quiz, grade + option_diagnostics
core/rag.py              Embeddings + FAISS Dual-Memory RAG (tài liệu + lỗi sai)
core/adaptive.py         Concept Map, Placement Test, Feynman Teach-Back, Study Plan
core/breakthrough.py     Phòng Lab: Cognitive Diff, Incident Simulator, Trojan Bug-Hunt, Devil's Advocate
core/coach_graph.py      LangGraph StateGraph 4 nhánh sư phạm + Tool Calling traces
core/vision.py           Chấm ảnh bài nháp / code đối chiếu tài liệu nguồn
core/ui_theme.py         Design tokens UI/UX Pro Max, SVG vector icons, Bento/Diff/Radar HTML
eval/                    questions.json (20 câu) + run_eval.py
samples/                 Văn bản mẫu + 2 ảnh mẫu trong samples/vision/
tests/                   34 test tự động (pytest + AppTest), chạy không cần API key
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
