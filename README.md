# Study Coach 2.0 — Gia sư AI Thích ứng theo Năng lực Cá nhân

Ứng dụng học tập cá nhân hóa chạy trực tiếp trên tài liệu của người học (PDF/TXT/văn bản dán), tích hợp trọn vẹn 5 lớp công nghệ: **OpenAI Structured Output**, **Function Calling & SQLite Telemetry**, **Embeddings & FAISS**, **LangChain Dual-Memory RAG**, và **LangGraph Socratic Agent + Vision**.

## Chức năng chính
1. **Tài khoản cá nhân, Bảo mật & Bảo toàn phiên:** Đăng ký / đăng nhập `username/password` (băm `PBKDF2-SHA256` 200.000 vòng + salt 16 byte), bọc `st.form` để đăng nhập 1-click hoặc nhấn `Enter`, lưu phiên qua **Browser Cookie (`sc_session`)** ký HMAC (thanh URL sạch, F5 / mở tab mới không bị đăng xuất, tự dọn profile cũ không mật khẩu).
2. **Nạp tài liệu & Lưu trữ bền vững theo tài khoản:** Tài khoản mới bắt đầu từ màn hình nạp tài liệu trống (PDF/TXT/dán văn bản hoặc dùng thử giáo trình mẫu RAG); tự động lưu tài liệu (`learner_documents`) và lịch sử hội thoại (`chat_messages`) vào SQLite theo `(learner_id, doc_id)`, cache FAISS index bằng `@st.cache_resource`.
3. **Bản đồ Năng lực & Chẩn đoán đầu vào động (3–30 câu):** Tự rút trích `DocumentConceptMap` có quan hệ tiên quyết (`prerequisites`), sinh Placement Test linh hoạt 3–30 câu tùy độ dài & số khái niệm tài liệu (`placement_question_count`, sinh theo lô 10 câu ở chế độ Live), hỗ trợ nút **"Bỏ qua bài kiểm tra (tính 0%)"** để vào học ngay từ mức *Mới bắt đầu*.
4. **Gia sư Socratic Context-First (LangGraph):** LLM Router (`_IntentRouterOutput`) đọc ngữ cảnh 6 lượt gần nhất + `concept_map`, cấp đầy đủ đoạn nguồn FAISS RAG + `concept_map` cho mọi nhánh hội thoại ở chế độ Live, ưu tiên chủ đề nền tảng khi chào hỏi, tránh lặp bài giảng/câu hỏi khi người học đáp `"ko biết"` / `"cho ví dụ"` / `"học chủ đề khác"`, gộp trace LangGraph + đoạn nguồn FAISS vào 1 thanh tra cứu gọn, kèm nút **Luyện ngay Quiz**.
5. **Giao diện AI Learning Workspace 5 tầng (Visual Hierarchy):** Thanh điều hướng 5 tab sage/emerald (`st.segmented_control`), dải bối cảnh học tập, thẻ Chatbot hợp nhất có header `Study Coach AI · ● Trực tuyến`, Quiz & Placement Test bọc `st.form` không chớp trang khi chọn đáp án.
6. **Vision chấm bài nháp, Quiz thích ứng + Feynman Teach-Back (tối đa 100% ở cấp Bloom Phân tích), Phòng Lab Đột Phá & Dashboard Số liệu:** Nhận ảnh chụp sơ đồ/code (2 ảnh mẫu trong `samples/vision/`); quiz bám nguồn có `evidence` nguyên văn + `option_diagnostics`, đo độ tự tin (*Đoán mò / Bình thường / Chắc chắn*), chấm bài tự giảng lại (`Teach-Back`); Phòng Lab gồm *Git Diff cho Não bộ*, *Ca Trực Cứu Hỏa Production*, *Săn Lỗi Cài Bẫy*, *Hội đồng Phản biện*; Dashboard có Spaced Repetition, Radar Ảo tưởng 2×2, telemetry token/độ trễ/chi phí và bộ eval 20 câu.

## Cài đặt & Chạy

```powershell
cd SourceCode
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

- **Chế độ Live AI Portal (giao diện chính):** App đọc `OPENAI_API_KEY` từ `.env`, từ `st.secrets` (khi deploy Streamlit Cloud qua `_get_setting` trong `config.py`), hoặc nhập trực tiếp khi đăng nhập / trong popover `⚙️ Cài đặt`. Người dùng đăng nhập bằng **tài khoản thật (`username/password`)** — không còn nút chuyển *Chế độ Demo* hay chọn hồ sơ mẫu (`Minh / Lan`) trên giao diện.
- **Test suite & eval chạy offline:** `pytest` và `eval/run_eval.py` chạy mock/offline, **không cần API key** — dùng để tiết kiệm ngân sách.

Chạy ứng dụng (dùng `python -m` để tương thích Windows App Control):

```powershell
python -m streamlit run app.py
```

Chạy toàn bộ test tự động (**223 test**) và bộ đánh giá 20 câu:

```powershell
python -m pytest -q          # 223 test tự động, chạy mock/offline không cần API key
python eval/run_eval.py
python eval/run_eval.py --openai-embeddings   # đo lại với embedding OpenAI thật (cần key)
```

### Triển khai Streamlit Community Cloud
Cấu hình được đọc qua `_get_setting` trong `config.py` (ưu tiên `.env` local, sau đó `st.secrets` trên Cloud) và giao diện tùy biến qua `.streamlit/config.toml`. Trên Cloud chỉ cần khai báo `OPENAI_API_KEY` (và tùy chọn `OPENAI_BASE_URL`) trong phần *Secrets*.

Lưu ý: `eval/` chỉ đo phần **truy xuất** (nhận biết câu trong/ngoài tài liệu, trúng đúng file nguồn), không đo độ đúng câu trả lời của model. Quyết định trong/ngoài tài liệu luôn dùng điểm của embedding cục bộ để kết quả không phụ thuộc thang điểm của embedding OpenAI.

## Cấu trúc mã nguồn

| File / Thư mục | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit AI Learning Workspace 5 tầng: đăng nhập/đăng ký `username/password` bọc `st.form`, phiên qua cookie `sc_session`, nạp tài liệu & onboarding trống, các tab Học/Quiz/Coach/Lab/Analytics |
| `config.py` | Đọc `.env` hoặc `st.secrets` qua `_get_setting` (ưu tiên key runtime nhập trên UI), bảng giá ước tính, `MAX_INPUT_CHARS`, `DB_PATH`, `MAX_SESSION_COST_USD` |
| `core/contracts.py` | Hợp đồng dữ liệu Pydantic (`Quiz`, `LearnerDNA`, `CoachTurnResult`, `TeachBackEvaluation`, `VisionCritique`, `DocumentConceptMap`) |
| `core/prompts.py` | Tập trung system prompts cho Quiz, Concept Map, Feynman, Socratic Coach, Intent Router & Vision |
| `core/db.py` | SQLite: `users` (`PBKDF2-SHA256` + salt), `learners`, `concept_mastery`, `misconceptions`, `quiz_attempts`, `confidence_stats`, `learner_documents`, `chat_messages`, `llm_calls`; tự dọn profile cũ không mật khẩu |
| `core/llm.py` | Gọi OpenAI (Structured Output, Function Calling, Streaming, Vision), retry backoff & kiểm soát chi phí |
| `core/demo_provider.py` | Bộ xử lý Grounded Offline Demo bám sát chunk thật và Learner DNA (dùng cho pytest/eval, không cần key) |
| `core/loader.py` | Đọc PDF/TXT kèm `[Trang n]` / `[Mục n]`, nạp sẵn giáo trình mẫu |
| `core/quiz.py` | Sinh quiz thích ứng, lọc câu không bám nguồn, chấm điểm & bắt bệnh đáp án sai |
| `core/rag.py` | Embeddings + FAISS + LangChain Dual-Memory RAG (tài liệu + ký ức lỗi sai, hỗ trợ tiếng Việt không dấu) |
| `core/adaptive.py` | Concept Map, Placement Test động 3–30 câu (`placement_question_count`, sinh lô 10 câu), chấm Feynman Teach-Back, lập lộ trình ôn tập |
| `core/breakthrough.py` | Phòng Lab Đột Phá: Cognitive Diff, Incident Simulator, Trojan Bug-Hunt, Devil's Advocate Q&A |
| `core/coach_graph.py` | Đồ thị LangGraph `StateGraph` Context-First: LLM Router `_IntentRouterOutput` (6 lượt hội thoại), 4 nhánh sư phạm, Streaming `stream_coach_turn` & trace Tool Calling |
| `core/vision.py` | Sinh ảnh bài nháp mẫu và chấm từng bước đối chiếu tài liệu nguồn |
| `core/ui_theme.py` | Design tokens CSS, SVG vector icons (`icon_svg`) và component HTML sạch cho giao diện |
| `eval/` | Bộ 20 câu hỏi đánh giá (`questions.json`) và script `run_eval.py` (đo phần truy xuất, kèm cấu hình đối chứng và bộ giữ riêng `holdout.json` 24 câu) |
| `tests/` | **223 test tự động** (`test_quiz.py`, `test_fixes.py`, `test_adaptive_coach.py`, `test_breakthrough.py`, `test_chatbot_deep.py`, `test_chatbot_exhaustive.py`, `test_app_ui.py`, `test_core_fixes.py`, `test_ui_fixes.py`) |
