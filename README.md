# Study Coach 2.0 — Gia sư AI Thích ứng theo Năng lực Cá nhân

Ứng dụng học tập cá nhân hóa chạy trực tiếp trên tài liệu của người học, tích hợp trọn vẹn 5 lớp công nghệ: **OpenAI Structured Output**, **Function Calling & SQLite Telemetry**, **Embeddings & FAISS**, **LangChain Dual-Memory RAG**, và **LangGraph Socratic Agent + Vision**.

## Chức năng chính
1. **Tài khoản và phiên đăng nhập:** Đăng ký / đăng nhập `username/password` (băm `PBKDF2-SHA256` 200.000 vòng + salt 16 byte), form đăng nhập 1-click hoặc nhấn `Enter`, phiên lưu qua cookie `sc_session` ký HMAC bằng khóa bí mật `SESSION_SECRET` (URL sạch, F5 hoặc mở tab mới không bị đăng xuất).
2. **Nạp và quản lý tài liệu:** Nhận PDF, TXT, MD, DOCX, PPTX hoặc văn bản dán; báo rõ khi PDF là bản scan. Tài liệu lưu theo tài khoản (`learner_documents`), đổi tài liệu ngay trên thanh bên, có trang **Tài liệu** để đọc toàn bộ nội dung kèm mục lục và tìm kiếm.
3. **Bản đồ năng lực và chẩn đoán đầu vào:** Tự rút `DocumentConceptMap` có quan hệ tiên quyết, sinh bài chẩn đoán 3–30 câu theo thang Bloom tùy độ dài tài liệu, có nút "Bỏ qua bài kiểm tra (tính 0%)".
4. **Gia sư AI (LangGraph):** Bộ định tuyến ý định đọc 6 lượt hội thoại gần nhất và bản đồ khái niệm rồi rẽ 4 nhánh sư phạm. Câu trả lời có **trích dẫn bấm được** mở đúng đoạn gốc được tô sáng, từ chối câu hỏi ngoài tài liệu. Hỗ trợ **nhiều cuộc trò chuyện** cho một tài liệu, phản hồi 👍/👎, "Trả lời lại", "Giải thích đơn giản hơn", và chấm ảnh bài nháp bằng Vision.
5. **Luyện tập:** Quiz thích ứng theo điểm yếu, **làm từng câu** có thanh tiến độ; ba dạng câu hỏi (trắc nghiệm, điền khuyết, trả lời ngắn do LLM chấm); mỗi đáp án sai gắn một lỗi tư duy; thang tự tin (*Đoán mò / Bình thường / Chắc chắn*); vấn đáp ngược Feynman.
6. **Ôn hôm nay:** Gom các khái niệm đến hạn ôn và lỗi tư duy đang mở thành một phiên ôn ngắn (tối đa 8 câu), kèm chuỗi ngày học liên tục.
7. **Phòng Lab:** Git Diff cho Não bộ (đối chiếu ghi chú với giáo trình), Ca trực sự cố, Săn lỗi cài bẫy, Hội đồng phản biện.
8. **Lộ trình & Số liệu:** Biểu đồ tiến bộ theo thời gian, mức thành thạo theo khái niệm, radar ảo tưởng hiểu bài, lộ trình ôn tập ưu tiên, sổ tay lỗi tư duy (tải Markdown, flashcard CSV/Markdown), telemetry token/độ trễ/chi phí và bộ eval 20 câu.
9. **Giao diện:** Thanh điều hướng dọc 7 trang, khung chat dạng bong bóng cao hết màn hình với chỉ báo "đang suy nghĩ", toàn bộ CSS và design tokens nằm trong `core/ui_theme.css`.

## Cài đặt & Chạy

```powershell
cd SourceCode
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

- **Cấu hình:** App đọc `OPENAI_API_KEY` và `OPENAI_BASE_URL` từ `.env`, hoặc từ `st.secrets` khi deploy Streamlit Cloud (qua `_get_setting` trong `config.py`). Giao diện **không có ô nhập API key hay Base URL**: mọi cấu hình kết nối chỉ lấy từ môi trường.
- **Khóa ký phiên:** đặt `SESSION_SECRET` trong `.env` khi triển khai; để trống thì app tự sinh một khóa ngẫu nhiên và lưu ở `data/.session_secret`.
- **Test suite & eval chạy offline:** `pytest` và `eval/run_eval.py` chạy mock/offline, **không cần API key**.

Chạy ứng dụng (dùng `python -m` để tương thích Windows App Control):

```powershell
python -m streamlit run app.py
```

Chạy toàn bộ test tự động (**249 test**) và bộ đánh giá 20 câu:

```powershell
python -m pytest -q          # 249 test tự động, chạy mock/offline không cần API key
python eval/run_eval.py
python eval/run_eval.py --openai-embeddings   # đo lại với embedding OpenAI thật (cần key)
```

### Triển khai Streamlit Community Cloud
Cấu hình được đọc qua `_get_setting` trong `config.py` (ưu tiên biến môi trường, sau đó `st.secrets`) và màu giao diện qua `.streamlit/config.toml`. Trên Cloud khai báo `OPENAI_API_KEY`, `SESSION_SECRET` (và tùy chọn `OPENAI_BASE_URL`) trong phần *Secrets*.

Lưu ý: `eval/` chỉ đo phần **truy xuất** (nhận biết câu trong/ngoài tài liệu, trúng đúng file nguồn), không đo độ đúng câu trả lời của model. Quyết định trong/ngoài tài liệu luôn dùng điểm của embedding cục bộ để kết quả không phụ thuộc thang điểm của embedding OpenAI.

## Cấu trúc mã nguồn

| File / Thư mục | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit: đăng nhập/đăng ký, thanh điều hướng dọc (`_render_sidebar`), 7 trang (Gia sư AI, Ôn hôm nay, Bản đồ năng lực, Luyện tập, Phòng Lab, Tài liệu, Lộ trình & Số liệu) |
| `config.py` | Đọc `.env` hoặc `st.secrets` qua `_get_setting`, bảng giá ước tính, `MAX_INPUT_CHARS`, `DB_PATH`, `MAX_SESSION_COST_USD`, khóa ký phiên (`get_session_secret`) |
| `core/contracts.py` | Hợp đồng dữ liệu Pydantic (`Quiz`, `LearnerDNA`, `CoachTurnResult`, `TeachBackEvaluation`, `VisionCritique`, `DocumentConceptMap`) |
| `core/prompts.py` | Tập trung system prompts cho Quiz, Concept Map, Feynman, Socratic Coach, Intent Router, chấm câu mở & Vision |
| `core/db.py` | SQLite 11 bảng: `users`, `learners`, `concept_mastery`, `misconceptions`, `quiz_attempts`, `confidence_stats`, `learner_documents`, `conversations`, `chat_messages`, `mastery_history`, `llm_calls` |
| `core/llm.py` | Gọi OpenAI (Structured Output, Function Calling, Streaming, Vision), retry backoff & kiểm soát chi phí |
| `core/demo_provider.py` | Bộ xử lý offline bám sát chunk thật và Learner DNA (dùng cho pytest/eval, không cần key) |
| `core/loader.py` | Đọc PDF/TXT/MD/DOCX/PPTX kèm `[Trang n]` / `[Mục n]`, nạp sẵn giáo trình mẫu |
| `core/quiz.py` | Sinh quiz thích ứng, lọc câu không bám nguồn, chấm trắc nghiệm và câu mở (`grade_open_answers`), bắt bệnh đáp án sai |
| `core/rag.py` | Embeddings + FAISS + LangChain Dual-Memory RAG (tài liệu + ký ức lỗi sai, hỗ trợ tiếng Việt không dấu) |
| `core/adaptive.py` | Concept Map, bài chẩn đoán đầu vào, chấm Feynman Teach-Back, lộ trình ôn tập, phiên Ôn hôm nay (`build_review_session`) |
| `core/breakthrough.py` | Phòng Lab: Cognitive Diff, Incident Simulator, Trojan Bug-Hunt, Devil's Advocate Q&A |
| `core/coach_graph.py` | Đồ thị LangGraph `StateGraph`: LLM Router `_IntentRouterOutput`, 4 nhánh sư phạm, streaming `stream_coach_turn` & trace tool calling |
| `core/vision.py` | Sinh ảnh bài nháp mẫu và chấm từng bước đối chiếu tài liệu nguồn |
| `core/export.py` | Xuất sổ tay lỗi tư duy, flashcard và ghi chú đã hiệu đính ra Markdown/CSV |
| `core/ui_theme.py` | Icon SVG và các hàm dựng HTML cho giao diện (`render_*_html`) |
| `core/ui_theme.css` | Design tokens (màu, bo góc, bóng) và toàn bộ CSS giao diện |
| `eval/` | Bộ 20 câu hỏi đánh giá (`questions.json`) và script `run_eval.py` (đo phần truy xuất, kèm cấu hình đối chứng và bộ giữ riêng `holdout.json` 24 câu) |
| `tests/` | **249 test tự động** trong 10 file (`test_quiz.py`, `test_fixes.py`, `test_adaptive_coach.py`, `test_breakthrough.py`, `test_chatbot_deep.py`, `test_chatbot_exhaustive.py`, `test_app_ui.py`, `test_core_fixes.py`, `test_ui_fixes.py`, `test_features.py`) |
