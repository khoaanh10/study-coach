# Study Coach 2.0 — Gia sư AI Thích ứng theo Năng lực Cá nhân

Ứng dụng học tập cá nhân hóa chạy trực tiếp trên tài liệu của người học (PDF/TXT/văn bản dán), tích hợp trọn vẹn 5 lớp công nghệ: **OpenAI Structured Output**, **Function Calling & SQLite Telemetry**, **Embeddings & FAISS**, **LangChain Dual-Memory RAG**, và **LangGraph Socratic Agent + Vision**.

## Chức năng chính
1. **Bản đồ Năng lực & Chẩn đoán 60s:** Tự rút trích khái niệm cốt lõi kèm quan hệ tiên quyết, bài test đầu vào 3 câu leo thang Bloom, và gợi ý bước học ưu tiên tiếp theo.
2. **Gia sư Socratic (LangGraph) & Đối chiếu A/B:** Agent tự đọc `LearnerDNA` và ký ức lỗi sai trong SQLite/FAISS để rẽ 4 nhánh sư phạm (*từ chối ngoài tài liệu*, *gỡ rối lỗi tư duy Socratic*, *vá lỗ hổng nền tảng*, *giảng thích ứng theo trình độ Mới bắt đầu / Trung bình / Nâng cao*).
3. **Vision chấm bài nháp từng bước:** Nhận ảnh chụp sơ đồ/code học viên (kèm 2 ảnh mẫu trong `samples/vision/`), đối chiếu tài liệu nguồn `[Trang/Mục]` và gợi ý sửa vừa tầm.
4. **Quiz Thích ứng, Vấn đáp Feynman & Radar Điểm mù:** Sinh đề bám nguồn có `evidence` nguyên văn, chẩn đoán lỗi tư duy trên từng đáp án sai, đo độ tự tin (*Đoán mò / Bình thường / Chắc chắn*) để phát hiện *Điểm mù tự tin sai*; chấm bài tự giảng lại (`Teach-Back`).
5. **Phòng Lab Đột Phá (`core/breakthrough.py`):** *Git Diff cho Não bộ* (soi mâu thuẫn giữa ghi chú cá nhân và giáo trình gốc), *Ca Trực Cứu Hỏa Production* (Ticket sự cố + 3 tool điều tra), *Săn Lỗi Cài Bẫy* và *Hội đồng Phản biện Q&A* (hỗ trợ nạp `Documents/INFRA.md` luyện bảo vệ đồ án).
6. **Lộ trình Ôn tập & Dashboard Số liệu:** Lập lịch ôn Spaced Repetition, Radar Ảo tưởng Hiểu bài 2×2, sổ tay lỗi tư duy, telemetry token/độ trễ/chi phí và bộ đánh giá 20 câu (`eval/questions.json`).

## Cài đặt & Chạy

```powershell
cd SourceCode
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
copy .env.example .env
```

- **Chế độ Live AI Portal:** Mở `.env` điền `OPENAI_API_KEY` (hoặc nhập trực tiếp trong popover *Cấu hình AI & Key* trên giao diện).
- **Chế độ Demo Offline (không cần key):** App tự chạy dựa trên các đoạn nguồn FAISS và hồ sơ SQLite thật khi chưa có API key.

Chạy ứng dụng (dùng `python -m` để tương thích Windows App Control):

```powershell
python -m streamlit run app.py
```

Chạy toàn bộ test tự động và bộ đánh giá 20 câu:

```powershell
python -m pytest -v
python eval/run_eval.py
python eval/run_eval.py --openai-embeddings   # đo lại với embedding OpenAI thật (cần key)
```

Lưu ý: `eval/` chỉ đo phần **truy xuất** (nhận biết câu trong/ngoài tài liệu, trúng đúng file nguồn), không đo độ đúng câu trả lời của model. Quyết định trong/ngoài tài liệu luôn dùng điểm của embedding cục bộ để kết quả không phụ thuộc thang điểm của embedding OpenAI.

## Cấu trúc mã nguồn

| File / Thư mục | Vai trò |
|---|---|
| `app.py` | Giao diện Streamlit 5 không gian làm việc chuẩn UI/UX Pro Max |
| `config.py` | Đọc `.env`, runtime API key, bảng giá ước tính & hằng số |
| `core/contracts.py` | Hợp đồng dữ liệu Pydantic (`Quiz`, `LearnerDNA`, `CoachTurnResult`, `TeachBackEvaluation`, `VisionCritique`) |
| `core/prompts.py` | Tập trung system prompts cho Quiz, Concept Map, Feynman, Socratic Coach & Vision |
| `core/db.py` | Lưu trữ SQLite: hồ sơ người học mẫu, `concept_mastery`, `misconceptions`, `quiz_attempts`, `llm_calls` |
| `core/llm.py` | Gọi OpenAI (Structured Output, Function Calling, Streaming, Vision), retry backoff & kiểm soát chi phí |
| `core/demo_provider.py` | Bộ xử lý Grounded Offline Demo bám sát chunk thật và Learner DNA |
| `core/loader.py` | Đọc PDF/TXT kèm `[Trang n]` / `[Mục n]`, nạp sẵn giáo trình mẫu |
| `core/quiz.py` | Sinh quiz thích ứng, lọc câu không bám nguồn, chấm điểm & bắt bệnh đáp án sai |
| `core/rag.py` | Embeddings + FAISS + LangChain Dual-Memory RAG (tài liệu + ký ức lỗi sai, hỗ trợ tiếng Việt không dấu) |
| `core/adaptive.py` | Concept Map, Placement Test 3 câu, chấm Feynman Teach-Back, lập lộ trình ôn tập |
| `core/breakthrough.py` | Phòng Lab Đột Phá: Cognitive Diff, Incident Simulator, Trojan Bug-Hunt, Devil's Advocate Q&A |
| `core/coach_graph.py` | Đồ thị LangGraph `StateGraph` (hỗ trợ đa lượt + Streaming `stream_coach_turn`) & ghi trace Tool Calling |
| `core/vision.py` | Sinh ảnh bài nháp mẫu và chấm từng bước đối chiếu tài liệu nguồn |
| `core/ui_theme.py` | CSS tokens, SVG vector icons và các component HTML sạch cho giao diện |
| `eval/` | Bộ 20 câu hỏi đánh giá (`questions.json`) và script `run_eval.py` (đo phần truy xuất, kèm cấu hình đối chứng và bộ giữ riêng `holdout.json`) |
| `tests/` | Bộ kiểm thử tự động toàn diện (`test_quiz.py`, `test_fixes.py`, `test_adaptive_coach.py`, `test_breakthrough.py`, `test_chatbot_deep.py`, `test_app_ui.py`) |
