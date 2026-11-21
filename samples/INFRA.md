# Infra: Study Coach 2.0 — Hạ tầng hoàn chỉnh & Lộ trình 5 tuần đã triển khai

Tài liệu mô tả hạ tầng thực tế của Study Coach 2.0 sau khi hoàn thiện 5 tuần: mọi thành phần dưới đây đã được cài đặt và chạy được, không còn là kế hoạch.

## 1. Tổng quan

```
 Máy dev / máy demo (Windows/Mac/Linux, không cần GPU)
 ┌──────────────────────────────────────────────────────────────────┐
 │ Streamlit UI (app.py · ui_theme.py)                              │
 │   st.form (Login/Register/Quiz/Placement) · @st.cache_resource   │
 │   st.context.cookies (phiên đăng nhập qua cookie sc_session)     │
 │      │                                                            │
 │ core/ (Python 3.12)                                              │
 │   contracts · prompts · loader · llm · rag · quiz · adaptive ·   │
 │   coach_graph (LangGraph StateGraph + Context-First Router) ·    │
 │   breakthrough · vision · db · ui_theme                         │
 │      │                     │                       │             │
 │   SQLite                Dual-Memory FAISS        .env / st.secrets│
 │   data/study_coach.db   (doc_store +             (đọc qua          │
 │   9 bảng                mistake_store +          config._get_setting)│
 │                         scope_store)                              │
 └──────┼─────────────────────┼───────────────────────┼─────────────┘
        │                     │                       │
        ▼                     ▼                       ▼
   (file local)         (index local,           AI Portal (cổng API khóa học)
                         không dịch vụ cloud)   └► GPT-4o-mini · GPT-4.1-mini
                                                   text-embedding-3-small
 Deploy: Streamlit Community Cloud (từ GitHub subtree SourceCode/)
```

Không có server riêng: ứng dụng chạy trên máy người dùng (hoặc Streamlit Community Cloud), chỉ gọi ra ngoài tới **AI Portal**. Vector store dùng **FAISS** lưu cục bộ (Dual-Memory), không phụ thuộc dịch vụ vector cloud.

## 2. Thành phần cố định

| Thành phần | Chi tiết |
|---|---|
| Ngôn ngữ / môi trường | Python 3.12, `venv` trong `.venv/` (không commit) |
| Giao diện | Streamlit, `streamlit run app.py`, cổng mặc định 8501; dùng `st.form`, `@st.cache_resource`, `st.context.cookies` |
| LLM | OpenAI API qua **AI Portal**: `OPENAI_BASE_URL=https://aiportalapi.stu-platform.live/jpe` |
| Model trên cổng | `GPT-4o-mini` (mặc định), `GPT-4.1-mini`, `text-embedding-3-small`. Tên đúng như portal, phân biệt hoa thường |
| Hạn mức | $5/tháng mỗi key. `MAX_INPUT_CHARS = 12000` giới hạn chi phí mỗi lần gọi |
| Lưu trữ | **SQLite** `data/study_coach.db` (9 bảng) + **FAISS** Dual-Memory cục bộ (`doc_store` + `mistake_store`, thêm `scope_store` khi dùng embedding OpenAI) |
| Mã nguồn | GitLab nội bộ Fsoft Academy `origin/main` (repo đầy đủ) + GitHub `khoaanh10/study-coach` (chỉ chứa `SourceCode/`, để deploy) |
| Kiểm thử | `pytest`, 223 test mock/offline, không cần key |
| Dữ liệu mẫu | `samples/` (trong git) |
| Dữ liệu chạy | `data/` (SQLite, FAISS index, cache): **không commit** |

## 3. Cấu hình và bí mật

`.env` (đã trong `.gitignore`) cho môi trường local; `st.secrets` cho Streamlit Cloud. Mọi giá trị đọc qua `config._get_setting` (ưu tiên biến môi trường, sau đó `st.secrets`).

| Biến | Ý nghĩa |
|---|---|
| `OPENAI_API_KEY` | Key AI Portal của từng người |
| `OPENAI_BASE_URL` | Địa chỉ cổng API khóa học |
| `OPENAI_MODEL` | Model mặc định (`GPT-4o-mini`) |
| `OPENAI_EMBEDDING_MODEL` | Model embedding, mặc định `text-embedding-3-small` |
| `DB_PATH` | Đường dẫn file SQLite, mặc định `data/study_coach.db` |
| `MAX_SESSION_COST_USD` | Ngân sách tối đa mỗi phiên (mặc định 1.0 USD), chặn khi vượt ngưỡng |

Ngoài ra người dùng có thể nhập key tạm thời ngay trên giao diện (`config.set_runtime_api_key`, cô lập theo `session_api_key` của từng phiên).

Quy tắc: mỗi người dùng **key riêng**; không commit credential (`.env`, `.streamlit/secrets.toml`, `*.db`, `.venv/`); không dán key vào chat, issue, ảnh chụp màn hình hay video demo; nghi lộ thì tạo key mới.

## 4. Repo, nhánh, CI/CD

- **GitLab nội bộ (Fsoft Academy):** `origin/main` chứa toàn bộ repo (Documents, TaskList, Slides, SourceCode…).
- **GitHub Deploy repo:** `khoaanh10/study-coach`, nhánh `main`, chỉ chứa thư mục `SourceCode/`, được đồng bộ bằng `git subtree split --prefix=SourceCode`.
- **CD tự động:** mỗi lần đẩy `SourceCode/` lên GitHub Deploy repo, **Streamlit Community Cloud** tự build lại từ `app.py`.
- **CI:** cài `requirements.txt`, `ruff check`, `pytest`; CI **không dùng API key** nên test phải mock LLM.

## 5. Chi phí và giới hạn

- $5/tháng mỗi key; chia nhỏ thành quota theo tuần để không cạn trước hackathon.
- `GPT-4o-mini` mặc định; `GPT-4.1-mini` chỉ khi so sánh hoặc cần chất lượng cao; embedding bằng `text-embedding-3-small`.
- Giữ `MAX_INPUT_CHARS = 12000` để chặn chi phí; cache FAISS KB bằng `@st.cache_resource` (`_build_cached_doc_kb`) và cache đồ thị LangGraph đã compile bằng `_GRAPH_CACHE`, giảm rerun và lời gọi embedding lặp lại.
- Telemetry mỗi lời gọi (token vào/ra, độ trễ, chi phí ước tính) ghi vào bảng `llm_calls`; `MAX_SESSION_COST_USD` chặn khi vượt ngân sách phiên.

## 6. Kiểm thử, độ bền và đo chất lượng

- **Test:** `pytest` với 223 test mock/offline, không cần API key, chạy được cả khi tách riêng `SourceCode/`.
- **Eval:** `eval/questions.json` (20 câu) và `eval/holdout.json` (24 câu giữ riêng); kết quả lưu ở `eval/results/` để lập bảng trước/sau. Có thể chạy với embedding OpenAI thật qua `--openai-embeddings`.
- **Độ bền:** retry có backoff khi lỗi tạm thời (quá tải, hết thời gian, mất mạng); chặn khi vượt ngân sách phiên; thông báo tiếng Việt rõ ràng khi PDF scan, văn bản quá ngắn, model trả sai định dạng; từ chối đúng khi câu hỏi ngoài tài liệu; `st.form` giữ thao tác không làm refresh giữa chừng.

## 7. Bảo mật (Bổ sung Auth & Session)

- **Tài khoản:** đăng ký/đăng nhập `username/password`; mật khẩu băm `PBKDF2-SHA256` với 200.000 vòng lặp và salt 16 byte (bảng `users`).
- **Phiên đăng nhập:** Browser Cookie `sc_session` ký **HMAC-SHA256** (16 ký tự hex), đọc qua `st.context.cookies`, đặt `SameSite=Lax`, hết hạn sau 7 ngày; **không đưa token lên URL** (query param chỉ được xoá nếu còn sót).
- **Dọn profile cũ:** tự xoá các hồ sơ `user_*` tạo trước đây không có mật khẩu (không nằm trong bảng `users`) khi khởi tạo DB.
- Repo public: không có key, không có dữ liệu cá nhân. Địa chỉ cổng AI Portal xuất hiện trong repo (không phải bí mật nhưng ai cũng thấy).
- File upload xử lý trong bộ nhớ, không lưu lên cloud. Không đưa tài liệu nhạy cảm hay thông tin khách hàng vào demo.

## 8. Tiến trình hạ tầng qua 5 tuần (Đã hoàn thiện)

| Tuần | Nội dung đã cài đặt |
|---|---|
| **Tuần 1** | Khung repo, `.venv`, `.env.example`, Streamlit UI, `contracts.py` (Pydantic), `llm.py` (structured output + interface LLM), `quiz.py` (sinh & chấm quiz, `_verify`, `_is_grounded`), `loader.py` (PDF/TXT), AI Portal qua `.env`; SQLite `db.py` khởi tạo |
| **Tuần 2** | Bảng `llm_calls` (telemetry token/độ trễ/chi phí), bảng giá ước tính trong `config.py`, function calling + streaming (`complete_with_tools`, `stream_chat_completion`), retry backoff |
| **Tuần 3** | Embeddings `text-embedding-3-small` + fallback `LocalNGramEmbeddings` chạy CPU, **FAISS** cục bộ, `RecursiveCharacterTextSplitter` chunk có overlap, metadata `[Trang n]`, `extract_concept_map`, chấm Vấn đáp Feynman |
| **Tuần 4** | **FAISS Dual-Memory** (`doc_store` + `mistake_store`) + `scope_store` quyết định trong/ngoài tài liệu độc lập thang điểm embedding; cổng ngoài-tài-liệu; chọn **FAISS Dual-Memory thay cho Pinecone** để chạy không phụ thuộc dịch vụ vector cloud |
| **Tuần 5** | LangGraph `StateGraph` cho Gia sư Socratic (`coach_graph.py`) + Context-First LLM Router `_IntentRouterOutput`; `breakthrough.py` (Cognitive Diff, Production Incident, Trojan Bug-Hunt, Devil's Advocate nạp `INFRA.md`); `vision.py` chấm ảnh bài làm; Auth username/password + cookie `sc_session`; Visual Hierarchy 5 tầng |

## 9. Checklist hạ tầng (Hoàn thành)

- [x] Mỗi người có key riêng và chạy được app trên máy mình
- [x] CI xanh, `main` được bảo vệ, có tag cho từng tuần
- [x] Dữ liệu demo cố định, FAISS index dựng sẵn
- [x] Bảng số liệu eval từ tuần 2 đến tuần 5 đã đủ
- [x] Máy trình bày đã cài đủ thư viện, chạy offline với cache
- [x] Còn ngân sách key cho ngày demo
- [x] Không có key nào xuất hiện trong repo, ảnh chụp hay video
- [x] 223 test pytest mock/offline đạt, không cần API key
- [x] Deploy Streamlit Community Cloud tự động từ GitHub `SourceCode/`
