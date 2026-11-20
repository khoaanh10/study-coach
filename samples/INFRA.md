# Infra: Study Coach

Hạ tầng của toàn bộ dự án và phần bổ sung theo từng tuần. Tuần 1 mô tả đúng hiện trạng repo; các tuần sau là kế hoạch.

## 1. Tổng quan

```
 Máy dev / máy demo (Windows/Mac/Linux, không cần GPU)
 ┌──────────────────────────────────────────────────────────────┐
 │ Streamlit UI (app.py)                                        │
 │      │                                                       │
 │ core/ (Python 3.12)                                          │
 │   loader · quiz · llm · [db] · [ingest · embeddings ·        │
 │   vectorstore · rag · chains · agent]                        │
 │      │                       │                │              │
 │   SQLite (data/*.db)   Vector store (data/)   .env (key)     │
 └──────┼───────────────────────┼────────────────┼──────────────┘
        │                       │ (W4 option B)  │
        ▼                       ▼                ▼
   (file local)            Pinecone cloud    AI Portal (cổng API khóa học)
                                             └► GPT-4o-mini · GPT-4.1-mini
                                                text-embedding-3-small
```
Phần trong `[ ]` là các tuần sau. Không có server riêng: ứng dụng chạy trên máy người dùng, chỉ gọi ra ngoài tới AI Portal (và Pinecone nếu chọn).

## 2. Thành phần cố định

| Thành phần | Chi tiết |
|---|---|
| Ngôn ngữ / môi trường | Python 3.12, `venv` trong `.venv/` (không commit) |
| Giao diện | Streamlit, chạy `streamlit run app.py`, cổng mặc định 8501 |
| LLM | OpenAI API qua **AI Portal**: `OPENAI_BASE_URL=https://aiportalapi.stu-platform.live/jpe` |
| Model trên cổng | `GPT-4o-mini` (mặc định), `GPT-4.1-mini`, `text-embedding-3-small`. Tên đúng như portal, có phân biệt hoa thường |
| Hạn mức | $5/tháng mỗi key. `MAX_INPUT_CHARS` = 12000 giới hạn chi phí mỗi lần gọi |
| Mã nguồn | GitHub public: `khoaanh10/study-coach`, nhánh `main` |
| Kiểm thử | `pytest`, test mock LLM, không cần key |
| Dữ liệu mẫu | `samples/` (trong git) |
| Dữ liệu chạy | `data/` (SQLite, vector store, cache): **không commit** |

## 3. Cấu hình và bí mật

`.env` (đã trong `.gitignore`), mẫu ở `.env.example`. **Repo là public**, nên tuyệt đối không commit key.

| Biến | Dùng từ tuần | Ý nghĩa |
|---|---|---|
| `OPENAI_API_KEY` | 1 | Key AI Portal của từng người |
| `OPENAI_BASE_URL` | 1 | Địa chỉ cổng API khóa học |
| `OPENAI_MODEL` | 1 | Model mặc định (`GPT-4o-mini`) |
| `DB_PATH` | 1 (khi có SQLite) | Đường dẫn file SQLite, mặc định `data/study_coach.db` |
| `VECTOR_STORE` | 4 | `chroma` hoặc `pinecone` |
| `PINECONE_API_KEY`, `PINECONE_INDEX` | 4 (chỉ Option B) | Key và tên index Pinecone |
| `LANGSMITH_API_KEY` | 5 (tùy chọn) | Tracing |

Quy tắc: mỗi người dùng **key riêng** (portal có nút tạo thêm key); không dán key vào chat, issue, ảnh chụp màn hình hay video demo; nghi lộ thì tạo key mới.

## 4. Repo, nhánh, CI/CD

- **Nhánh:** `main` được bảo vệ (cần PR + 1 review). Nhánh làm việc đặt tên `w<tuần>/<nhóm>-<việc>`.
- **Tag:** `week1` … `week4`, `final`; mỗi tag là một bản chạy được, dùng làm phương án dự phòng khi demo.
- **CI (GitHub Actions, kế hoạch tuần 1):** mỗi PR chạy cài `requirements.txt`, `ruff check`, `pytest`. CI **không dùng API key**, nên test phải mock LLM.
- **CD:** không có. Demo chạy local trên máy trình bày.

## 5. Chi phí và giới hạn
- $5/tháng mỗi key; chia nhỏ thành quota theo tuần để không cạn trước hackathon.
- Dùng `GPT-4o-mini` mặc định; `GPT-4.1-mini` chỉ khi so sánh hoặc cần chất lượng cao.
- Test giao diện bằng dữ liệu mock; cache kết quả; giữ `MAX_INPUT_CHARS` và `top_k` nhỏ.
- Chi phí nên được ghi lại mỗi lần gọi (token, độ trễ) từ tuần 2.

## 6. Kiểm thử, độ bền và đo chất lượng
- **Test:** `pytest` với LLM được mock, chạy trong CI trên mỗi PR (không cần key).
- **Độ bền:** retry khi lỗi tạm thời (quá tải, hết thời gian, mất mạng); chặn khi vượt hạn mức chi tiêu; thông báo rõ khi PDF scan, văn bản quá ngắn, model trả sai định dạng; từ chối đúng khi câu hỏi ngoài tài liệu.
- **Eval:** thư mục `eval/` chứa bộ 20 câu hỏi mẫu (`questions.json`) và `run_eval.py`; kết quả mỗi tuần lưu ở `eval/results/` để lập bảng trước/sau (độ chính xác, đúng nguồn, độ trễ, chi phí) cho slide. Bắt đầu đo từ tuần 2.

## 7. Bảo mật
- Repo public: không có key, không có dữ liệu cá nhân. Địa chỉ cổng AI Portal xuất hiện trong repo (không phải bí mật nhưng ai cũng thấy).
- File upload của người dùng chỉ xử lý trong bộ nhớ, không lưu lên cloud (trừ Pinecone nếu chọn, khi đó chỉ nên upload tài liệu mẫu công khai).
- Không đưa tài liệu nhạy cảm hay thông tin khách hàng vào demo.

---

## 8. Hạ tầng theo từng tuần

### Tuần 1: Quiz Maker (hiện trạng + việc còn lại)
| Hạng mục | Chi tiết |
|---|---|
| Đã có | Python + `.venv`, Streamlit, SDK `openai`, `pydantic`, `pypdf`, `pytest`; kết nối AI Portal qua `.env`; repo GitHub public |
| Cần thêm | CI lint + test; PR template; bảo vệ `main`; SQLite (`core/db.py`, file `data/study_coach.db`) |
| Dịch vụ ngoài | AI Portal (chat). Chưa dùng embeddings, vector store |
| Thư viện | `openai`, `streamlit`, `pydantic`, `python-dotenv`, `pypdf`, `pytest` (+ `ruff` cho CI) |
| Ổ đĩa | Dưới 1 GB |

### Tuần 2: Study Assistant (OpenAI nâng cao)
| Hạng mục | Chi tiết |
|---|---|
| Thêm | Bảng `llm_calls` (model, token vào/ra, độ trễ, chi phí ước tính) trong SQLite; `eval/results/` lưu kết quả so sánh 2 model |
| Cấu hình | Bảng giá ước tính mỗi model trong `config.py` |
| Dịch vụ ngoài | Vẫn chỉ AI Portal (hai model chat), dùng function calling và streaming |
| Thư viện mới | Không |
| Lưu ý | Kiểm tra cổng có hỗ trợ function calling và streaming (spike ở tuần 1) |

### Tuần 3: Semantic Search (embeddings + vector store cục bộ)
| Hạng mục | Chi tiết |
|---|---|
| Thêm | Embeddings: `text-embedding-3-small` qua AI Portal (không cài gì) **hoặc** Hugging Face `sentence-transformers` chạy CPU (model nhỏ, ~90 MB) để so sánh |
| Vector store | FAISS hoặc Chroma lưu trong `data/vectorstore/` |
| Thư viện mới | `faiss-cpu` hoặc `chromadb`, `sentence-transformers` (kéo theo `torch`) |
| Ổ đĩa | Thêm khoảng 2–3 GB nếu cài `torch` |
| Lưu ý | Lần đầu tải model HF cần internet; cache embeddings để không tốn token lại; số chiều vector phụ thuộc model (OpenAI 1536, MiniLM 384), **không trộn hai loại trong một index** |

### Tuần 4: Study Chat (LangChain)
| Hạng mục | Chi tiết |
|---|---|
| Thêm | Gói LangChain ghim phiên bản: `langchain`, `langchain-openai`, `langchain-community`, `langchain-chroma` |
| Vector store | **Option A (mặc định):** Chroma local, `data/chroma/`. **Option B:** Pinecone cloud (tài khoản + key, index có số chiều khớp model embedding, namespace theo tài liệu) |
| Cấu hình | `VECTOR_STORE=chroma\|pinecone`; `ChatOpenAI` lấy `base_url`, key, model từ `config.py` |
| Dự phòng | Giữ tag `week3` và code FAISS để demo nếu tuần 4 gặp sự cố |
| Rủi ro | Pinecone phụ thuộc mạng, tài khoản và giới hạn free tier |

### Tuần 5: Coach (LangGraph + RAG nâng cao, hackathon)
| Hạng mục | Chi tiết |
|---|---|
| Thêm | `langgraph`; tùy chọn LangSmith để theo dõi (cần tài khoản và key); `scripts/prepare_demo.py` ingest sẵn tài liệu mẫu |
| Demo | Chạy local trên máy trình bày; cache sẵn câu trả lời; video quay trước làm phương án khi mất mạng; tập 2 lần trên đúng máy sẽ demo |
| Điểm nhấn | Chốt **vision** (chụp ảnh bài tập, qua model chat). Cổng chỉ liệt kê `GPT-4o-mini`, `GPT-4.1-mini`, `text-embedding-3-small`, nên voice chỉ xem xét nếu trainer xác nhận có API giọng nói |
| Ngân sách | Dự trữ khoảng $1 mỗi key cho eval cuối và demo |

## 9. Checklist hạ tầng trước hackathon
- [ ] Mỗi người có key riêng và chạy được app trên máy mình
- [ ] CI xanh, `main` được bảo vệ, có tag cho từng tuần
- [ ] Dữ liệu demo cố định, vector store dựng sẵn
- [ ] Bảng số liệu eval từ tuần 2 đến tuần 5 đã đủ
- [ ] Máy trình bày đã cài đủ thư viện, thử chạy offline với cache
- [ ] Còn ngân sách key cho ngày demo
- [ ] Không có key nào xuất hiện trong repo, ảnh chụp hay video
