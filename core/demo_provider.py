"""Chế độ Demo Offline bám tài liệu thật (Grounded Offline Provider).

Khi chưa cấu hình OPENAI_API_KEY hoặc người dùng chọn chế độ Demo Offline, module này vận hành
hoàn toàn dựa trên các chunk thật từ FAISS/Loader và LearnerDNA thật trong SQLite để đảm bảo
mọi luồng Adaptive Quiz, Concept Map, LangGraph Coach, Feynman Teach-Back và Vision hoạt động trung thực.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from core import db
from core.contracts import (
    ConceptNode,
    DocumentConceptMap,
    LearnerDNA,
    Question,
    Quiz,
    RetrievedChunk,
    TeachBackEvaluation,
    VisionCritique,
    VisionStepCheck,
)
from core.rag import _STOP_WORDS_FOLDED, _SYNONYM_BRIDGES, _strip_accents, _tokenize_vi

_SECTION_RE = re.compile(r"\[(?:Trang|Mục)\s+([^\]]+)\]")
_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*[.)]\s+|#{1,3}\s+)([^\n]{4,80})$", re.MULTILINE)


def _log_demo(purpose: str, db_path: str | Path | None = None, latency_ms: int = 45) -> None:
    try:
        db.log_llm_call(
            model="Offline-Demo-Grounded",
            purpose=purpose,
            tokens_in=0,
            tokens_out=0,
            latency_ms=latency_ms,
            cost_usd=0.0,
            is_demo=True,
            db_path=db_path,
        )
    except Exception:
        pass


def _is_default_curriculum(text: str) -> bool:
    low = text.lower()
    return "chunk quá lớn" in low and "overlap" in low and "openai api" in low


_DEFAULT_SOURCES = {"rag-co-ban.txt", "openai-api-co-ban.txt"}
# Phủ định "mạnh" (không bao gồm "không" trơn vì nhiều cụm sai tự bắt đầu bằng "không cần").
_NEG_STRONG = ("không nên", "không được", "đừng", "tránh", "không làm", "chớ", "không bao giờ", "không phải", "không thể")
_NEG_ANY = _NEG_STRONG + ("không", "chẳng", "chưa")
_CLAUSE_SPLIT_RE = re.compile(r"[.!?;,\n]|(?:\bcòn\b|\bmà\b|\bnhưng\b|\bngược lại\b|\btrong khi\b)")


def _is_default_source(sources: Iterable[str], text: str = "") -> bool:
    """Tài liệu là giáo trình mặc định (theo tên file mẫu hoặc nội dung đặc trưng)."""
    return any(s in _DEFAULT_SOURCES for s in sources) or _is_default_curriculum(text)


def _split_clauses(text_low: str) -> list[str]:
    """Tách thành mệnh đề theo dấu câu và các từ nối đối lập (còn/mà/nhưng...)."""
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text_low) if c and c.strip()]


def _phrase_asserted(clause: str, phrases: Iterable[str]) -> bool:
    """True nếu một cụm (sai) xuất hiện trong mệnh đề và KHÔNG bị phủ định đứng trước nó."""
    for ph in phrases:
        idx = clause.find(ph)
        if idx < 0:
            continue
        prefix = clause[:idx]
        negs = _NEG_STRONG if ph.startswith("không") else _NEG_ANY
        if not any(n in prefix for n in negs):
            return True
    return False


def _claim_has_wrong_phrase(
    text_low: str,
    bad_phrases: Iterable[str],
    require: Iterable[str] = (),
    forbid: Iterable[str] = (),
) -> bool:
    """Có mệnh đề KHẲNG ĐỊNH chứa cụm sai (đủ từ bắt buộc `require`, không có từ `forbid`) hay không."""
    bad = tuple(bad_phrases)
    req = tuple(require)
    fbd = tuple(forbid)
    for cl in _split_clauses(text_low):
        if all(r in cl for r in req) and not any(f in cl for f in fbd) and _phrase_asserted(cl, bad):
            return True
    return False


def demo_extract_concept_map(
    text: str,
    doc_id: str,
    doc_title: str,
    db_path: str | Path | None = None,
) -> DocumentConceptMap:
    _log_demo("concept_map", db_path=db_path, latency_ms=32)
    if _is_default_curriculum(text):
        return DocumentConceptMap(
            doc_id=doc_id,
            doc_title=doc_title,
            concepts=[
                ConceptNode(
                    name="RAG là gì",
                    summary="Mô hình tìm đoạn tài liệu liên quan trước khi sinh câu trả lời để giảm bịa đặt và dẫn nguồn.",
                    page_ref="[Mục 1]",
                    prerequisites=[],
                    key_terms=["rag", "retrieval", "truy xuất", "dẫn nguồn", "bịa"],
                ),
                ConceptNode(
                    name="Chunking và Overlap",
                    summary="Chia nhỏ tài liệu có phần gối đầu (overlap) để tránh cắt đứt ý và giữ cân bằng độ chính xác truy xuất.",
                    page_ref="[Mục 2]",
                    prerequisites=["RAG là gì"],
                    key_terms=["chunk", "overlap", "ngữ cảnh", "metadata", "quá lớn", "quá nhỏ"],
                ),
                ConceptNode(
                    name="Embeddings và Cosine",
                    summary="Biến văn bản thành vector số để tìm theo ngữ nghĩa thay vì chỉ khớp từ khóa nhờ cosine similarity.",
                    page_ref="[Mục 3]",
                    prerequisites=["Chunking và Overlap"],
                    key_terms=["embedding", "vector", "cosine", "ngữ nghĩa", "từ khóa"],
                ),
                ConceptNode(
                    name="Vector store và truy xuất",
                    summary="Lưu vector vào FAISS/Chroma/Pinecone và lấy top-k đoạn gần nhất (3–5) để giữ prompt gọn.",
                    page_ref="[Mục 4]",
                    prerequisites=["Embeddings và Cosine"],
                    key_terms=["faiss", "chroma", "pinecone", "top-k", "vector store"],
                ),
                ConceptNode(
                    name="Cấu trúc lời gọi & Messages",
                    summary="Gồm system/user/assistant; mô hình không tự nhớ hội thoại nên phải gửi lại lịch sử messages và chỉnh temperature.",
                    page_ref="[Mục 2–3 · API]",
                    prerequisites=[],
                    key_terms=["system", "user", "assistant", "messages", "temperature", "max tokens"],
                ),
                ConceptNode(
                    name="Structured outputs & Function calling",
                    summary="Ép kết quả JSON theo schema Pydantic và cho phép model tự gọi hàm công cụ rồi trả kết quả cuối.",
                    page_ref="[Mục 4–5 · API]",
                    prerequisites=["Cấu trúc lời gọi & Messages"],
                    key_terms=["structured", "pydantic", "schema", "function calling", "tool"],
                ),
                ConceptNode(
                    name="Đánh giá chất lượng RAG",
                    summary="Đo tỉ lệ trả lời đúng và tỉ lệ trích đúng nguồn trên bộ câu hỏi mẫu mỗi khi đổi chunk hay rerank.",
                    page_ref="[Mục 6]",
                    prerequisites=["Vector store và truy xuất"],
                    key_terms=["đánh giá", "bộ câu hỏi", "trích đúng nguồn", "rerank", "tỉ lệ"],
                ),
            ],
        )

    # Trích xuất tự động từ bất kỳ tài liệu PDF/TXT tùy chỉnh nào.
    headings = [m.group(1).strip() for m in _HEADING_RE.finditer(text)]
    nodes: list[ConceptNode] = []
    if headings:
        prev_name: str | None = None
        for idx, h in enumerate(headings[:7], start=1):
            pos = text.find(h)
            snippet = text[pos + len(h) : pos + len(h) + 260].strip().split("\n")[0]
            page_match = _SECTION_RE.findall(text[: max(pos, 1)])
            page_ref = f"[Trang/Mục {page_match[-1]}]" if page_match else f"[Mục {idx}]"
            words = [w.strip(".,:;()[]\"'").lower() for w in h.split() if len(w) > 2]
            nodes.append(
                ConceptNode(
                    name=h[:45],
                    summary=snippet[:180] or f"Nội dung trọng tâm của mục {h}.",
                    page_ref=page_ref,
                    prerequisites=[prev_name] if prev_name else [],
                    key_terms=words[:5],
                )
            )
            prev_name = h[:45]
    else:
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 60][:5]
        prev_name = None
        for idx, p in enumerate(paras, start=1):
            page_match = _SECTION_RE.search(p)
            page_ref = f"[{page_match.group(0).strip('[]')}]" if page_match else f"[Đoạn {idx}]"
            clean_p = _SECTION_RE.sub("", p).strip()
            title_words = " ".join(clean_p.split()[:5]).rstrip(".,:;")
            name = f"{title_words}..." if len(clean_p.split()) > 5 else f"Chủ đề {idx}"
            words = [w.strip(".,:;()[]\"'").lower() for w in clean_p.split() if len(w) >= 4]
            nodes.append(
                ConceptNode(
                    name=name,
                    summary=clean_p[:180],
                    page_ref=page_ref,
                    prerequisites=[prev_name] if prev_name else [],
                    key_terms=list(dict.fromkeys(words))[:5],
                )
            )
            prev_name = name

    return DocumentConceptMap(doc_id=doc_id, doc_title=doc_title, concepts=nodes)


_CURRICULUM_QUIZ_BANK: list[Question] = [
    Question(
        question="Khi chọn kích thước chunk trong RAG, điều gì xảy ra nếu đặt chunk quá lớn?",
        options=[
            "Việc truy xuất kém chính xác vì chứa nhiều ý lẫn lộn",
            "Làm mất hoàn toàn ngữ cảnh giữa các câu lân cận",
            "Không thể tính toán cosine similarity được nữa",
            "Tự động bỏ qua metadata tên tài liệu và số trang",
        ],
        evidence="Chunk quá lớn làm việc truy xuất kém chính xác vì chứa nhiều ý lẫn lộn, còn chunk quá nhỏ làm mất ngữ cảnh.",
        explanation="Theo tài liệu, chunk quá lớn gom nhiều chủ đề khác nhau vào cùng một đoạn khiến truy xuất nhiễu; ngược lại 'mất ngữ cảnh' là hệ quả của chunk quá nhỏ.",
        correct_index=0,
        topic="Chunking và Overlap",
        bloom_level="Hiểu",
        page_ref="[Mục 2]",
        option_diagnostics=[
            "Chính xác",
            "Gán nhầm hậu quả mất ngữ cảnh của chunk quá nhỏ sang chunk quá lớn",
            "Hiểu sai vai trò vector hóa: cosine vẫn tính được nhưng độ nhiễu ngữ nghĩa tăng",
            "Nhầm lẫn giữa kích thước chunk và bước lưu metadata",
        ],
    ),
    Question(
        question="Mục đích chính của việc thiết lập overlap (gối đầu) giữa các chunk là gì?",
        options=[
            "Để ý nằm ở ranh giới giữa hai chunk không bị cắt đứt",
            "Để giảm dung lượng lưu trữ của vector store xuống một nửa",
            "Để thay thế hoàn toàn bước tìm kiếm theo embeddings",
            "Để loại bỏ nhu cầu gắn số trang vào metadata",
        ],
        evidence="Người ta thường để các chunk gối lên nhau một phần, gọi là overlap, để một ý nằm ở ranh giới giữa hai chunk không bị cắt đứt.",
        explanation="Overlap giữ lại một phần nội dung giao nhau giữa hai đoạn liên tiếp nhằm bảo toàn trọn vẹn mạch ý nằm ngay ranh giới cắt.",
        correct_index=0,
        topic="Chunking và Overlap",
        bloom_level="Hiểu",
        page_ref="[Mục 2]",
        option_diagnostics=[
            "Chính xác",
            "Hiểu ngược tác dụng: overlap làm tăng nhẹ số token/dung lượng chứ không giảm",
            "Nhầm vai trò của bước chia văn bản với bước embedding truy xuất",
            "Nhầm cơ chế giữ mạch ý với việc bỏ metadata nguồn",
        ],
    ),
    Question(
        question="Vì sao hệ thống RAG dùng Embeddings có thể tìm ra đoạn nói về 'giá thành' khi người dùng hỏi về 'chi phí'?",
        options=[
            "Vì các đoạn có ý nghĩa gần nhau sẽ có vector số gần nhau theo cosine similarity",
            "Vì hệ thống tự động dịch tất cả từ tiếng Việt sang từ điển đồng nghĩa cứng",
            "Vì prompt bắt buộc LLM phải đoán nghĩa từ ngoài tài liệu",
            "Vì số trang của hai đoạn văn luôn trùng khớp với nhau",
        ],
        evidence="Embedding là cách biến một đoạn văn bản thành một vector số, sao cho các đoạn có ý nghĩa gần nhau thì vector gần nhau.",
        explanation="Embeddings ánh xạ ngữ nghĩa vào không gian vector, nên hai khái niệm đồng nghĩa như 'chi phí' và 'giá thành' có khoảng cách vector rất gần.",
        correct_index=0,
        topic="Embeddings và Cosine",
        bloom_level="Hiểu",
        page_ref="[Mục 3]",
        option_diagnostics=[
            "Chính xác",
            "Nhầm tìm kiếm ngữ nghĩa (semantic search) với khớp từ khóa/từ điển cứng",
            "Hiểu sai nguồn gốc khả năng tìm kiếm: đây là tác dụng của embedding trước khi gọi LLM",
            "Nhầm lẫn giữa vị trí trang tài liệu với độ tương đồng vector",
        ],
    ),
    Question(
        question="Trong truy xuất Vector store, tại sao thường chọn giá trị top-k nhỏ (ví dụ 3 đến 5 đoạn)?",
        options=[
            "Để prompt không quá dài và kiểm soát chi phí mỗi lần gọi",
            "Vì FAISS và Chroma chỉ cho phép lưu tối đa 5 vector",
            "Để ép mô hình trả lời bằng kiến thức bên ngoài nhiều hơn",
            "Vì k lớn hơn 5 sẽ làm mất luôn role system trong messages",
        ],
        evidence="Giá trị k thường nhỏ, ví dụ 3 đến 5, để prompt không quá dài và tốn chi phí.",
        explanation="Đưa quá nhiều đoạn vào prompt vừa tốn token chi phí vừa làm loãng thông tin chính; k=3..5 giữ ngữ cảnh gọn và tập trung.",
        correct_index=0,
        topic="Vector store và truy xuất",
        bloom_level="Vận dụng",
        page_ref="[Mục 4]",
        option_diagnostics=[
            "Chính xác",
            "Nhầm giới hạn tham số truy xuất top-k với sức chứa của vector database",
            "Hiểu ngược mục tiêu RAG: RAG luôn ưu tiên bám đoạn truy xuất chứ không dùng kiến thức ngoài",
            "Nhầm lẫn cấu trúc messages API với tham số top-k của vector store",
        ],
    ),
    Question(
        question="Một ứng dụng chat dùng OpenAI API trả lời câu sau nhưng quên sạch câu trước đó của người dùng. Nguyên nhân đúng theo tài liệu là gì?",
        options=[
            "Mô hình không tự nhớ cuộc trò chuyện nên chương trình phải gửi lại lịch sử messages ở mỗi lần gọi",
            "Do quên đặt temperature = 0 khi khởi tạo client OpenAI",
            "Do dùng Structured outputs nên model từ chối lưu bộ nhớ",
            "Do hàm load_text chưa thêm nhãn [Trang n] cho file chat",
        ],
        evidence="Mô hình không tự nhớ cuộc trò chuyện, nên nếu muốn hội thoại nhiều lượt thì chương trình phải gửi lại lịch sử messages trong mỗi lần gọi.",
        explanation="OpenAI Chat API là stateless (không trạng thái): nếu code không gửi kèm các lượt user/assistant trước đó trong danh sách messages thì mô hình không nhớ.",
        correct_index=0,
        topic="Cấu trúc lời gọi & Messages",
        bloom_level="Vận dụng",
        page_ref="[Mục 2 · API]",
        option_diagnostics=[
            "Chính xác",
            "Nhầm tham số kiểm soát độ ngẫu nhiên (temperature) với bộ nhớ hội thoại",
            "Nhầm định dạng đầu ra (Structured outputs) với cơ chế stateless của API",
            "Nhầm định dạng đánh dấu trang tài liệu với lịch sử hội thoại messages",
        ],
    ),
    Question(
        question="Khi xây dựng module trích xuất câu hỏi trắc nghiệm cần đúng 4 đáp án và chỉ số đáp án đúng, kỹ thuật nào giúp đảm bảo định dạng đầu ra ổn định nhất?",
        options=[
            "Dùng Structured outputs theo schema khai báo bằng Pydantic",
            "Tăng tham số temperature lên thật cao để model tự sáng tạo cấu trúc",
            "Bỏ hẳn role system để prompt ngắn và tiết kiệm token",
            "Dùng max_tokens = 10 để cắt gọn câu trả lời của model",
        ],
        evidence="Thay vì nhận văn bản tự do rồi tự phân tích, ta có thể yêu cầu mô hình trả về dữ liệu theo một schema định sẵn, ví dụ khai báo bằng Pydantic.",
        explanation="Structured outputs ép mô hình trả về đúng cấu trúc Pydantic mong muốn thay vì phải parse chuỗi văn bản tự do dễ lỗi.",
        correct_index=0,
        topic="Structured outputs & Function calling",
        bloom_level="Vận dụng",
        page_ref="[Mục 4 · API]",
        option_diagnostics=[
            "Chính xác",
            "Hiểu ngược vai trò temperature: temperature cao làm câu trả lời ngẫu nhiên, khó ổn định hơn",
            "Bỏ system prompt làm mất quy tắc kiểm soát định dạng và tính trung thực",
            "Đặt max_tokens quá nhỏ sẽ làm JSON bị cắt cụt giữa chừng",
        ],
    ),
    Question(
        question="Trong cơ chế Function calling, bên nào trực tiếp thực thi code của hàm công cụ (ví dụ truy vấn SQLite)?",
        options=[
            "Chương trình của bạn thực thi hàm rồi gửi kết quả ngược lại cho mô hình",
            "Máy chủ OpenAI tự kết nối trực tiếp vào cơ sở dữ liệu local của bạn",
            "Pydantic tự động thực thi hàm mà không cần code xử lý",
            "PyPDF chạy hàm công cụ ngay khi đọc trang PDF",
        ],
        evidence="Mô hình sẽ quyết định khi nào cần gọi hàm và trả về tên hàm cùng tham số. Chương trình thực thi hàm rồi gửi kết quả ngược lại cho mô hình để nó đưa ra câu trả lời cuối cùng.",
        explanation="LLM chỉ sinh tên hàm và JSON tham số; ứng dụng Python mới là nơi thực thi hàm an toàn rồi trả kết quả lại cho LLM.",
        correct_index=0,
        topic="Structured outputs & Function calling",
        bloom_level="Hiểu",
        page_ref="[Mục 5 · API]",
        option_diagnostics=[
            "Chính xác",
            "Hiểu sai kiến trúc bảo mật/chạy hàm: LLM trên cloud không tự chạy code trên máy bạn",
            "Nhầm thư viện xác thực dữ liệu Pydantic với bộ điều phối thực thi tool",
            "Nhầm bộ đọc PDF với cơ chế Function calling",
        ],
    ),
    Question(
        question="Khi muốn đánh giá xem một thay đổi về chunk size hay bước rerank có thực sự cải thiện hệ thống RAG hay không, nhóm cần đo cả hai chỉ số nào?",
        options=[
            "Tỉ lệ câu trả lời đúng và tỉ lệ trích đúng nguồn trên cùng bộ câu hỏi mẫu",
            "Chỉ cần đếm số file TXT trong thư mục samples/",
            "Chỉ đo màu sắc giao diện và tốc độ gõ phím của người dùng",
            "Chỉ kiểm tra số dòng code trong file config.py",
        ],
        evidence="Ta đo tỉ lệ câu trả lời đúng và tỉ lệ trích đúng nguồn, rồi chạy lại sau mỗi lần thay đổi như đổi kích thước chunk, đổi mô hình embedding hoặc thêm bước rerank.",
        explanation="Trả lời đúng thôi chưa đủ trong RAG; phải đo cả tỉ lệ trích đúng nguồn để chắc chắn câu trả lời thật sự dựa vào đúng đoạn tài liệu gốc.",
        correct_index=0,
        topic="Đánh giá chất lượng RAG",
        bloom_level="Phân tích",
        page_ref="[Mục 6]",
        option_diagnostics=[
            "Chính xác",
            "Chỉ kiểm tra dữ liệu đầu vào không đo được chất lượng truy xuất và sinh câu trả lời",
            "Nhầm tiêu chí đánh giá giao diện với đánh giá chất lượng mô hình RAG",
            "Đếm dòng code không phản ánh độ chính xác câu trả lời và trích nguồn",
        ],
    ),
    Question(
        question="Kỹ thuật RAG giải quyết hiện tượng mô hình ngôn ngữ bịa thông tin bằng cách nào?",
        options=[
            "Tìm các đoạn tài liệu liên quan trước rồi đưa vào prompt buộc mô hình trả lời bám nguồn",
            "Huấn luyện lại toàn bộ trọng số mô hình mỗi khi người dùng gửi một file PDF",
            "Tăng kích thước max_tokens lên gấp 10 lần mặc định",
            "Cấm người dùng đặt câu hỏi bằng tiếng Việt có dấu",
        ],
        evidence="hệ thống trước tiên tìm các đoạn tài liệu liên quan đến câu hỏi, sau đó đưa các đoạn đó vào prompt để mô hình trả lời. Nhờ vậy câu trả lời bám sát tài liệu và có thể dẫn nguồn",
        explanation="RAG không huấn luyện lại mô hình mà cung cấp ngữ cảnh truy xuất theo thời gian thực kèm yêu cầu dẫn nguồn và nói không biết nếu thiếu thông tin.",
        correct_index=0,
        topic="RAG là gì",
        bloom_level="Nhớ",
        page_ref="[Mục 1]",
        option_diagnostics=[
            "Chính xác",
            "Nhầm RAG (Retrieval-Augmented Generation) với Fine-tuning trọng số mô hình",
            "Nhầm giới hạn độ dài đầu ra với cơ chế bám nguồn tài liệu",
            "Nhầm chuẩn hóa ngôn ngữ đầu vào với kỹ thuật truy xuất tài liệu",
        ],
    ),
    Question(
        question="Vì sao file tài liệu trong RAG cần lưu metadata như tên tài liệu và số trang đi kèm mỗi chunk?",
        options=[
            "Để truy xuất xong có thể trích dẫn nguồn cho người dùng kiểm tra lại",
            "Để nén file PDF xuống dung lượng nhỏ hơn",
            "Để biến file scan ảnh thành file Word tự động",
            "Để tăng độ ngẫu nhiên temperature của mô hình",
        ],
        evidence="Mỗi chunk nên lưu kèm metadata như tên tài liệu và số trang để trích nguồn.",
        explanation="Metadata là cầu nối giúp hiển thị chính xác đoạn câu trả lời đến từ tài liệu nào, trang mấy để người học kiểm chứng.",
        correct_index=0,
        topic="Chunking và Overlap",
        bloom_level="Nhớ",
        page_ref="[Mục 2]",
        option_diagnostics=[
            "Chính xác",
            "Nhầm metadata trích dẫn với thuật toán nén dữ liệu",
            "Nhầm metadata trang với công nghệ OCR nhận diện chữ trên ảnh",
            "Nhầm thông tin vị trí tài liệu với tham số sinh văn bản",
        ],
    ),
]


def _extract_grounded_sentences(text: str) -> list[tuple[str, str]]:
    """Tách các câu thông tin thật kèm nhãn [Trang/Mục] từ bất kỳ tài liệu nào."""
    results: list[tuple[str, str]] = []
    cur_page = "[Tài liệu]"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_page = _SECTION_RE.search(line)
        if m_page:
            cur_page = f"[{m_page.group(0).strip('[]')}]"
        clean_line = _SECTION_RE.sub("", line).strip()
        for sent in re.split(r"(?<=[.!?])\s+", clean_line):
            sent_clean = sent.strip()
            if 45 <= len(sent_clean) <= 240 and sum(c.isalpha() for c in sent_clean) >= 25:
                results.append((sent_clean, cur_page))
    return results


def _build_dynamic_questions(text: str, n: int, difficulty: str, focus_topic: str | None) -> list[Question]:
    sents = _extract_grounded_sentences(text)
    if not sents:
        return []
    bloom = {"Dễ": "Nhớ", "Trung bình": "Hiểu", "Khó": "Vận dụng"}.get(difficulty, "Hiểu")
    topic_label = focus_topic or "Nội dung tài liệu"
    out: list[Question] = []
    pool_texts = [s for s, _ in sents]

    # Ưu tiên câu có chứa từ khóa của focus_topic.
    if focus_topic:
        kws = [w.lower() for w in focus_topic.split() if len(w) > 2]
        sents = sorted(sents, key=lambda item: sum(kw in item[0].lower() for kw in kws), reverse=True)

    for idx, (sent, page_ref) in enumerate(sents[: max(n, 3)]):
        words = sent.split()
        lead = " ".join(words[: min(7, len(words) // 2)]).rstrip(".,:;")
        other_opts = [
            f"Thông tin trái ngược hoàn toàn với nội dung nêu tại {page_ref}",
            "Một giả định bên ngoài không được đề cập trong đoạn tài liệu này",
        ]
        if len(pool_texts) > 1:
            alt = pool_texts[(idx + 1) % len(pool_texts)]
            other_opts.append(f"Nhầm sang ý khác: {alt[:95]}...")
        else:
            other_opts.append("Kết quả chỉ phụ thuộc vào cấu hình giao diện người dùng")

        if difficulty == "Khó":
            q_text = f"Dựa trên cơ chế được trình bày trong '{lead}...', nhận định nào phản ánh đúng bản chất theo {page_ref}?"
        elif difficulty == "Dễ":
            q_text = f"Theo tài liệu ({page_ref}), nội dung nào đúng khi nói về '{lead}'?"
        else:
            q_text = f"Ý chính nào mô tả đúng nội dung liên quan đến '{lead}...' tại {page_ref}?"

        correct_ans = sent[:155] + ("..." if len(sent) > 155 else "")
        options = [correct_ans, other_opts[0], other_opts[1], other_opts[2]]
        # Xoay vị trí đáp án đúng để không luôn ở A.
        shift = idx % 4
        rotated = options[-shift:] + options[:-shift] if shift else options
        diag_base = [
            "Chính xác",
            "Hiểu ngược chiều khẳng định của tài liệu nguồn",
            "Suy diễn bằng kiến thức ngoài thay vì bám sát văn bản",
            "Đánh tráo khái niệm từ một ý khác sang ngữ cảnh này",
        ]
        rotated_diag = diag_base[-shift:] + diag_base[:-shift] if shift else diag_base
        out.append(
            Question(
                question=q_text,
                options=rotated,
                evidence=sent,
                explanation=f"Đối chiếu nguyên văn tại {page_ref}: \"{sent}\"",
                correct_index=shift,
                topic=topic_label,
                bloom_level=bloom,
                page_ref=page_ref,
                option_diagnostics=rotated_diag,
            )
        )
        if len(out) >= n:
            break
    return out


def demo_generate_quiz(
    text: str,
    n: int = 5,
    difficulty: str = "Trung bình",
    focus_topic: str | None = None,
    learner_dna: LearnerDNA | None = None,
    db_path: str | Path | None = None,
) -> Quiz:
    _log_demo("quiz_generation", db_path=db_path, latency_ms=58)
    if _is_default_curriculum(text):
        norm_text = " ".join(text.lower().split())
        available = [q for q in _CURRICULUM_QUIZ_BANK if q.evidence.lower()[:30] in norm_text]
        if not available:
            available = list(_CURRICULUM_QUIZ_BANK)

        target_bloom = {"Dễ": "Nhớ", "Trung bình": "Hiểu", "Khó": "Vận dụng"}.get(difficulty, "Hiểu")
        weak_set = {
            c.concept.lower()
            for c in (learner_dna.concepts if learner_dna else [])
            if c.status in {"misconception", "learning"} and c.mastery_score < 60
        }

        def _score_q(q: Question) -> tuple[int, int, int]:
            focus_hit = 1 if (focus_topic and focus_topic.lower() in q.topic.lower()) else 0
            weak_hit = 1 if q.topic.lower() in weak_set else 0
            bloom_hit = 1 if (q.bloom_level == target_bloom or (difficulty == "Khó" and q.bloom_level in {"Vận dụng", "Phân tích"})) else 0
            return (focus_hit, weak_hit, bloom_hit)

        ranked = sorted(available, key=_score_q, reverse=True)
        chosen = ranked[: min(n, len(ranked))]
        return Quiz(questions=[q.model_copy(deep=True) for q in chosen])

    dynamic = _build_dynamic_questions(text, n=n, difficulty=difficulty, focus_topic=focus_topic)
    return Quiz(questions=dynamic)


def _clean_chunk_sentences(content: str) -> list[str]:
    """Tách văn bản chunk thành các câu trọn vẹn, loại bỏ dòng tiêu đề mục đứng riêng."""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    body_parts: list[str] = []
    for ln in lines:
        if re.match(r"^(?:\d+(?:\.\d+)*[.)]\s+|#{1,3}\s+)[^\n]{2,70}$", ln) or (len(ln) < 55 and ln.isupper()):
            continue
        body_parts.append(ln)
    body = " ".join(body_parts) if body_parts else " ".join(lines)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) >= 15]
    return sents or ([body] if body else [])


def _select_relevant_excerpt(
    query: str,
    chunks: list[RetrievedChunk],
    max_chars: int = 420,
) -> tuple[str, str]:
    """Chọn 1-3 câu trọn vẹn sát câu hỏi nhất từ chunk tốt nhất, không bao giờ cắt cụt giữa từ."""
    if not chunks:
        return "", "[Tài liệu]"

    q_low = query.lower()
    q_folded = _strip_accents(q_low)
    q_terms = {
        _strip_accents(t)
        for t in _tokenize_vi(query)
        if len(_strip_accents(t)) >= 2 and _strip_accents(t) not in _STOP_WORDS_FOLDED
    }
    for k, syns in _SYNONYM_BRIDGES.items():
        if k in q_low or _strip_accents(k) in q_folded:
            for syn in syns:
                q_terms.update(_strip_accents(t) for t in _tokenize_vi(syn))

    best_chunk = chunks[0]
    best_sents: list[str] = []
    best_score = -1.0

    for ch in chunks[:3]:
        sents = _clean_chunk_sentences(ch.content)
        if not sents:
            continue
        scored_sents: list[tuple[float, int, str]] = []
        top_sent_score = 0.0
        for idx, s in enumerate(sents):
            s_terms = {
                _strip_accents(t)
                for t in _tokenize_vi(s)
                if len(_strip_accents(t)) >= 2 and _strip_accents(t) not in _STOP_WORDS_FOLDED
            }
            overlap = len(q_terms & s_terms)
            bonus = 0.25 if idx == 0 else 0.0
            sc = overlap + bonus + float(ch.score)
            scored_sents.append((overlap + bonus, idx, s))
            if sc > top_sent_score:
                top_sent_score = sc
        if top_sent_score > best_score:
            best_score = top_sent_score
            best_chunk = ch
            # Ưu tiên các câu có từ khóa khớp, giữ theo đúng thứ tự xuất hiện trong văn bản
            picked = [item for item in sorted(scored_sents, key=lambda x: (-x[0], x[1]))[:3]]
            picked.sort(key=lambda x: x[1])
            best_sents = [item[2] for item in picked]

    if not best_sents:
        best_sents = _clean_chunk_sentences(best_chunk.content)

    excerpt_parts: list[str] = []
    total_len = 0
    for s in best_sents:
        added_len = len(s) + (1 if excerpt_parts else 0)
        if excerpt_parts and total_len + added_len > max_chars:
            break
        excerpt_parts.append(s)
        total_len += added_len

    excerpt = " ".join(excerpt_parts).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "..."

    cite = best_chunk.page if best_chunk.page else "[Tài liệu]"
    return excerpt, cite


def _extract_direct_focus_sentence(query: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Chọn đúng một câu gốc trả lời trực diện nhất cho câu hỏi cụ thể kèm nhãn [Trang/Mục]."""
    if not chunks:
        return "", "[Tài liệu]"
    q_low = query.lower()
    q_fold = _strip_accents(q_low)
    q_terms = {
        _strip_accents(t)
        for t in _tokenize_vi(query)
        if len(_strip_accents(t)) >= 2 and _strip_accents(t) not in _STOP_WORDS_FOLDED
    }
    for k, syns in _SYNONYM_BRIDGES.items():
        if k in q_low or _strip_accents(k) in q_fold:
            for syn in syns:
                q_terms.update(_strip_accents(t) for t in _tokenize_vi(syn))

    best_sent = ""
    best_cite = chunks[0].page or "[Tài liệu]"
    best_sc = -1.0
    for rank, ch in enumerate(chunks[:3]):
        sents = _clean_chunk_sentences(ch.content)
        for s_idx, s in enumerate(sents):
            s_terms = {
                _strip_accents(t)
                for t in _tokenize_vi(s)
                if len(_strip_accents(t)) >= 2 and _strip_accents(t) not in _STOP_WORDS_FOLDED
            }
            overlap = len(q_terms & s_terms)
            sc = overlap * 2.0 + float(ch.score) - 0.1 * rank - 0.02 * s_idx
            if sc > best_sc:
                best_sc = sc
                best_sent = s
                best_cite = ch.page or "[Tài liệu]"
    if not best_sent:
        return _select_relevant_excerpt(query, chunks, max_chars=300)
    return best_sent, best_cite


def _build_contextual_followup(
    query: str,
    topic: str,
    cite: str,
    level: str,
    used_followups: set[str] | None = None,
) -> str:
    """Sinh câu hỏi Socratic ngắn gọn bám sát chính khía cạnh người học vừa hỏi và tránh lặp lại câu vừa đặt."""
    q_fold = _strip_accents(query.lower())
    seen = {u.strip() for u in (used_followups or set()) if u.strip()}

    if any(k in q_fold for k in ("role", "vai tro", "system", "user", "assistant")):
        cand = f"Dựa theo {cite}, nếu bạn bỏ qua role `system` hoặc quên gửi lại các lượt `assistant` cũ thì phản hồi của model sẽ thay đổi thế nào?"
    elif any(k in q_fold for k in ("bia", "khong co trong tai lieu", "ngoai tai lieu", "khong xuat hien")):
        cand = f"Tại sao prompt ở {cite} bắt buộc model nói rõ 'không có trong tài liệu' thay vì tự đoán khi gặp câu ngoài nguồn?"
    elif any(k in q_fold for k in ("chi phi", "gia thanh")):
        cand = f"So với tìm kiếm khớp từ khóa thông thường, cơ chế embedding ở {cite} khác ở điểm mấu chốt nào giúp tìm đúng đoạn không trùng chữ?"
    elif any(k in q_fold for k in ("cosine", "similarity")):
        cand = f"Theo {cite}, hai đoạn văn có ý nghĩa gần nhau sẽ có độ đo vector thế nào so với hai đoạn khác chủ đề?"
    elif any(k in q_fold for k in ("metadata", "so trang", "trich nguon")):
        cand = f"Nếu quên lưu metadata tên file và số trang cùng mỗi chunk ở {cite}, người dùng cuối sẽ gặp bất lợi gì?"
    elif any(k in q_fold for k in ("top-k", "top k", "faiss", "vector store")):
        cand = f"Theo {cite}, nếu tăng `top-k` từ 4 lên 30 thì prompt gửi cho LLM sẽ gặp hai vấn đề gì?"
    elif any(k in q_fold for k in ("temperature", "pydantic", "structured", "function calling")):
        cand = f"Ở phần {topic} ({cite}), vì sao hệ thống Python phải tự thực thi hàm công cụ hoặc ép Pydantic schema thay vì để LLM trả chữ tự do?"
    elif level == "Nâng cao":
        cand = f"Trong thiết kế thực tế dựa trên {cite}, bạn sẽ kiểm thử điều kiện biên nào của `{topic}` để vừa giữ độ chính xác vừa không đội chi phí token?"
    else:
        cand = f"Dựa vào ý cốt lõi vừa trích ở {cite}, nếu bỏ qua hoặc cấu hình sai bước `{topic}` thì khâu nào trong hệ thống sẽ hỏng đầu tiên?"

    if cand in seen:
        cand = f"Bạn thử nêu một tình huống cụ thể trong tài liệu {cite} để áp dụng đúng nguyên tắc của `{topic}` vừa nhắc ở trên xem?"
    return cand


def demo_coach_conversational_reply(
    intent: str,
    query: str,
    topic: str,
    retrieved_chunks: list[RetrievedChunk],
    learner_dna: LearnerDNA,
    concept_map: DocumentConceptMap,
    db_path: str | Path | None = None,
) -> tuple[str, str]:
    """Sinh phản hồi hội thoại đa kịch bản (chào hỏi/hồ sơ, tóm tắt tài liệu, xin ví dụ, đào sâu)."""
    _log_demo(f"coach_{intent}", db_path=db_path, latency_ms=42)
    excerpt, cite = _select_relevant_excerpt(f"{topic} {query}", retrieved_chunks, max_chars=460)

    if intent == "greeting_or_meta":
        top_misc = learner_dna.active_misconceptions[0] if learner_dna.active_misconceptions else None
        weakest = (
            min(learner_dna.concepts, key=lambda c: (0 if c.status == "misconception" else 1, c.mastery_score))
            if learner_dna.concepts
            else None
        )
        priority_topic = top_misc.topic if top_misc else (weakest.concept if weakest else "nội dung đầu tiên")
        priority_detail = (
            f"Bạn đang có lỗi tư duy cần gỡ ở **{top_misc.topic}** (*{top_misc.description}*)."
            if top_misc
            else (
                f"Chủ đề ưu tiên hiện tại của bạn là **{weakest.concept}** (thành thạo **{weakest.mastery_score:.0f}%**)."
                if weakest
                else "Bạn có thể hỏi bất kỳ khái niệm nào trong tài liệu đang nạp."
            )
        )
        sample_topics = ", ".join(f"**{c.concept}**" for c in learner_dna.concepts[:4]) or "**tài liệu hiện tại**"
        answer = (
            f"Chào **{learner_dna.name}**! Mình là **Gia sư Socratic Study Coach** đồng hành cùng bạn trên tài liệu **{concept_map.doc_title}**.\n\n"
            f"- **Hồ sơ hiện tại:** Trình độ **{learner_dna.overall_level}** · Điểm trung bình **{learner_dna.mastery_avg:.0f}%** · "
            f"**{len(learner_dna.active_misconceptions)}** lỗi tư duy đang mở.\n"
            f"- **Ưu tiên ngay lúc này:** {priority_detail}\n\n"
            f"Bạn có thể hỏi mình giải thích từng phần ({sample_topics}), nhờ tóm tắt giáo trình, "
            "xin ví dụ thực tế hoặc trả lời trực tiếp câu hỏi vặn để mình chấm điểm & cập nhật hồ sơ cho bạn."
        )
        follow_up = f"Bạn muốn bắt đầu ngay với chủ đề **{priority_topic}** hay muốn mình tóm tắt nhanh toàn bộ tài liệu trước?"
        return answer, follow_up

    if intent == "doc_summary":
        mastery_map = {c.concept.lower(): c for c in learner_dna.concepts}
        lines = []
        for idx, node in enumerate(concept_map.concepts, start=1):
            m = mastery_map.get(node.name.lower())
            score_tag = f" · Đã đạt `{m.mastery_score:.0f}%`" if m and m.attempts > 0 else " · `Chưa học`"
            lines.append(f"{idx}. **{node.name}** `{node.page_ref or '[Tài liệu]'}`{score_tag} — {node.summary}")
        answer = (
            f"**Tóm tắt cấu trúc tài liệu: {concept_map.doc_title}** ({len(concept_map.concepts)} chủ đề cốt lõi)\n\n"
            + "\n".join(lines)
        )
        next_c = next((c.concept for c in learner_dna.concepts if c.status == "misconception" or c.mastery_score < 60), None)
        if not next_c and concept_map.concepts:
            next_c = concept_map.concepts[0].name
        follow_up = f"Trong danh sách trên, bạn muốn mình đi sâu vào phần **{next_c}** ngay bây giờ không?"
        return answer, follow_up

    if intent == "followup_example":
        t_low = topic.lower()
        if "chunk" in t_low:
            concrete = (
                "Giả sử đoạn tài liệu ở trang 3 viết định nghĩa Overlap ở cuối trang và ví dụ minh họa ở đầu trang 4:\n"
                "- **Nếu `overlap = 0`:** Điểm cắt rơi đúng giữa hai trang khiến chunk 1 chỉ có nửa đầu khái niệm, chunk 2 mất chủ ngữ — khi FAISS tìm kiếm sẽ thiếu ngữ cảnh.\n"
                "- **Nếu đặt `chunk_size = 500`, `overlap = 80–100`:** 80–100 ký tự cuối của chunk 1 lặp lại ở đầu chunk 2, giữ trọn mạch ý qua ranh giới kèm metadata `[Trang 3]`."
            )
        elif "rag" in t_low:
            concrete = (
                "Ví dụ khi nhân viên hỏi nội quy công ty *\"Nghỉ phép năm còn dư có được cộng dồn không?\"*:\n"
                "- **Không dùng RAG:** LLM trả lời theo kiến thức chung trên mạng, dễ bịa sai con số ngày phép của công ty.\n"
                "- **Dùng RAG:** Hệ thống tìm đúng 3 đoạn từ file `NoiQuy.pdf` `[Trang 12]`, nhét vào prompt và ép LLM chỉ trả lời dựa trên 3 đoạn đó kèm trích dẫn `[Trang 12]`."
            )
        elif "embedding" in t_low or "cosine" in t_low:
            concrete = (
                "Ví dụ hệ thống lưu câu *\"Giá thành gọi model lớn khá cao\"* thành một vector 384/1536 chiều:\n"
                "- Khi người dùng hỏi *\"Chi phí vận hành hết bao nhiêu?\"*, từ khóa `'giá thành'` không trùng mặt chữ với `'chi phí'`, "
                "nhưng vector của hai câu gần nhau theo `cosine similarity` nên Embedding vẫn tìm ra đúng đoạn."
            )
        elif "vector" in t_low or "top" in t_low:
            concrete = (
                "Ví dụ kho tài liệu dài 100 trang được chia thành 200 chunk và lưu vector vào **FAISS**:\n"
                "- Thay vì gửi cả 200 chunk vào prompt (vượt giới hạn token và rất tốn tiền), hệ thống chỉ lấy **`top_k = 3..5`** chunk có vector gần câu hỏi nhất để đưa cho LLM."
            )
        elif "message" in t_low or "cấu trúc" in t_low:
            concrete = (
                "Ví dụ hội thoại 2 lượt với OpenAI Chat API:\n"
                "- **Lượt 1:** gửi `messages = [system, user: 'RAG là gì?']` -> nhận `assistant: 'RAG là...'`.\n"
                "- **Lượt 2 (nếu quên gửi lịch sử):** chỉ gửi `[user: 'Ví dụ cụ thể?']` thì model không biết 'cụ thể' cho cái gì.\n"
                "- **Đúng chuẩn:** phải gửi kèm cả lượt 1 (`[system, user1, assistant1, user2]`)."
            )
        elif "structured" in t_low or "function" in t_low:
            concrete = (
                "Ví dụ module cần chấm quiz 4 đáp án hoặc tra điểm học viên:\n"
                "- **Structured outputs:** Khai báo Pydantic `Quiz`, model bắt buộc trả JSON đúng 4 lựa chọn và `correct_index` kiểu `int` thay vì văn xuôi.\n"
                "- **Function calling:** Model trả yêu cầu gọi `get_learner_dna(learner_id='me')`, code Python của bạn tự truy vấn SQLite rồi gửi kết quả lại cho model viết lời khuyên."
            )
        elif "đánh giá" in t_low or "eval" in t_low:
            concrete = (
                "Ví dụ nhóm thay đổi `chunk_size` từ 1000 xuống 500 hoặc thêm bước `rerank`:\n"
                "- Chạy lại **20 câu hỏi mẫu** cố định: trước sửa đúng `14/20` và trích đúng nguồn `12/20`; sau sửa đạt `18/20` đúng và `18/20` đúng nguồn -> chứng minh cấu hình mới thực sự tốt hơn."
            )
        else:
            concrete = (
                f"Áp dụng trực tiếp nguyên lý của **{topic}** vào thực tế vận hành:\n"
                f"- **Đầu vào cần chuẩn hóa:** Kiểm tra đúng dữ liệu/tham số được mô tả tại {cite}.\n"
                f"- **Kết quả mong đợi:** Hệ thống giữ đúng ràng buộc kỹ thuật thay vì tự đoán mò ngoài tài liệu."
            )
        answer = (
            f"**Ví dụ thực tế cho chủ đề `{topic}`** ({cite}):\n\n"
            f"{concrete}\n\n"
            f"**Căn cứ trong tài liệu ({cite}):**\n"
            f"> \"{excerpt}\""
        )
        follow_up = f"Nhìn vào ví dụ trên ở chủ đề **{topic}**, theo bạn nếu đảo ngược tham số chính ở bước này thì lỗi gì sẽ xảy ra đầu tiên?"
        return answer, follow_up

    # intent == "followup_elaborate"
    answer = (
        f"**Đào sâu cơ chế `{topic}` ({learner_dna.name} · Trình độ {learner_dna.overall_level}):**\n\n"
        f"**Nguyên văn trích xuất từ {cite}:**\n"
        f"> \"{excerpt}\"\n\n"
        "**Phân tích 3 lớp hoạt động:**\n"
        "1. **Điểm kích hoạt (Trigger):** Bước này xử lý trực tiếp dữ liệu trước khi chuyển tiếp trong pipeline.\n"
        "2. **Cơ chế bên trong:** Các tham số được thiết kế để khóa chặt tính chính xác và giữ nguyên ngữ cảnh/nguồn gốc dữ liệu.\n"
        "3. **Tác động dây chuyền:** Cấu hình lệch ở đây sẽ làm bước truy xuất hoặc bước sinh câu trả lời phía sau bị nhiễu hoặc mất kiểm chứng."
    )
    follow_up = f"Dựa trên cơ chế 3 lớp vừa phân tích tại {cite}, bạn thử tự tóm tắt lại vai trò quan trọng nhất của **{topic}** xem?"
    return answer, follow_up


def demo_review_response(
    topic: str,
    suggested: bool,
    retrieved_chunks: list[RetrievedChunk],
    learner_dna: LearnerDNA,
    concept_map: DocumentConceptMap,
    db_path: str | Path | None = None,
) -> tuple[str, str]:
    """Ôn nhanh một chủ đề: ý chính trích từ tài liệu + từ khóa + một dòng nhắc điểm dễ nhầm (không chất vấn)."""
    _log_demo("coach_review", db_path=db_path, latency_ms=60)
    node = next((c for c in concept_map.concepts if c.name == topic), None)
    # Chỉ lấy đoạn nguồn đúng với mục của khái niệm này (không để đoạn của mục khác chen vào phần ôn tập).
    page_ref = node.page_ref if node else ""
    own_chunks = [c for c in retrieved_chunks if page_ref and c.page == page_ref]
    excerpt, cite = _select_relevant_excerpt(topic, own_chunks or retrieved_chunks[:1], max_chars=520)
    mastery = next((c for c in learner_dna.concepts if c.concept == topic), None)

    lines: list[str] = []
    if suggested:
        why = (
            f" vì hồ sơ cho thấy bạn mới đạt **{mastery.mastery_score:.0f}%** ở phần này"
            if mastery and mastery.attempts > 0
            else ""
        )
        lines.append(
            f"Bạn chưa nói rõ phần nào nên mình chọn **{topic}**{why}. Muốn ôn phần khác thì cứ nêu tên phần đó nhé."
        )
    lines.append(f"**Ôn nhanh `{topic}`** ({cite})")
    if excerpt:
        lines.append(f"**Ý chính theo tài liệu:**\n> \"{excerpt}\"")
    if node and node.key_terms:
        lines.append("**Từ khóa cần nhớ:** " + ", ".join(node.key_terms[:5]))
    mis = [m for m in learner_dna.active_misconceptions if m.topic.lower() == topic.lower()]
    if mis:
        lines.append(f"**Điểm dễ nhầm cần để ý:** {mis[0].description.rstrip('.')}.")
    follow = f"Bạn thử tự tóm tắt `{topic}` bằng một hai câu, hoặc bấm tạo quiz về chủ đề này để kiểm tra nhé?"
    return "\n\n".join(lines), follow


def demo_coach_response(
    route: str,
    query: str,
    topic: str,
    retrieved_chunks: list[RetrievedChunk],
    past_mistakes: list[RetrievedChunk],
    learner_dna: LearnerDNA,
    prerequisite_topic: str | None = None,
    prerequisite_page: str | None = None,
    is_followup_turn: bool = False,
    used_followups: set[str] | None = None,
    db_path: str | Path | None = None,
) -> tuple[str, str]:
    """Sinh câu trả lời Gia sư Socratic bám chặt chunk truy xuất và phân biệt rõ theo trình độ."""
    _log_demo(f"coach_{route}", db_path=db_path, latency_ms=65)
    level = learner_dna.overall_level
    excerpt_query = query if any(t in _strip_accents(query.lower()) for t in _strip_accents(topic.lower()).split() if len(t) >= 3) else f"{topic}. {query}"
    excerpt, cite = _select_relevant_excerpt(excerpt_query, retrieved_chunks, max_chars=420)
    direct_sent, direct_cite = _extract_direct_focus_sentence(excerpt_query, retrieved_chunks)

    second_chunk = retrieved_chunks[1] if len(retrieved_chunks) > 1 else None
    cite2 = second_chunk.page if second_chunk and second_chunk.page else cite

    if route == "refuse_out_of_doc":
        topics_list = ", ".join(c.concept for c in learner_dna.concepts[:5]) or "tài liệu hiện tại"
        return (
            f"Thông tin về câu hỏi **\"{query}\"** không có trong tài liệu bạn đã nạp, nên mình không tự bịa câu trả lời ngoài nguồn.\n\n"
            f"Hiện tại tài liệu của bạn tập trung vào các chủ đề: **{topics_list}**. "
            "Nếu muốn, bạn hãy hỏi về một trong các phần trên hoặc upload thêm tài liệu liên quan nhé.",
            "Bạn muốn chuyển sang ôn phần nào có sẵn trong tài liệu hiện tại?",
        )

    if route == "bridge_prerequisite":
        prereq = prerequisite_topic or "kiến thức nền tảng"
        prereq_cite = prerequisite_page or cite
        prereq_score = next((c.mastery_score for c in learner_dna.concepts if c.concept == prereq), 0)
        answer = (
            f"Trước khi đi sâu vào **{topic}**, hồ sơ của bạn cho thấy phần nền tảng **{prereq}** "
            f"hiện mới đạt **{prereq_score:.0f}%** — mà hai phần này móc nối trực tiếp với nhau.\n\n"
            f"**1. Củng cố nhanh nền tảng ({prereq} · {prereq_cite}):**\n"
            f"Nếu dữ liệu đầu vào ở bước `{prereq}` chưa được chuẩn hóa đúng nguyên tắc ({prereq_cite}) thì bước `{topic}` phía sau sẽ xử lý lệch hoặc truy xuất nhiễu theo.\n\n"
            f"**2. Trả lời trực tiếp câu hỏi của bạn ({direct_cite}):**\n"
            f"> \"{direct_sent or excerpt}\""
        )
        follow_up = f"Theo bạn, nếu làm sai bước `{prereq}` ({prereq_cite}) thì kết quả ở phần `{topic}` ({direct_cite}) sẽ bị ảnh hưởng thế nào?"
        return answer, follow_up

    if route == "socratic_remediation":
        misc_desc = (
            past_mistakes[0].content
            if past_mistakes
            else (
                learner_dna.active_misconceptions[0].description
                if learner_dna.active_misconceptions
                else "có sự nhầm lẫn giữa các điều kiện kỹ thuật của chủ đề"
            )
        )
        is_inversion_misc = any(k in misc_desc.lower() for k in ("mất ngữ cảnh", "ngược", "nhỏ", "lớn"))
        caution_note = (
            "Hãy chú ý ranh giới nhân – quả trong câu trích trên: đừng gán hệ quả của trường hợp này sang trường hợp ngược lại."
            if is_inversion_misc
            else "Hãy đối chiếu kỹ câu vừa trích ở trên để tránh lặp lại đúng lỗi tư duy đang mở trong hồ sơ."
        )
        quote_block = excerpt if len(excerpt) >= len(direct_sent) else direct_sent
        answer = (
            f"**Trả lời trực tiếp câu hỏi ({cite}):**\n"
            f"> \"{quote_block}\"\n\n"
            f"**Lưu ý gỡ lỗi tư duy cho {learner_dna.name} (`{topic}`):** *{misc_desc}*\n\n"
            f"{caution_note}"
        )
        follow_up = _build_contextual_followup(query, topic, direct_cite, level, used_followups)
        if not any(k in _strip_accents(query.lower()) for k in ("role", "system", "user", "assistant", "metadata", "cosine")):
            if is_inversion_misc:
                follow_up = f"Dựa vào đúng câu trích tại {cite}, bạn thử phân biệt ngắn gọn hai chiều tác động đó bằng lời của bạn xem?"
        return answer, follow_up

    # route == "adaptive_explain" — trả lời bám sát câu hỏi & phân biệt rõ theo trình độ!
    q_fold = _strip_accents(query.lower())
    is_specific_subquestion = is_followup_turn or any(
        marker in q_fold
        for marker in (
            "tai sao", "vi sao", "bang cach nao", "khac nhau", "khac gi", "dung de",
            "lam gi", "khi nao", "neu ", "role", "vai tro", "cosine", "chi phi",
            "gia thanh", "metadata", "top-k", "top k", "temperature", "stateless",
        )
    )

    if is_specific_subquestion and direct_sent:
        detail_tail = (
            "- **Góc nhìn nâng cao:** Cần theo dõi đánh đổi giữa độ chính xác truy xuất, số token đưa vào prompt và tỉ lệ trích đúng nguồn trên bộ eval."
            if level == "Nâng cao"
            else (
                "- **Điểm cần nhớ:** Đây chính là cơ chế trực tiếp giải quyết tình huống bạn vừa hỏi; lưu ý bám sát quy tắc trong đoạn trích để hệ thống không bị sai lệch ở bước tiếp theo."
                if level == "Trung bình"
                else "- **Hiểu ngắn gọn:** Bạn chỉ cần nhớ đúng nguyên tắc trong câu trích trên để biết vì sao bước này phải làm như vậy."
            )
        )
        context_extra = f"\n\n**Ngữ cảnh bổ sung ({cite}):**\n> \"{excerpt}\"" if direct_sent != excerpt and len(excerpt) > len(direct_sent) + 25 else ""
        answer = (
            f"**[Giải đáp cho {learner_dna.name} · Trình độ {level} · {direct_cite}]**\n\n"
            f"> \"{direct_sent}\"\n\n"
            f"{detail_tail}"
            f"{context_extra}"
        )
        follow_up = _build_contextual_followup(query, topic, direct_cite, level, used_followups)
        return answer, follow_up

    if level == "Mới bắt đầu":
        answer = (
            f"**[Giải thích cho {learner_dna.name} · Trình độ Mới bắt đầu]**\n\n"
            f"Hãy hình dung chủ đề **{topic}** giống như cách bạn tìm nhanh một trang sách trong thư viện bằng thẻ mục lục thay vì ngồi đọc lại từ trang đầu đến trang cuối.\n\n"
            f"**Trong tài liệu ({cite}), cơ chế cốt lõi được mô tả rất gọn:**\n"
            f"> \"{excerpt}\"\n\n"
            "Bạn chỉ cần nhớ 2 ý mấu chốt: *(1) Mục đích nó sinh ra để giải quyết vấn đề gì* và *(2) Dữ liệu đi qua nó thay đổi ra sao*. "
            "Chưa cần nhớ hết công thức vội đâu!"
        )
        follow_up = f"Nếu giải thích lại khái niệm **{topic}** cho một người bạn trong 1 câu ngắn, bạn sẽ nói thế nào?"
    elif level == "Nâng cao":
        answer = (
            f"**[Phân tích kỹ thuật cho {learner_dna.name} · Trình độ Nâng cao]**\n\n"
            f"Vì bạn đã nắm vững phần khái niệm cơ bản ({learner_dna.mastery_avg:.0f}% trung bình), mình đi thẳng vào **đánh đổi kỹ thuật (trade-off) và điều kiện biên** của **{topic}** theo tài liệu ({cite}, {cite2}):\n\n"
            f"> \"{excerpt}\"\n\n"
            "- **Điểm nghẽn cần cân bằng:** Mọi tham số (như kích thước chunk, overlap, top-k, temperature) đều đánh đổi giữa *độ chính xác / ngữ cảnh* và *chi phí token / độ nhiễu prompt*.\n"
            "- **Kiểm chứng thực chiến:** Khi tinh chỉnh tham số này, đừng chỉ nhìn câu trả lời hay mà phải chạy lại bộ câu hỏi đo cả tỉ lệ trả lời đúng lẫn tỉ lệ trích đúng nguồn."
        )
        follow_up = (
            f"Trong hệ thống thật dựa trên {cite}, nếu người dùng hỏi câu dài chứa hai chủ đề giao nhau ở ranh giới đoạn, bạn sẽ tinh chỉnh thông số nào để không đánh đổi quá nhiều token?"
        )
    else:
        answer = (
            f"**[Giải thích cho {learner_dna.name} · Trình độ Trung bình]**\n\n"
            f"Về **{topic}**, tài liệu ({cite}) chỉ ra cơ chế hoạt động trực tiếp như sau:\n\n"
            f"> \"{excerpt}\"\n\n"
            "Điểm quan trọng cần nắm là vai trò của bước này trong chuỗi xử lý: đầu vào của nó là gì, xử lý ra kết quả gì và ảnh hưởng thế nào tới bước kế tiếp."
        )
        follow_up = f"Theo tài liệu {cite}, điều gì sẽ xảy ra nếu ta cấu hình quá mức hoặc bỏ qua bước **{topic}** này?"

    return answer, follow_up


def demo_evaluate_teach_back(
    concept: str,
    learner_answer: str,
    retrieved_chunks: list[RetrievedChunk],
    learner_dna: LearnerDNA,
    concept_map: DocumentConceptMap | None = None,
    question_context: str | None = None,
    db_path: str | Path | None = None,
) -> TeachBackEvaluation:
    _log_demo("feynman_teach_back", db_path=db_path, latency_ms=48)
    ans = learner_answer.strip()
    ans_low = ans.lower()
    node = (
        next((c for c in concept_map.concepts if c.name.lower() == concept.lower()), None)
        if concept_map
        else None
    )
    target_page = node.page_ref if node and node.page_ref else ""

    # Thu thập câu nguồn: ưu tiên đúng mục của khái niệm thay vì gom câu từ mục khác.
    prioritized_chunks = (
        sorted(retrieved_chunks, key=lambda c: 0 if target_page and c.page == target_page else 1)
        if target_page
        else list(retrieved_chunks)
    )
    own_citations = [c.page for c in prioritized_chunks if target_page and c.page == target_page]
    citations = list(dict.fromkeys(own_citations or [c.page for c in prioritized_chunks if c.page])) or ["[Tài liệu]"]

    source_sents: list[tuple[str, str]] = []
    for ch in prioritized_chunks:
        for s in _clean_chunk_sentences(ch.content):
            if len(s.strip()) > 20:
                source_sents.append((s.strip(), ch.page or "[Tài liệu]"))

    q_ctx_terms: set[str] = set()
    if question_context:
        q_ctx_terms = {
            _strip_accents(t)
            for t in _tokenize_vi(question_context)
            if len(_strip_accents(t)) >= 3 and _strip_accents(t) not in _STOP_WORDS_FOLDED
        }

    if question_context and len(source_sents) > 1:
        def _sent_rel(item: tuple[str, str]) -> tuple[int, int]:
            sent, page = item
            same_p = 1 if (not target_page or page == target_page) else 0
            s_terms = {
                _strip_accents(t)
                for t in _tokenize_vi(sent)
                if len(_strip_accents(t)) >= 3 and _strip_accents(t) not in _STOP_WORDS_FOLDED
            }
            ctx_hit = len((q_ctx_terms & s_terms))
            ans_terms = {
                _strip_accents(t)
                for t in _tokenize_vi(ans)
                if len(_strip_accents(t)) >= 3 and _strip_accents(t) not in _STOP_WORDS_FOLDED
            }
            return (same_p, ctx_hit + len(ans_terms & s_terms))

        source_sents.sort(key=_sent_rel, reverse=True)

    stop_words = {"là", "và", "của", "các", "cho", "trong", "khi", "để", "này", "một", "những", "với", "như", "thì", "được"}
    ans_fold = _strip_accents(ans_low)
    correct_points: list[str] = []
    missing_points: list[str] = []
    best_ratio = 0.0

    for sent, page in source_sents[:5]:
        tokens = [
            w.strip(".,:;()\"'").lower()
            for w in sent.split()
            if len(w.strip(".,:;()\"'")) >= 3 and w.lower() not in stop_words
        ]
        if not tokens:
            continue
        uniq_tokens = set(tokens)
        hits = sum(1 for t in uniq_tokens if t in ans_low or _strip_accents(t) in ans_fold)
        ratio = hits / max(len(uniq_tokens), 1)
        if ratio > best_ratio:
            best_ratio = ratio
        if ratio >= 0.25:
            correct_points.append(f"Bạn đã nhắc được ý cốt lõi ({page}): {sent}")
        else:
            missing_points.append(f"Cần bổ sung ý ở {page}: {sent}")

    misconceptions: list[str] = []
    # Quy tắc cứng chỉ áp cho giáo trình mặc định; tài liệu khác chỉ dùng so khớp chung ở trên.
    if _is_default_source((c.source for c in retrieved_chunks), " ".join(c.content for c in retrieved_chunks)):
        if "chunk" in concept.lower():
            if _claim_has_wrong_phrase(ans_low, ("mất ngữ cảnh",), require=("lớn",), forbid=("nhỏ",)):
                misconceptions.append("Gán nhầm hậu quả mất ngữ cảnh của chunk quá nhỏ sang chunk quá lớn")
            if _claim_has_wrong_phrase(ans_low, ("bằng 0", "= 0", "không cần", "bỏ qua"), require=("overlap",)):
                misconceptions.append("Hiểu sai tác dụng của overlap: không có overlap sẽ làm cắt đứt ý ở ranh giới chunk")
        if "messages" in concept.lower():
            if _claim_has_wrong_phrase(ans_low, ("tự nhớ", "không cần gửi lại")):
                misconceptions.append("Cho rằng mô hình OpenAI tự nhớ hội thoại trước đó mà không cần gửi lại messages.")
    if len(ans.split()) < 8:
        misconceptions.append("Câu giải thích còn quá ngắn, chưa thể hiện được cơ chế hoạt động.")

    if len(ans.split()) < 5:
        score = 18
    elif question_context and best_ratio >= 0.35 and not misconceptions and len(ans.split()) >= 7:
        # Đang trả lời một câu hỏi Socratic hẹp trong chủ đề -> chấm theo độ chuẩn của câu mục tiêu thay vì trừ điểm các ý ở câu hỏi khác.
        score = max(82, min(96, int(round(65 + 28 * best_ratio + min(10, len(ans.split()) // 2)))))
    else:
        total_checked = max(len(correct_points) + len(missing_points), 1)
        base = int(round((len(correct_points) / total_checked) * 85)) + min(15, len(ans.split()) // 2)
        score = max(15, min(96, base - 22 * len(misconceptions)))

    if score >= 80 and not misconceptions:
        verdict = "Vững bản chất"
        follow_up = f"Tuyệt vời! Để kiểm tra mức Vận dụng ở {citations[0]}, bạn sẽ xử lý thế nào nếu gặp tình huống biên làm giảm độ chính xác của bước này?"
    elif score >= 50:
        verdict = "Hiểu một phần"
        target_miss = missing_points[0] if missing_points else f"chi tiết kỹ thuật tại {citations[0]}"
        follow_up = f"Bạn đã nắm được phần nền khá ổn. Tuy nhiên nhìn vào mục thiếu ({target_miss}), bạn có thể bổ sung cơ chế đó hoạt động ra sao không?"
    else:
        verdict = "Đang hiểu lệch" if misconceptions else "Hiểu một phần"
        follow_up = (
            f"Hãy đọc lại gợi ý ở {citations[0]} rồi thử giải thích lại ngắn gọn: "
            f"nếu bỏ qua nguyên tắc chính của '{concept}', hậu quả trực tiếp là gì?"
        )

    return TeachBackEvaluation(
        topic=concept,
        understanding_score=score,
        verdict=verdict,
        correct_points=correct_points[:3],
        missing_points=missing_points[:3],
        misconceptions_found=misconceptions,
        socratic_follow_up=follow_up,
        citations=citations[:3],
    )


def demo_critique_image(
    image_bytes: bytes,
    note: str,
    learner_dna: LearnerDNA,
    sample_id: str | None = None,
    db_path: str | Path | None = None,
) -> VisionCritique:
    _log_demo("vision_critique", db_path=db_path, latency_ms=80)
    digest = hashlib.sha256(image_bytes).hexdigest()[:12]

    if sample_id not in ("code-openai-loi", "so-do-rag-loi"):
        # Ảnh người dùng tải lên: Demo Offline KHÔNG đọc được ảnh -> nói thật, không bịa nội dung/lỗi.
        return VisionCritique(
            extracted_work="Chế độ Demo Offline không đọc được nội dung ảnh; hãy bật Live để chấm ảnh của bạn.",
            topic="Chưa đọc được ảnh",
            overall_assessment=(
                "Chế độ Demo Offline không phân tích được ảnh bạn tải lên nên chưa có nhận xét nào. "
                "Bật chế độ Live (có API key) để chấm bài làm trong ảnh."
            ),
            steps=[],
            hint_for_learner="Bật Live để chấm ảnh của bạn, hoặc chọn một trong hai ảnh mẫu để xem phê bình minh họa.",
            detected_misconception="",
            citations=[],
            is_demo=True,
        )

    is_code_sample = sample_id == "code-openai-loi"

    if is_code_sample:
        hint = (
            "Gợi ý cho bạn: Tách API key ra file `.env`, gom các lượt chat vào danh sách `messages`, và đưa `temperature` về khoảng `0.0–0.2` nếu muốn trích xuất ổn định."
            if learner_dna.overall_level == "Mới bắt đầu"
            else "Đọc lại 3 điểm cấu hình ở [Mục 1–3 · API]: lưu ý vòng đời stateful của `messages` phía client và tác động của `temperature > 1.0` lên JSON parsing."
        )
        return VisionCritique(
            extracted_work=(
                "Đoạn code học viên viết: client = OpenAI(api_key='sk-live-12345...'); "
                "client.chat.completions.create(model='GPT-4o-mini', messages=[{'role':'user','content': q}], temperature=1.8)"
            ),
            topic="Cấu trúc lời gọi & Messages",
            overall_assessment="Đoạn code gọi được API nhưng mắc 3 lỗi nghiêm trọng về bảo mật key, bộ nhớ hội thoại và độ ổn định.",
            steps=[
                VisionStepCheck(
                    step_name="Bước 1: Khởi tạo kết nối OpenAI",
                    status="Sai",
                    feedback="Bạn đang ghi cứng API key trực tiếp trong mã nguồn ('sk-live-...'). Theo tài liệu, key là bí mật phải đọc từ `.env`.",
                    page_ref="[Mục 1 · API]",
                ),
                VisionStepCheck(
                    step_name="Bước 2: Truyền ngữ cảnh hội thoại (messages)",
                    status="Cần chỉnh",
                    feedback="Bạn chỉ truyền 1 message 'user' mới nhất. Mô hình không tự nhớ lượt trước nên phải gửi kèm lịch sử các lượt trước.",
                    page_ref="[Mục 2 · API]",
                ),
                VisionStepCheck(
                    step_name="Bước 3: Tham số temperature = 1.8",
                    status="Sai",
                    feedback="Đặt temperature = 1.8 làm câu trả lời rất ngẫu nhiên và dễ bịa, trái nguyên tắc dùng giá trị thấp để trích xuất ổn định.",
                    page_ref="[Mục 3 · API]",
                ),
            ],
            hint_for_learner=hint,
            detected_misconception="Tưởng mô hình OpenAI tự nhớ hội thoại trước đó và ghi cứng API key trong code",
            citations=["[Mục 1 · API]", "[Mục 2 · API]", "[Mục 3 · API]"],
            is_demo=True,
        )

    hint = (
        "Gợi ý từng bước: Hãy đưa mũi tên truy xuất Vector Store lên TRƯỚC khi gọi LLM, giảm chunk xuống vừa phải và bật overlap để ý không bị cắt đứt."
        if learner_dna.overall_level == "Mới bắt đầu"
        else "Kiểm tra lại thứ tự Pipeline ở [Mục 1–2]: độ nhiễu sẽ tăng vọt khi chunk 50.000 ký tự chứa nhiều chủ đề giao nhau và không có overlap ở biên."
    )
    return VisionCritique(
        extracted_work=(
            f"Sơ đồ học viên phác thảo (hash ảnh {digest}): "
            "Bước 1: Chia chunk 50.000 ký tự, overlap = 0 -> Bước 2: Câu hỏi đi thẳng vào LLM -> Bước 3: Lưu Vector Store cuối cùng."
        ),
        topic="Chunking và Overlap",
        overall_assessment="Học viên đã nhận ra các thành phần của RAG nhưng sắp xếp ngược thứ tự truy xuất và chọn sai cấu hình chunking.",
        steps=[
            VisionStepCheck(
                step_name="Bước 1: Thiết lập Chunking (size = 50.000, overlap = 0)",
                status="Sai",
                feedback="Chunk quá lớn (50.000 ký tự) sẽ gộp nhiều ý lẫn lộn làm truy xuất kém chính xác; overlap = 0 khiến ý nằm ở ranh giới bị cắt đứt.",
                page_ref="[Mục 2]",
            ),
            VisionStepCheck(
                step_name="Bước 2: Thứ tự luồng Câu hỏi -> LLM trước Vector Store",
                status="Sai",
                feedback="Trái nguyên tắc 'Truy xuất rồi mới trả lời': phải tìm top-k chunk liên quan từ Vector Store trước rồi mới đưa vào prompt cho LLM.",
                page_ref="[Mục 1, Mục 4]",
            ),
            VisionStepCheck(
                step_name="Bước 3: Đưa top-k nhỏ và gắn metadata nguồn",
                status="Đúng",
                feedback="Ý tưởng giữ top-k nhỏ và đánh dấu tên tài liệu + số trang hợp lý để trích nguồn.",
                page_ref="[Mục 2, Mục 4]",
            ),
        ],
        hint_for_learner=hint,
        detected_misconception="Nhầm thứ tự truy xuất RAG và đặt chunk quá lớn không có overlap",
        citations=["[Mục 1]", "[Mục 2]", "[Mục 4]"],
        is_demo=True,
    )
