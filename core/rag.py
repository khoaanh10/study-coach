"""Tầng RAG Dual-Memory: Embeddings + FAISS + LangChain Text Splitter + Ký ức lỗi sai cá nhân."""
from __future__ import annotations

import hashlib
import re
import unicodedata
import warnings
from typing import Iterable

warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

import numpy as np  # noqa: E402
from openai import OpenAIError  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

import config  # noqa: E402
from core.contracts import ConceptMastery, LearnerDNA, RetrievedChunk  # noqa: E402
from core.llm import LLMError, friendly_error  # noqa: E402

_STOP_WORDS = {
    "là", "gì", "nào", "thế", "sao", "như", "của", "và", "các", "những", "cho", "trong",
    "khi", "để", "này", "một", "với", "tại", "về", "hay", "hoặc", "được", "không", "có",
    "tôi", "bạn", "mình", "giải", "thích", "hãy", "hỏi", "câu", "phần", "theo", "tài", "liệu",
    "xin", "chào", "giúp", "khác", "nhau", "cách", "làm", "việc", "hơn", "đi", "nhé",
    "em", "anh", "thầy", "cô", "ví", "dụ", "tóm", "tắt", "toàn", "bộ", "cụ", "thể",
    "ngắn", "gọn", "kỹ", "rõ", "phải", "cần", "nên", "đang", "vừa", "thử", "xem",
    "thêm", "nữa", "rằng", "thì", "vào", "ra", "lên", "xuống", "lắm", "bao", "nhiêu", "hôm", "nay", "gồm",
    "what", "is", "are", "the", "a", "an", "in", "on", "of", "for", "to", "and", "or", "how", "why",
}


def _strip_accents(text: str) -> str:
    """Chuẩn hóa tiếng Việt có dấu sang không dấu để hỗ trợ tìm kiếm ngữ nghĩa khi người dùng gõ không dấu."""
    nfd = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


_STOP_WORDS_FOLDED = {_strip_accents(w) for w in _STOP_WORDS}

# Cầu nối đồng nghĩa nhẹ giúp embedding cục bộ nhận diện ngữ nghĩa Việt-Anh phổ biến trong tài liệu AI.
_SYNONYM_BRIDGES: dict[str, list[str]] = {
    "chi phí": ["giá thành", "tiền", "tốn token", "cost", "budget"],
    "giá thành": ["chi phí", "tốn token", "cost"],
    "bịa": ["hallucination", "sai sự thật", "không bám nguồn"],
    "trí nhớ": ["nhớ hội thoại", "messages", "lịch sử"],
    "nhớ": ["lịch sử messages", "hội thoại nhiều lượt", "stateless"],
    "lỗi": ["nhầm lẫn", "misconception", "ngoại lệ", "rate limit"],
    "tìm kiếm": ["truy xuất", "retrieval", "semantic search"],
    "truy xuất": ["retrieval", "tìm kiếm", "top-k"],
    "ngữ nghĩa": ["semantic", "embedding", "vector", "cosine"],
    "độ chính xác": ["tỉ lệ đúng", "trích đúng nguồn", "chất lượng"],
}


# Âm tiết tiếng Việt viết không dấu (vd "chung", "ghi", "tin"): không phải thuật ngữ kỹ thuật Anh/mã.
_VN_SYLLABLE_RE = re.compile(r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?[aeiouy]+(ng|nh|ch|c|m|n|p|t)?$")


def _is_technical_term(tok: str) -> bool:
    """Token ASCII ≥3 ký tự trông giống thuật ngữ Anh/mã (docker, chunk, top-k...) chứ không phải âm tiết Việt không dấu."""
    return len(tok) >= 3 and bool(re.match(r"^[a-z0-9_-]+$", tok)) and not _VN_SYLLABLE_RE.match(tok)


def _tokenize_vi(text: str) -> list[str]:
    clean = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ-]", " ", text.lower())
    return [t for t in clean.split() if len(t) >= 2]


class LocalNGramEmbeddings(Embeddings):
    """Embedding thuần NumPy (384 chiều chuẩn hóa L2) kế thừa chuẩn LangChain Embeddings.

    Dùng cho Chế độ Demo Offline, test tự động, và dự phòng khi cổng không cấp endpoint embedding.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _add_feature(self, vec: np.ndarray, feat: str, weight: float) -> None:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "little")
        idx = h % self.dim
        sign = 1.0 if ((h >> 17) & 1) == 0 else -1.0
        vec[idx] += sign * weight

    def embed_query(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        low = text.lower()
        folded_low = _strip_accents(low)
        expanded = [low]
        for k, syns in _SYNONYM_BRIDGES.items():
            if k in low or _strip_accents(k) in folded_low:
                expanded.extend(syns)
        merged = " ".join(expanded)

        words = _tokenize_vi(merged)
        for w in words:
            folded_w = _strip_accents(w)
            is_stop = (w in _STOP_WORDS) or (folded_w in _STOP_WORDS_FOLDED)
            if not is_stop:
                self._add_feature(vec, f"w:{w}", 2.2)
                if len(folded_w) >= 2:
                    self._add_feature(vec, f"wf:{folded_w}", 1.85)
                padded = f"<{w}>"
                for n in (3, 4):
                    for i in range(max(0, len(padded) - n + 1)):
                        self._add_feature(vec, f"c{n}:{padded[i:i + n]}", 0.7)
            else:
                self._add_feature(vec, f"s:{w}", 0.25)

        for i in range(len(words) - 1):
            fw1, fw2 = _strip_accents(words[i]), _strip_accents(words[i + 1])
            if fw1 in _STOP_WORDS_FOLDED and fw2 in _STOP_WORDS_FOLDED:
                continue
            self._add_feature(vec, f"b:{words[i]}_{words[i + 1]}", 1.5)
            if len(fw1) >= 2 and len(fw2) >= 2:
                self._add_feature(vec, f"bf:{fw1}_{fw2}", 1.3)

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def build_embeddings(demo_mode: bool = False) -> Embeddings:
    """Dùng OpenAIEmbeddings khi có live API key và không ở chế độ Demo Offline."""
    if not demo_mode and config.has_live_api_key():
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.get_effective_api_key(),
            base_url=config.get_effective_base_url() or None,
        )
    return LocalNGramEmbeddings()


class DualMemoryKnowledgeBase:
    """Cơ sở tri thức RAG 2 ngăn trên LangChain + FAISS:
    - Ngăn 1 (`doc_store`): chunk tài liệu học kèm metadata trang/mục.
    - Ngăn 2 (`mistake_store`): vector lỗi tư duy cá nhân từ SQLite để cá nhân hóa truy hồi.
    """

    def __init__(
        self,
        sections: list[dict],
        embeddings: Embeddings | None = None,
        fallback_text: str = "",
    ) -> None:
        self.embeddings = embeddings or LocalNGramEmbeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=550,
            chunk_overlap=90,
            separators=["\n\n", "\n", ". ", " "],
        )
        self.documents: list[Document] = []
        self.corpus_vocab: set[str] = set()
        self.corpus_vocab_folded: set[str] = set()
        self.doc_store: FAISS | None = None
        self.mistake_store: FAISS | None = None
        # Chỉ có khi embedding chính KHÔNG phải bản cục bộ: dùng riêng cho cổng ngoài-tài-liệu (xem _build_doc_index).
        self.scope_store: FAISS | None = None
        self._build_doc_index(sections, fallback_text)

    def _build_doc_index(self, sections: list[dict], fallback_text: str) -> None:
        raw_docs: list[Document] = []
        if sections:
            for sec in sections:
                txt = (sec.get("text") or "").strip()
                if not txt:
                    continue
                raw_docs.append(
                    Document(
                        page_content=txt,
                        metadata={
                            "source": sec.get("source", "Tài liệu"),
                            "page": sec.get("page", "[Tài liệu]"),
                            "topic": sec.get("topic_hint", "Kiến thức chung"),
                        },
                    )
                )
        elif fallback_text.strip():
            raw_docs.append(
                Document(
                    page_content=fallback_text.strip(),
                    metadata={"source": "Văn bản", "page": "[Tài liệu]", "topic": "Kiến thức chung"},
                )
            )

        if not raw_docs:
            return

        self.documents = self.splitter.split_documents(raw_docs)
        for doc in self.documents:
            for tok in _tokenize_vi(doc.page_content):
                if tok not in _STOP_WORDS:
                    self.corpus_vocab.add(tok)
                ftok = _strip_accents(tok)
                if len(ftok) >= 3 and ftok not in _STOP_WORDS_FOLDED:
                    self.corpus_vocab_folded.add(ftok)

        try:
            self.doc_store = FAISS.from_documents(self.documents, self.embeddings)
        except Exception:
            # Dự phòng tự động sang embedding cục bộ nếu cổng không hỗ trợ endpoint embedding.
            self.embeddings = LocalNGramEmbeddings()
            self.doc_store = FAISS.from_documents(self.documents, self.embeddings)

        if not isinstance(self.embeddings, LocalNGramEmbeddings):
            # Các ngưỡng của cổng ngoài-tài-liệu trong retrieve() được chỉnh theo thang điểm của embedding cục bộ.
            # Điểm cosine của embedding OpenAI cao hơn hẳn (câu ngoài tài liệu vẫn đạt 0,2 đến 0,4), nên dùng chung
            # ngưỡng sẽ để lọt câu ngoài tài liệu. Vì vậy: OpenAI chỉ xếp hạng đoạn trích, còn quyết định
            # trong/ngoài tài liệu dùng điểm của bản cục bộ (chạy offline, rất nhẹ).
            self.scope_store = FAISS.from_documents(self.documents, LocalNGramEmbeddings())

    def sync_learner_mistakes(self, learner_dna: LearnerDNA) -> None:
        """Đồng bộ vector store ngăn 2 từ danh sách lỗi tư duy của người học."""
        mistake_docs: list[Document] = []
        for m in learner_dna.active_misconceptions:
            mistake_docs.append(
                Document(
                    page_content=f"Chủ đề {m.topic}: {m.description}",
                    metadata={"topic": m.topic, "resolved": False, "occurrences": m.occurrences},
                )
            )
        if not mistake_docs:
            self.mistake_store = None
            return
        try:
            self.mistake_store = FAISS.from_documents(mistake_docs, self.embeddings)
        except Exception:
            self.mistake_store = FAISS.from_documents(mistake_docs, LocalNGramEmbeddings())

    def retrieve(self, query: str, top_k: int = 4) -> tuple[list[RetrievedChunk], bool]:
        """Truy xuất top-k đoạn tài liệu liên quan và xác định câu hỏi có nằm trong tài liệu không."""
        if not self.doc_store:
            return [], False

        try:
            pairs = self.doc_store.similarity_search_with_score(query, k=top_k)
        except OpenAIError as e:  # key/mạng/rate limit khi embed câu hỏi bằng OpenAIEmbeddings
            raise LLMError(friendly_error(e)) from e
        chunks: list[RetrievedChunk] = []
        for doc, l2_sq in pairs:
            # Với vector chuẩn hóa L2: cosine = 1 - ||u - v||^2 / 2
            sim = max(0.0, min(1.0, 1.0 - float(l2_sq) / 2.0))
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    source=str(doc.metadata.get("source", "")),
                    page=str(doc.metadata.get("page", "[Tài liệu]")),
                    topic=str(doc.metadata.get("topic", "")),
                    score=round(sim, 3),
                    memory_type="document",
                )
            )

        if self.scope_store is not None:
            scope_pairs = self.scope_store.similarity_search_with_score(query, k=1)
            best_score = max(0.0, min(1.0, 1.0 - float(scope_pairs[0][1]) / 2.0)) if scope_pairs else 0.0
        else:
            best_score = chunks[0].score if chunks else 0.0
        q_tokens = [
            t for t in _tokenize_vi(query)
            if t not in _STOP_WORDS and _strip_accents(t) not in _STOP_WORDS_FOLDED
        ]
        expanded_q_tokens = set(q_tokens)
        q_low = query.lower()
        folded_q = _strip_accents(q_low)
        for k, syns in _SYNONYM_BRIDGES.items():
            if k in q_low or _strip_accents(k) in folded_q:
                for s in syns:
                    expanded_q_tokens.update(_tokenize_vi(s))

        has_vn_accents = folded_q != q_low
        if has_vn_accents:
            lexical_hits = sum(1 for t in expanded_q_tokens if t in self.corpus_vocab)
            english_or_code_terms = [
                t for t in q_tokens
                if _is_technical_term(t) and _strip_accents(t) not in _STOP_WORDS_FOLDED
            ]
            unknown_tech_terms = [t for t in english_or_code_terms if t not in self.corpus_vocab]
            folded_content_a = [_strip_accents(t) for t in q_tokens if len(_strip_accents(t)) >= 3]
            unknown_folded_a = [t for t in folded_content_a if t not in self.corpus_vocab_folded]
            strong_hits = sum(1 for t in english_or_code_terms if t in self.corpus_vocab)

            if english_or_code_terms and len(unknown_tech_terms) == len(english_or_code_terms) and best_score < 0.42:
                is_in_scope = False
            elif lexical_hits == 0 and best_score < 0.28:
                is_in_scope = False
            elif strong_hits == 0 and best_score < 0.2:
                # Không có thuật ngữ kỹ thuật (Anh/mã) nào trùng corpus và độ tương đồng thấp:
                # chỉ trùng vài âm tiết phổ thông ("giá", "tin", "chính") thì không đủ chứng cứ trong tài liệu.
                is_in_scope = False
            elif strong_hits == 0 and best_score < 0.25 and (
                (folded_content_a and len(unknown_folded_a) / len(folded_content_a) >= 0.5) or lexical_hits <= 1
            ):
                is_in_scope = False
            else:
                is_in_scope = best_score >= 0.14 or lexical_hits >= 1
        else:
            if not q_tokens:
                is_in_scope = False
            else:
                folded_content = [_strip_accents(t) for t in q_tokens if len(_strip_accents(t)) >= 3]
                expanded_folded = {_strip_accents(t) for t in expanded_q_tokens if len(_strip_accents(t)) >= 3}
                folded_hits = sum(1 for t in expanded_folded if t in self.corpus_vocab_folded)
                unknown_tokens = [t for t in folded_content if t not in self.corpus_vocab_folded]
                if folded_hits == 0 or best_score < 0.09:
                    is_in_scope = False
                elif folded_content and (len(unknown_tokens) / len(folded_content)) > 0.5 and best_score < 0.42:
                    is_in_scope = False
                else:
                    is_in_scope = folded_hits >= max(1, (len(q_tokens) + 1) // 2) or best_score >= 0.28

        return chunks, is_in_scope

    def retrieve_past_mistakes(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
        if not self.mistake_store:
            return []
        q_content = {
            _strip_accents(t)
            for t in _tokenize_vi(query)
            if len(_strip_accents(t)) >= 3 and _strip_accents(t) not in _STOP_WORDS_FOLDED
        }
        if not q_content:
            return []
        try:
            pairs = self.mistake_store.similarity_search_with_score(query, k=top_k)
        except OpenAIError as e:
            raise LLMError(friendly_error(e)) from e
        out: list[RetrievedChunk] = []
        for doc, l2_sq in pairs:
            sim = max(0.0, min(1.0, 1.0 - float(l2_sq) / 2.0))
            if sim < 0.24:
                continue
            doc_tokens = {
                _strip_accents(t)
                for t in _tokenize_vi(doc.page_content)
                if len(_strip_accents(t)) >= 3 and _strip_accents(t) not in _STOP_WORDS_FOLDED
            }
            if q_content & doc_tokens:
                out.append(
                    RetrievedChunk(
                        content=doc.page_content,
                        topic=str(doc.metadata.get("topic", "")),
                        page="Hồ sơ lỗi tư duy",
                        score=round(sim, 3),
                        memory_type="mistake",
                    )
                )
        return out


def compute_coverage_stats(masteries: Iterable[ConceptMastery]) -> dict:
    items = list(masteries)
    total = len(items)
    if total == 0:
        return {
            "total": 0,
            "mastered": 0,
            "learning": 0,
            "misconception": 0,
            "unseen": 0,
            "due": 0,
            "coverage_pct": 0.0,
        }
    mastered = sum(1 for c in items if c.status == "mastered")
    learning = sum(1 for c in items if c.status == "learning")
    misconception = sum(1 for c in items if c.status == "misconception")
    unseen = sum(1 for c in items if c.status == "unseen")
    due = sum(1 for c in items if c.due_for_review)
    weighted = sum(min(c.mastery_score, 100.0) for c in items if c.attempts > 0) / total
    return {
        "total": total,
        "mastered": mastered,
        "learning": learning,
        "misconception": misconception,
        "unseen": unseen,
        "due": due,
        "coverage_pct": round(weighted, 1),
    }
