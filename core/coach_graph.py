"""Agent Gia sư Socratic điều phối bằng LangGraph (Tuần 5) + Function Calling (Tuần 2) + LangChain Prompts."""
from __future__ import annotations

import re
import time
import weakref
from pathlib import Path
from typing import Iterator, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import config
from core import adaptive, db
from core import llm as _llm
from core.contracts import (
    CoachTurnResult,
    DocumentConceptMap,
    LearnerDNA,
    RetrievedChunk,
    ToolCallTrace,
)
from core.demo_provider import demo_coach_conversational_reply, demo_coach_response, demo_review_response
from core.llm import complete_structured, complete_with_tools, stream_chat_completion
from core.prompts import SOCRATIC_COACH_PROMPTS
from core.rag import (
    _STOP_WORDS_FOLDED,
    DualMemoryKnowledgeBase,
    _is_technical_term,
    _strip_accents,
    _tokenize_vi,
)

# Từ "đệm" của câu hỏi nối tiếp (đã bỏ dấu): không mang nội dung chủ đề mới.
_FOLLOWUP_FILLER = {
    "vi", "du", "minh", "hoa", "thuc", "te", "tinh", "huong", "chi", "tiet", "hon", "ky", "ro",
    "hieu", "cu", "the", "noi", "them", "sau", "chua", "giai", "thich", "lai", "dieu", "nay",
    "kia", "cai", "phan", "tren", "vua", "roi", "nua", "khac", "toi", "ban", "cho", "hay",
    "ko", "k", "khong", "biet", "em", "di", "nhe", "nha", "giup", "cau", "do", "giang",
}
# Cụm mở đầu thể hiện YÊU CẦU (không phải câu trả lời cho câu hỏi Socratic).
_REQUEST_PREFIXES = (
    "minh muon", "toi muon", "em muon", "cho minh", "cho toi", "cho em", "hay ", "giai thich",
    "muon hoc", "muon biet", "nhac lai", "tom tat", "hoc them", "chi cho", "ban co the",
)
# Ý định hội thoại thuần: không tra tài liệu, trả lời bám lịch sử hội thoại.
_CONVERSATIONAL_INTENTS = {"acknowledgement", "confirm_answer", "conversation_repair", "next_question"}
_GREETING_WORDS = {"hi", "hello", "hey", "chao", "thank", "thanks"}
_GREETING_PHRASES = (("xin", "chao"), ("cam", "on"))


def _fold_words(text: str) -> list[str]:
    """Tách từ (đã bỏ dấu, chữ thường, bỏ dấu câu) để so khớp THEO TỪ thay vì chuỗi con."""
    return re.findall(r"[a-z0-9]+", _strip_accents(text.lower()))


def _is_greeting_text(q_fold: str) -> bool:
    toks = _fold_words(q_fold)
    if any(t in _GREETING_WORDS for t in toks):
        return True
    return any(
        toks[i:i + len(ph)] == list(ph) for ph in _GREETING_PHRASES for i in range(len(toks) - len(ph) + 1)
    )


# Phản hồi ngắn của người học cho lượt trước ("ok", "vâng", "hiểu rồi", "tiếp đi"...). Đây KHÔNG phải câu hỏi
# về tài liệu nên không được đi qua cổng ngoài-tài-liệu.
_ACK_RE = re.compile(
    r"^(?:ok(?:ay|e|ie)?|vang|da|u|uh|um|duoc|co|dung|chuan|tot|hay|tuyet|ro|phai|"
    r"hieu|da hieu|minh hieu|em hieu|toi hieu|ro roi|tiep|tiep tuc|tiep di|tiep nao|"
    r"duoc roi|dung roi|phai roi|chac chan|tat nhien|hieu roi)"
    r"(?:\s+(?:nhe|nha|a|ha|nhi|roi|di|ban|em|thay|co|ne|nao|qua|lam|that|day|vay|chu|dau))*$"
)
_NOT_UNDERSTOOD_RE = re.compile(
    r"^(?:khong|chua|khong hieu|chua hieu|chua ro|khong ro|em chua hieu|minh chua hieu|toi chua hieu)"
    r"(?:\s+(?:a|nhe|nha|lam|ro|ha))*$"
)


def _is_acknowledgement(q_fold: str) -> str:
    """Trả về 'ack' (đồng ý/đã hiểu), 'confused' (chưa hiểu) hoặc '' nếu không phải phản hồi ngắn."""
    toks = _fold_words(q_fold)
    if not toks or len(toks) > 5 or q_fold.strip().endswith("?"):
        return ""
    text = " ".join(toks)
    if _NOT_UNDERSTOOD_RE.match(text):
        return "confused"
    if _ACK_RE.match(text):
        return "ack"
    return ""


# Yêu cầu ôn tập ("ôn phần chunking", "ôn lại RAG", "củng cố embeddings"): người học muốn được TÓM TẮT/ÔN LẠI chủ đề,
# không phải bị chất vấn về lỗi tư duy.
_REVIEW_FOLDED_RE = re.compile(
    r"\b(?:on tap|on lai|on bai|on phan|on muc|on chu de|on giup|on them|cung co|hoc lai|review)\b"
)
_REVIEW_FILLER = {
    "on", "tap", "lai", "phan", "bai", "muc", "chu", "de", "giup", "them", "cung", "co", "hoc", "review", "nao",
    "do", "kia", "nhe", "nha", "di", "muon", "can", "toi", "minh", "em", "cho", "mot", "vai", "nua", "voi",
}


def _is_review_request(query: str) -> bool:
    q_low = query.lower()
    if re.search(r"\bôn\b", q_low):  # "ôn" có dấu: không nhầm với "on" tiếng Anh
        return True
    return bool(_REVIEW_FOLDED_RE.search(_strip_accents(q_low)))


def _match_topic_in_query(query: str, concept_map: DocumentConceptMap) -> str:
    """Tên khái niệm mà câu hỏi nhắc tới rõ ràng (theo tên, từ khóa hoặc từ đặc trưng trong tên); '' nếu không có."""
    q_low = query.lower()
    q_fold = _strip_accents(q_low)
    best_name, best_score = "", 0
    for node in concept_map.concepts:
        score = 0
        if node.name.lower() in q_low or _strip_accents(node.name.lower()) in q_fold:
            score += 5
        score += sum(2 for kw in node.key_terms if len(kw.strip()) >= 3 and _strip_accents(kw.lower()) in q_fold)
        name_tokens = [t for t in _fold_words(node.name) if len(t) >= 4 and t not in _STOP_WORDS_FOLDED]
        score += sum(1 for t in name_tokens if t in q_fold)
        if score > best_score:
            best_name, best_score = node.name, score
    return best_name


def _pick_review_topic(learner_dna: LearnerDNA, concept_map: DocumentConceptMap) -> str:
    """Khi người học không nêu phần nào: ưu tiên phần đến hạn ôn hoặc điểm thấp nhất trong hồ sơ."""
    names = [c.name for c in concept_map.concepts]
    cands = [c for c in learner_dna.concepts if c.concept in names and c.attempts > 0]
    if cands:
        cands.sort(key=lambda c: (0 if c.due_for_review else 1, c.mastery_score))
        return cands[0].concept
    return names[0] if names else "Kiến thức chung"


def _followup_content_terms(query: str) -> list[str]:
    """Các từ nội dung (không phải từ đệm/stopword) trong câu hỏi nối tiếp."""
    return [
        t for t in _fold_words(query)
        if len(t) >= 3 and t not in _STOP_WORDS_FOLDED and t not in _FOLLOWUP_FILLER
    ]


class CoachState(TypedDict):
    query: str
    learner_id: str
    doc_id: str
    focus_topic: str | None
    remediation_context: str | None
    model: str | None
    demo_mode: bool
    chat_history: list[dict]
    learner_dna: LearnerDNA
    concept_map: DocumentConceptMap
    conv_intent: str
    last_topic: str
    last_route: str
    last_follow_up: str
    topic: str
    is_in_scope: bool
    prerequisite_topic: str | None
    retrieved_chunks: list[RetrievedChunk]
    past_mistakes: list[RetrievedChunk]
    tool_traces: list[ToolCallTrace]
    route_taken: str
    pedagogical_reason: str
    answer: str
    follow_up_question: str
    suggested_quiz_topic: str
    start_ts: float
    cost_usd: float


def _last_question_in(text: str) -> str:
    """Câu hỏi cuối cùng (kết thúc bằng '?') trong một đoạn văn; '' nếu không có."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text.strip()) if s.strip()]
    for s in reversed(sentences):
        if s.endswith("?"):
            return s[:260]
    return ""


# Người học phàn nàn/nhắc lại mạch hội thoại ("bạn hỏi xong mình trả lời có mà", "sao lặp lại vậy"...).
_REPAIR_MARKERS = (
    "ban hoi", "minh tra loi", "em tra loi", "toi tra loi", "da tra loi", "tra loi roi", "tra loi co",
    "sao lai", "lap lai", "noi lai", "nhac lai hoai", "ban khong nghe", "khong nghe", "khong doc",
    "khong lien quan", "lac de", "tra loi sai", "hoi gi", "hieu sai", "vua noi", "noi roi",
)


def _is_conversation_repair(q_fold: str) -> bool:
    return any(m in q_fold for m in _REPAIR_MARKERS) and not any(
        t in q_fold for t in (" la gi", "giai thich", "tai lieu")
    )


# Cách nói đời thường/teencode ("k hiểu", "ý là sao z", "chịu luôn", "nói lại dễ hiểu đi") sau lượt Socratic:
# vẫn là nối tiếp chủ đề, không phải câu hỏi kiến thức mới nên không được rơi vào cổng ngoài-tài-liệu.
_CONFUSION_RE = re.compile(
    r"\b(?:k|ko|khong|chua|hok|hong)\s+(?:hieu|ro|biet)\b"
    r"|\bhieu\s+(?:lam|khong)\b"
    r"|\bchiu(?:\s+(?:luon|thua|roi|hong))?\b"
    r"|\b(?:mo\s+ho|kho\s+hieu|roi\s+loan)\b"
)
_MEANING_QUESTION_RE = re.compile(
    r"\b(?:y|nghia)(?:\s+la)?\s+(?:sao|gi)\b"
    r"|\bla\s+sao\s+(?:z|za|vay|the|v)\b"
)
_EXPLAIN_REQUEST_RE = re.compile(
    r"\b(?:noi\s+lai|nhac\s+lai|giai\s+thich|giang|di[eê]n\s+giai|noi\s+ro|noi\s+them|"
    r"de\s+hieu|don\s+gia[n]?\s+hon|chi\s+tiet|cu\s+the|ro\s+hon|gium|giup)\b"
)
_EXAMPLE_REQUEST_RE = re.compile(r"\b(?:vi\s+du|vd|minh\s+hoa|doi\s+thuong)\b")
# Dạng câu hỏi kiến thức MỚI (không phải xin giảng lại chủ đề đang bàn).
_NEW_QUESTION_RE = re.compile(
    r"\b(?:la\s+gi|the\s+nao|ra\s+sao|tai\s+sao|vi\s+sao|khac\s+gi|hoat\s+dong\s+sao|bang\s+cach\s+nao)\b"
)
_HEDGE_GUESS_RE = re.compile(
    r"^(?:(?:o|oi|u|uh|um|a|thi|thoi)\s+)*"
    r"(?:chac(?:\s+(?:vay|v|the))?|co\s+le|minh\s+doan|em\s+doan|toi\s+doan|"
    r"doan\s+vay|khong\s+chac|khong\s+han|tuy(?:\s+truong\s+hop)?)"
    r"(?:\s+(?:a|ha|nhi|nhe|nha|vay|v|do|day|luon|qua))?$"
)


_CONTINUE_PHRASES = (
    "bat dau", "giang luon", "di sau", "phan do", "muc do", "hoc luon",
    "cai dau tien", "phan dau tien", "muc dau", "hoc phan", "luon di", "luon nhe",
)
_NAV_OFFER_PHRASES = (
    "ban muon bat dau", "ban muon minh di sau", "ban muon chuyen sang", "tom tat nhanh",
    "ban muon on phan nao", "dien dat lai",
)
_COREF_RE = re.compile(
    r"\b(?:no|cai do|phan do|buoc nay|buoc do|viec nay|nhu vay|the con|vay con|vay no|"
    r"tai sao vay|vi sao vay|khi nao vay)\b"
)


def _is_continuation_accept(
    query: str, q_fold: str, last_topic: str, last_route: str, last_follow_up: str
) -> str:
    """Trả về 'doc_summary' hoặc 'continue_topic' khi người học đồng ý với đề xuất ở lượt trước."""
    if query.strip().endswith("?") or len(_fold_words(q_fold)) > 12:
        return ""
    if any(end_w in q_fold for end_w in (" la gi", " the nao", " ra sao", " khac gi", " tai sao", " vi sao")):
        return ""
    follow_fold = _strip_accents(last_follow_up.lower())
    if "tom tat" in q_fold and ("tom tat" in follow_fold or last_route in {"coach_guide", "refuse_out_of_doc"}):
        return "doc_summary"
    if any(p in q_fold for p in _CONTINUE_PHRASES) and (last_follow_up or last_topic or last_route in {"coach_guide", "refuse_out_of_doc"}):
        return "continue_topic"
    if _is_acknowledgement(q_fold) == "ack" and (
        last_route in {"coach_guide", "refuse_out_of_doc"} or any(p in follow_fold for p in _NAV_OFFER_PHRASES)
    ):
        return "continue_topic"
    return ""


def _is_yes_no_knowledge_question(last_follow_up: str) -> bool:
    """Chỉ coi là câu hỏi Có/Không khi thực sự hỏi một mệnh đề đúng/sai, không phải hỏi cảm nhận hoặc tự luận."""
    q_fold = _strip_accents(last_follow_up.lower().strip())
    if not q_fold.endswith("?"):
        return False
    if q_fold.endswith("khong?") and q_fold.startswith(
        ("ban co thay", "ban co nghi", "ban co dong y", "ban co cho rang")
    ):
        return True
    non_yn_markers = (
        "ban co hieu", "co hieu", "ban co nho", "hieu khong", "duoc khong",
        "ban muon", "muon minh", "bat dau", "xem?",
        "tai sao", "vi sao", "nhu the nao", "the nao", "phan biet",
        "giai thich", "tom tat", "vi du", "xu ly", "thong so nao", "cai gi", "la gi", "ra sao",
    )
    if any(m in q_fold for m in non_yn_markers):
        return False
    return any(m in q_fold for m in ("co phai", "co dung", "co nen", "co can", "co the", "khong?"))


def _has_unknown_tech_terms(query: str, kb: DualMemoryKnowledgeBase) -> bool:
    for t in _tokenize_vi(query):
        if _is_technical_term(t) and _strip_accents(t) not in _STOP_WORDS_FOLDED:
            if t not in kb.corpus_vocab and _strip_accents(t) not in kb.corpus_vocab_folded:
                return True
    return False


def _last_assistant_content(chat_history: list[dict]) -> str:
    """Nội dung câu trả lời gần nhất của Gia sư (kể cả khi câu hỏi gợi mở kết thúc bằng '!' thay vì '?')."""
    for msg in reversed(chat_history or []):
        if msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    return ""


def _extract_prior_context(chat_history: list[dict], concept_map: DocumentConceptMap) -> tuple[str, str, str]:
    """Lấy (last_topic, last_route, last_follow_up); giữ lại chủ đề hợp lệ trước đó nếu lượt gần nhất bị từ chối."""
    last_route = ""
    last_follow = ""
    last_topic = ""
    seen_assistant = False

    for msg in reversed(chat_history or []):
        if msg.get("role") != "assistant":
            continue
        turn = msg.get("turn")
        content = str(msg.get("content") or "")
        if not seen_assistant:
            seen_assistant = True
            if turn is not None:
                last_route = getattr(turn, "route_taken", "") or ""
                last_follow = getattr(turn, "follow_up_question", "") or _last_question_in(content)
            else:
                last_follow = _last_question_in(content)

        if turn is not None:
            t_route = getattr(turn, "route_taken", "") or ""
            if t_route == "refuse_out_of_doc":
                continue
            cand = getattr(turn, "topic", "") or getattr(turn, "suggested_quiz_topic", "") or ""
            if cand and cand != "Tổng quan tài liệu":
                last_topic = cand
                break
            from_follow = _match_topic_in_query(getattr(turn, "follow_up_question", "") or content, concept_map)
            if from_follow:
                last_topic = from_follow
                break
        else:
            low = content.lower()
            for node in concept_map.concepts:
                if node.name.lower() in low:
                    last_topic = node.name
                    break
            if last_topic:
                break

    return last_topic, last_route, last_follow


def _classify_conversation_intent(
    query: str,
    last_topic: str,
    last_route: str,
    last_follow_up: str,
    concept_map: DocumentConceptMap,
) -> str:
    """Phân loại ý định hội thoại tự nhiên để giữ mạch chủ đề và kích hoạt chấm Socratic đa lượt."""
    q = query.strip()
    q_low = q.lower()
    q_fold = _strip_accents(q_low)
    words = q_fold.split()

    # Nhận diện từ khóa kỹ thuật cụ thể trong tài liệu hiện tại để không nhầm câu hỏi chuyên môn thành chào hỏi.
    domain_terms = {"rag", "chunk", "chunking", "overlap", "embedding", "embeddings", "cosine", "vector", "faiss", "top-k", "openai", "api key", "messages", "temperature", "structured", "pydantic", "function calling", "rerank"}
    for node in concept_map.concepts:
        for kw in node.key_terms[:4]:
            if len(kw.strip()) >= 3:
                domain_terms.add(_strip_accents(kw.lower().strip()))
    has_domain_term = any(t in q_fold for t in domain_terms if len(t) >= 3)

    next_question_phrases = ("hoi cau khac", "doi cau", "cau khac", "bo qua", "skip", "cau de hon", "chuyen cau")
    if last_topic and "?" not in q and len(words) <= 10 and any(p in q_fold for p in next_question_phrases):
        return "next_question"
    # Cụm đổi câu hỏi / bỏ cuộc rõ ràng: chấp nhận cả khi có dấu '?' ở cuối.
    q_plain = re.sub(r"[^\w\s]", " ", q_fold).split()
    q_plain_str = " ".join(q_plain)
    swap_phrases = ("hoi cau khac", "doi cau hoi", "doi cau di", "doi cau")
    giveup_phrases = ("minh khong biet", "em khong biet", "toi khong biet", "chiu", "bo qua cau nay")
    if last_topic and len(q_plain) <= 8 and not has_domain_term and not (
        _EXPLAIN_REQUEST_RE.search(q_fold) or _EXAMPLE_REQUEST_RE.search(q_fold) or _MEANING_QUESTION_RE.search(q_fold)
    ) and (
        any(q_plain_str.startswith(p) or q_plain_str == p for p in swap_phrases)
        or q_plain_str in giveup_phrases
        or any(q_plain_str.startswith(p + " ") for p in giveup_phrases)
    ):
        return "next_question"

    summary_phrases = (
        "tom tat tai lieu", "tom tat toan bo", "tom tat giao trinh", "tong quan tai lieu",
        "tai lieu gom nhung", "gom nhung phan nao", "gom nhung chu de", "noi dung chinh",
        "liet ke cac phan", "liet ke chu de", "tom tat giup",
    )
    if any(p in q_fold for p in summary_phrases) or ("tom tat" in q_fold and not has_domain_term):
        return "doc_summary"

    meta_phrases = (
        "ban la ai", "may la ai", "gia su la ai", "giup gi duoc", "giup duoc gi",
        "lam duoc gi", "co chuc nang gi", "yeu phan nao", "diem yeu cua toi",
        "hoc gi truoc", "nen hoc gi", "lo trinh hoc", "ho so cua toi",
    )
    if any(p in q_fold for p in meta_phrases):
        return "greeting_or_meta"

    cont = _is_continuation_accept(q, q_fold, last_topic, last_route, last_follow_up)
    if cont and not has_domain_term:
        return cont

    thanks_toks = _fold_words(q_fold)
    if (
        thanks_toks
        and len(thanks_toks) <= 6
        and ({"thanks", "thank"} & set(thanks_toks) or {"cam", "on"} <= set(thanks_toks))
        and not has_domain_term
        and not q.endswith("?")
    ):
        return "acknowledgement"
    if _is_greeting_text(q_fold) and not has_domain_term and len(words) <= 10:
        return "greeting_or_meta"

    ack_kind = _is_acknowledgement(q_fold)
    if ack_kind and _is_yes_no_knowledge_question(last_follow_up) and last_route != "acknowledge":
        if ack_kind == "ack" or ("hieu" not in q_fold and "ro" not in q_fold):
            return "confirm_answer"
    if ack_kind == "confused" and last_topic:
        return "followup_elaborate"  # "chưa hiểu" -> giải thích lại kỹ hơn chủ đề vừa nói
    if ack_kind:
        return "acknowledgement"

    if last_topic and _is_conversation_repair(q_fold):
        return "conversation_repair"

    if _is_review_request(q):
        return "review_request"

    # Câu đoán mò/đồng ý nửa vời sau câu hỏi Có/Không ("ờ chắc v", "chắc vậy", "tùy trường hợp").
    if (
        last_topic
        and last_follow_up.strip().endswith("?")
        and _HEDGE_GUESS_RE.match(q_plain_str)
        and not has_domain_term
    ):
        return "confirm_answer"

    example_markers = ("vi du", "minh hoa", "tinh huong thuc te", "vi du cu the")
    if (
        (any(m in q_fold for m in example_markers) or _EXAMPLE_REQUEST_RE.search(q_fold))
        and len(words) <= 16
        and (last_topic or has_domain_term)
    ):
        return "followup_example"

    elaborate_markers = (
        "ky hon", "ro hon", "chi tiet hon", "de hieu hon", "cu the hon",
        "noi them", "sau hon", "chua hieu", "giai thich lai", "giai thich them",
    )
    if (
        (
            any(m in q_fold for m in elaborate_markers)
            or _CONFUSION_RE.search(q_fold)
            or _MEANING_QUESTION_RE.search(q_fold)
            or (
                _EXPLAIN_REQUEST_RE.search(q_fold)
                and not _EXAMPLE_REQUEST_RE.search(q_fold)
                and not _NEW_QUESTION_RE.search(q_fold)
            )
            or (
                any(k in q_fold for k in ("ko biet", "khong biet", "k biet", "chua biet"))
                and any(k in q_fold for k in ("giai thich", "giang", "chi"))
            )
        )
        and last_topic
        and len(words) <= 16
    ):
        return "followup_elaborate"

    is_question_form = q.endswith("?") or q.endswith("？") or any(
        q_fold.startswith(st_w)
        for st_w in (
            "giai thich", "tai sao", "vi sao", "nhu the nao", "the nao", "lam sao",
            "so sanh", "phan biet", "hay ", "cho hoi", "cho minh hoi", "liet ke", "trinh bay", "tom tat",
        )
    ) or any(
        end_w in q_fold
        for end_w in (" la gi", " the nao", " ra sao", " bang cach nao", " khac gi", " hoat dong sao")
    )

    if (
        last_topic
        and last_follow_up
        and last_route in {"adaptive_explain", "socratic_remediation", "bridge_prerequisite", "socratic_eval", "acknowledge"}
        and not is_question_form
        and last_follow_up.strip().endswith("?")
        and not any(q_fold.startswith(p) for p in _REQUEST_PREFIXES)
        and (
            len(words) >= 5
            or any(q_fold.startswith(p) for p in ("theo minh", "theo em", "em nghi", "minh nghi", "em hieu", "minh hieu", "do la", "vi "))
        )
    ):
        return "socratic_answer"

    return "standard_qa"


_IntentLiteral = Literal[
    "acknowledgement",
    "confirm_answer",
    "conversation_repair",
    "greeting_or_meta",
    "doc_summary",
    "review_request",
    "continue_topic",
    "followup_example",
    "followup_elaborate",
    "next_question",
    "socratic_answer",
    "standard_qa",
]


class _IntentRouterOutput(BaseModel):
    intent: _IntentLiteral
    in_scope: bool = Field(default=True)
    standalone_query: str = Field(default="")
    matched_topic: str = Field(default="")
    reason: str = Field(default="")


def _format_recent_history(chat_history: list[dict] | None, limit: int = 6, max_chars: int = 280) -> str:
    if not chat_history:
        return "(Chưa có lịch sử hội thoại)"
    lines: list[str] = []
    for m in chat_history[-limit:]:
        role = "Học viên" if m.get("role") == "user" else "Gia sư"
        content = str(m.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars:
            content = content[:max_chars] + "…"
        turn = m.get("turn")
        fq = (getattr(turn, "follow_up_question", "") or "").strip() if turn is not None else ""
        if fq and fq not in content:
            content = f"{content} [Câu hỏi Gia sư: {fq}]"
        lines.append(f"- {role}: {content}")
    return "\n".join(lines) or "(Chưa có lịch sử hội thoại)"


def _classify_intent_with_llm(
    query: str,
    last_topic: str,
    last_route: str,
    last_follow_up: str,
    concept_map: DocumentConceptMap,
    db_path: str | Path | None = None,
    chat_history: list[dict] | None = None,
) -> _IntentRouterOutput | None:
    """Context-First Agent Router (chạy ở Live mode): đọc cả lịch sử hội thoại + mục lục tài liệu để quyết định intent, in_scope và search_query."""
    concepts_desc = "\n".join(
        f"- {c.name}: {(c.summary or '')[:120]} (từ khóa: {', '.join(c.key_terms[:4])})"
        for c in concept_map.concepts[:12]
    ) or "- (Chưa có danh sách khái niệm)"
    history_block = _format_recent_history(chat_history, limit=6)
    system_prompt = (
        "Bạn là Context-First Agent Router cho Gia sư Socratic.\n"
        "Nhiệm vụ: đọc TOÀN BỘ lịch sử hội thoại gần nhất và mục lục tài liệu để hiểu đúng ý đồ của học viên.\n\n"
        "NGUYÊN TẮC BẮT BUỘC:\n"
        "1. LUÔN đặt `in_scope = true` cho MỌI câu nói nối mạch hội thoại:\n"
        "   - Trả lời câu hỏi Socratic (kể cả ví dụ sáng tạo ngắn 2–4 từ như 'skill làm video', 'tóm tắt email', 'chia sách') -> `socratic_answer`, `matched_topic = last_topic`.\n"
        "   - Đồng ý/phản đối/đoán ngắn ('có', 'có chứ', 'ờ chắc v', 'mình đoán là có', 'không hẳn') -> `confirm_answer`.\n"
        "   - Nói không biết / chịu / chưa hiểu / nhờ giải thích ('ko biết, giải thích', 'chịu, nói đáp án đi', 'ý là sao') -> `followup_elaborate`, `matched_topic = last_topic`.\n"
        "   - Xin ví dụ ('cho 1 vd đời thường đi') -> `followup_example`, `matched_topic = last_topic`.\n"
        "   - Muốn đổi câu hỏi / bỏ qua ('hỏi câu khác', 'đổi câu dễ hơn', 'bỏ qua') -> `next_question`.\n"
        "   - Đồng ý bắt đầu/học tiếp ('bắt đầu', 'học luôn', 'tiếp đi') -> `continue_topic`.\n"
        "   - Chào hỏi, hỏi về hồ sơ/tiến độ/điểm số/trình độ bản thân ('trình độ của tôi có tăng không', 'điểm mình bao nhiêu rồi', kể cả có 'ok,' ở đầu) -> `greeting_or_meta`; tóm tắt tài liệu -> `doc_summary`; ôn tập -> `review_request`; xác nhận/cảm ơn -> `acknowledgement`; phàn nàn hiểu nhầm -> `conversation_repair`.\n"
        "2. CHỈ đặt `in_scope = false` (và `intent = 'standard_qa'`) khi học viên hỏi một câu kiến thức MỚI RÕ RÀNG nằm ngoài hoàn toàn chủ đề tài liệu (ví dụ: 'giá vàng hôm nay', 'thủ đô của Pháp', 'cài Kubernetes thế nào' khi tài liệu không nói về chúng).\n"
        "3. `standalone_query`: viết câu truy vấn rõ nghĩa để tìm đoạn nguồn trong tài liệu. Nếu tài liệu bằng tiếng Anh mà học viên hỏi tiếng Việt, viết `standalone_query` bằng CÙNG NGÔN NGỮ với tài liệu (hoặc song ngữ Anh-Viet) để truy xuất chính xác.\n"
        "4. `matched_topic`: chọn đúng tên 1 chủ đề trong mục lục tài liệu nếu liên quan, hoặc để trống."
    )
    user_prompt = (
        f"TÀI LIỆU ĐANG NẠP: {getattr(concept_map, 'doc_title', '') or '(Tài liệu học tập)'}\n"
        f"MỤC LỤC KHÁI NIỆM:\n{concepts_desc}\n\n"
        f"LỊCH SỬ HỘI THOẠI GẦN NHẤT:\n{history_block}\n\n"
        f"NGỮ CẢNH LƯỢT TRƯỚC: last_topic={last_topic or '(trống)'} | last_route={last_route or '(trống)'} | last_follow_up={last_follow_up or '(trống)'}\n\n"
        f"TIN NHẮN MỚI NHẤT CỦA HỌC VIÊN: {query}"
    )
    try:
        return complete_structured(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema=_IntentRouterOutput,
            model=config.DEFAULT_MODEL,
            temperature=0.0,
            purpose="intent_router",
            db_path=db_path,
        )
    except Exception:
        return None


# Dòng chỉ chứa nhãn định dạng ("### Câu hỏi kiểm tra:", "**Gợi mở:**") đứng trước câu hỏi thật.
_QUESTION_LABEL_LINE_RE = re.compile(
    r"^(?:#{1,6}\s+|\*\*[^*]{1,80}\*\*\s*$|(?:câu hỏi|gợi mở|kiểm tra nhanh)[^?]{0,60}:\s*$)",
    re.IGNORECASE,
)
# Nhãn nằm cùng dòng với câu hỏi ("**Câu hỏi gợi mở:** Bạn ...?").
_QUESTION_LABEL_INLINE_RE = re.compile(
    r"^(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?:câu hỏi(?:\s+(?:kiểm tra|gợi mở|vặn|socratic|dành cho bạn))?|gợi mở|kiểm tra nhanh)"
    r"\s*:\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)


def _clean_follow_up_question(raw: str) -> str:
    """Bỏ nhãn markdown để UI chỉ hiển thị đúng câu hỏi, không lặp tiêu đề của model."""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    while len(lines) > 1 and _QUESTION_LABEL_LINE_RE.match(lines[0]):
        lines.pop(0)
    text = " ".join(lines)
    text = _QUESTION_LABEL_INLINE_RE.sub("", text, count=1)
    return re.sub(r"^[\-\*•>]\s*", "", text).strip()


def _extract_trailing_question(text: str) -> tuple[str, str]:
    """Tách câu hỏi gợi mở ở cuối câu trả lời từ LLM (nếu có) để hiển thị khung Socratic thống nhất."""
    cleaned = text.strip()
    if not cleaned:
        return "", ""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    if len(paragraphs) < 2:
        return cleaned, ""
    q = _clean_follow_up_question(paragraphs[-1])
    if q.endswith("?") and len(q) <= 260:
        body = "\n\n".join(paragraphs[:-1]).strip()
        return body, q
    return cleaned, ""


def _detect_topic(
    query: str,
    focus_topic: str | None,
    concept_map: DocumentConceptMap,
    chunks: list[RetrievedChunk],
) -> str:
    if focus_topic:
        return focus_topic
    q_low = query.lower()
    q_fold = _strip_accents(q_low)
    best_name = ""
    best_score = 0
    for node in concept_map.concepts:
        score = 0
        if node.name.lower() in q_low or _strip_accents(node.name.lower()) in q_fold:
            score += 5
        score += sum(2 for kw in node.key_terms if kw.lower() in q_low or _strip_accents(kw.lower()) in q_fold)
        name_tokens = [t for t in _fold_words(node.name) if len(t) >= 4 and t not in _STOP_WORDS_FOLDED]
        score += sum(2 for t in name_tokens if t in q_fold)
        for ch in chunks[:2]:
            ch_low = ch.content.lower()
            if node.name.lower() in ch_low or any(kw.lower() in ch_low for kw in node.key_terms[:2]):
                score += 1
        if score > best_score:
            best_score = score
            best_name = node.name
    if best_name:
        return best_name
    if chunks and chunks[0].topic:
        return chunks[0].topic
    return concept_map.concepts[0].name if concept_map.concepts else "Kiến thức chung"


def _find_prerequisite_gap(
    topic: str,
    concept_map: DocumentConceptMap,
    learner_dna: LearnerDNA,
) -> str | None:
    node = next((c for c in concept_map.concepts if c.name.lower() == topic.lower()), None)
    if not node or not node.prerequisites:
        return None
    mastery_map = {c.concept.lower(): c for c in learner_dna.concepts}
    for pre in node.prerequisites:
        m = mastery_map.get(pre.lower())
        if m and m.attempts > 0 and m.mastery_score < 45.0:
            return pre
    return None


def _inspect_state(state: CoachState, kb: DualMemoryKnowledgeBase) -> dict:
    """Thực thi pha đọc LearnerDNA + tìm FAISS Dual-Memory và xác định ngữ cảnh hội thoại."""
    dna = state["learner_dna"]
    cmap = state["concept_map"]
    traces: list[ToolCallTrace] = []
    last_topic, last_route, last_follow_up = _extract_prior_context(state.get("chat_history", []), cmap)
    rule_intent = _classify_conversation_intent(
        query=state["query"],
        last_topic=last_topic,
        last_route=last_route,
        last_follow_up=last_follow_up,
        concept_map=cmap,
    )
    llm_router: _IntentRouterOutput | None = None
    if not state.get("demo_mode") and config.has_live_api_key():
        llm_router = _classify_intent_with_llm(
            query=state["query"],
            last_topic=last_topic,
            last_route=last_route,
            last_follow_up=last_follow_up,
            concept_map=cmap,
            db_path=state.get("db_path"),
            chat_history=state.get("chat_history", []),
        )
    conv_intent = llm_router.intent if llm_router is not None else rule_intent
    if (
        llm_router is not None
        and llm_router.in_scope
        and conv_intent == "standard_qa"
        and rule_intent in {"followup_elaborate", "followup_example", "confirm_answer", "next_question", "continue_topic", "acknowledgement"}
        and not _has_unknown_tech_terms(state["query"], kb)
        and not _match_topic_in_query(state["query"], cmap)
    ):
        conv_intent = rule_intent
    if llm_router is not None:
        traces.append(
            ToolCallTrace(
                tool_name="classify_intent_llm",
                arguments={"intent": conv_intent, "in_scope": llm_router.in_scope, "matched_topic": llm_router.matched_topic},
                summary=f"LLM Router: {conv_intent} · in_scope={llm_router.in_scope}" + (f" · {llm_router.reason[:80]}" if llm_router.reason else ""),
            )
        )

    traces.append(
        ToolCallTrace(
            tool_name="get_learner_dna",
            arguments={"learner_id": dna.learner_id, "doc_id": state["doc_id"]},
            summary=f"Trình độ: {dna.overall_level} · Trung bình: {dna.mastery_avg:.0f}% · {len(dna.active_misconceptions)} lỗi đang mở",
        )
    )

    if conv_intent in _CONVERSATIONAL_INTENTS:
        chunks: list[RetrievedChunk] = []
        in_scope = True
        topic = last_topic or (cmap.concepts[0].name if cmap.concepts else "Kiến thức chung")
        traces.append(
            ToolCallTrace(
                tool_name="inspect_conversation",
                arguments={"intent": conv_intent, "last_topic": last_topic},
                summary=f"Phản hồi ngắn của người học, giữ mạch chủ đề '{topic}' (không tra tài liệu)",
            )
        )
        merged_mistakes: list[RetrievedChunk] = []
        prereq_gap = None
    elif conv_intent == "greeting_or_meta":
        mastery_map = {c.concept: c for c in dna.concepts}
        first_foundation = next(
            (
                node.name
                for node in cmap.concepts
                if (node.name not in mastery_map) or mastery_map[node.name].attempts == 0 or mastery_map[node.name].mastery_score < 60
            ),
            "",
        )
        focus_name = first_foundation or (
            dna.active_misconceptions[0].topic
            if dna.active_misconceptions
            else (cmap.concepts[0].name if cmap.concepts else state["query"])
        )
        chunks = []
        in_scope = True
        traces.append(
            ToolCallTrace(
                tool_name="inspect_concept_map",
                arguments={"doc_id": state["doc_id"], "intent": conv_intent},
                summary=f"Kiểm tra hồ sơ '{dna.name}' ({dna.overall_level}) và ưu tiên chủ đề '{focus_name}'",
            )
        )
        topic = focus_name
        merged_mistakes = []
        prereq_gap = None
    elif conv_intent == "review_request":
        explicit = _match_topic_in_query(state["query"], cmap)
        content = [
            t for t in _followup_content_terms(state["query"]) if t not in _REVIEW_FILLER
        ]
        chunks = []
        in_scope = True
        if explicit:
            topic = explicit
        elif content:
            # Người học nêu một chủ đề không trùng khái niệm nào: chỉ ôn nếu thật sự có trong tài liệu.
            chunks, in_scope = kb.retrieve(" ".join(content), top_k=3)
            topic = _detect_topic(state["query"], None, cmap, chunks) if in_scope else (
                cmap.concepts[0].name if cmap.concepts else "Kiến thức chung"
            )
        else:
            topic = _pick_review_topic(dna, cmap)
        if in_scope:
            node = next((c for c in cmap.concepts if c.name == topic), None)
            chunks, _ = kb.retrieve(f"{topic} {node.summary}" if node else topic, top_k=3)
        traces.append(
            ToolCallTrace(
                tool_name="search_study_material",
                arguments={"query": f"ôn tập: {topic}", "top_k": 3},
                summary=(
                    f"Yêu cầu ôn tập '{topic}' ({'người học nêu rõ' if explicit else 'hệ thống đề xuất'}) · "
                    f"{len(chunks)} đoạn nguồn · trong tài liệu={in_scope}"
                ),
            )
        )
        merged_mistakes = [
            RetrievedChunk(
                content=m.description,
                topic=m.topic,
                page="Hồ sơ lỗi tư duy",
                score=1.0,
                memory_type="mistake",
            )
            for m in dna.active_misconceptions
            if m.topic.lower() == topic.lower()
        ]
        prereq_gap = None
    elif conv_intent == "continue_topic":
        # Câu gợi mở có thể kết thúc bằng '!' nên không trích được tên chủ đề: lùi về nội dung câu trả lời gần nhất.
        last_assistant_content = _last_assistant_content(state.get("chat_history", []))
        offered = _match_topic_in_query(last_follow_up, cmap) or _match_topic_in_query(last_assistant_content, cmap)
        topic = offered or (last_topic if any(c.name == last_topic for c in cmap.concepts) else _pick_review_topic(dna, cmap))
        node = next((c for c in cmap.concepts if c.name == topic), None)
        chunks, _ = kb.retrieve(f"{topic} {node.summary}" if node else topic, top_k=4)
        in_scope = True
        merged_mistakes = [
            RetrievedChunk(
                content=m.description,
                topic=m.topic,
                page="Hồ sơ lỗi tư duy",
                score=1.0,
                memory_type="mistake",
            )
            for m in dna.active_misconceptions
            if m.topic.lower() == topic.lower()
        ]
        prereq_gap = _find_prerequisite_gap(topic, cmap, dna)
        traces.append(
            ToolCallTrace(
                tool_name="search_study_material",
                arguments={"query": f"tiếp tục chủ đề: {topic}", "top_k": 4},
                summary=f"Người học đồng ý bắt đầu/đi sâu vào chủ đề '{topic}' ({len(chunks)} đoạn nguồn)",
            )
        )
    elif conv_intent == "doc_summary":
        chunks = []
        seen_pages: set[str] = set()
        for node in cmap.concepts[:6]:
            c_list, _ = kb.retrieve(f"{node.name} {node.summary}", top_k=1)
            if c_list and c_list[0].page not in seen_pages:
                seen_pages.add(c_list[0].page)
                chunks.append(c_list[0])
        in_scope = True
        topic = "Tổng quan tài liệu"
        merged_mistakes = []
        prereq_gap = None
        traces.append(
            ToolCallTrace(
                tool_name="search_study_material",
                arguments={"query": "Tổng quan toàn bộ chủ đề", "top_k": len(chunks)},
                summary=f"Tổng hợp {len(cmap.concepts)} chủ đề từ {len(chunks)} phần tài liệu nguồn",
            )
        )
    elif llm_router is not None and not llm_router.in_scope:
        chunks = []
        in_scope = False
        topic = last_topic or (cmap.concepts[0].name if cmap.concepts else "Kiến thức chung")
        merged_mistakes = []
        prereq_gap = None
        traces.append(
            ToolCallTrace(
                tool_name="search_study_material",
                arguments={"query": state["query"][:80], "top_k": 0},
                summary=f"LLM Router xác định câu hỏi ngoài phạm vi tài liệu ({llm_router.reason[:80] or 'out_of_scope'})",
            )
        )
    else:
        retrieval_query = state["query"]
        if llm_router is not None and llm_router.standalone_query.strip() and conv_intent in {"standard_qa", "followup_example", "followup_elaborate", "socratic_answer"}:
            retrieval_query = llm_router.standalone_query.strip()
        q_fold = _strip_accents(state["query"].lower())
        explicit_in_q = _match_topic_in_query(state["query"], cmap)
        llm_matched = ""
        if llm_router is not None and llm_router.matched_topic:
            llm_matched = next((c.name for c in cmap.concepts if c.name.lower() == llm_router.matched_topic.strip().lower()), "")
        if conv_intent in {"followup_example", "followup_elaborate"} and last_topic and not state.get("focus_topic"):
            target_ctx = explicit_in_q or llm_matched or last_topic
            if llm_router is None and _has_unknown_tech_terms(state["query"], kb) and not explicit_in_q:
                retrieval_query = state["query"]
            elif target_ctx.lower() not in retrieval_query.lower():
                retrieval_query = f"{target_ctx}. {retrieval_query}"
        elif (
            conv_intent == "standard_qa"
            and last_topic
            and not state.get("focus_topic")
            and not explicit_in_q
            and _COREF_RE.search(q_fold)
            and not _has_unknown_tech_terms(state["query"], kb)
        ):
            # Câu hỏi nối tiếp dùng đại từ ("nó", "phần đó", "vậy còn"): gắn chủ đề đang bàn để truy xuất đúng mạch.
            target_ctx = last_topic
            if last_topic.lower() not in retrieval_query.lower():
                retrieval_query = f"{last_topic}. {retrieval_query}"
        else:
            target_ctx = state.get("focus_topic") or llm_matched or ""

        if llm_router is not None and llm_router.in_scope:
            # Live Context-First mode: tin tưởng quyết định in_scope của LLM Router, dùng FAISS để lấy đoạn trích nguồn.
            if conv_intent == "socratic_answer":
                target_ctx = target_ctx or last_topic
                q_search = llm_router.standalone_query.strip() or f"{target_ctx}. {last_follow_up or state['query']}"
                chunks, _ = kb.retrieve(q_search, top_k=4)
            else:
                chunks, _ = kb.retrieve(retrieval_query, top_k=4)
                if not chunks:
                    fallback_q = target_ctx or (cmap.concepts[0].name if cmap.concepts else state["query"])
                    chunks, _ = kb.retrieve(fallback_q, top_k=4)
            in_scope = True
        else:
            chunks, in_scope = kb.retrieve(retrieval_query, top_k=4)
            if conv_intent == "socratic_answer" and last_topic:
                node = next((c for c in cmap.concepts if c.name.lower() == last_topic.lower()), None)
                q_fold = _strip_accents(state["query"].lower())
                topic_terms = {_strip_accents(last_topic.lower())}
                topic_terms.update(t for t in _fold_words(last_topic) if len(t) >= 3)
                if node:
                    topic_terms.update(_strip_accents(k.lower()) for k in node.key_terms if len(k.strip()) >= 2)
                overlaps_topic = any(t in q_fold for t in topic_terms if len(t) >= 2)
                if in_scope or overlaps_topic:
                    in_scope = True
                    target_ctx = last_topic
                    chunks, _ = kb.retrieve(f"{last_topic}. {last_follow_up or state['query']}", top_k=4)

        cites = ", ".join(dict.fromkeys(c.page for c in chunks if c.page)) or "Không khớp"
        top_sim = chunks[0].score if chunks else 0.0
        traces.append(
            ToolCallTrace(
                tool_name="search_study_material",
                arguments={"query": retrieval_query[:80], "top_k": 4},
                summary=f"Tìm thấy {len(chunks)} đoạn ({cites}) · độ tương đồng cao nhất {top_sim:.2f} · trong tài liệu={in_scope}",
            )
        )

        kb.sync_learner_mistakes(dna)
        past_mistakes = kb.retrieve_past_mistakes(state["query"], top_k=2)
        topic = _detect_topic(state["query"], target_ctx or None, cmap, chunks)
        matching_active = [
            RetrievedChunk(
                content=m.description,
                topic=m.topic,
                page="Hồ sơ lỗi tư duy",
                score=1.0,
                memory_type="mistake",
            )
            for m in dna.active_misconceptions
            if m.topic.lower() == topic.lower()
        ]
        merged_mistakes = matching_active or past_mistakes
        traces.append(
            ToolCallTrace(
                tool_name="search_past_mistakes",
                arguments={"topic": topic, "learner_id": dna.learner_id},
                summary=f"Phát hiện {len(merged_mistakes)} lỗi tư duy liên quan tới '{topic}'",
            )
        )
        prereq_gap = _find_prerequisite_gap(topic, cmap, dna) if in_scope else None

    return {
        "conv_intent": conv_intent,
        "last_topic": last_topic,
        "last_route": last_route,
        "last_follow_up": last_follow_up,
        "topic": topic,
        "is_in_scope": in_scope,
        "prerequisite_topic": prereq_gap,
        "retrieved_chunks": chunks,
        "past_mistakes": merged_mistakes,
        "tool_traces": traces,
    }


def _prior_routes_for_topic(state: CoachState) -> set[str]:
    target = (state.get("topic") or "").lower()
    if not target:
        return set()
    routes: set[str] = set()
    for m in state.get("chat_history", []):
        if m.get("role") != "assistant":
            continue
        t = m.get("turn")
        if t is None:
            continue
        t_topic = (getattr(t, "topic", "") or getattr(t, "suggested_quiz_topic", "") or "").lower()
        if t_topic == target:
            r = getattr(t, "route_taken", "") or ""
            if r:
                routes.add(r)
    return routes


def _route_from_state(state: CoachState) -> str:
    intent = state["conv_intent"]
    if intent in _CONVERSATIONAL_INTENTS:
        return "acknowledge"
    if intent in {"greeting_or_meta", "doc_summary"}:
        return "coach_guide"
    if not state["is_in_scope"]:
        return "refuse_out_of_doc"
    if intent == "review_request":
        return "adaptive_explain"
    if intent == "continue_topic":
        return "adaptive_explain"
    if intent == "socratic_answer":
        return "socratic_eval"
    if intent in {"followup_example", "followup_elaborate"}:
        return "adaptive_explain"

    seen_routes = _prior_routes_for_topic(state)
    if state.get("remediation_context"):
        return "socratic_remediation"
    if state["past_mistakes"] and "socratic_remediation" not in seen_routes:
        return "socratic_remediation"
    if state.get("prerequisite_topic") and "bridge_prerequisite" not in seen_routes:
        return "bridge_prerequisite"
    return "adaptive_explain"


def _resolve_live_prompt_spec(state: CoachState, route: str) -> tuple[str, str, str]:
    """Trả về (prompt_key, extra_instruction, pedagogical_reason) dùng chung cho cả batch & stream."""
    intent = state["conv_intent"]
    level = state["learner_dna"].overall_level
    if route == "coach_guide":
        extra = (
            "Hãy tóm tắt đầy đủ các chủ đề trong tài liệu kèm nhãn [Trang/Mục]."
            if intent == "doc_summary"
            else (
                f"Hãy chào học viên theo tên, nêu trình độ hiện tại ({level}, điểm TB {state['learner_dna'].mastery_avg:.0f}%) "
                f"và gợi ý bắt đầu học ngay chủ đề nền tảng tiếp theo là '{state['topic']}' (theo đúng thứ tự lộ trình từ đầu tài liệu, không nhảy cóc qua chủ đề chưa học)."
            )
        )
        reason = (
            "Học viên yêu cầu tóm tắt cấu trúc toàn bộ tài liệu đang nạp."
            if intent == "doc_summary"
            else f"Học viên chào hỏi hoặc hỏi về hồ sơ năng lực; đề xuất chủ đề tiếp theo theo lộ trình: '{state['topic']}'."
        )
        return "coach_guide", extra, reason
    if route == "bridge_prerequisite":
        prereq = state["prerequisite_topic"] or ""
        return (
            "bridge_prerequisite",
            f"Học viên đang hổng khái niệm tiên quyết '{prereq}'. Hãy nhắc lại khái niệm này trước khi trả lời câu chính.",
            f"Học viên đang yếu khái niệm nền tảng '{prereq}' liên quan trực tiếp đến '{state['topic']}'.",
        )
    if route == "socratic_remediation":
        ctx = state.get("remediation_context") or ""
        return (
            "socratic_remediation",
            f"Ngữ cảnh câu hỏi vừa làm sai (nếu có): {ctx}",
            "Phát hiện lỗi tư duy đang mở hoặc yêu cầu gỡ rối câu sai; chuyển sang chế độ hỏi vặn Socratic.",
        )

    prompt_key = {
        "Mới bắt đầu": "adaptive_explain_beginner",
        "Nâng cao": "adaptive_explain_advanced",
    }.get(level, "adaptive_explain_intermediate")
    extra_inst = f"Trả lời phù hợp tuyệt đối với trình độ {level} của học viên."
    if intent == "review_request":
        reason = f"Học viên yêu cầu ÔN TẬP chủ đề '{state['topic']}'; tóm tắt ý chính, không chuyển sang chất vấn lỗi tư duy."
        extra_inst += (
            f" Học viên muốn ÔN TẬP chủ đề '{state['topic']}': tóm tắt các ý chính theo tài liệu, nêu 2-3 điểm dễ nhầm; "
            "nếu hồ sơ có lỗi tư duy liên quan thì chỉ nhắc ngắn một dòng, KHÔNG chất vấn hay chẩn đoán dài; "
            "kết bằng một câu hỏi kiểm tra ngắn."
        )
    elif intent == "followup_example":
        reason = f"Nối tiếp mạch chủ đề '{state['topic']}'; đưa ví dụ thực tế minh họa theo trình độ '{level}'."
        last_q = state.get("last_follow_up") or ""
        extra_inst += (
            f" Học viên đang xin VÍ DỤ THỰC TẾ cho chủ đề '{state['topic']}'"
            + (f" (xoay quanh câu hỏi vừa đặt: \"{last_q}\"). " if last_q else ". ")
            + "QUY TẮC BẮT BUỘC: (1) Không mở đầu bằng 'Rất tốt!'; (2) Đi thẳng vào 1-2 ví dụ cụ thể, ngắn gọn bám sát tài liệu; "
            "(3) KHÔNG lặp lại phần định nghĩa lý thuyết hay mục 'Nhắc lại' đã giảng ở lượt trước; "
            "(4) Kết thúc bằng 1 câu hỏi ngắn gắn trực tiếp với ví dụ vừa nêu (KHÔNG lặp lại câu hỏi cũ)."
        )
    elif intent == "followup_elaborate":
        reason = f"Nối tiếp mạch chủ đề '{state['topic']}'; giải đáp câu hỏi Socratic hoặc làm rõ ý theo trình độ '{level}'."
        last_q = state.get("last_follow_up") or ""
        extra_inst += (
            f" Học viên vừa phản hồi '{state['query']}' về chủ đề '{state['topic']}'"
            + (f" sau khi bạn hỏi: \"{last_q}\". " if last_q else ". ")
            + "QUY TẮC BẮT BUỘC: (1) TUYỆT ĐỐI KHÔNG khen 'Rất tốt!' hay 'Tuyệt vời!' khi học viên nói không biết/chưa hiểu; "
            "(2) TUYỆT ĐỐI KHÔNG giảng lại toàn bộ bài học từ đầu, không lặp lại các đoạn 'Giải thích từng bước' hay 'Nhắc lại' vừa nói ở lượt trước; "
            "(3) Hãy giải đáp hoặc gợi ý trực tiếp, ngắn gọn (3-5 câu) cho chính câu hỏi bạn vừa đặt kèm 1 ví dụ cụ thể từ tài liệu; "
            "(4) Kết thúc bằng MỘT câu hỏi kiểm tra MỚI dễ hơn và KHÁC hẳn câu hỏi cũ."
        )
    else:
        reason = f"Câu hỏi nằm trong tài liệu; điều chỉnh giọng giảng và độ sâu kỹ thuật cho trình độ '{level}'."
    return prompt_key, extra_inst, reason


def _build_coach_openai_messages(state: CoachState, prompt_key: str, extra_instruction: str) -> list[dict]:
    dna = state["learner_dna"]
    chunks = state["retrieved_chunks"]
    context_block = "\n\n".join(f"{c.page} ({c.source}):\n{c.content}" for c in chunks) or "Không có đoạn trích cần thiết."
    mistakes_block = "\n".join(f"- {m.content}" for m in state["past_mistakes"]) or "Không có lỗi ghi nhận."
    mastery = {c.concept: c for c in dna.concepts}
    concept_lines = []
    for node in state["concept_map"].concepts[:10]:
        m = mastery.get(node.name)
        score = f" ({m.mastery_score:.0f}%)" if m and m.attempts > 0 else " (0% - chưa học)"
        concept_lines.append(f"- {node.name}{score}")
    concepts_block = "\n".join(concept_lines) or "Chưa có khái niệm nào."

    prompt_tmpl = ChatPromptTemplate.from_messages(
        [
            ("system", SOCRATIC_COACH_PROMPTS[prompt_key]),
            (
                "human",
                "HỒ SƠ HỌC VIÊN: {name} · Trình độ: {level} · Điểm TB: {avg}%\n"
                "NGỮ CẢNH LƯỢT TRƯỚC: Chủ đề = {last_topic} | Câu hỏi Gia sư vừa đặt = {last_follow_up}\n"
                "CÁC KHÁI NIỆM TRONG TÀI LIỆU ĐANG NẠP (chỉ được nhắc tới các chủ đề này):\n{concepts}\n\n"
                "LỖI TƯ DUY LIÊN QUAN:\n{mistakes}\n\n"
                "ĐOẠN TÀI LIỆU NGUỒN (RAG):\n{context}\n\n"
                "CHỈ DẪN SƯ PHẠM BỔ SUNG: {extra}\n\n"
                "CÂU HỎI HỌC VIÊN: {query}",
            ),
        ]
    )
    lc_msgs = prompt_tmpl.format_messages(
        name=dna.name,
        level=dna.overall_level,
        avg=f"{dna.mastery_avg:.0f}",
        last_topic=state.get("last_topic") or "Chưa có",
        last_follow_up=state.get("last_follow_up") or "Không có",
        concepts=concepts_block,
        mistakes=mistakes_block,
        context=context_block,
        extra=extra_instruction,
        query=state["query"],
    )
    openai_msgs: list[dict] = [{"role": "system", "content": lc_msgs[0].content}]
    for hist in state.get("chat_history", [])[-6:]:
        role = hist.get("role")
        if role not in {"user", "assistant"} or not hist.get("content"):
            continue
        content = str(hist["content"])
        if role == "assistant":
            turn_obj = hist.get("turn")
            fq = (getattr(turn_obj, "follow_up_question", "") or "").strip() if turn_obj is not None else ""
            if fq and fq not in content:
                content = f"{content}\n\n[Câu hỏi Socratic Gia sư đã hỏi]: {fq}"
        openai_msgs.append({"role": role, "content": content[:1200]})
    openai_msgs.append({"role": "user", "content": lc_msgs[1].content})
    return openai_msgs


# Cache đồ thị đã compile theo (id(kb), db_path, return_nodes) để mỗi lượt chat không phải dựng lại StateGraph.
_GRAPH_CACHE: dict[tuple[int, str, bool], object] = {}
_GRAPH_CACHE_LIMIT = 16


def _build_coach_graph(kb: DualMemoryKnowledgeBase, db_path: str | Path | None = None, return_nodes: bool = False):
    cache_key = (id(kb), str(db_path), return_nodes)
    cached = _GRAPH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    def inspect_and_retrieve(state: CoachState) -> dict:
        return _inspect_state(state, kb)

    def route_pedagogy(state: CoachState) -> str:
        return _route_from_state(state)

    def _run_llm_with_tools(state: CoachState, prompt_key: str, extra_instruction: str) -> tuple[str, str, list[ToolCallTrace], float]:
        dna = state["learner_dna"]
        openai_msgs = _build_coach_openai_messages(state, prompt_key, extra_instruction)

        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_learner_weaknesses",
                    "description": "Kiểm tra danh sách chủ đề yếu và lỗi tư duy gần đây của học viên.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_extra_passage",
                    "description": "Tìm thêm một đoạn tài liệu theo từ khóa cụ thể nếu cần mở rộng giải thích.",
                    "parameters": {
                        "type": "object",
                        "properties": {"keyword": {"type": "string", "description": "Từ khóa cần tra cứu"}},
                        "required": ["keyword"],
                    },
                },
            },
        ]

        def _tool_lookup() -> str:
            weak = [f"{c.concept} ({c.mastery_score:.0f}%)" for c in dna.concepts if c.mastery_score < 60]
            misc = [m.description for m in dna.active_misconceptions]
            return f"Chủ đề cần cải thiện: {', '.join(weak) or 'Không có'}. Lỗi tư duy: {', '.join(misc) or 'Không có'}."

        def _tool_fetch(keyword: str = "") -> str:
            extra_chunks, _ = kb.retrieve(keyword or state["query"], top_k=2)
            return "\n".join(f"{c.page}: {c.content[:280]}" for c in extra_chunks) or "Không tìm thêm được đoạn phù hợp."

        cost_before = db.get_session_cost_usd(db_path=db_path)
        raw_answer, extra_traces = complete_with_tools(
            messages=openai_msgs,
            tools_schema=tools_schema,
            tool_dispatch={
                "lookup_learner_weaknesses": _tool_lookup,
                "fetch_extra_passage": _tool_fetch,
            },
            model=state.get("model") or config.DEFAULT_MODEL,
            temperature=0.3,
            purpose=f"coach_{prompt_key}",
            db_path=db_path,
        )
        cost_delta = max(0.0, round(db.get_session_cost_usd(db_path=db_path) - cost_before, 6))
        body, trailing_q = _extract_trailing_question(raw_answer)
        return body, trailing_q, extra_traces, cost_delta

    def coach_guide(state: CoachState) -> dict:
        intent = state["conv_intent"]
        reason = (
            "Học viên yêu cầu tóm tắt cấu trúc toàn bộ tài liệu đang nạp."
            if intent == "doc_summary"
            else "Học viên chào hỏi hoặc hỏi về hồ sơ năng lực, điểm yếu và lộ trình cá nhân."
        )
        if state["demo_mode"] or not config.has_live_api_key():
            ans, follow = demo_coach_conversational_reply(
                intent=intent,
                query=state["query"],
                topic=state["topic"],
                retrieved_chunks=state["retrieved_chunks"],
                learner_dna=state["learner_dna"],
                concept_map=state["concept_map"],
                db_path=db_path,
            )
            return {
                "route_taken": "coach_guide",
                "pedagogical_reason": reason,
                "answer": ans,
                "follow_up_question": follow,
                "cost_usd": 0.0,
            }

        extra = (
            "Hãy tóm tắt đầy đủ các chủ đề trong tài liệu kèm nhãn [Trang/Mục]."
            if intent == "doc_summary"
            else "Hãy chào học viên theo tên, nêu trình độ và chủ đề cần ưu tiên nhất dựa trên hồ sơ."
        )
        ans, follow, extra_traces, cost_delta = _run_llm_with_tools(state, "coach_guide", extra)
        return {
            "route_taken": "coach_guide",
            "pedagogical_reason": reason,
            "answer": ans,
            "follow_up_question": follow,
            "tool_traces": [*state["tool_traces"], *extra_traces],
            "cost_usd": cost_delta,
        }

    def socratic_eval(state: CoachState) -> dict:
        target_topic = state["topic"]
        cmap = state["concept_map"]
        cost_before = db.get_session_cost_usd(db_path=db_path)
        ev, updated_dna = adaptive.evaluate_teach_back(
            concept=target_topic,
            learner_answer=state["query"],
            kb=kb,
            learner_dna=state["learner_dna"],
            doc_id=state["doc_id"],
            concepts_order=[c.name for c in cmap.concepts],
            concept_map=cmap,
            question_context=state.get("last_follow_up") or "",
            model=state.get("model"),
            demo_mode=state["demo_mode"],
            db_path=db_path,
        )
        cost_delta = max(0.0, round(db.get_session_cost_usd(db_path=db_path) - cost_before, 6))

        lines = [f"**Chấm câu trả lời Socratic — Chủ đề `{target_topic}`:** `{ev.understanding_score}/100` (*{ev.verdict}*)"]
        if ev.correct_points:
            lines.append("**Điểm bạn đã nắm đúng:**")
            lines.extend(f"- {pt}" for pt in ev.correct_points[:2])
        if ev.misconceptions_found:
            lines.append("**Chỗ cần chỉnh ngay:**")
            lines.extend(f"- {m}" for m in ev.misconceptions_found)
        if ev.missing_points and ev.understanding_score < 80:
            lines.append("**Ý còn thiếu theo tài liệu:**")
            lines.extend(f"- {ms}" for ms in ev.missing_points[:2])
        if ev.understanding_score >= 80 and not ev.misconceptions_found:
            lines.append(
                f"Đã cập nhật tiến bộ vào **Learner DNA** (`{target_topic}` nâng điểm thành thạo và tự động đánh dấu khắc phục lỗi tư duy liên quan)."
            )

        traces = [
            *state["tool_traces"],
            ToolCallTrace(
                tool_name="evaluate_socratic_response",
                arguments={"topic": target_topic, "learner_id": updated_dna.learner_id},
                summary=f"Chấm phản hồi: {ev.understanding_score}/100 ({ev.verdict}) · {len(ev.misconceptions_found)} lỗi tư duy",
            ),
        ]
        # Giữ đoạn trích khớp đúng mục đang chấm để citations không bị lẫn mục khác.
        node = next((c for c in cmap.concepts if c.name.lower() == target_topic.lower()), None)
        own_chunks = [c for c in state["retrieved_chunks"] if node and node.page_ref and c.page == node.page_ref]
        return {
            "learner_dna": updated_dna,
            "retrieved_chunks": own_chunks or state["retrieved_chunks"],
            "route_taken": "socratic_eval",
            "pedagogical_reason": f"Học viên vừa trả lời câu hỏi Socratic cho chủ đề '{target_topic}'; tự động chấm bám nguồn và cập nhật SQLite Learner DNA.",
            "answer": "\n\n".join(lines),
            "follow_up_question": ev.socratic_follow_up,
            "tool_traces": traces,
            "cost_usd": cost_delta,
        }

    def acknowledge(state: CoachState) -> dict:
        base = _acknowledge_template(state)
        if state["demo_mode"] or not config.has_live_api_key():
            return base
        try:
            cost_before = db.get_session_cost_usd(db_path=db_path)
            history_lines = []
            for hist in (state.get("chat_history") or [])[-6:]:
                if hist.get("role") in {"user", "assistant"} and hist.get("content"):
                    who = "Học viên" if hist["role"] == "user" else "Gia sư"
                    history_lines.append(f"{who}: {str(hist['content'])[:400]}")
            sys_prompt = (
                "Bạn là Gia sư Socratic nghiêm túc nhưng gần gũi, trả lời bằng tiếng Việt, ngắn gọn (2-4 câu). "
                "QUY TẮC QUAN TRỌNG: KHÔNG BAO GIỜ tin suông hay khen 'tuyệt vời, bạn đã hiểu rồi' khi học viên chỉ nói 'ok', 'hiểu rồi', 'có'. "
                "Hãy nói thẳng rằng chỉ nói 'hiểu rồi' chưa chứng minh được mức hiểu (và chưa được cộng điểm trình độ); "
                "gợi ý ngắn họ có thể tự giải thích 1–2 câu ngay tại đây hoặc bấm nút 'Luyện ngay Quiz' bên dưới để chấm điểm nâng trình độ, "
                "rồi đặt ĐÚNG MỘT câu hỏi kiểm tra thực chất, ngắn gọn về chủ đề hiện tại (không lặp nguyên văn câu cũ, không bịa ngoài tài liệu)."
            )
            user_prompt = (
                f"Chủ đề hiện tại: {base.get('topic') or state['topic']}\n"
                f"Câu hỏi Gia sư vừa đặt: {state.get('last_follow_up') or 'Không có'}\n"
                f"Ý định hội thoại: {state.get('conv_intent', '')}\n"
                f"Hội thoại gần đây:\n{chr(10).join(history_lines) or '(trống)'}\n\n"
                f"Học viên vừa nói: {state['query']}"
            )
            client = _llm.get_client()
            _llm._check_budget(db_path)
            target_model = state.get("model") or config.DEFAULT_MODEL
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
            resp, latency_ms = _llm._call_with_retry(
                lambda: client.chat.completions.create(model=target_model, messages=msgs, temperature=0.5)
            )
            _llm._record_usage(resp, target_model, "coach_acknowledge", latency_ms, db_path=db_path)
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                return base
            body, trailing_q = _extract_trailing_question(text)
            cost_delta = max(0.0, round(db.get_session_cost_usd(db_path=db_path) - cost_before, 6))
            return {**base, "answer": body or text, "follow_up_question": trailing_q, "cost_usd": cost_delta}
        except Exception:  # noqa: BLE001 - LLM lỗi thì dùng câu mẫu deterministic
            return base

    def _acknowledge_template(state: CoachState) -> dict:
        topic = state["topic"]
        has_prior = bool(state.get("last_topic"))
        words = set(_fold_words(state["query"]))
        names = [c.name for c in state["concept_map"].concepts]
        next_topic = ""
        if topic in names and names.index(topic) + 1 < len(names):
            next_topic = names[names.index(topic) + 1]
        said = state["query"].strip()[:30]

        last_q = (state.get("last_follow_up") or "").strip()
        last_q = last_q if last_q.endswith("?") and len(last_q) > 1 else ""
        intent = state.get("conv_intent", "")
        new_topic = topic
        if intent == "next_question":
            node = next((c for c in state["concept_map"].concepts if c.name == topic), None)
            summary = (node.summary or "").strip() if node else ""
            key_terms = [k.strip() for k in (node.key_terms if node else []) if k.strip()]
            candidates: list[str] = []
            if key_terms:
                candidates.append(
                    f"Đổi câu nhé: theo bạn `{key_terms[0]}` đóng vai trò gì trong `{topic}`, và nếu bỏ đi thì điều gì sẽ hỏng?"
                )
            if summary:
                candidates.append(
                    f"Đổi câu nhé: tài liệu nói `{topic}` — {summary[:120]}. Bạn thấy ý nào cốt lõi nhất và vì sao?"
                )
            candidates.append(
                f"Đổi câu nhé: bạn thử kể một tình huống ngắn liên quan tới `{topic}` mà bạn hình dung được, rồi nói bước xử lý tiếp theo."
            )
            candidates.append(
                f"Đổi câu nhé: nếu chỉ được nhớ một ý duy nhất về `{topic}`, bạn chọn ý nào và giải thích ngắn vì sao?"
            )
            follow = next((c for c in candidates if c != last_q), candidates[-1])
            ans = (
                f"Ok, mình đổi sang câu hỏi khác cho `{topic}`."
                if has_prior
                else f"Ok, mình đặt một câu hỏi khác về `{topic}` nhé."
            )
        elif intent == "conversation_repair" and has_prior:
            ans = (
                "Xin lỗi, bạn nói đúng: câu trả lời của bạn cần được ghi nhận chứ không phải bị hỏi lại máy móc."
                + (f" Câu hỏi mình vừa đặt là: \"{last_q}\"." if last_q else "")
                + f" Mình sẽ tiếp tục đúng mạch chủ đề `{topic}`."
            )
            follow = (
                f"Để mình chấm đúng ý bạn, bạn cho mình thêm một hai câu: theo bạn vì sao như vậy trong `{topic}`?"
                if last_q
                else f"Bạn muốn mình giải thích tiếp `{topic}` hay hỏi bạn một câu kiểm tra ngắn?"
            )
        elif words & {"cam", "thanks", "thank"}:
            ans = "Không có gì! Mình rất vui khi giúp được bạn."
            follow = (
                f"Bạn muốn ôn tiếp `{topic}`, làm quiz về chủ đề này, hay hỏi sang một phần khác của tài liệu?"
                if has_prior
                else "Khi cần ôn hay làm quiz, bạn cứ nói tên phần muốn học nhé."
            )
        elif state.get("last_route") == "acknowledge" and intent in {"acknowledgement", "confirm_answer"}:
            ack_streak = 0
            for m in reversed(state.get("chat_history", [])):
                if m.get("role") != "assistant":
                    continue
                t = m.get("turn")
                if t is not None and getattr(t, "route_taken", "") == "acknowledge":
                    ack_streak += 1
                else:
                    break
            if ack_streak % 2 == 1:
                ans = f"Ok, mình không hỏi lại y câu cũ nữa. Mình hỏi gọn hơn một chút về `{topic}` nhé."
                follow = f"Để đỡ mất thời gian, bạn thử nêu 1 từ khóa hoặc 1 ý ngắn về `{topic}` mà bạn nhớ nhất nhé?"
            else:
                next_hint = f" hoặc chuyển sang `{next_topic}`" if next_topic else ""
                ans = f"Mình hiểu bạn muốn đi nhanh qua `{topic}`. Nếu chưa muốn tự diễn đạt ngay, mình có thể đổi sang ví dụ thực tế{next_hint}."
                follow = f"Bạn muốn mình đưa 1 ví dụ ngắn về `{topic}`, hay chuyển sang làm quiz kiểm tra nhanh?"
        elif intent == "confirm_answer" and has_prior and last_q:
            ans = (
                f"Ok, bạn trả lời \"{said}\" cho câu hỏi: \"{last_q}\". Mình ghi nhận ý đó. "
                "Câu trả lời có/không chưa đủ để thấy bạn hiểu cơ chế phía sau."
            )
            follow = f"Bạn nói thêm lý do hoặc ví dụ ngắn cho lựa chọn \"{said}\" của mình nhé?"
        elif not has_prior:
            ans = "Ok. Bạn muốn mình giải thích hoặc kiểm tra phần nào của tài liệu? Bạn có thể hỏi một khái niệm cụ thể."
            follow = ""
        elif words & {"tiep"} and next_topic:
            new_topic = next_topic
            ans = f"Ok, mình sang phần tiếp theo của tài liệu: `{next_topic}`."
            follow = f"Trước khi vào, bạn thử nói `{next_topic}` theo bạn là để làm gì, rồi mình đối chiếu với tài liệu nhé?"
        elif words & {"hieu"}:
            ans = (
                f"Tốt. Bạn nói đã hiểu `{topic}`, nhưng chỉ qua câu \"{said}\" thì mình chưa biết bạn nắm đến đâu, "
                "nên mình kiểm tra nhanh thay vì cho qua."
            )
            follow = f"Bạn thử giải thích lại `{topic}` bằng một hai câu của chính mình để mình chấm và chỉ chỗ cần chỉnh nhé?"
        elif last_q:
            ans = f"Ok! Để mình biết bạn nắm được ý chính của `{topic}` chưa, bạn trả lời ngắn câu hỏi vừa rồi bằng 1–2 câu của bạn nhé:"
            follow = last_q
        else:
            ans = f"Ok. Mình muốn chắc là bạn nắm `{topic}` chứ không chỉ đồng ý cho qua, nên mình kiểm tra nhanh một chút."
            follow = f"Bạn thử giải thích lại `{topic}` bằng một hai câu của chính mình để mình chấm và chỉ chỗ cần chỉnh nhé?"
        return {
            "topic": new_topic,
            "route_taken": "acknowledge",
            "pedagogical_reason": (
                "Người học chỉ xác nhận ngắn gọn, chưa cho thấy mức hiểu; chuyển sang kiểm tra bằng cách yêu cầu tự diễn đạt "
                "(không chấm điểm, không coi là câu hỏi ngoài tài liệu)."
            ),
            "answer": ans,
            "follow_up_question": follow,
            "cost_usd": 0.0,
        }

    def refuse_out_of_doc(state: CoachState) -> dict:
        ans, follow = demo_coach_response(
            route="refuse_out_of_doc",
            query=state["query"],
            topic=state["topic"],
            retrieved_chunks=state["retrieved_chunks"],
            past_mistakes=state["past_mistakes"],
            learner_dna=state["learner_dna"],
            db_path=db_path,
        )
        return {
            "route_taken": "refuse_out_of_doc",
            "pedagogical_reason": "Câu hỏi nằm ngoài phạm vi tài liệu đang nạp; từ chối trả lời để chống bịa thông tin.",
            "answer": ans,
            "follow_up_question": follow,
            "cost_usd": 0.0,
        }

    def bridge_prerequisite(state: CoachState) -> dict:
        prereq = state["prerequisite_topic"]
        pre_node = next((c for c in state["concept_map"].concepts if c.name.lower() == (prereq or "").lower()), None)
        prereq_page = pre_node.page_ref if pre_node and pre_node.page_ref else ""
        reason = f"Học viên đang yếu khái niệm nền tảng '{prereq}' liên quan trực tiếp đến '{state['topic']}'."
        if state["demo_mode"] or not config.has_live_api_key():
            ans, follow = demo_coach_response(
                route="bridge_prerequisite",
                query=state["query"],
                topic=state["topic"],
                retrieved_chunks=state["retrieved_chunks"],
                past_mistakes=state["past_mistakes"],
                learner_dna=state["learner_dna"],
                prerequisite_topic=prereq,
                prerequisite_page=prereq_page,
                db_path=db_path,
            )
            return {
                "route_taken": "bridge_prerequisite",
                "pedagogical_reason": reason,
                "answer": ans,
                "follow_up_question": follow,
                "cost_usd": 0.0,
            }

        ans, follow, extra_traces, cost_delta = _run_llm_with_tools(
            state,
            "bridge_prerequisite",
            f"Học viên đang hổng khái niệm tiên quyết '{prereq}'. Hãy nhắc lại khái niệm này trước khi trả lời câu chính.",
        )
        return {
            "route_taken": "bridge_prerequisite",
            "pedagogical_reason": reason,
            "answer": ans,
            "follow_up_question": follow,
            "tool_traces": [*state["tool_traces"], *extra_traces],
            "cost_usd": cost_delta,
        }

    def socratic_remediation(state: CoachState) -> dict:
        ctx = state.get("remediation_context") or ""
        reason = "Phát hiện lỗi tư duy đang mở hoặc yêu cầu gỡ rối câu sai; chuyển sang chế độ hỏi vặn Socratic."
        if state["demo_mode"] or not config.has_live_api_key():
            used_followups = {
                (getattr(m.get("turn"), "follow_up_question", "") or "").strip()
                for m in state.get("chat_history", [])
                if m.get("role") == "assistant" and m.get("turn") is not None
            }
            ans, follow = demo_coach_response(
                route="socratic_remediation",
                query=state["query"],
                topic=state["topic"],
                retrieved_chunks=state["retrieved_chunks"],
                past_mistakes=state["past_mistakes"],
                learner_dna=state["learner_dna"],
                used_followups=used_followups,
                db_path=db_path,
            )
            return {
                "route_taken": "socratic_remediation",
                "pedagogical_reason": reason,
                "answer": ans,
                "follow_up_question": follow,
                "cost_usd": 0.0,
            }

        ans, follow, extra_traces, cost_delta = _run_llm_with_tools(
            state,
            "socratic_remediation",
            f"Ngữ cảnh câu hỏi vừa làm sai (nếu có): {ctx}",
        )
        return {
            "route_taken": "socratic_remediation",
            "pedagogical_reason": reason,
            "answer": ans,
            "follow_up_question": follow,
            "tool_traces": [*state["tool_traces"], *extra_traces],
            "cost_usd": cost_delta,
        }

    def adaptive_explain(state: CoachState) -> dict:
        level = state["learner_dna"].overall_level
        intent = state["conv_intent"]
        prompt_key = {
            "Mới bắt đầu": "adaptive_explain_beginner",
            "Nâng cao": "adaptive_explain_advanced",
        }.get(level, "adaptive_explain_intermediate")
        if intent == "review_request":
            reason = f"Học viên yêu cầu ÔN TẬP chủ đề '{state['topic']}'; tóm tắt ý chính, không chuyển sang chất vấn lỗi tư duy."
        elif intent == "continue_topic":
            reason = f"Tiếp nối theo gợi ý của Gia sư, bắt đầu giải thích chủ đề '{state['topic']}' cho trình độ '{level}'."
        elif intent == "followup_example":
            reason = f"Nối tiếp mạch chủ đề '{state['topic']}'; đưa ví dụ thực tế minh họa theo trình độ '{level}'."
        elif intent == "followup_elaborate":
            reason = f"Nối tiếp mạch chủ đề '{state['topic']}'; đào sâu cơ chế hoạt động theo trình độ '{level}'."
        else:
            reason = f"Câu hỏi nằm trong tài liệu; điều chỉnh giọng giảng và độ sâu kỹ thuật cho trình độ '{level}'."

        if state["demo_mode"] or not config.has_live_api_key():
            used_followups = {
                (getattr(m.get("turn"), "follow_up_question", "") or "").strip()
                for m in state.get("chat_history", [])
                if m.get("role") == "assistant" and m.get("turn") is not None
            }
            is_followup = bool(_prior_routes_for_topic(state)) or (
                intent != "continue_topic" and bool(state.get("last_topic")) and _COREF_RE.search(_strip_accents(state["query"].lower())) is not None
            )
            if intent == "review_request":
                ans, follow = demo_review_response(
                    topic=state["topic"],
                    suggested=not _match_topic_in_query(state["query"], state["concept_map"]),
                    retrieved_chunks=state["retrieved_chunks"],
                    learner_dna=state["learner_dna"],
                    concept_map=state["concept_map"],
                    db_path=db_path,
                )
            elif intent in {"followup_example", "followup_elaborate"}:
                ans, follow = demo_coach_conversational_reply(
                    intent=intent,
                    query=state["query"],
                    topic=state["topic"],
                    retrieved_chunks=state["retrieved_chunks"],
                    learner_dna=state["learner_dna"],
                    concept_map=state["concept_map"],
                    db_path=db_path,
                )
            else:
                ans, follow = demo_coach_response(
                    route="adaptive_explain",
                    query=state["query"],
                    topic=state["topic"],
                    retrieved_chunks=state["retrieved_chunks"],
                    past_mistakes=state["past_mistakes"],
                    learner_dna=state["learner_dna"],
                    is_followup_turn=is_followup,
                    used_followups=used_followups,
                    db_path=db_path,
                )
            return {
                "route_taken": "adaptive_explain",
                "pedagogical_reason": reason,
                "answer": ans,
                "follow_up_question": follow,
                "cost_usd": 0.0,
            }

        extra_inst = f"Trả lời phù hợp tuyệt đối với trình độ {level} của học viên."
        if intent == "review_request":
            extra_inst += (
                f" Học viên muốn ÔN TẬP chủ đề '{state['topic']}': tóm tắt các ý chính theo tài liệu, nêu 2-3 điểm dễ nhầm; "
                "nếu hồ sơ có lỗi tư duy liên quan thì chỉ nhắc ngắn một dòng, KHÔNG chất vấn hay chẩn đoán dài; "
                "kết bằng một câu hỏi kiểm tra ngắn."
            )
        elif intent == "followup_example":
            extra_inst += f" Học viên đang xin VÍ DỤ THỰC TẾ cho chủ đề '{state['topic']}'."
        elif intent == "followup_elaborate":
            extra_inst += f" Học viên muốn ĐÀO SÂU CƠ CHẾ chi tiết hơn cho chủ đề '{state['topic']}'."
        ans, follow, extra_traces, cost_delta = _run_llm_with_tools(state, prompt_key, extra_inst)
        return {
            "route_taken": "adaptive_explain",
            "pedagogical_reason": reason,
            "answer": ans,
            "follow_up_question": follow,
            "tool_traces": [*state["tool_traces"], *extra_traces],
            "cost_usd": cost_delta,
        }

    def update_learner_memory(state: CoachState) -> dict:
        traces = list(state["tool_traces"])
        suggested_topic = ""
        if state["route_taken"] not in {"refuse_out_of_doc", "coach_guide"}:
            if state["route_taken"] == "acknowledge":
                suggested_topic = state["topic"] if state.get("last_topic") else ""
            else:
                suggested_topic = state.get("prerequisite_topic") or state["topic"]
                traces.append(
                    ToolCallTrace(
                        tool_name="record_learning_signal",
                        arguments={"topic": state["topic"], "route": state["route_taken"]},
                        summary=f"Đã ghi nhận tương tác Socratic cho chủ đề '{state['topic']}' (nhánh {state['route_taken']})",
                    )
                )
        return {
            "tool_traces": traces,
            "suggested_quiz_topic": suggested_topic,
        }

    graph = StateGraph(CoachState)
    graph.add_node("inspect_and_retrieve", inspect_and_retrieve)
    graph.add_node("coach_guide", coach_guide)
    graph.add_node("socratic_eval", socratic_eval)
    graph.add_node("refuse_out_of_doc", refuse_out_of_doc)
    graph.add_node("acknowledge", acknowledge)
    graph.add_node("bridge_prerequisite", bridge_prerequisite)
    graph.add_node("socratic_remediation", socratic_remediation)
    graph.add_node("adaptive_explain", adaptive_explain)
    graph.add_node("update_learner_memory", update_learner_memory)

    graph.add_edge(START, "inspect_and_retrieve")
    graph.add_conditional_edges(
        "inspect_and_retrieve",
        route_pedagogy,
        {
            "coach_guide": "coach_guide",
            "socratic_eval": "socratic_eval",
            "refuse_out_of_doc": "refuse_out_of_doc",
            "acknowledge": "acknowledge",
            "bridge_prerequisite": "bridge_prerequisite",
            "socratic_remediation": "socratic_remediation",
            "adaptive_explain": "adaptive_explain",
        },
    )
    for node_name in (
        "coach_guide",
        "socratic_eval",
        "refuse_out_of_doc",
        "acknowledge",
        "bridge_prerequisite",
        "socratic_remediation",
        "adaptive_explain",
    ):
        graph.add_edge(node_name, "update_learner_memory")
    graph.add_edge("update_learner_memory", END)
    compiled = graph.compile()
    result = (
        (
            compiled,
            {
                "refuse_out_of_doc": refuse_out_of_doc,
                "socratic_eval": socratic_eval,
                "acknowledge": acknowledge,
                "update_learner_memory": update_learner_memory,
            },
        )
        if return_nodes
        else compiled
    )
    if len(_GRAPH_CACHE) >= _GRAPH_CACHE_LIMIT:
        _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
    _GRAPH_CACHE[cache_key] = result
    return result


def _make_init_state(
    query: str,
    learner_dna: LearnerDNA,
    concept_map: DocumentConceptMap,
    doc_id: str,
    focus_topic: str | None = None,
    remediation_context: str | None = None,
    model: str | None = None,
    demo_mode: bool = False,
    chat_history: list[dict] | None = None,
) -> CoachState:
    return {
        "query": query.strip(),
        "learner_id": learner_dna.learner_id,
        "doc_id": doc_id,
        "focus_topic": focus_topic,
        "remediation_context": remediation_context,
        "model": model,
        "demo_mode": demo_mode or (not config.has_live_api_key()),
        "chat_history": chat_history or [],
        "learner_dna": learner_dna,
        "concept_map": concept_map,
        "conv_intent": "standard_qa",
        "last_topic": "",
        "last_route": "",
        "last_follow_up": "",
        "topic": "",
        "is_in_scope": True,
        "prerequisite_topic": None,
        "retrieved_chunks": [],
        "past_mistakes": [],
        "tool_traces": [],
        "route_taken": "",
        "pedagogical_reason": "",
        "answer": "",
        "follow_up_question": "",
        "suggested_quiz_topic": "",
        "start_ts": time.perf_counter(),
        "cost_usd": 0.0,
    }


def _iter_word_chunks(text: str, words_per_chunk: int = 8) -> Iterator[str]:
    tokens = text.split(" ")
    for i in range(0, len(tokens), words_per_chunk):
        piece = " ".join(tokens[i:i + words_per_chunk])
        if i + words_per_chunk < len(tokens):
            piece += " "
        yield piece


def run_coach_turn(
    query: str,
    kb: DualMemoryKnowledgeBase,
    learner_dna: LearnerDNA,
    concept_map: DocumentConceptMap,
    doc_id: str,
    focus_topic: str | None = None,
    remediation_context: str | None = None,
    model: str | None = None,
    demo_mode: bool = False,
    chat_history: list[dict] | None = None,
    db_path: str | Path | None = None,
) -> CoachTurnResult:
    """Chạy trọn một lượt hội thoại qua đồ thị LangGraph Socratic Coach."""
    init_state = _make_init_state(
        query=query,
        learner_dna=learner_dna,
        concept_map=concept_map,
        doc_id=doc_id,
        focus_topic=focus_topic,
        remediation_context=remediation_context,
        model=model,
        demo_mode=demo_mode,
        chat_history=chat_history,
    )
    app_graph = _build_coach_graph(kb, db_path=db_path)
    final = app_graph.invoke(init_state)
    latency_ms = int((time.perf_counter() - init_state["start_ts"]) * 1000)

    chunks = final["retrieved_chunks"] if final["route_taken"] != "refuse_out_of_doc" else []
    citations = list(dict.fromkeys([c.page for c in chunks if c.page]))

    return CoachTurnResult(
        answer=final["answer"],
        route_taken=final["route_taken"],
        pedagogical_reason=final["pedagogical_reason"],
        learner_level_used=learner_dna.overall_level,
        citations=citations,
        retrieved_chunks=chunks,
        tool_traces=final["tool_traces"],
        follow_up_question=final["follow_up_question"],
        suggested_quiz_topic=final["suggested_quiz_topic"],
        topic="" if final["route_taken"] == "refuse_out_of_doc" else final.get("topic", ""),
        latency_ms=latency_ms,
        cost_usd=final.get("cost_usd", 0.0),
        is_demo=init_state["demo_mode"],
    )


def stream_coach_turn(
    query: str,
    kb: DualMemoryKnowledgeBase,
    learner_dna: LearnerDNA,
    concept_map: DocumentConceptMap,
    doc_id: str,
    focus_topic: str | None = None,
    remediation_context: str | None = None,
    model: str | None = None,
    demo_mode: bool = False,
    chat_history: list[dict] | None = None,
    db_path: str | Path | None = None,
) -> tuple[Iterator[str], dict[str, CoachTurnResult]]:
    """Stream từng cụm văn bản phản hồi cho UI; sau khi đọc hết generator, result_box['turn'] chứa kết quả đầy đủ."""
    result_box: dict[str, CoachTurnResult] = {}
    is_demo = demo_mode or (not config.has_live_api_key())

    if is_demo:
        def _demo_gen() -> Iterator[str]:
            turn = run_coach_turn(
                query=query,
                kb=kb,
                learner_dna=learner_dna,
                concept_map=concept_map,
                doc_id=doc_id,
                focus_topic=focus_topic,
                remediation_context=remediation_context,
                model=model,
                demo_mode=True,
                chat_history=chat_history,
                db_path=db_path,
            )
            yield from _iter_word_chunks(turn.answer)
            result_box["turn"] = turn

        return _demo_gen(), result_box

    init_state = _make_init_state(
        query=query,
        learner_dna=learner_dna,
        concept_map=concept_map,
        doc_id=doc_id,
        focus_topic=focus_topic,
        remediation_context=remediation_context,
        model=model,
        demo_mode=False,
        chat_history=chat_history,
    )
    inspected = _inspect_state(init_state, kb)
    state: CoachState = {**init_state, **inspected}  # type: ignore[typeddict-item]
    route = _route_from_state(state)

    if route in {"refuse_out_of_doc", "socratic_eval", "acknowledge"}:
        def _fallback_gen() -> Iterator[str]:
            _, nodes = _build_coach_graph(kb, db_path=db_path, return_nodes=True)
            cur: CoachState = {**state, **nodes[route](state)}  # type: ignore[typeddict-item]
            cur = {**cur, **nodes["update_learner_memory"](cur)}  # type: ignore[typeddict-item]
            latency_ms = int((time.perf_counter() - init_state["start_ts"]) * 1000)
            refused = cur["route_taken"] == "refuse_out_of_doc"
            chunks = [] if refused else cur["retrieved_chunks"]
            turn = CoachTurnResult(
                answer=cur["answer"],
                route_taken=cur["route_taken"],
                pedagogical_reason=cur["pedagogical_reason"],
                learner_level_used=learner_dna.overall_level,
                citations=list(dict.fromkeys([c.page for c in chunks if c.page])),
                retrieved_chunks=chunks,
                tool_traces=cur["tool_traces"],
                follow_up_question=cur["follow_up_question"],
                suggested_quiz_topic=cur["suggested_quiz_topic"],
                topic="" if refused else cur.get("topic", ""),
                latency_ms=latency_ms,
                cost_usd=cur.get("cost_usd", 0.0),
                is_demo=False,
            )
            yield from _iter_word_chunks(turn.answer)
            result_box["turn"] = turn

        return _fallback_gen(), result_box

    prompt_key, extra_inst, reason = _resolve_live_prompt_spec(state, route)
    weak = [f"{c.concept} ({c.mastery_score:.0f}%)" for c in learner_dna.concepts if c.mastery_score < 60]
    misc = [m.description for m in learner_dna.active_misconceptions]
    traces = list(state["tool_traces"])
    traces.append(
        ToolCallTrace(
            tool_name="lookup_learner_weaknesses",
            arguments={"learner_id": learner_dna.learner_id},
            summary=f"Chủ đề cần cải thiện: {', '.join(weak) or 'Không có'}. Lỗi: {', '.join(misc) or 'Không có'}."[:180],
        )
    )
    openai_msgs = _build_coach_openai_messages(state, prompt_key, extra_inst)

    def _live_gen() -> Iterator[str]:
        cost_before = db.get_session_cost_usd(db_path=db_path)
        parts: list[str] = []
        for token_chunk in stream_chat_completion(
            messages=openai_msgs,
            model=model or config.DEFAULT_MODEL,
            temperature=0.3,
            purpose=f"coach_stream_{route}",
            db_path=db_path,
        ):
            parts.append(token_chunk)
            yield token_chunk

        full_text = "".join(parts).strip()
        body, trailing_q = _extract_trailing_question(full_text)
        cost_delta = max(0.0, round(db.get_session_cost_usd(db_path=db_path) - cost_before, 6))
        latency_ms = int((time.perf_counter() - state["start_ts"]) * 1000)

        suggested_topic = ""
        if route != "coach_guide":
            suggested_topic = state.get("prerequisite_topic") or state["topic"]
            traces.append(
                ToolCallTrace(
                    tool_name="record_learning_signal",
                    arguments={"topic": state["topic"], "route": route},
                    summary=f"Đã ghi nhận tương tác Socratic cho chủ đề '{state['topic']}' (nhánh {route})",
                )
            )
        chunks = state["retrieved_chunks"]
        citations = list(dict.fromkeys([c.page for c in chunks if c.page]))
        result_box["turn"] = CoachTurnResult(
            answer=body,
            route_taken=route,
            pedagogical_reason=reason,
            learner_level_used=learner_dna.overall_level,
            citations=citations,
            retrieved_chunks=chunks,
            tool_traces=traces,
            follow_up_question=trailing_q,
            suggested_quiz_topic=suggested_topic,
            topic=state["topic"],
            latency_ms=latency_ms,
            cost_usd=cost_delta,
            is_demo=False,
        )

    return _live_gen(), result_box
