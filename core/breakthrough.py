"""Phòng Lab Đột Phá: 4 tính năng chưa có ngoài thị trường dùng LangChain, FAISS, LangGraph & Structured Output."""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
from core import db
from core.contracts import (
    CognitiveDiffReport,
    DefenseEvaluation,
    DevilsAdvocateQuestion,
    DiffBlindSpot,
    DiffContradiction,
    DiffVerifiedItem,
    DocumentConceptMap,
    IncidentClue,
    IncidentEvaluation,
    IncidentScenario,
    LearnerDNA,
    TrojanChallenge,
    TrojanStatement,
)
from core.demo_provider import _claim_has_wrong_phrase, _is_default_source
from core.llm import LLMError, complete_structured
from core.prompts import (
    COGNITIVE_DIFF_PROMPT,
    DEVILS_ADVOCATE_EVAL_PROMPT,
    DEVILS_ADVOCATE_GEN_PROMPT,
    INCIDENT_EVAL_PROMPT,
    INCIDENT_GEN_PROMPT,
    TROJAN_GEN_PROMPT,
)
from core.rag import DualMemoryKnowledgeBase

SAMPLE_NOTES: dict[str, dict[str, str]] = {
    "rag_note_loi": {
        "title": "Ghi chú ôn tập RAG (có 2 lỗi hiểu ngược & thiếu mục)",
        "content": (
            "1. RAG giúp mô hình trả lời bám tài liệu bên ngoài bằng cách tìm đoạn liên quan trước khi sinh câu trả lời.\n"
            "2. Khi chia chunk, đặt chunk quá lớn sẽ làm mất ngữ cảnh, còn chunk quá nhỏ chỉ làm tốn dung lượng.\n"
            "3. Nên đặt overlap bằng 0 để tiết kiệm bộ nhớ và tránh lặp từ.\n"
            "4. Embedding biến văn bản thành vector số để tìm theo ngữ nghĩa gần đúng."
        ),
    },
    "api_note_loi": {
        "title": "Ghi chú OpenAI API (có lỗi về messages & temperature)",
        "content": (
            "1. API key phải lưu trong biến môi trường hoặc file .env, không ghi thẳng vào code.\n"
            "2. Mô hình OpenAI tự động nhớ các lượt hỏi đáp trước đó nên chỉ cần gửi câu hỏi mới nhất.\n"
            "3. Khi cần trích xuất dữ liệu ổn định hoặc JSON, nên đặt temperature thật cao (khoảng 1.5) để model thông minh hơn.\n"
            "4. Structured outputs dùng Pydantic giúp đảm bảo đầu ra đúng định dạng."
        ),
    },
}


def _log_demo(purpose: str, db_path: str | Path | None = None, latency_ms: int = 55) -> None:
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


def load_project_infra_doc() -> tuple[str, str] | None:
    """Đọc file Documents/INFRA.md của dự án để đội ngũ luyện bảo vệ đồ án trực tiếp."""
    base = config.BASE_DIR
    candidates = (
        base / "samples" / "INFRA.md",
        base / "docs" / "INFRA.md",
        base.parent / "Documents" / "INFRA.md",
        base.parent / "docs" / "INFRA.md",
    )
    for infra_path in candidates:
        if infra_path.exists():
            return infra_path.read_text(encoding="utf-8"), "INFRA.md (Kiến trúc Hackathon Study Coach)"
    return None


# ==============================================================================
# 1. GIT DIFF CHO NÃO BỘ (COGNITIVE DIFF)
# ==============================================================================

def _offline_cognitive_diff(
    learner_note: str,
    kb: DualMemoryKnowledgeBase,
    concept_map: DocumentConceptMap,
    db_path: str | Path | None = None,
) -> CognitiveDiffReport:
    _log_demo("cognitive_diff", db_path=db_path, latency_ms=62)
    lines = [ln.strip() for ln in re.split(r"\n+|(?<=[.!?])\s+", learner_note.strip()) if len(ln.strip()) >= 12]
    contradictions: list[DiffContradiction] = []
    verified: list[DiffVerifiedItem] = []
    # Quy tắc/ground-truth cứng chỉ áp cho giáo trình mặc định; tài liệu khác dùng so khớp chung theo chunk truy xuất.
    is_default = _is_default_source(
        (str(d.metadata.get("source", "")) for d in kb.documents),
        " ".join(d.page_content for d in kb.documents),
    )

    for claim in lines:
        low = claim.lower()
        chunks, in_scope = kb.retrieve(claim, top_k=2)
        top_chunk = chunks[0] if chunks else None
        page_ref = top_chunk.page if top_chunk and top_chunk.page else "[Tài liệu]"

        # Bắt các quy luật mâu thuẫn kinh điển bám theo tài liệu truy xuất từ FAISS
        if is_default and (
            ("chunk" in low or "đoạn" in low)
            and _claim_has_wrong_phrase(low, ("mất ngữ cảnh",), require=("lớn",), forbid=("nhỏ",))
        ):
            contradictions.append(
                DiffContradiction(
                    topic="Chunking và Overlap",
                    learner_claim=claim,
                    ground_truth="Chunk quá lớn làm việc truy xuất kém chính xác vì chứa nhiều ý lẫn lộn, còn chunk quá nhỏ làm mất ngữ cảnh.",
                    page_ref="[Mục 2]",
                    severity="Cao",
                    explanation="Bạn đã đảo ngược nguyên nhân-hậu quả: mất ngữ cảnh là lỗi của chunk quá nhỏ, không phải chunk quá lớn.",
                )
            )
            continue

        if is_default and _claim_has_wrong_phrase(
            low, ("bằng 0", "= 0", "không cần", "bỏ qua", "tránh lặp"), require=("overlap",)
        ):
            contradictions.append(
                DiffContradiction(
                    topic="Chunking và Overlap",
                    learner_claim=claim,
                    ground_truth="Người ta thường để các chunk gối lên nhau một phần, gọi là overlap, để một ý nằm ở ranh giới giữa hai chunk không bị cắt đứt.",
                    page_ref="[Mục 2]",
                    severity="Cao",
                    explanation="Đặt overlap = 0 sẽ cắt đứt câu/ý nằm ngay ranh giới hai chunk khiến RAG truy xuất thiếu ngữ cảnh.",
                )
            )
            continue

        if is_default and _claim_has_wrong_phrase(
            low, ("tự động nhớ", "tự nhớ", "chỉ cần gửi câu hỏi mới")
        ) and any(k in low for k in ("openai", "mô hình", "lượt", "hội thoại", "messages")):
            contradictions.append(
                DiffContradiction(
                    topic="Cấu trúc lời gọi & Messages",
                    learner_claim=claim,
                    ground_truth="Mô hình không tự nhớ cuộc trò chuyện, nên nếu muốn hội thoại nhiều lượt thì chương trình phải gửi lại lịch sử messages trong mỗi lần gọi.",
                    page_ref="[Mục 2 · API]",
                    severity="Cao",
                    explanation="Chat Completions API không lưu trạng thái hội thoại; bỏ qua lịch sử messages sẽ làm bot quên sạch câu trước.",
                )
            )
            continue

        if is_default and "temperature" in low and _claim_has_wrong_phrase(
            low, ("cao", "1.5", "1.8", "2.0"), require=("temperature",)
        ) and any(k in low for k in ("ổn định", "trích xuất", "json")):
            contradictions.append(
                DiffContradiction(
                    topic="Cấu trúc lời gọi & Messages",
                    learner_claim=claim,
                    ground_truth="Tham số temperature điều khiển mức ngẫu nhiên của câu trả lời: giá trị thấp cho kết quả ổn định, phù hợp việc trích xuất thông tin.",
                    page_ref="[Mục 3 · API]",
                    severity="Cao",
                    explanation="Trích xuất dữ liệu/JSON cần temperature thấp (0.0–0.3); temperature cao làm câu trả lời ngẫu nhiên và dễ vỡ định dạng.",
                )
            )
            continue

        if is_default and "rag" in low and _claim_has_wrong_phrase(low, ("huấn luyện lại", "fine-tuning")):
            contradictions.append(
                DiffContradiction(
                    topic="RAG là gì",
                    learner_claim=claim,
                    ground_truth="RAG (Retrieval-Augmented Generation) là kỹ thuật cho mô hình ngôn ngữ trả lời dựa trên tài liệu bên ngoài bằng cách tìm đoạn liên quan đưa vào prompt.",
                    page_ref="[Mục 1]",
                    severity="Trung bình",
                    explanation="RAG bổ sung ngữ cảnh tại thời điểm truy vấn chứ không huấn luyện lại trọng số mô hình.",
                )
            )
            continue

        # Kiểm tra ngược cực tính tổng quát cho tài liệu tùy chỉnh bất kỳ
        if top_chunk and in_scope and top_chunk.score >= 0.22:
            chunk_low = top_chunk.content.lower()
            neg_claim = any(n in low for n in ("không cần", "không thể", "cấm", "bỏ qua"))
            neg_truth = any(n in chunk_low for n in ("không cần", "không thể", "cấm", "bỏ qua"))
            if neg_claim != neg_truth and ("không cần" in low or "bỏ qua" in low):
                contradictions.append(
                    DiffContradiction(
                        topic=top_chunk.topic or "Kiến thức chung",
                        learner_claim=claim,
                        ground_truth=top_chunk.content[:240],
                        page_ref=page_ref,
                        severity="Trung bình",
                        explanation="Phát hiện khác biệt về điều kiện khẳng định/phủ định so với tài liệu gốc.",
                    )
                )
                continue

        if in_scope and top_chunk:
            verified.append(
                DiffVerifiedItem(
                    learner_claim=claim,
                    page_ref=page_ref,
                    note="Khớp với nội dung tài liệu nguồn.",
                )
            )

    # Tìm điểm mù bỏ sót dựa trên Bản đồ Khái niệm
    note_low = learner_note.lower()
    blind_spots: list[DiffBlindSpot] = []
    for node in concept_map.concepts:
        terms = [t.lower() for t in node.key_terms] + [w.lower() for w in node.name.split() if len(w) >= 3]
        hits = sum(1 for t in set(terms) if t in note_low)
        if hits == 0 or (len(set(terms)) >= 3 and hits / len(set(terms)) < 0.18):
            blind_spots.append(
                DiffBlindSpot(
                    topic=node.name,
                    page_ref=node.page_ref or "[Tài liệu]",
                    importance_reason=node.summary,
                )
            )

    total_items = max(len(verified) + len(contradictions) + len(blind_spots), 1)
    raw_score = int(round(100.0 * (len(verified) / total_items) - 10.0 * len(contradictions)))
    score = max(12, min(98, raw_score))

    patched_lines = ["BẢN GHI CHÚ ĐÃ HIỆU ĐÍNH BÁM TÀI LIỆU GỐC:"]
    for v in verified:
        patched_lines.append(f"- [Đã kiểm chứng {v.page_ref}] {v.learner_claim}")
    for c in contradictions:
        patched_lines.append(f"- [Đã sửa lỗi {c.page_ref}] {c.ground_truth}")
    for b in blind_spots[:4]:
        patched_lines.append(f"- [Bổ sung điểm mù {b.page_ref} — {b.topic}] {b.importance_reason}")

    summary = (
        f"Phát hiện {len(contradictions)} câu hiểu ngược nguy hiểm và {len(blind_spots)} khái niệm bị bỏ sót "
        f"(khớp {len(verified)} điểm)."
    )
    return CognitiveDiffReport(
        alignment_score=score,
        summary=summary,
        contradictions=contradictions,
        blind_spots=blind_spots,
        verified_points=verified,
        patched_note="\n".join(patched_lines),
    )


def run_cognitive_diff(
    learner_note: str,
    kb: DualMemoryKnowledgeBase,
    concept_map: DocumentConceptMap,
    learner_dna: LearnerDNA,
    doc_id: str,
    model: str | None = None,
    demo_mode: bool = False,
    db_path: str | Path | None = None,
) -> tuple[CognitiveDiffReport, LearnerDNA]:
    """Soi mâu thuẫn & điểm mù giữa Ghi chú cá nhân và Giáo trình gốc."""
    if demo_mode or not config.has_live_api_key():
        report = _offline_cognitive_diff(learner_note, kb, concept_map, db_path=db_path)
    else:
        chunks, _ = kb.retrieve(learner_note[:800], top_k=6)
        ctx = "\n\n".join(f"{c.page} ({c.source}): {c.content}" for c in chunks)
        topics = "\n".join(f"- {n.name} ({n.page_ref}): {n.summary}" for n in concept_map.concepts)
        user_msg = (
            f"DANH SÁCH KHÁI NIỆM GIÁO TRÌNH:\n{topics}\n\n"
            f"ĐOẠN GIÁO TRÌNH NGUỒN:\n{ctx}\n\n"
            f"GHI CHÚ CÁ NHÂN CỦA HỌC VIÊN:\n{learner_note.strip()}"
        )
        try:
            report = complete_structured(
                [{"role": "system", "content": COGNITIVE_DIFF_PROMPT}, {"role": "user", "content": user_msg}],
                CognitiveDiffReport,
                model=model or config.DEFAULT_MODEL,
                temperature=0.2,
                purpose="cognitive_diff",
                db_path=db_path,
            )
        except LLMError:
            report = _offline_cognitive_diff(learner_note, kb, concept_map, db_path=db_path)

    contradicted_topics = {c.topic.strip().lower() for c in report.contradictions if c.topic}
    for c in report.contradictions:
        db.record_misconception(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            topic=c.topic,
            description=f"[Diff ghi chú] {c.explanation}",
            db_path=db_path,
        )
        db.update_concept_mastery(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            concept=c.topic,
            is_correct=False,
            bloom_level="Hiểu",
            total_concepts=len(concept_map.concepts) or None,
            db_path=db_path,
        )

    verified_topics: set[str] = set()
    for v in report.verified_points:
        v_low = v.learner_claim.lower()
        best_node = None
        for node in concept_map.concepts:
            if v.page_ref and node.page_ref and v.page_ref == node.page_ref:
                best_node = node
                break
            terms = [node.name.lower()] + [t.lower() for t in node.key_terms if len(t) >= 3]
            if any(t in v_low for t in terms):
                best_node = node
                break
        if best_node and best_node.name.strip().lower() not in contradicted_topics:
            verified_topics.add(best_node.name)

    for topic_name in sorted(verified_topics):
        db.update_concept_mastery(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            concept=topic_name,
            is_correct=True,
            bloom_level="Hiểu",
            total_concepts=len(concept_map.concepts) or None,
            db_path=db_path,
        )
        db.resolve_misconception(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            topic=topic_name,
            db_path=db_path,
        )

    updated = db.get_learner_dna(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concepts_order=[n.name for n in concept_map.concepts],
        db_path=db_path,
    )
    return report, updated


# ==============================================================================
# 2. CA TRỰC CỨU HỎA PRODUCTION (DOC-TO-INCIDENT SIMULATOR VIA LANGGRAPH)
# ==============================================================================

_CURRICULUM_INCIDENTS: list[IncidentScenario] = [
    IncidentScenario(
        ticket_id="INC-401",
        severity="SEV-1",
        title="Chatbot nội bộ trả lời thiếu nửa cuối quy trình & hóa đơn API tăng gấp 4 lần",
        symptom=(
            "Người dùng hỏi quy định nằm ở đoạn chuyển tiếp cuối trang 3 PDF nhưng bot luôn cắt mất vế sau của câu trả lời. "
            "Đồng thời chi phí token đầu vào mỗi lượt chat tăng vọt gấp 4 lần mức dự kiến."
        ),
        topic="Chunking và Overlap",
        page_ref="[Mục 2, Mục 4]",
        clues=[
            IncidentClue(
                tool_name="inspect_config",
                title="Cấu hình Ingest & Vector Store hiện tại",
                output="chunk_size = 400 | chunk_overlap = 0 | top_k = 18 | metadata_pages = enabled",
            ),
            IncidentClue(
                tool_name="inspect_logs",
                title="Log truy xuất đoạn tài liệu (FAISS)",
                output="Chunk #11 kết thúc giữa câu: 'Khi phát hiện lỗi mạng hoặc rate limit thì...' -> Mất phần sau ở Chunk #12!",
            ),
            IncidentClue(
                tool_name="inspect_metrics",
                title="Chỉ số prompt & chi phí trung bình",
                output="Avg retrieved prompt length: 7,400 chars/turn (k=18) -> Input tokens x4 · Latency: 4.8s",
            ),
        ],
        root_cause_options=[
            "overlap = 0 làm đứt mạch ý ở ranh giới chunk và top_k = 18 khiến prompt quá dài, đội chi phí token",
            "Do embedding cosine similarity chỉ hỗ trợ tiếng Anh nên cắt mất vế sau tiếng Việt",
            "Do dùng Structured Output Pydantic làm giảm độ dài câu trả lời của LLM",
            "Do dùng file .env để đọc OPENAI_API_KEY khiến tốc độ mạng chậm gấp 4 lần",
        ],
        correct_option_index=0,
        ground_truth_explanation=(
            "Theo [Mục 2], cần thiết lập overlap để ý ở ranh giới hai chunk không bị cắt đứt; và theo [Mục 4], "
            "giá trị top-k thường nhỏ (3 đến 5) để prompt không quá dài và kiểm soát chi phí."
        ),
    ),
    IncidentScenario(
        ticket_id="INC-402",
        severity="SEV-1",
        title="Trợ lý hội thoại quên sạch câu hỏi trước & parser JSON báo lỗi liên tục",
        symptom=(
            "Khi học viên hỏi câu nối tiếp 'Còn trường hợp ngược lại thì sao?', bot trả lời như người mới quen lần đầu. "
            "Ngoài ra 35% lượt gọi trích xuất dữ liệu trả về văn bản tự do làm crash hàm parse."
        ),
        topic="Cấu trúc lời gọi & Messages",
        page_ref="[Mục 2–4 · API]",
        clues=[
            IncidentClue(
                tool_name="inspect_config",
                title="Tham số gọi chat.completions hiện tại",
                output="endpoint = create() | response_format = None | temperature = 1.6 | messages = [system, current_user_only]",
            ),
            IncidentClue(
                tool_name="inspect_logs",
                title="Payload gửi lên AI Portal",
                output="len(messages) == 2 ở mọi lượt gọi; không có lịch sử role 'assistant' và 'user' trước đó.",
            ),
            IncidentClue(
                tool_name="inspect_metrics",
                title="Tỉ lệ lỗi định dạng đầu ra",
                output="JSONDecodeError: 35% requests · Avg temperature variance: High · Stateless session misses: 100%",
            ),
        ],
        root_cause_options=[
            "Không gửi lại lịch sử messages ở mỗi lần gọi làm mất ngữ cảnh, temperature=1.6 quá cao và thiếu Structured Outputs",
            "Do quên gán biến OPENAI_BASE_URL nên mỗi lượt gọi bị đổi sang model khác",
            "Do PyPDF không gắn [Trang n] vào từng tin nhắn hội thoại",
            "Do đặt max_tokens quá lớn nên model tự xóa lịch sử cuộc trò chuyện cũ",
        ],
        correct_option_index=0,
        ground_truth_explanation=(
            "Theo [Mục 2–4 · API], mô hình không tự nhớ hội thoại nên phải gửi lại lịch sử messages; "
            "trích xuất cần temperature thấp và dùng Structured Outputs (Pydantic) thay vì parse chuỗi tự do."
        ),
    ),
    IncidentScenario(
        ticket_id="INC-403",
        severity="SEV-2",
        title="Hệ thống RAG đạt 85% câu trả lời đúng nhưng vẫn bị người dùng khiếu nại bịa thông tin",
        symptom=(
            "Đội phát triển tăng kích thước chunk lên 4.000 ký tự và chỉ đo tỉ lệ trả lời đúng bằng LLM-judge, "
            "nhưng khi người dùng kiểm tra lại PDF thì không tìm thấy số trang hoặc thấy AI tự suy diễn câu ngoài tài liệu."
        ),
        topic="Đánh giá chất lượng RAG",
        page_ref="[Mục 2, Mục 5, Mục 6]",
        clues=[
            IncidentClue(
                tool_name="inspect_config",
                title="Cấu hình đánh giá & prompt hệ thống",
                output="eval_metrics = ['answer_correctness'] | citation_check = disabled | system_prompt = 'Hãy trả lời thật chi tiết câu hỏi'",
            ),
            IncidentClue(
                tool_name="inspect_logs",
                title="Mẫu kiểm tra truy xuất thực tế",
                output="Query: 'Chính sách hoàn học phí?' (không có trong tài liệu) -> AI trả lời tự tin theo kiến thức chung, không dẫn nguồn.",
            ),
            IncidentClue(
                tool_name="inspect_metrics",
                title="Báo cáo chất lượng tuần",
                output="Answer accuracy: 85% | Source citation accuracy: Not measured (actual ~32%) | Chunk impurity: High",
            ),
        ],
        root_cause_options=[
            "Thiếu đo tỉ lệ trích đúng nguồn khi eval, chunk quá lớn lẫn nhiều ý, và prompt thiếu lệnh nói 'không có trong tài liệu'",
            "Do FAISS chỉ hoạt động khi dùng với file .docx chứ không hỗ trợ file .txt",
            "Do chưa dùng Function calling để tính cosine similarity bằng tay",
            "Do chỉ số temperature = 0 khiến mô hình bịa thêm thông tin ngoài tài liệu",
        ],
        correct_option_index=0,
        ground_truth_explanation=(
            "Theo [Mục 5–6], prompt RAG phải yêu cầu trích dẫn số trang và nói rõ 'không có trong tài liệu' khi thiếu thông tin; "
            "khi đánh giá phải đo cả tỉ lệ trả lời đúng LẪN tỉ lệ trích đúng nguồn."
        ),
    ),
]


def _shuffle_incident_options(sc: IncidentScenario) -> IncidentScenario:
    """Xáo thứ tự phương án một cách xác định (seed theo ticket_id) để đáp án đúng không luôn là A."""
    n = len(sc.root_cause_options)
    if n < 2 or not 0 <= sc.correct_option_index < n:
        return sc
    seed = int(hashlib.sha256(sc.ticket_id.encode("utf-8")).hexdigest()[:8], 16)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    out = sc.model_copy(deep=True)
    out.root_cause_options = [sc.root_cause_options[i] for i in order]
    out.correct_option_index = order.index(sc.correct_option_index)
    return out


def _is_valid_incident(sc: IncidentScenario) -> bool:
    """Kịch bản Incident từ LLM hợp lệ: đủ 4 phương án (không rỗng), đáp án trong 0..3, đủ 3 clue."""
    return (
        len(sc.root_cause_options) == 4
        and all(o.strip() for o in sc.root_cause_options)
        and 0 <= sc.correct_option_index <= 3
        and len(sc.clues) == 3
        and bool(sc.title.strip())
        and bool(sc.ground_truth_explanation.strip())
    )


class _IncidentState(TypedDict):
    doc_text: str
    concept_map: DocumentConceptMap
    scenario_index: int
    scenario: IncidentScenario | None


def generate_incident_scenario(
    doc_text: str,
    concept_map: DocumentConceptMap,
    kb: DualMemoryKnowledgeBase,
    scenario_index: int = 0,
    model: str | None = None,
    demo_mode: bool = False,
) -> IncidentScenario:
    """Dựng Ticket sự cố Production từ tài liệu học tập qua LangGraph StateGraph."""
    if not demo_mode and config.has_live_api_key():
        topics = ", ".join(c.name for c in concept_map.concepts[:5])
        try:
            live_sc = complete_structured(
                [
                    {"role": "system", "content": INCIDENT_GEN_PROMPT},
                    {"role": "user", "content": f"CHỦ ĐỀ: {topics}\n\nTÀI LIỆU:\n{doc_text[:config.MAX_INPUT_CHARS]}"},
                ],
                IncidentScenario,
                model=model or config.DEFAULT_MODEL,
                temperature=0.35,
                purpose="incident_gen",
            )
            if _is_valid_incident(live_sc):
                return _shuffle_incident_options(live_sc)
            # Dữ liệu LLM hỏng: dùng generator offline thay vì chấm người học theo dữ liệu lỗi.
        except LLMError:
            pass

    def build_incident_node(state: _IncidentState) -> dict:
        low = state["doc_text"].lower()
        if "chunk quá lớn" in low and "openai api" in low:
            sc = _CURRICULUM_INCIDENTS[state["scenario_index"] % len(_CURRICULUM_INCIDENTS)].model_copy(deep=True)
            return {"scenario": _shuffle_incident_options(sc)}

        # Sinh sự cố động từ bất kỳ tài liệu tùy chỉnh nào
        nodes = state["concept_map"].concepts
        target = nodes[state["scenario_index"] % len(nodes)] if nodes else None
        t_name = target.name if target else "Cơ chế cốt lõi"
        t_page = target.page_ref if target and target.page_ref else "[Tài liệu]"
        chunks, _ = kb.retrieve(t_name, top_k=2)
        snippet = chunks[0].content[:200] if chunks else (target.summary if target else state["doc_text"][:200])

        sc = IncidentScenario(
            ticket_id=f"INC-{501 + state['scenario_index']}",
            severity="SEV-1",
            title=f"Lỗi vận hành tính năng liên quan đến '{t_name}' trên Production",
            symptom=(
                f"Hệ thống áp dụng nội dung '{t_name}' gặp sai lệch kết quả đầu ra và độ tin cậy suy giảm nghiêm trọng "
                f"so với mô tả chuẩn trong tài liệu ({t_page})."
            ),
            topic=t_name,
            page_ref=t_page,
            clues=[
                IncidentClue(
                    tool_name="inspect_config",
                    title="Cấu hình mô-đun đang chạy",
                    output=f"module='{t_name}' | strict_grounding=False | boundary_guard=Disabled",
                ),
                IncidentClue(
                    tool_name="inspect_logs",
                    title="Log đối chiếu tài liệu nguồn",
                    output=f"Tài liệu gốc ({t_page}) quy định: \"{snippet}...\"",
                ),
                IncidentClue(
                    tool_name="inspect_metrics",
                    title="Chỉ số sự cố ghi nhận",
                    output="Error rate: 42% | Misaligned outputs vs specification: High",
                ),
            ],
            root_cause_options=[
                f"Hệ thống vi phạm nguyên tắc của '{t_name}' ({t_page}) và tắt kiểm tra điều kiện biên",
                "Do lỗi định dạng font chữ trên màn hình giao diện người dùng",
                "Do tốc độ quạt tản nhiệt CPU tăng cao vào ban đêm",
                "Do trình duyệt không hỗ trợ file UTF-8",
            ],
            correct_option_index=0,
            ground_truth_explanation=f"Theo tài liệu ({t_page}): {snippet}",
        )
        return {"scenario": _shuffle_incident_options(sc)}

    g = StateGraph(_IncidentState)
    g.add_node("build_incident", build_incident_node)
    g.add_edge(START, "build_incident")
    g.add_edge("build_incident", END)
    app_g = g.compile()
    out = app_g.invoke(
        {"doc_text": doc_text, "concept_map": concept_map, "scenario_index": scenario_index, "scenario": None}
    )
    return out["scenario"]


def evaluate_incident_resolution(
    scenario: IncidentScenario,
    chosen_option_index: int | None,
    learner_fix: str,
    tools_used: list[str],
    learner_dna: LearnerDNA,
    doc_id: str,
    concepts_order: list[str] | None = None,
    model: str | None = None,
    demo_mode: bool = False,
    db_path: str | Path | None = None,
) -> tuple[IncidentEvaluation, LearnerDNA]:
    """Chấm kết quả xử lý sự cố Production và cập nhật Learner DNA ở mức Bloom Phân tích."""
    is_right_cause = chosen_option_index == scenario.correct_option_index
    bonus = min(15, len(set(tools_used)) * 5)

    if not demo_mode and config.has_live_api_key():
        try:
            chosen_txt = (
                scenario.root_cause_options[chosen_option_index]
                if chosen_option_index is not None and chosen_option_index < len(scenario.root_cause_options)
                else "Không chọn"
            )
            ev = complete_structured(
                [
                    {"role": "system", "content": INCIDENT_EVAL_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"TICKET: {scenario.title}\nNGUYÊN NHÂN CHUẨN: {scenario.ground_truth_explanation}\n"
                            f"HỌC VIÊN CHỌN: {chosen_txt} (Đúng={is_right_cause})\n"
                            f"ĐỀ XUẤT SỬA: {learner_fix}\nSỐ TOOL ĐÃ ĐIỀU TRA: {len(set(tools_used))}/3"
                        ),
                    },
                ],
                IncidentEvaluation,
                model=model or config.DEFAULT_MODEL,
                temperature=0.2,
                purpose="incident_eval",
                db_path=db_path,
            )
            ev.investigation_bonus = bonus
            ev.score = max(0, min(100, int(ev.score)))
            # Đồng bộ kết luận với việc người học có chọn đúng nguyên nhân gốc hay không.
            ev.resolved_correctly = is_right_cause
        except LLMError:
            ev = _offline_eval_incident(scenario, is_right_cause, learner_fix, bonus, db_path)
    else:
        ev = _offline_eval_incident(scenario, is_right_cause, learner_fix, bonus, db_path)

    db.update_concept_mastery(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concept=scenario.topic,
        is_correct=ev.resolved_correctly,
        score_override=float(ev.score),
        bloom_level="Phân tích",
        total_concepts=len(concepts_order) if concepts_order else None,
        db_path=db_path,
    )
    if ev.resolved_correctly and ev.score >= 75:
        db.resolve_misconception(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            topic=scenario.topic,
            db_path=db_path,
        )
    elif not ev.resolved_correctly:
        db.record_misconception(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            topic=scenario.topic,
            description=f"[Sự cố {scenario.ticket_id}] Chưa xác định đúng nguyên nhân gốc: {scenario.ground_truth_explanation[:120]}",
            db_path=db_path,
        )

    updated = db.get_learner_dna(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concepts_order=concepts_order,
        db_path=db_path,
    )
    return ev, updated


def _offline_eval_incident(
    scenario: IncidentScenario,
    is_right_cause: bool,
    learner_fix: str,
    bonus: int,
    db_path: str | Path | None = None,
) -> IncidentEvaluation:
    _log_demo("incident_eval", db_path=db_path, latency_ms=42)
    fix_low = re.sub(r"[-_]", " ", learner_fix.strip().lower())
    gt_norm = re.sub(r"[-_]", " ", scenario.ground_truth_explanation.lower())
    stop = {"theo", "mục", "cần", "thiết", "lập", "để", "nằm", "không", "cho", "quá", "nên", "với", "tài", "liệu"}
    gt_tokens = {w.strip(".,:;()[]") for w in gt_norm.split() if len(w.strip(".,:;()[]")) >= 3 and w.strip(".,:;()[]") not in stop}
    fix_hits = sum(1 for t in gt_tokens if t in fix_low)
    fix_quality = min(22, fix_hits * 6) if len(fix_low) >= 12 else 0

    if is_right_cause:
        score = min(100, 65 + bonus + fix_quality)
        feedback = (
            f"Chính xác! Bạn đã bắt đúng nguyên nhân gốc của sự cố {scenario.ticket_id}"
            + (f" (cộng +{bonus} điểm thưởng vì đã chạy đủ công cụ điều tra trước khi kết luận)." if bonus else ".")
        )
        post = f"Sau khi vá cấu hình cho {scenario.ticket_id}, bạn sẽ bổ sung chỉ số giám sát nào để sự cố này không tái diễn âm thầm?"
    else:
        score = max(15, 18 + bonus + fix_quality // 2)
        feedback = (
            f"Chưa trúng nguyên nhân gốc của {scenario.ticket_id}. "
            "Hãy đối chiếu lại kết quả từ 3 công cụ điều tra (Config / Log / Metrics) với tài liệu nguồn."
        )
        post = f"Nhìn lại log của {scenario.ticket_id}, dấu hiệu nào phản ánh trực tiếp nguyên lý ở {scenario.page_ref}?"

    return IncidentEvaluation(
        resolved_correctly=is_right_cause,
        score=score,
        investigation_bonus=bonus,
        feedback=feedback,
        violated_principle=scenario.ground_truth_explanation,
        page_ref=scenario.page_ref,
        post_mortem_question=post,
    )


# ==============================================================================
# 3. SĂN LỖI CÀI BẪY (TROJAN BUG-HUNT)
# ==============================================================================

_CURRICULUM_TROJANS: list[TrojanChallenge] = [
    TrojanChallenge(
        title="Audit bản tóm tắt kiến thức RAG & OpenAI API do AI sinh",
        context_summary="Có đúng 2/5 mệnh đề dưới đây bị cài lỗi sai tinh vi dù đọc rất hợp lý. Hãy khoanh vùng như một Tech Lead review tài liệu!",
        statements=[
            TrojanStatement(
                id=1,
                topic="RAG là gì",
                statement="RAG tìm kiếm các đoạn tài liệu liên quan trước rồi ghép vào prompt để giảm hiện tượng mô hình bịa thông tin và cho phép dẫn nguồn.",
                is_buggy=False,
                page_ref="[Mục 1]",
            ),
            TrojanStatement(
                id=2,
                topic="Chunking và Overlap",
                statement="Đặt kích thước chunk quá lớn sẽ giúp tăng độ chính xác truy xuất nhưng dễ gây mất ngữ cảnh giữa các câu lân cận.",
                is_buggy=True,
                bug_explanation="Bẫy đảo ngược 2 chiều! Chunk quá lớn làm truy xuất KÉM chính xác (lẫn nhiều ý); mất ngữ cảnh là hậu quả của chunk quá NHỎ.",
                page_ref="[Mục 2]",
            ),
            TrojanStatement(
                id=3,
                topic="Embeddings và Cosine",
                statement="Embeddings ánh xạ đoạn văn thành vector số sao cho các câu đồng nghĩa như 'chi phí' và 'giá thành' có độ tương đồng cosine cao.",
                is_buggy=False,
                page_ref="[Mục 3]",
            ),
            TrojanStatement(
                id=4,
                topic="Cấu trúc lời gọi & Messages",
                statement="Để xây dựng chatbot hội thoại nhiều lượt trên OpenAI Chat API, ta chỉ cần gửi message của người dùng hiện tại vì server OpenAI tự giữ ngữ cảnh.",
                is_buggy=True,
                bug_explanation="Mô hình không tự nhớ cuộc trò chuyện; ứng dụng phải tự gửi lại lịch sử messages ở mỗi lần gọi.",
                page_ref="[Mục 2 · API]",
            ),
            TrojanStatement(
                id=5,
                topic="Đánh giá chất lượng RAG",
                statement="Sau khi đổi chunk size hoặc thêm bước rerank, cần đo lại cả tỉ lệ trả lời đúng lẫn tỉ lệ trích đúng nguồn trên cùng một bộ câu hỏi mẫu.",
                is_buggy=False,
                page_ref="[Mục 6]",
            ),
        ],
    ),
]


def _is_valid_trojan(ch: TrojanChallenge) -> bool:
    """Trojan hợp lệ: đủ 5 mệnh đề, id duy nhất, đúng 2 mệnh đề bẫy (có giải thích) và nội dung không rỗng."""
    stmts = ch.statements
    if len(stmts) != 5 or len({st.id for st in stmts}) != 5:
        return False
    if sum(1 for st in stmts if st.is_buggy) != 2:
        return False
    return all(st.statement.strip() and (not st.is_buggy or st.bug_explanation.strip()) for st in stmts)


def generate_trojan_challenge(
    doc_text: str,
    concept_map: DocumentConceptMap,
    kb: DualMemoryKnowledgeBase,
    challenge_index: int = 0,
    model: str | None = None,
    demo_mode: bool = False,
) -> TrojanChallenge:
    if not demo_mode and config.has_live_api_key():
        try:
            live_ch = complete_structured(
                [
                    {"role": "system", "content": TROJAN_GEN_PROMPT},
                    {"role": "user", "content": f"TÀI LIỆU NGUỒN:\n{doc_text[:config.MAX_INPUT_CHARS]}"},
                ],
                TrojanChallenge,
                model=model or config.DEFAULT_MODEL,
                temperature=0.35,
                purpose="trojan_gen",
            )
            if _is_valid_trojan(live_ch):
                return live_ch
            # Dữ liệu LLM hỏng: dùng generator offline thay vì chấm người học theo dữ liệu lỗi.
        except LLMError:
            pass

    _log_demo("trojan_gen", latency_ms=35)
    low = doc_text.lower()
    if "chunk quá lớn" in low and "openai api" in low:
        return _CURRICULUM_TROJANS[challenge_index % len(_CURRICULUM_TROJANS)].model_copy(deep=True)

    # Tạo động 5 mệnh đề (3 đúng, 2 cài bẫy) cho tài liệu tùy chỉnh bất kỳ
    stmts: list[TrojanStatement] = []
    nodes = concept_map.concepts[:5]
    if not nodes:
        # Không có khái niệm nào: dùng bộ mẫu (luôn đủ 5 mệnh đề, đúng 2 bẫy).
        return _CURRICULUM_TROJANS[challenge_index % len(_CURRICULUM_TROJANS)].model_copy(deep=True)
    # Luôn đủ 5 mệnh đề: lặp lại khái niệm nếu thiếu, và luôn cài đúng 2 bẫy (vị trí 2 và 4).
    nodes = [nodes[k % len(nodes)] for k in range(5)]
    for i, node in enumerate(nodes, start=1):
        is_buggy = i in (2, 4)
        if is_buggy:
            stmt_txt = f"{node.name}: {node.summary} Tuy nhiên nguyên tắc này hoàn toàn không cần kiểm chứng với tài liệu gốc và nên bỏ qua bước kiểm tra biên."
            bug_exp = f"Mệnh đề đã tự ý thêm khẳng định 'không cần kiểm chứng và bỏ qua bước kiểm tra biên' trái với nội dung chuẩn tại {node.page_ref}."
        else:
            stmt_txt = f"{node.name}: {node.summary}"
            bug_exp = ""
        stmts.append(
            TrojanStatement(
                id=i,
                topic=node.name,
                statement=stmt_txt,
                is_buggy=is_buggy,
                bug_explanation=bug_exp,
                page_ref=node.page_ref or "[Tài liệu]",
            )
        )
    return TrojanChallenge(
        title=f"Audit kiến thức tài liệu: {concept_map.doc_title}",
        context_summary="Kiểm tra kỹ từng mệnh đề dưới đây và đánh dấu các câu có chi tiết bị bóp méo hoặc cài bẫy.",
        statements=stmts,
    )


def grade_trojan_submission(
    challenge: TrojanChallenge,
    flagged_ids: list[int],
    learner_dna: LearnerDNA,
    doc_id: str,
    concepts_order: list[str] | None = None,
    db_path: str | Path | None = None,
) -> tuple[dict, LearnerDNA]:
    actual_bugs = {s.id for s in challenge.statements if s.is_buggy}
    flagged = set(flagged_ids)
    true_pos = actual_bugs & flagged
    false_pos = flagged - actual_bugs
    missed = actual_bugs - flagged

    raw = (len(true_pos) / max(len(actual_bugs), 1)) * 100.0 - len(false_pos) * 20.0
    score = max(0, min(100, int(round(raw))))

    total_concepts = len(concepts_order) if concepts_order else None
    for s in challenge.statements:
        if s.is_buggy:
            caught = s.id in flagged
            db.update_concept_mastery(
                learner_id=learner_dna.learner_id,
                doc_id=doc_id,
                concept=s.topic,
                is_correct=caught,
                bloom_level="Phân tích",
                total_concepts=total_concepts,
                db_path=db_path,
            )
            if caught and len(false_pos) <= 1:
                db.resolve_misconception(
                    learner_id=learner_dna.learner_id,
                    doc_id=doc_id,
                    topic=s.topic,
                    db_path=db_path,
                )
            elif not caught:
                db.record_misconception(
                    learner_id=learner_dna.learner_id,
                    doc_id=doc_id,
                    topic=s.topic,
                    description=f"[Bỏ sót lỗi cài bẫy] {s.bug_explanation}",
                    db_path=db_path,
                )

    updated = db.get_learner_dna(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concepts_order=concepts_order,
        db_path=db_path,
    )
    return {
        "score": score,
        "caught": sorted(true_pos),
        "missed": sorted(missed),
        "false_alarms": sorted(false_pos),
        "total_bugs": len(actual_bugs),
    }, updated


# ==============================================================================
# 4. HỘI ĐỒNG PHẢN BIỆN Q&A (DEVIL'S ADVOCATE)
# ==============================================================================

_ADVOCATE_BANK_INFRA: list[DevilsAdvocateQuestion] = [
    DevilsAdvocateQuestion(
        persona="Giám khảo Kiến trúc Hệ thống",
        question=(
            "Trong INFRA.md, nhóm giới hạn MAX_INPUT_CHARS = 12.000 để giữ ngân sách $5/tháng. "
            "Vậy nếu giám khảo đưa một giáo trình PDF 50 trang (~90.000 ký tự) rồi hỏi thông tin nằm ở trang 42, "
            "hệ thống xử lý thế nào để không bị mất nội dung cuối sách?"
        ),
        conflicting_sections=["Mục 2: MAX_INPUT_CHARS = 12.000", "Mục 8: Chunking & Vector store Tuần 3–4"],
        hidden_trap="Nhầm lẫn giữa giới hạn prompt 12.000 ký tự của Quiz Maker Tuần 1 với toàn bộ pipeline RAG FAISS/Chroma.",
        sample_strong_answer=(
            "Giới hạn MAX_INPUT_CHARS=12.000 chỉ áp dụng khi đưa nguyên văn bản vào 1 lần gọi tạo Quiz trực tiếp để kiểm soát chi phí. "
            "Còn đối với hỏi đáp dài hạn, toàn bộ 50 trang PDF được tách thành chunk (~550 ký tự, có overlap) kèm metadata [Trang n] "
            "và index hết vào vector store (FAISS/Chroma); khi hỏi ở trang 42 chỉ lấy top-k (3–5) chunk liên quan đưa vào prompt."
        ),
    ),
    DevilsAdvocateQuestion(
        persona="Giám khảo Hiệu năng & Độ bền (Robustness)",
        question=(
            "Để chống bịa đáp án (BUG-01), nhóm thêm lớp _verify dùng GPT-4.1-mini khiến thời gian tạo quiz tăng gấp đôi (~18 giây). "
            "Tại sao nhóm chấp nhận đánh đổi tốc độ này, và có cơ chế dự phòng nào nếu API bị quá tải hoặc mất kết nối?"
        ),
        conflicting_sections=["Mục 5: Chi phí & Giới hạn", "Mục 6: Kiểm thử & Độ bền"],
        hidden_trap="Chỉ khen độ chính xác mà quên nêu cơ chế retry có backoff, fallback/giữ UX loading rõ ràng và bộ 20 câu eval đo trước/sau.",
        sample_strong_answer=(
            "Với sản phẩm học tập, 1 câu trả lời sai kiến thức làm người học hiểu sai bản chất tốn kém hơn nhiều so với 9 giây chờ; "
            "nhóm ưu tiên độ chính xác (0 câu gán nhầm trên 10 lần test hồi quy BUG-01), bù lại bằng giao diện trạng thái tải rõ ràng, "
            "chỉ gọi model đắt ở bước soát ngắn và bảo vệ bằng retry có backoff + chặn ngân sách phiên."
        ),
    ),
]

_ADVOCATE_BANK_CURRICULUM: list[DevilsAdvocateQuestion] = [
    DevilsAdvocateQuestion(
        persona="Tech Lead Phản biện Kỹ thuật",
        question=(
            "Tài liệu nói 'Chunk quá nhỏ làm mất ngữ cảnh' (Mục 2) nhưng lại nói 'Top-k cần nhỏ 3–5 để prompt gọn' (Mục 4). "
            "Nếu tăng chunk lên thật to để giữ ngữ cảnh và giữ k=3 thì có giải quyết triệt để được cả hai không?"
        ),
        conflicting_sections=["Mục 2: Chunking & Overlap", "Mục 4: Vector store & top-k"],
        hidden_trap="Quên vế đối lập của Mục 2: chunk quá lớn làm truy xuất kém chính xác vì chứa nhiều ý lẫn lộn.",
        sample_strong_answer=(
            "Không! Vì theo Mục 2, chunk quá lớn lại làm việc truy xuất kém chính xác do gom nhiều ý lẫn lộn trong một vector và vẫn làm phình prompt. "
            "Giải pháp đúng là giữ chunk kích thước vừa phải và thêm phần gối đầu (overlap) để nối mạch ý ở ranh giới kết hợp top-k=3..5."
        ),
    ),
    DevilsAdvocateQuestion(
        persona="Giám khảo AI Engineering",
        question=(
            "Nếu ta đã yêu cầu LLM trả về JSON bằng Structured Outputs (Mục 4 · API), tại sao tài liệu vẫn cảnh báo vấn đề bịa thông tin và cần RAG + đánh giá trích nguồn?"
        ),
        conflicting_sections=["Mục 4 · API: Structured outputs", "Mục 5–6 RAG: Trích nguồn & Đánh giá"],
        hidden_trap="Đánh đồng 'đúng định dạng cấu trúc JSON' (Structured Output) với 'đúng sự thật nội dung' (Groundedness).",
        sample_strong_answer=(
            "Structured outputs với Pydantic chỉ đảm bảo đúng khuôn dạng các trường JSON (không bị vỡ schema khi parse), chứ không ngăn được LLM bịa nội dung bên trong các trường đó. "
            "Do đó vẫn phải kết hợp RAG truy xuất đoạn nguồn, ép trích nguyên văn bằng chứng và đo cả tỉ lệ trích đúng nguồn."
        ),
    ),
]


def generate_devils_advocate(
    doc_text: str,
    concept_map: DocumentConceptMap,
    kb: DualMemoryKnowledgeBase,
    round_index: int = 0,
    model: str | None = None,
    demo_mode: bool = False,
) -> DevilsAdvocateQuestion:
    if not demo_mode and config.has_live_api_key():
        try:
            return complete_structured(
                [
                    {"role": "system", "content": DEVILS_ADVOCATE_GEN_PROMPT},
                    {"role": "user", "content": f"TÀI LIỆU:\n{doc_text[:config.MAX_INPUT_CHARS]}"},
                ],
                DevilsAdvocateQuestion,
                model=model or config.DEFAULT_MODEL,
                temperature=0.4,
                purpose="devils_advocate_gen",
            )
        except LLMError:
            pass

    _log_demo("devils_advocate_gen", latency_ms=38)
    low = doc_text.lower()
    if "ai portal" in low and "max_input_chars" in low and "pinecone" in low:
        return _ADVOCATE_BANK_INFRA[round_index % len(_ADVOCATE_BANK_INFRA)].model_copy(deep=True)
    if "chunk quá lớn" in low and "overlap" in low:
        return _ADVOCATE_BANK_CURRICULUM[round_index % len(_ADVOCATE_BANK_CURRICULUM)].model_copy(deep=True)

    nodes = concept_map.concepts
    c1 = nodes[0] if nodes else None
    c2 = nodes[min(len(nodes) - 1, 1 + (round_index % max(len(nodes) - 1, 1)))] if len(nodes) > 1 else c1
    return DevilsAdvocateQuestion(
        persona="Giám khảo Phản biện Hệ thống",
        question=(
            f"Trong tài liệu '{concept_map.doc_title}', phần '{c1.name if c1 else 'Cơ sở'}' ({c1.page_ref if c1 else ''}) "
            f"và phần '{c2.name if c2 else 'Mở rộng'}' ({c2.page_ref if c2 else ''}) có sự đánh đổi nào khi triển khai thực tế, "
            "và điều gì sẽ xảy ra nếu gặp dữ liệu biên?"
        ),
        conflicting_sections=[
            f"{c1.name} ({c1.page_ref})" if c1 else "Phần 1",
            f"{c2.name} ({c2.page_ref})" if c2 else "Phần 2",
        ],
        hidden_trap="Trả lời chung chung mà không nhắc được sự đánh đổi và điều kiện từ cả hai phần.",
        sample_strong_answer=(
            f"Cần cân bằng giữa yêu cầu nền tảng của {c1.name if c1 else 'phần 1'} ({c1.summary if c1 else ''}) "
            f"với ràng buộc triển khai ở {c2.name if c2 else 'phần 2'} ({c2.summary if c2 else ''}), đồng thời kiểm chứng qua bộ test biên."
        ),
    )


def _target_topic_for_defense(question: DevilsAdvocateQuestion, learner_dna: LearnerDNA) -> str:
    """Tìm khái niệm trong hồ sơ Learner DNA khớp nhất với câu hỏi phản biện."""
    haystack = " ".join(question.conflicting_sections + [question.question]).lower()
    best_name = ""
    best_hits = 0
    for c in learner_dna.concepts:
        cname = c.concept
        if cname.lower() in haystack:
            return cname
        words = [w.strip(".,:;()[]{}") for w in cname.lower().split() if len(w.strip(".,:;()[]{}")) >= 4]
        hits = sum(1 for w in words if w in haystack)
        if hits > best_hits:
            best_hits = hits
            best_name = cname
    if best_name and best_hits > 0:
        return best_name
    if learner_dna.concepts:
        return learner_dna.concepts[0].concept
    return "Kiến thức tổng hợp"


def evaluate_defense(
    question: DevilsAdvocateQuestion,
    learner_defense: str,
    kb: DualMemoryKnowledgeBase,
    learner_dna: LearnerDNA,
    doc_id: str,
    model: str | None = None,
    demo_mode: bool = False,
    db_path: str | Path | None = None,
) -> tuple[DefenseEvaluation, LearnerDNA]:
    chunks, _ = kb.retrieve(question.question, top_k=3)
    ev: DefenseEvaluation | None = None
    if not demo_mode and config.has_live_api_key():
        ctx = "\n\n".join(f"{c.page}: {c.content}" for c in chunks)
        try:
            ev = complete_structured(
                [
                    {"role": "system", "content": DEVILS_ADVOCATE_EVAL_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"CÂU HỎI PHẢN BIỆN: {question.question}\n"
                            f"BẪY ẨN: {question.hidden_trap}\n"
                            f"CÂU TRẢ LỜI BẢO VỆ CỦA HỌC VIÊN: {learner_defense}\n\n"
                            f"TÀI LIỆU NGUỒN:\n{ctx}"
                        ),
                    },
                ],
                DefenseEvaluation,
                model=model or config.DEFAULT_MODEL,
                temperature=0.2,
                purpose="devils_advocate_eval",
                db_path=db_path,
            )
            ev.defense_score = max(0, min(100, int(ev.defense_score)))
        except LLMError:
            ev = None

    if ev is None:
        _log_demo("devils_advocate_eval", db_path=db_path, latency_ms=46)
        ans_low = learner_defense.strip().lower()
        ref_tokens = {
            w.strip(".,:;()").lower()
            for w in question.sample_strong_answer.split()
            if len(w.strip(".,:;()")) >= 3
        }
        stop = {"cần", "và", "của", "cho", "với", "này", "khi", "để", "các", "những", "trong", "không"}
        key_ref = {t for t in ref_tokens if t not in stop}
        hits = sum(1 for t in key_ref if t in ans_low)
        ratio = hits / max(len(key_ref), 1)

        contradict = False
        if "chunk" in ans_low and "chunk quá lớn" in ans_low and "chính xác hơn" in ans_low:
            contradict = True
        if "structured" in ans_low and "không cần rag" in ans_low:
            contradict = True

        if len(learner_defense.split()) < 8:
            score = 22
            verdict = "Cần chặt chẽ hơn"
        elif contradict:
            score = 32
            verdict = "Mâu thuẫn tài liệu"
        else:
            score = max(35, min(95, int(round(35 + ratio * 75))))
            verdict = "Thuyết phục" if score >= 75 else "Cần chặt chẽ hơn"

        strengths = []
        gaps = []
        if hits >= 4:
            strengths.append("Lập luận của bạn đã chạm tới các khái niệm cốt lõi trong tài liệu.")
        if len(learner_defense.split()) >= 18 and not contradict:
            strengths.append("Cách trình bày có cấu trúc nguyên nhân–giải pháp rõ ràng.")
        if contradict:
            gaps.append("Câu trả lời có ý mâu thuẫn trực tiếp với khẳng định trong tài liệu gốc.")
        if ratio < 0.35:
            gaps.append(f"Còn thiếu vế xử lý bẫy ẩn: {question.hidden_trap}")

        ev = DefenseEvaluation(
            defense_score=score,
            verdict=verdict,
            strengths=strengths or ["Đã đưa ra quan điểm phản hồi câu hỏi phản biện."],
            gaps_or_contradictions=gaps,
            strong_answer_outline=question.sample_strong_answer,
            next_hard_follow_up="Nếu phải chứng minh hiệu quả của phương án bạn vừa bảo vệ bằng số liệu cụ thể trước hội đồng, bạn sẽ đo chỉ số nào?",
        )

    target_topic = _target_topic_for_defense(question, learner_dna)
    db.update_concept_mastery(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concept=target_topic,
        is_correct=ev.defense_score >= 60,
        score_override=float(ev.defense_score),
        bloom_level="Đánh giá",
        total_concepts=len(learner_dna.concepts) or None,
        db_path=db_path,
    )
    if ev.defense_score >= 75:
        db.resolve_misconception(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            topic=target_topic,
            db_path=db_path,
        )
    updated = db.get_learner_dna(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concepts_order=[c.concept for c in learner_dna.concepts] or None,
        db_path=db_path,
    )
    return ev, updated
