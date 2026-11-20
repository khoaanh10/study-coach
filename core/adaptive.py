"""Điều phối năng lực cá nhân: Bản đồ khái niệm, Chẩn đoán đầu vào, Vấn đáp Feynman và Lộ trình ôn tập."""
from __future__ import annotations

from pathlib import Path

import config
from core import db
from core.contracts import (
    DocumentConceptMap,
    LearnerDNA,
    Quiz,
    QuizResult,
    TeachBackEvaluation,
)
from core.demo_provider import (
    demo_evaluate_teach_back,
    demo_extract_concept_map,
    demo_generate_quiz,
)
from core.llm import LLMError, complete_structured
from core.prompts import CONCEPT_MAP_PROMPT, SYSTEM_PROMPT, TEACH_BACK_PROMPT
from core.quiz import _is_valid
from core.rag import _STOP_WORDS, DualMemoryKnowledgeBase

_BLOOM_RANK = {"Nhớ": 1, "Hiểu": 2, "Vận dụng": 3, "Phân tích": 4}


def extract_concept_map(
    text: str,
    doc_id: str,
    doc_title: str,
    model: str | None = None,
    demo_mode: bool = False,
    db_path: str | Path | None = None,
) -> DocumentConceptMap:
    """Trích xuất bản đồ khái niệm (Concept Map) kèm số trang và quan hệ tiên quyết."""
    if demo_mode or not config.has_live_api_key():
        return demo_extract_concept_map(text=text, doc_id=doc_id, doc_title=doc_title, db_path=db_path)

    messages = [
        {"role": "system", "content": CONCEPT_MAP_PROMPT},
        {
            "role": "user",
            "content": f"Mã tài liệu: {doc_id}\nTiêu đề: {doc_title}\n\nVĂN BẢN:\n{text[:config.MAX_INPUT_CHARS]}",
        },
    ]
    try:
        cmap = complete_structured(
            messages,
            DocumentConceptMap,
            model=model or config.DEFAULT_MODEL,
            temperature=0.2,
            purpose="concept_map",
            db_path=db_path,
        )
        cmap.doc_id = doc_id
        cmap.doc_title = doc_title
        if cmap.concepts:
            return cmap
    except LLMError:
        pass
    return demo_extract_concept_map(text=text, doc_id=doc_id, doc_title=doc_title, db_path=db_path)


def generate_placement_test(
    text: str,
    concept_map: DocumentConceptMap,
    model: str | None = None,
    demo_mode: bool = False,
    db_path: str | Path | None = None,
) -> Quiz:
    """Sinh bài chẩn đoán năng lực đầu vào 3 câu leo thang Bloom: Nhớ -> Hiểu -> Vận dụng."""
    concept_names = [c.name for c in concept_map.concepts[:5]]
    n_target = min(max(len(concept_names), 5), 10)
    bloom_cycle = ("Nhớ", "Hiểu", "Vận dụng", "Phân tích")
    if demo_mode or not config.has_live_api_key():
        # Lấy đủ câu phủ tối đa các chủ đề của tài liệu và trải đều 4 cấp độ Bloom.
        pool = demo_generate_quiz(text=text, n=max(n_target, 10), difficulty="Trung bình").questions
        picked = []
        used_q: set[str] = set()
        used_topics: set[str] = set()
        for idx in range(n_target):
            target_bloom = bloom_cycle[idx % len(bloom_cycle)]
            cand = next(
                (
                    q
                    for q in pool
                    if q.bloom_level == target_bloom and q.question not in used_q and q.topic not in used_topics
                ),
                None,
            )
            if cand is None:
                cand = next((q for q in pool if q.question not in used_q and q.topic not in used_topics), None)
            if cand is None:
                cand = next((q for q in pool if q.bloom_level == target_bloom and q.question not in used_q), None)
            if cand is None:
                cand = next((q for q in pool if q.question not in used_q), None)
            if cand is not None:
                used_q.add(cand.question)
                if cand.topic:
                    used_topics.add(cand.topic)
                picked.append(cand)
        return Quiz(questions=picked[:n_target])

    topic_lines = "\n".join(
        f"- Câu {i + 1}: chủ đề '{concept_names[i % len(concept_names)]}' — mức Bloom '{bloom_cycle[i % len(bloom_cycle)]}'"
        for i in range(n_target)
    ) if concept_names else "\n".join(
        f"- Câu {i + 1}: mức Bloom '{bloom_cycle[i % len(bloom_cycle)]}'"
        for i in range(n_target)
    )
    user_prompt = (
        f"Hãy tạo đúng {n_target} câu hỏi CHẨN ĐOÁN TRÌNH ĐỘ ĐẦU VÀO phủ đều toàn bộ các chủ đề chính của tài liệu:\n"
        f"{topic_lines}\n"
        f"Mỗi câu phải bám sát văn bản và ghi đúng tên chủ đề tương ứng.\n\nVĂN BẢN:\n{text[:config.MAX_INPUT_CHARS]}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    quiz = complete_structured(
        messages, Quiz, model=model or config.DEFAULT_MODEL, temperature=0.2, purpose="placement_test", db_path=db_path
    )
    # Chỉ giữ câu hợp lệ (đủ 4 đáp án, chỉ số đúng trong 0..3), không chấm người học theo câu hỏng.
    return Quiz(questions=[q for q in quiz.questions if _is_valid(q)][:n_target])


def _match_concept_name(raw_topic: str, concepts_order: list[str] | None) -> str:
    if not concepts_order:
        return raw_topic or "Kiến thức chung"
    low = (raw_topic or "").strip().lower()
    if not low:
        return concepts_order[0]
    for cname in concepts_order:
        clow = cname.lower()
        if low == clow or low in clow or clow in low:
            return cname
    # Khớp theo từ khóa giao nhau.
    # Bỏ stopword ("và", "của"...) để không gán nhầm khái niệm chỉ vì trùng từ nối.
    raw_words = {w for w in low.split() if w not in _STOP_WORDS}
    best_name = raw_topic
    best_overlap = 0
    for cname in concepts_order:
        overlap = len(raw_words & {w for w in cname.lower().split() if w not in _STOP_WORDS})
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = cname
    return best_name


def record_quiz_outcome(
    learner_id: str,
    doc_id: str,
    result: QuizResult,
    model: str,
    difficulty: str,
    concepts_order: list[str] | None = None,
    db_path: str | Path | None = None,
) -> LearnerDNA:
    """Lưu lịch sử làm bài và cập nhật điểm thành thạo từng khái niệm + lỗi tư duy vào SQLite."""
    db.save_quiz_attempt(
        learner_id=learner_id,
        doc_id=doc_id,
        model=model,
        difficulty=difficulty,
        result=result,
        db_path=db_path,
    )

    is_placement = difficulty == "Chẩn đoán"
    placement_ratio = result.score / max(result.total, 1)
    total_concepts = len(concepts_order) if concepts_order else None

    topic_results: dict[str, list[bool]] = {}
    solid_cnt = lucky_cnt = aware_cnt = blind_cnt = 0
    for item in result.items:
        mapped_topic = _match_concept_name(item.topic, concepts_order)
        # Chỉ tính để "khắc phục lỗi cũ" khi đúng ở mức Bloom >= Hiểu và không phải câu "Đoán mò".
        eligible = (
            item.is_correct
            and _BLOOM_RANK.get(item.bloom_level or "Hiểu", 2) >= 2
            and item.confidence != "Đoán mò"
        )
        topic_results.setdefault(mapped_topic, []).append(eligible)

        score_override: float | None = None
        if is_placement:
            if item.is_correct:
                if item.confidence == "Đoán mò":
                    score_override = 55.0
                elif placement_ratio >= 0.8:
                    score_override = (
                        78.0 if item.bloom_level == "Nhớ" else (82.0 if item.bloom_level == "Hiểu" else 86.0)
                    )
                elif placement_ratio >= 0.5:
                    score_override = 64.0 if item.bloom_level == "Nhớ" else 68.0
                else:
                    score_override = 50.0
            else:
                score_override = 30.0 if placement_ratio >= 0.5 else 24.0

        db.update_concept_mastery(
            learner_id=learner_id,
            doc_id=doc_id,
            concept=mapped_topic,
            is_correct=item.is_correct,
            score_override=score_override,
            bloom_level=item.bloom_level or "Hiểu",
            confidence=item.confidence,
            total_concepts=total_concepts,
            db_path=db_path,
        )
        if item.is_correct:
            if item.confidence == "Đoán mò":
                lucky_cnt += 1
            else:
                solid_cnt += 1
        else:
            if item.chosen_index is not None and item.confidence == "Chắc chắn":
                blind_cnt += 1
            else:
                aware_cnt += 1

        if not item.is_correct and item.diagnosed_misconception and not item.diagnosed_misconception.startswith("Bỏ trống"):
            desc = item.diagnosed_misconception
            if item.confidence == "Chắc chắn":
                desc = f"[Điểm mù tự tin cao] {desc}"
            db.record_misconception(
                learner_id=learner_id,
                doc_id=doc_id,
                topic=mapped_topic,
                description=desc,
                db_path=db_path,
            )

    db.update_confidence_stats(
        learner_id=learner_id,
        doc_id=doc_id,
        solid=solid_cnt,
        lucky=lucky_cnt,
        aware=aware_cnt,
        blind=blind_cnt,
        db_path=db_path,
    )

    # Nếu làm đúng toàn bộ các câu của một chủ đề ở mức Hiểu trở lên (và không đoán mò), đánh dấu đã khắc phục lỗi cũ.
    for topic, outcomes in topic_results.items():
        if outcomes and all(outcomes):
            db.resolve_misconception(learner_id, doc_id, topic, db_path=db_path)

    return db.get_learner_dna(
        learner_id=learner_id,
        doc_id=doc_id,
        concepts_order=concepts_order,
        db_path=db_path,
    )


def evaluate_teach_back(
    concept: str,
    learner_answer: str,
    kb: DualMemoryKnowledgeBase,
    learner_dna: LearnerDNA,
    doc_id: str,
    concepts_order: list[str] | None = None,
    concept_map: DocumentConceptMap | None = None,
    question_context: str | None = None,
    model: str | None = None,
    demo_mode: bool = False,
    db_path: str | Path | None = None,
) -> tuple[TeachBackEvaluation, LearnerDNA]:
    """Chấm bài học viên tự giảng lại khái niệm theo phương pháp Feynman hoặc trả lời câu hỏi Socratic."""
    node = (
        next((c for c in concept_map.concepts if c.name.lower() == concept.lower()), None)
        if concept_map
        else None
    )
    query_parts = [concept]
    if node and node.summary:
        query_parts.append(node.summary)
    if question_context:
        query_parts.append(question_context)
    if learner_answer.strip():
        query_parts.append(learner_answer.strip())
    chunks, _ = kb.retrieve(". ".join(query_parts), top_k=4)
    if node and node.page_ref:
        own = [c for c in chunks if c.page == node.page_ref]
        other = [c for c in chunks if c.page != node.page_ref]
        chunks = (own + other)[:3]
    else:
        chunks = chunks[:3]

    if demo_mode or not config.has_live_api_key():
        ev = demo_evaluate_teach_back(
            concept=concept,
            learner_answer=learner_answer,
            retrieved_chunks=chunks,
            learner_dna=learner_dna,
            concept_map=concept_map,
            question_context=question_context,
            db_path=db_path,
        )
    else:
        context = "\n\n".join(f"{c.page}: {c.content}" for c in chunks)
        user_msg = (
            f"KHÁI NIỆM ĐANG KIỂM TRA: {concept}\n"
            + (f"CÂU HỎI ĐÃ ĐẶT CHO HỌC VIÊN: {question_context}\n" if question_context else "")
            + f"TRÌNH ĐỘ HIỆN TẠI CỦA HỌC VIÊN: {learner_dna.overall_level}\n\n"
            + f"CÂU GIẢI THÍCH CỦA HỌC VIÊN:\n{learner_answer.strip()}\n\n"
            + f"ĐOẠN TÀI LIỆU NGUỒN ĐỐI CHIẾU:\n{context}"
        )
        ev = complete_structured(
            [
                {"role": "system", "content": TEACH_BACK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            TeachBackEvaluation,
            model=model or config.DEFAULT_MODEL,
            temperature=0.2,
            purpose="feynman_teach_back",
            db_path=db_path,
        )

    mapped_concept = _match_concept_name(concept, concepts_order)
    if ev.understanding_score >= 95 and not ev.misconceptions_found:
        bloom = "Phân tích"
    elif ev.understanding_score >= 80:
        bloom = "Vận dụng"
    else:
        bloom = "Hiểu"
    total_concepts = len(concepts_order) if concepts_order else (len(concept_map.concepts) if concept_map else None)
    db.update_concept_mastery(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concept=mapped_concept,
        score_override=float(ev.understanding_score),
        bloom_level=bloom,
        total_concepts=total_concepts,
        db_path=db_path,
    )
    for m in ev.misconceptions_found:
        db.record_misconception(learner_dna.learner_id, doc_id, mapped_concept, m, db_path=db_path)

    if ev.understanding_score >= 80 and not ev.misconceptions_found:
        db.resolve_misconception(learner_dna.learner_id, doc_id, mapped_concept, db_path=db_path)

    updated_dna = db.get_learner_dna(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concepts_order=concepts_order,
        db_path=db_path,
    )
    return ev, updated_dna


def build_study_plan(learner_dna: LearnerDNA, concept_map: DocumentConceptMap) -> list[dict]:
    """Lập lộ trình ôn tập ưu tiên theo lỗ hổng nền tảng, lỗi tư duy và Spaced Repetition."""
    mastery_by_name = {c.concept: c for c in learner_dna.concepts}
    attempted_or_active = {
        c.concept
        for c in learner_dna.concepts
        if c.attempts > 0 or c.status == "misconception"
    }

    # Chỉ coi một khái niệm là "lỗ hổng nền tảng đang chặn" nếu nó thực sự là tiên quyết
    # của ít nhất một khái niệm khác đang được học/yếu, hoặc chính nó là gốc đã học nhưng yếu (< 55%).
    blocking_prereqs: set[str] = set()
    for node in concept_map.concepts:
        node_m = mastery_by_name.get(node.name)
        node_needs_prereq = (node.name in attempted_or_active) or (node_m is not None and node_m.mastery_score < 60.0)
        if node_needs_prereq:
            for pre in node.prerequisites:
                pre_m = mastery_by_name.get(pre)
                if pre_m and pre_m.mastery_score < 55.0 and pre_m.attempts > 0:
                    blocking_prereqs.add(pre)

    plan_items: list[dict] = []
    for topo_idx, node in enumerate(concept_map.concepts):
        m = mastery_by_name.get(node.name)
        score = m.mastery_score if m else 0.0
        status = m.status if m else "unseen"
        due = m.due_for_review if m else False

        if node.name in blocking_prereqs:
            rank = 1
            priority = "Vá lỗ hổng nền tảng"
            reason = "Là kiến thức gốc cho các mục phía sau nhưng hiện còn yếu."
            action = "Học lại với Gia sư Socratic"
        elif status == "misconception":
            rank = 2
            priority = "Gỡ lỗi tư duy"
            misc = next((x.description for x in learner_dna.active_misconceptions if x.topic == node.name), "Cần sửa lỗi hiểu nhầm.")
            reason = misc
            action = "Làm quiz gỡ lỗi hoặc vấn đáp Feynman"
        elif due:
            rank = 3
            priority = "Đến hạn ôn tập"
            reason = "Đã tới hạn ôn lại theo đường cong quên lãng để giữ trí nhớ dài hạn."
            action = "Ôn nhanh bằng Vấn đáp Feynman"
        elif status == "unseen" or score < 75.0:
            rank = 4
            priority = "Mục tiêu tiếp theo"
            reason = "Chưa học sâu hoặc điểm thành thạo dưới ngưỡng vững (75%)."
            action = "Hỏi Gia sư hoặc làm Quiz 3 câu"
        else:
            rank = 5
            priority = "Đã làm chủ"
            reason = f"Đạt {score:.0f}% ở mức Bloom {m.bloom_level if m else 'Hiểu'}."
            action = "Thử thách câu Khó / Phân tích"

        plan_items.append(
            {
                "rank": rank,
                "topo_idx": topo_idx,
                "concept": node.name,
                "page_ref": node.page_ref,
                "mastery_score": score,
                "status": status,
                "priority_label": priority,
                "reason": reason,
                "recommended_action": action,
            }
        )

    # Trong cùng mức ưu tiên, mục đang hiểu lệch & xuất hiện sớm hơn trong giáo trình đứng trước.
    plan_items.sort(key=lambda x: (x["rank"], 0 if x["status"] == "misconception" else 1, x["topo_idx"]))
    return plan_items
