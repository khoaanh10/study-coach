"""Sinh và chấm quiz trắc nghiệm thích ứng từ văn bản tài liệu và Learner DNA."""
from __future__ import annotations

import re

from pydantic import BaseModel

import config
from core.contracts import (
    GradedQuestion,
    LearnerDNA,
    Question,
    Quiz,
    QuizResult,
)
from core.demo_provider import demo_generate_quiz
from core.llm import LLMError, complete_structured
from core.prompts import (
    DIFFICULTY_HINTS,
    GATE_PROMPT,
    SYSTEM_PROMPT,
    VERIFY_PROMPT,
)

NO_CONTENT_MESSAGE = (
    "Văn bản này không có nội dung kiến thức rõ ràng để ra câu hỏi. "
    "Hãy dán một đoạn tài liệu, bài học hoặc ghi chú có ý nghĩa."
)

_VOWELS = set("aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ")


def _looks_meaningless(text: str) -> bool:
    """Chặn sớm văn bản gõ bừa mà không tốn lần gọi API nào.

    Chỉ bắt các trường hợp rõ ràng; trường hợp khó hơn do model và bước soát đáp án xử lý.
    """
    letters = [c for c in text if c.isalpha()]
    latin = sum(ord(c) < 0x250 or 0x1EA0 <= ord(c) <= 0x1EFF for c in letters)
    if letters and latin < len(letters) * 0.5:
        return False  # chữ không phải Latin (Nhật, Trung, Hàn...): các phép đo dưới đây không áp dụng
    words = [w for w in text.lower().split() if any(c.isalpha() for c in w)]
    if len(words) < 15 or len(set(words)) < 10:
        return True  # quá ít từ, hoặc vài từ lặp đi lặp lại
    if max(len(w) for w in words) > 40 and sum(len(w) > 25 for w in words) > len(words) * 0.2:
        return True  # nhiều "từ" dài bất thường
    with_vowel = sum(any(c in _VOWELS for c in w) for w in words)
    return with_vowel < len(words) * 0.7


_CURLY_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
}
_DASHES = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
}


def _norm(s: str) -> str:
    """Chuẩn hóa văn bản để so khớp: quotes thẳng, bỏ ngắt dòng PDF, gom khoảng trắng."""
    text = s.translate({**_CURLY_QUOTES, **_DASHES})
    text = text.replace("-\r\n", "").replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip(" .\"'")


def _is_grounded(q: Question, source: str) -> bool:
    """Bằng chứng của câu hỏi có thật sự nằm trong văn bản nguồn (đã chuẩn hóa) không."""
    ev = _norm(q.evidence)
    return len(ev) >= 15 and ev in source


def _is_valid(q: Question) -> bool:
    return len(q.options) == 4 and 0 <= q.correct_index < 4 and bool(q.question.strip())


def generate_quiz(
    text: str,
    n: int = 5,
    difficulty: str = "Trung bình",
    model: str | None = None,
    max_attempts: int = 2,
    demo_mode: bool = False,
    focus_topic: str | None = None,
    learner_dna: LearnerDNA | None = None,
) -> Quiz:
    text = text.strip()
    if len(text) < 100:
        raise LLMError("Văn bản quá ngắn để ra quiz (cần ít nhất khoảng 100 ký tự).")
    text = text[: config.MAX_INPUT_CHARS]
    if _looks_meaningless(text):
        raise LLMError(NO_CONTENT_MESSAGE)

    if demo_mode:
        quiz = demo_generate_quiz(
            text=text,
            n=n,
            difficulty=difficulty,
            focus_topic=focus_topic,
            learner_dna=learner_dna,
        )
        if not quiz.questions:
            raise LLMError(NO_CONTENT_MESSAGE)
        return quiz

    if not _has_knowledge(text, model):
        raise LLMError(NO_CONTENT_MESSAGE)

    adaptive_note = ""
    if focus_topic:
        adaptive_note = (
            f"\nƯU TIÊN CÁ NHÂN HÓA: Tập trung ít nhất một nửa số câu hỏi vào chủ đề '{focus_topic}'."
        )
    if learner_dna and learner_dna.active_misconceptions:
        misc_txt = "; ".join(m.description for m in learner_dna.active_misconceptions[:2])
        adaptive_note += (
            f"\nĐIỂM YẾU NGƯỜI HỌC: Người học hay mắc lỗi tư duy ({misc_txt}); "
            "hãy tạo đáp án nhiễu kiểm tra đúng ranh giới dễ nhầm này."
        )

    user_prompt = (
        # Xin dư 2 câu để còn câu thay thế khi có câu bị loại ở bước kiểm tra.
        f"Hãy tạo {n + 2} câu trắc nghiệm, mức độ: {difficulty}. "
        f"{DIFFICULTY_HINTS.get(difficulty, '')}{adaptive_note}\n\nVĂN BẢN:\n{text}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    source = _norm(text)
    good: list[Question] = []  # qua cả hai bước kiểm tra, gom qua các lần gọi
    for _ in range(max_attempts):
        quiz = complete_structured(messages, Quiz, model=model, temperature=0.2)
        if not quiz.questions:
            # Model cho rằng văn bản không có gì để hỏi; không gọi lại để ép ra câu hỏi.
            break
        valid = [q for q in quiz.questions if _is_valid(q)]
        if not valid:
            continue
        agreed = _verify(valid, text, model)
        seen = {_norm(q.question) for q in good}
        for q, ok in zip(valid, agreed):
            # Câu "tốt": bằng chứng trích đúng từ văn bản và người chấm độc lập ra cùng đáp án.
            if _is_grounded(q, source) and ok:
                if _norm(q.question) not in seen:
                    seen.add(_norm(q.question))
                    good.append(q)
        if len(good) >= n:
            break
    # Thà ra ít câu hơn yêu cầu, hoặc không ra quiz, còn hơn đưa câu có đáp án đáng ngờ:
    # câu bị loại không bao giờ được dùng để lấp chỗ trống.
    if not good:
        raise LLMError(NO_CONTENT_MESSAGE)
    return Quiz(questions=good[:n])


def generate_adaptive_quiz(
    text: str,
    learner_dna: LearnerDNA,
    n: int = 5,
    focus_topic: str | None = None,
    difficulty_override: str | None = None,
    model: str | None = None,
    demo_mode: bool = False,
) -> tuple[Quiz, str]:
    """Tự động chọn chủ đề yếu & độ khó phù hợp với hồ sơ Learner DNA rồi sinh quiz."""
    chosen_focus = focus_topic
    reason_parts: list[str] = []

    if not chosen_focus:
        if learner_dna.active_misconceptions:
            chosen_focus = learner_dna.active_misconceptions[0].topic
            reason_parts.append(f"xoáy vào chủ đề đang hiểu lệch '{chosen_focus}'")
        else:
            # Chọn khái niệm chưa vững có điểm thấp nhất (hoặc chưa học).
            ranked = sorted(
                learner_dna.concepts,
                key=lambda c: (0 if c.status == "misconception" else (1 if c.attempts > 0 else 2), c.mastery_score),
            )
            if ranked:
                chosen_focus = ranked[0].concept
                reason_parts.append(f"tập trung củng cố mục '{chosen_focus}' ({ranked[0].mastery_score:.0f}%)")

    if difficulty_override and difficulty_override in config.DIFFICULTIES:
        chosen_diff = difficulty_override
        reason_parts.append(f"độ khó chọn tay '{chosen_diff}'")
    else:
        focus_score = next(
            (c.mastery_score for c in learner_dna.concepts if c.concept == chosen_focus),
            learner_dna.mastery_avg,
        )
        if learner_dna.overall_level == "Nâng cao" and focus_score >= 65:
            chosen_diff = "Khó"
        elif learner_dna.overall_level == "Mới bắt đầu" or focus_score < 35:
            chosen_diff = "Dễ"
        else:
            chosen_diff = "Trung bình"
        reason_parts.append(f"tự điều chỉnh mức '{chosen_diff}' theo trình độ {learner_dna.overall_level}")

    quiz = generate_quiz(
        text=text,
        n=n,
        difficulty=chosen_diff,
        model=model,
        demo_mode=demo_mode,
        focus_topic=chosen_focus,
        learner_dna=learner_dna,
    )
    strategy = "Hệ thống đã " + " và ".join(reason_parts) + "."
    return quiz, strategy


class _Gate(BaseModel):
    reason: str  # lập luận ngắn, viết trước khi kết luận
    has_knowledge: bool


def _has_knowledge(text: str, model: str | None) -> bool:
    """Hỏi model xem văn bản có phải tài liệu để học không. Lỗi thì cho qua, không chặn người dùng."""
    messages = [
        {"role": "system", "content": GATE_PROMPT},
        {"role": "user", "content": f"VĂN BẢN:\n{text[:2000]}"},
    ]
    try:
        return complete_structured(messages, _Gate, model=model, temperature=0).has_knowledge
    except LLMError:
        return True


class _Check(BaseModel):
    reason: str  # lập luận ngắn, viết trước khi kết luận
    supported: bool


class _Checks(BaseModel):
    checks: list[_Check]


def _verify(questions: list[Question], text: str, model: str | None) -> list[bool]:
    """Soát lại từng câu: đáp án được đánh dấu đúng có thật sự được văn bản khẳng định không.

    Dùng model mạnh hơn để soát. Nếu bước soát lỗi thì coi như mọi câu đều qua,
    để không chặn việc ra quiz.
    """
    listing = "\n\n".join(
        f"Câu {i + 1}: {q.question}\nĐáp án được chọn là đúng: {q.options[q.correct_index]}"
        for i, q in enumerate(questions)
    )
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": f"VĂN BẢN:\n{text}\n\nCÁC CÂU CẦN SOÁT:\n{listing}"},
    ]
    try:
        got = complete_structured(messages, _Checks, model=config.STRONG_MODEL, temperature=0).checks
    except LLMError:
        return [True] * len(questions)
    if len(got) != len(questions):
        return [True] * len(questions)
    return [c.supported for c in got]


def grade(
    quiz: Quiz,
    answers: list[int | None],
    confidences: list[str] | None = None,
) -> QuizResult:
    """answers[i] là chỉ số đáp án đã chọn cho câu i, hoặc None nếu bỏ trống."""
    items: list[GradedQuestion] = []
    weak_topics: list[str] = []
    detected_misc: list[str] = []
    blind_spots = 0
    lucky_guesses = 0

    for i, q in enumerate(quiz.questions):
        chosen = answers[i] if i < len(answers) else None
        conf = (confidences[i] if confidences and i < len(confidences) else "Bình thường") or "Bình thường"
        is_ok = chosen == q.correct_index
        diagnosed = ""

        if is_ok:
            if conf == "Đoán mò":
                cal_tag = "Đúng do đoán mò (Kiến thức giả)"
                lucky_guesses += 1
            else:
                cal_tag = "Vững thực chất"
        else:
            if q.topic and q.topic not in weak_topics:
                weak_topics.append(q.topic)
            if chosen is None:
                diagnosed = "Bỏ trống — cần kiểm tra lại khái niệm này"
                cal_tag = "Biết mình chưa biết"
            else:
                if conf == "Chắc chắn":
                    cal_tag = "Điểm mù tự tin sai"
                    blind_spots += 1
                else:
                    cal_tag = "Biết mình chưa biết"
                if q.option_diagnostics and 0 <= chosen < len(q.option_diagnostics):
                    cand = q.option_diagnostics[chosen].strip()
                    if cand and cand.lower() != "chính xác":
                        diagnosed = cand
                        if cand not in detected_misc:
                            detected_misc.append(cand)

        items.append(
            GradedQuestion(
                question=q.question,
                options=q.options,
                chosen_index=chosen,
                correct_index=q.correct_index,
                is_correct=is_ok,
                explanation=q.explanation,
                topic=q.topic,
                bloom_level=q.bloom_level,
                evidence=q.evidence,
                page_ref=q.page_ref,
                diagnosed_misconception=diagnosed,
                confidence=conf,
                calibration_tag=cal_tag,
            )
        )

    score = sum(it.is_correct for it in items)
    total = len(items)
    ratio = score / total if total else 0.0
    recommended_level = "Nâng cao" if ratio >= 0.8 else ("Trung bình" if ratio >= 0.5 else "Mới bắt đầu")

    return QuizResult(
        score=score,
        total=total,
        items=items,
        weak_topics=weak_topics,
        detected_misconceptions=detected_misc,
        recommended_level=recommended_level,
        blind_spot_count=blind_spots,
        lucky_guess_count=lucky_guesses,
    )
