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
    OPEN_GRADE_PROMPT,
    SYSTEM_PROMPT,
    VERIFY_PROMPT,
)
from core.rag import _STOP_WORDS_FOLDED, _is_technical_term, _strip_accents

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
        valid = [_as_mcq(q) for q in quiz.questions if _is_valid(q)]
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


# ---------- Dạng câu hỏi mở: điền vào chỗ trống & tự luận ngắn ----------
QUESTION_MODES = ["Trắc nghiệm", "Điền vào chỗ trống", "Tự luận ngắn", "Kết hợp cả ba dạng"]
QTYPE_LABELS = {"mcq": "Trắc nghiệm", "fill_blank": "Điền vào chỗ trống", "short_answer": "Tự luận ngắn"}
BLANK_MARK = "_____"
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ_]+(?:-[0-9A-Za-z_]+)*")


def _as_mcq(q: Question) -> Question:
    """Câu do model sinh luôn là trắc nghiệm: xóa các trường dạng mở model có thể tự điền."""
    q.qtype, q.blank_prompt, q.answer_text = "mcq", "", ""
    return q


def _fold(text: str) -> str:
    """Chuẩn hóa để so khớp câu trả lời tự gõ: bỏ dấu, chữ thường, chỉ giữ chữ/số."""
    return " ".join(re.findall(r"[a-z0-9_]+", _strip_accents((text or "").lower())))


def _pick_blank_term(evidence: str, prefer: str = "") -> str:
    """Chọn (tất định) một thuật ngữ then chốt trong câu bằng chứng để đục lỗ.

    Ưu tiên thuật ngữ kỹ thuật/số (chunk, overlap, top-k, 0.2...), kế đến từ dài nhất không phải từ nối;
    trong cùng nhóm ưu tiên từ cũng xuất hiện trong đáp án đúng."""
    prefer_fold = set(_fold(prefer).split())
    best: tuple[tuple[int, int, int], str] | None = None
    for pos, tok in enumerate(_TOKEN_RE.findall(evidence or "")):
        low = tok.lower()
        folded = _strip_accents(low)
        if folded in _STOP_WORDS_FOLDED or len(tok) < 3:
            continue
        technical = _is_technical_term(low) or any(ch.isdigit() for ch in tok)
        if not technical and len(tok) < 5:
            continue
        rank = (2 if technical else 1, 1 if folded in prefer_fold else 0, len(tok))
        # Bằng hạng thì giữ từ xuất hiện trước (pos nhỏ hơn) để kết quả ổn định.
        if best is None or rank > best[0]:
            best = (rank, tok)
    return best[1] if best else ""


def to_fill_blank(q: Question) -> Question | None:
    """Chuyển câu trắc nghiệm đã có bằng chứng thành câu điền vào chỗ trống; None nếu không đục lỗ được."""
    evidence = (q.evidence or "").strip()
    if len(evidence) < 25:
        return None
    term = _pick_blank_term(evidence, q.options[q.correct_index] if _is_valid(q) else "")
    if not term:
        return None
    pattern = re.compile(rf"(?<![0-9A-Za-zÀ-ỹ_]){re.escape(term)}(?![0-9A-Za-zÀ-ỹ_])", re.IGNORECASE)
    blanked = pattern.sub(BLANK_MARK, evidence)
    if BLANK_MARK not in blanked:
        return None
    out = q.model_copy(deep=True)
    out.qtype, out.blank_prompt, out.answer_text = "fill_blank", blanked, term
    return out


def to_short_answer(q: Question) -> Question:
    out = q.model_copy(deep=True)
    out.qtype, out.blank_prompt = "short_answer", ""
    out.answer_text = q.options[q.correct_index] if _is_valid(q) else (q.evidence or "")
    return out


def apply_question_types(quiz: Quiz, mode: str = "Trắc nghiệm") -> Quiz:
    """Đổi dạng câu hỏi của quiz theo lựa chọn trên giao diện.

    Câu dạng mở được dựng (tất định) từ chính câu trắc nghiệm đã qua kiểm chứng bằng chứng, nên vẫn bám
    tài liệu và dùng được cả khi không có API key. Câu không đục lỗ được thì chuyển thành tự luận ngắn."""
    if mode not in QUESTION_MODES or mode == "Trắc nghiệm":
        return quiz
    out: list[Question] = []
    for i, q in enumerate(quiz.questions):
        if mode == "Kết hợp cả ba dạng":
            want = ("mcq", "fill_blank", "short_answer")[i % 3]
        else:
            want = "fill_blank" if mode == "Điền vào chỗ trống" else "short_answer"
        if want == "mcq":
            out.append(q)
        elif want == "fill_blank":
            out.append(to_fill_blank(q) or to_short_answer(q))
        else:
            out.append(to_short_answer(q))
    return Quiz(questions=out)


def display_question(q: Question) -> str:
    """Nội dung câu hỏi hiển thị cho người học theo đúng dạng câu."""
    if q.qtype == "fill_blank" and q.blank_prompt:
        return f"Điền từ/cụm từ còn thiếu vào chỗ trống: «{q.blank_prompt}»"
    return q.question


def _content_tokens(text: str) -> list[str]:
    return [t for t in _fold(text).split() if len(t) >= 2 and t not in _STOP_WORDS_FOLDED]


def grade_open_answer_offline(q: Question, learner_text: str) -> tuple[bool, str]:
    """Chấm tất định (không gọi LLM) một câu trả lời tự gõ. Trả (đúng?, nhận xét)."""
    answer = _fold(learner_text)
    expected = _fold(q.answer_text)
    if not answer:
        return False, "Bỏ trống."
    if q.qtype == "fill_blank":
        # Đúng khi gõ đúng thuật ngữ (không phân biệt hoa/thường, dấu), cho phép kèm 1-2 từ đệm.
        ok = bool(expected) and (
            answer == expected
            or (f" {expected} " in f" {answer} " and len(answer.split()) <= len(expected.split()) + 2)
        )
        return ok, ("Khớp đúng thuật ngữ trong tài liệu." if ok else f"Chưa khớp thuật ngữ cần điền: «{q.answer_text}».")
    keys = list(dict.fromkeys(_content_tokens(q.answer_text)))
    if not keys:
        keys = list(dict.fromkeys(_content_tokens(q.evidence)))
    got = set(_content_tokens(learner_text))
    hit = [k for k in keys if k in got]
    ratio = len(hit) / len(keys) if keys else 0.0
    ok = ratio >= 0.5 and len(got) >= min(3, len(keys))
    note = f"Nêu được {len(hit)}/{len(keys)} ý từ khóa của đáp án mẫu (chấm offline theo từ khóa)."
    return ok, note


class _OpenGrade(BaseModel):
    reason: str  # lập luận ngắn, viết trước khi kết luận
    is_correct: bool
    misconception: str = ""


class _OpenGrades(BaseModel):
    grades: list[_OpenGrade]


def grade_open_answers(
    questions: list[Question],
    learner_texts: list[str],
    model: str | None = None,
    demo_mode: bool = True,
) -> list[tuple[bool, str, str]]:
    """Chấm loạt câu trả lời tự gõ. Trả danh sách (đúng?, nhận xét, lỗi tư duy chẩn đoán).

    Có API key và không ở demo_mode: một lời gọi LLM (structured output) chấm tất cả các câu.
    demo_mode hoặc LLM lỗi: dùng bộ chấm tất định `grade_open_answer_offline`."""
    offline = [(*grade_open_answer_offline(q, t), "") for q, t in zip(questions, learner_texts)]
    todo = [i for i, t in enumerate(learner_texts) if (t or "").strip()]
    if demo_mode or not config.has_live_api_key() or not todo:
        return offline
    listing = "\n\n".join(
        f"Câu {n + 1} ({QTYPE_LABELS.get(questions[i].qtype, 'Tự luận ngắn')}): {display_question(questions[i])}\n"
        f"Đáp án mẫu: {questions[i].answer_text}\n"
        f"Bằng chứng: {questions[i].evidence}\n"
        f"Câu trả lời của người học: {learner_texts[i].strip()[:600]}"
        for n, i in enumerate(todo)
    )
    try:
        got = complete_structured(
            [
                {"role": "system", "content": OPEN_GRADE_PROMPT},
                {"role": "user", "content": listing},
            ],
            _OpenGrades,
            model=model,
            temperature=0,
            purpose="open_answer_grading",
        ).grades
    except Exception:  # noqa: BLE001 - lỗi mạng/định dạng bất kỳ đều lùi về bộ chấm offline, không chặn nộp bài
        return offline
    if len(got) != len(todo):
        return offline
    out = list(offline)
    for i, g in zip(todo, got):
        out[i] = (bool(g.is_correct), g.reason.strip(), (g.misconception or "").strip())
    return out


def grade(
    quiz: Quiz,
    answers: list[int | None],
    confidences: list[str] | None = None,
    text_answers: list[str | None] | None = None,
    demo_mode: bool = True,
    model: str | None = None,
) -> QuizResult:
    """answers[i] là chỉ số đáp án đã chọn cho câu i, hoặc None nếu bỏ trống.

    text_answers[i] là câu trả lời tự gõ cho câu dạng mở (fill_blank / short_answer). Câu dạng mở chỉ
    gọi LLM khi demo_mode=False và có API key; mặc định chấm tất định để test/CI không chạm API."""
    items: list[GradedQuestion] = []
    weak_topics: list[str] = []
    detected_misc: list[str] = []
    blind_spots = 0
    lucky_guesses = 0

    open_idx = [i for i, q in enumerate(quiz.questions) if q.qtype in ("fill_blank", "short_answer")]
    open_texts = [
        ((text_answers[i] if text_answers and i < len(text_answers) else "") or "").strip() for i in open_idx
    ]
    open_grades = dict(
        zip(
            open_idx,
            grade_open_answers([quiz.questions[i] for i in open_idx], open_texts, model=model, demo_mode=demo_mode),
        )
    ) if open_idx else {}
    open_text_by_idx = dict(zip(open_idx, open_texts))

    for i, q in enumerate(quiz.questions):
        chosen = answers[i] if i < len(answers) else None
        conf = (confidences[i] if confidences and i < len(confidences) else "Bình thường") or "Bình thường"
        is_open = i in open_grades
        learner_text = open_text_by_idx.get(i, "")
        grader_note = ""
        open_misc = ""
        if is_open:
            open_ok, grader_note, open_misc = open_grades[i]
            # Câu dạng mở: đúng thì coi như chọn đáp án đúng; sai/bỏ trống thì không có lựa chọn nào.
            chosen = q.correct_index if (open_ok and learner_text) else None
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
            if is_open and learner_text:
                cal_tag = "Điểm mù tự tin sai" if conf == "Chắc chắn" else "Biết mình chưa biết"
                if conf == "Chắc chắn":
                    blind_spots += 1
                if open_misc and open_misc.lower() != "chính xác":
                    diagnosed = open_misc
                    if open_misc not in detected_misc:
                        detected_misc.append(open_misc)
            elif chosen is None:
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
                question=display_question(q),
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
                qtype=q.qtype if is_open else "mcq",
                learner_text=learner_text,
                expected_text=q.answer_text if is_open else "",
                grader_feedback=grader_note,
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
