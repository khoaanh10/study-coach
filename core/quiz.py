"""Sinh và chấm quiz trắc nghiệm từ một đoạn văn bản."""
from pydantic import BaseModel

import config
from core.contracts import GradedQuestion, Question, Quiz, QuizResult
from core.llm import LLMError, complete_structured

SYSTEM_PROMPT = (
    "Bạn là giáo viên ra đề trắc nghiệm. Chỉ dựa trên VĂN BẢN người dùng cung cấp, "
    "không dùng kiến thức bên ngoài. Mỗi câu có đúng 4 đáp án, chỉ 1 đáp án đúng, "
    "các đáp án sai phải hợp lý nhưng rõ ràng sai theo văn bản. "
    "Viết câu hỏi bằng cùng ngôn ngữ với văn bản. "
    "Với mỗi câu, làm theo thứ tự: (1) chọn một câu trong văn bản làm bằng chứng và chép "
    "NGUYÊN VĂN vào evidence; (2) viết đáp án đúng khớp chính xác với bằng chứng đó; "
    "(3) explanation giải thích ngắn dựa trên evidence; (4) correct_index (0 đến 3) là vị trí "
    "của đáp án khớp với evidence. "
    "Cẩn thận với các ý đối lập trong văn bản (ví dụ 'quá lớn' và 'quá nhỏ', 'trước' và 'sau'): "
    "không được gán nguyên nhân hay hậu quả của ý này cho ý kia. "
    "Chỉ hỏi 'vì sao' khi văn bản có nêu rõ lý do; đáp án đúng không được lặp lại nguyên câu hỏi. "
    "Nếu văn bản vô nghĩa (ký tự gõ bừa, chuỗi lặp lại) hoặc không chứa thông tin, kiến thức nào "
    "để hỏi, trả về questions là danh sách rỗng. Không bịa câu hỏi, không hỏi về hình thức của "
    "văn bản (ký tự nào xuất hiện, lặp bao nhiêu lần)."
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

DIFFICULTY_HINTS = {
    "Dễ": (
        "Câu hỏi nhớ lại một thông tin nêu trực tiếp trong văn bản (định nghĩa, tên gọi, con số). "
        "Đáp án sai khác hẳn đáp án đúng."
    ),
    "Trung bình": (
        "Câu hỏi cần hiểu ý chính: hỏi vì sao, để làm gì, khác nhau thế nào. "
        "Không hỏi lại định nghĩa theo đúng câu chữ trong văn bản."
    ),
    "Khó": (
        "KHÔNG hỏi định nghĩa và không hỏi lại một câu có sẵn trong văn bản. Mỗi câu phải thuộc "
        "một trong các dạng: đưa ra một tình huống cụ thể mới rồi hỏi nên làm gì hoặc điều gì xảy ra; "
        "so sánh hai khái niệm trong văn bản; suy ra hệ quả từ ít nhất hai ý ở hai chỗ khác nhau. "
        "Cả 4 đáp án đều dùng thuật ngữ trong văn bản và dài tương đương nhau, đáp án sai chỉ sai ở "
        "một chi tiết."
    ),
}


def _norm(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .\"'“”")


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
) -> Quiz:
    text = text.strip()
    if len(text) < 100:
        raise LLMError("Văn bản quá ngắn để ra quiz (cần ít nhất khoảng 100 ký tự).")
    text = text[: config.MAX_INPUT_CHARS]
    if _looks_meaningless(text) or not _has_knowledge(text, model):
        raise LLMError(NO_CONTENT_MESSAGE)

    user_prompt = (
        # Xin dư 2 câu để còn câu thay thế khi có câu bị loại ở bước kiểm tra.
        f"Hãy tạo {n + 2} câu trắc nghiệm, mức độ: {difficulty}. "
        f"{DIFFICULTY_HINTS.get(difficulty, '')}\n\nVĂN BẢN:\n{text}"
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


class _Gate(BaseModel):
    reason: str  # lập luận ngắn, viết trước khi kết luận
    has_knowledge: bool


GATE_PROMPT = (
    "Bạn kiểm tra đầu vào cho một công cụ ra đề trắc nghiệm ôn tập. Đọc VĂN BẢN và quyết định nó có "
    "phải là tài liệu chứa thông tin, kiến thức để học và ra câu hỏi hay không. "
    "has_knowledge=true khi văn bản trình bày sự kiện, khái niệm, quy trình, số liệu hoặc lập luận "
    "(bài học, ghi chú, bài báo, tài liệu kỹ thuật, kể cả viết chưa trau chuốt). "
    "has_knowledge=false khi văn bản là: ký tự gõ bừa; các từ ngẫu nhiên không ghép thành câu có nghĩa; "
    "văn bản giữ chỗ như lorem ipsum; lời tán gẫu, than thở, chào hỏi không có thông tin gì để học. "
    "Viết reason một câu rồi mới đặt has_knowledge."
)


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


VERIFY_PROMPT = (
    "Bạn là người soát đề trắc nghiệm, rất khắt khe. Với mỗi câu, bạn nhận CÂU HỎI và ĐÁP ÁN ĐƯỢC "
    "CHỌN LÀ ĐÚNG. Chỉ dựa trên VĂN BẢN, kiểm tra xem văn bản có NÓI RÕ điều đó không. "
    "Đặt supported=false nếu gặp một trong các trường hợp: "
    "(a) văn bản không nêu thông tin mà đáp án khẳng định (kể cả khi nghe hợp lý theo kiến thức chung); "
    "(b) câu hỏi hỏi lý do nhưng văn bản không nêu lý do; "
    "(c) đáp án lấy đặc điểm của một ý khác trong văn bản, ví dụ văn bản nói 'A gây ra X, còn B gây ra Y' "
    "mà đáp án lại nói A gây ra Y; "
    "(d) đáp án chỉ lặp lại câu hỏi; "
    "(e) văn bản vô nghĩa, hoặc câu hỏi chỉ hỏi về hình thức của văn bản (ký tự, từ nào xuất hiện) "
    "chứ không hỏi về một thông tin, kiến thức. "
    "Với mỗi câu, viết reason (một câu, trích phần văn bản liên quan) rồi mới đặt supported. "
    "Trả về checks theo đúng thứ tự và đúng số lượng câu hỏi."
)


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


def grade(quiz: Quiz, answers: list[int | None]) -> QuizResult:
    """answers[i] là chỉ số đáp án đã chọn cho câu i, hoặc None nếu bỏ trống."""
    items = []
    for i, q in enumerate(quiz.questions):
        chosen = answers[i] if i < len(answers) else None
        items.append(
            GradedQuestion(
                question=q.question,
                options=q.options,
                chosen_index=chosen,
                correct_index=q.correct_index,
                is_correct=chosen == q.correct_index,
                explanation=q.explanation,
            )
        )
    return QuizResult(
        score=sum(it.is_correct for it in items),
        total=len(items),
        items=items,
    )
