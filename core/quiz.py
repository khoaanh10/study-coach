"""Sinh và chấm quiz trắc nghiệm từ một đoạn văn bản."""
import config
from core.contracts import GradedQuestion, Question, Quiz, QuizResult
from core.llm import LLMError, complete_structured

SYSTEM_PROMPT = (
    "Bạn là giáo viên ra đề trắc nghiệm. Chỉ dựa trên VĂN BẢN người dùng cung cấp, "
    "không dùng kiến thức bên ngoài. Mỗi câu có đúng 4 đáp án, chỉ 1 đáp án đúng, "
    "các đáp án sai phải hợp lý nhưng rõ ràng sai theo văn bản. "
    "Viết câu hỏi bằng cùng ngôn ngữ với văn bản. "
    "Phần explanation giải thích ngắn vì sao đáp án đúng, dẫn lại ý trong văn bản. "
    "correct_index là số từ 0 đến 3."
)

DIFFICULTY_HINTS = {
    "Dễ": "Câu hỏi nhớ lại thông tin trực tiếp có trong văn bản.",
    "Trung bình": "Câu hỏi cần hiểu ý chính và so sánh các chi tiết.",
    "Khó": "Câu hỏi cần suy luận, kết nối nhiều ý trong văn bản; đáp án sai rất gần đáp án đúng.",
}


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

    user_prompt = (
        f"Hãy tạo {n} câu trắc nghiệm, mức độ: {difficulty}. "
        f"{DIFFICULTY_HINTS.get(difficulty, '')}\n\nVĂN BẢN:\n{text}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    valid: list[Question] = []
    for _ in range(max_attempts):
        quiz = complete_structured(messages, Quiz, model=model)
        valid = [q for q in quiz.questions if _is_valid(q)]
        if len(valid) >= n:
            break
    if not valid:
        raise LLMError("Không tạo được câu hỏi hợp lệ. Thử lại hoặc dùng văn bản dài hơn.")
    return Quiz(questions=valid[:n])


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
