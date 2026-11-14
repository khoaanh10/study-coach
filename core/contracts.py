"""Hợp đồng dữ liệu dùng chung giữa các module (UI, quiz, sau này là RAG/agent)."""
from pydantic import BaseModel


class Question(BaseModel):
    question: str
    options: list[str]  # đúng 4 đáp án
    # evidence đứng trước correct_index để model tìm bằng chứng rồi mới chốt đáp án.
    evidence: str = ""  # câu trích nguyên văn từ văn bản nguồn
    explanation: str
    correct_index: int  # 0..3


class Quiz(BaseModel):
    questions: list[Question]


class GradedQuestion(BaseModel):
    question: str
    options: list[str]
    chosen_index: int | None
    correct_index: int
    is_correct: bool
    explanation: str


class QuizResult(BaseModel):
    score: int
    total: int
    items: list[GradedQuestion]
