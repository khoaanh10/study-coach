import pytest

from core.contracts import Question, Quiz
from core.llm import LLMError
from core.quiz import generate_quiz, grade


def make_quiz():
    qs = [
        Question(question="Q1?", options=["a", "b", "c", "d"], correct_index=1, explanation="e1"),
        Question(question="Q2?", options=["a", "b", "c", "d"], correct_index=3, explanation="e2"),
    ]
    return Quiz(questions=qs)


def test_grade_counts_correct_and_blank():
    result = grade(make_quiz(), [1, None])
    assert result.score == 1
    assert result.total == 2
    assert result.items[0].is_correct
    assert result.items[1].chosen_index is None


def test_grade_all_wrong():
    assert grade(make_quiz(), [0, 0]).score == 0


def test_short_text_rejected():
    with pytest.raises(LLMError):
        generate_quiz("quá ngắn")
