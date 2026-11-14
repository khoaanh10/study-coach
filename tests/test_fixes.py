"""Test cho các lỗi BUG-01 đến BUG-03 (không gọi API thật)."""
from io import BytesIO

import pytest
from openai import APIConnectionError, AuthenticationError
from pypdf import PdfWriter

from core import quiz as quiz_module
from core.contracts import Question, Quiz
from core.llm import friendly_error
from core.loader import load_text

TEXT = (
    "Chunk quá lớn làm việc truy xuất kém chính xác vì chứa nhiều ý lẫn lộn, "
    "còn chunk quá nhỏ làm mất ngữ cảnh. Các chunk thường gối lên nhau một phần, gọi là overlap."
)


def q(evidence: str, name: str = "Q?") -> Question:
    return Question(question=name, options=["a", "b", "c", "d"], evidence=evidence, explanation="e", correct_index=0)


def test_scan_pdf_has_clear_message():
    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    buf = BytesIO()
    w.write(buf)
    with pytest.raises(ValueError, match="không có lớp chữ"):
        load_text("scan.pdf", buf.getvalue())


def test_broken_pdf_has_clear_message():
    with pytest.raises(ValueError, match="Không đọc được file"):
        load_text("hong.pdf", b"khong phai pdf")


def make(cls):
    """Tạo lỗi của OpenAI mà không cần dựng request/response thật."""
    return cls.__new__(cls)


def test_auth_error_is_vietnamese_without_raw_detail():
    err = make(AuthenticationError)
    err.args = ("Invalid proxy server token sk-abc",)
    msg = friendly_error(err)
    assert "API key không hợp lệ" in msg
    assert "sk-abc" not in msg and "proxy" not in msg


def test_connection_error_is_vietnamese():
    assert "Không kết nối được" in friendly_error(make(APIConnectionError))


def test_grounded_evidence_detected():
    source = quiz_module._norm(TEXT)
    assert quiz_module._is_grounded(q("còn chunk quá nhỏ làm mất ngữ cảnh."), source)
    assert not quiz_module._is_grounded(q("chunk quá lớn làm mất ngữ cảnh"), source)


def fake_llm(monkeypatch, questions, verifier_answers, has_knowledge=True):
    def fake(messages, schema, **kwargs):
        if schema is Quiz:
            return Quiz(questions=questions)
        if schema is quiz_module._Gate:
            return schema(reason="r", has_knowledge=has_knowledge)
        return schema(checks=[{"reason": "r", "supported": v} for v in verifier_answers])

    monkeypatch.setattr(quiz_module, "complete_structured", fake)


def test_grounded_questions_are_preferred(monkeypatch):
    fake_llm(
        monkeypatch,
        [q("câu này không có trong văn bản nguồn", "bịa"), q("gối lên nhau một phần, gọi là overlap", "thật")],
        [True, True],
    )
    assert quiz_module.generate_quiz(TEXT, n=1).questions[0].question == "thật"


def test_question_rejected_by_verifier_is_ranked_last(monkeypatch):
    ev = "còn chunk quá nhỏ làm mất ngữ cảnh"
    # Người soát đề bác câu đầu và chấp nhận câu sau.
    fake_llm(monkeypatch, [q(ev, "lệch đáp án"), q(ev, "khớp đáp án")], [False, True])
    assert quiz_module.generate_quiz(TEXT, n=1).questions[0].question == "khớp đáp án"


@pytest.mark.parametrize(
    "text",
    [
        "asdasdasd " * 20,
        "sdfghjkl qwrtp zxcvbnm " * 10,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "1234567890 !@#$%^&*() " * 10,
        "hjkdshfjkdshfjkhsdjkfhsdjkhfjksdhfjkhsdkjfhjksdhfjkshdfjkhsdjkfhsdjkfhjksdhfjksdhfjkhsdjkfhsdjkfhsdjk " * 3,
    ],
)
def test_gibberish_is_rejected_without_calling_llm(monkeypatch, text):
    def boom(*a, **k):
        raise AssertionError("không được gọi LLM với văn bản vô nghĩa")

    monkeypatch.setattr(quiz_module, "complete_structured", boom)
    with pytest.raises(quiz_module.LLMError, match="không có nội dung kiến thức"):
        quiz_module.generate_quiz(text)


def test_real_text_is_not_flagged_as_gibberish():
    assert not quiz_module._looks_meaningless(TEXT)
    assert not quiz_module._looks_meaningless(
        "Retrieval augmented generation lets a language model answer from external documents. "
        "The system first retrieves relevant passages and then adds them to the prompt."
    )
    assert not quiz_module._looks_meaningless("機械学習はデータからパターンを学習する技術です。" * 5)


def test_model_returning_no_questions_gives_clear_error(monkeypatch):
    fake_llm(monkeypatch, [], [])
    with pytest.raises(quiz_module.LLMError, match="không có nội dung kiến thức"):
        quiz_module.generate_quiz(TEXT)


def test_all_questions_rejected_means_no_quiz(monkeypatch):
    fake_llm(monkeypatch, [q("còn chunk quá nhỏ làm mất ngữ cảnh", "đáng ngờ")], [False])
    with pytest.raises(quiz_module.LLMError, match="không có nội dung kiến thức"):
        quiz_module.generate_quiz(TEXT)


def test_text_without_knowledge_is_rejected_by_gate(monkeypatch):
    fake_llm(monkeypatch, [q("còn chunk quá nhỏ làm mất ngữ cảnh", "không nên xuất hiện")], [True], has_knowledge=False)
    with pytest.raises(quiz_module.LLMError, match="không có nội dung kiến thức"):
        quiz_module.generate_quiz(TEXT)
