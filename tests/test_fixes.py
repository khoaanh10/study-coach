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


def fake_llm(monkeypatch, questions, verifier_answers):
    def fake(messages, schema, **kwargs):
        if schema is Quiz:
            return Quiz(questions=questions)
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
