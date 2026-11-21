"""Kiểm thử các tính năng bổ sung: Ôn hôm nay, trích dẫn, lịch sử tiến bộ, phản hồi, hội thoại, dạng câu hỏi mới,
export, loader docx/pptx và các bản sửa lỗi nền. Không gọi API thật."""
from __future__ import annotations

import io
import sqlite3
import zipfile

import pytest
from streamlit.testing.v1 import AppTest

import config
from core import adaptive, db, export, loader, quiz
from core.contracts import CoachTurnResult
from core.loader import load_default_samples
from core.ui_theme import (
    find_highlight_ranges,
    highlight_text_html,
    render_bento_strip_html,
    render_doc_section_html,
)

APP = str(config.BASE_DIR / "app.py")


@pytest.fixture
def tdb(tmp_path):
    path = tmp_path / "f.db"
    db.init_db(path)
    return path


def _turn(text="Trả lời mẫu"):
    return CoachTurnResult(answer=text, route_taken="adaptive_explain", pedagogical_reason="x", learner_level_used="Mới bắt đầu")


# ---------- 9. Hội thoại ----------
def test_conversations_create_rename_delete(tdb):
    c1 = db.create_conversation("u1", "d1", db_path=tdb)
    db.append_chat_message("u1", "d1", "user", "Chunk là gì?", conversation_id=c1, db_path=tdb)
    c2 = db.create_conversation("u1", "d1", db_path=tdb)
    db.append_chat_message("u1", "d1", "user", "Overlap?", conversation_id=c2, db_path=tdb)
    convs = db.list_conversations("u1", "d1", db_path=tdb)
    assert {c["id"] for c in convs} == {c1, c2}
    assert next(c for c in convs if c["id"] == c1)["title"] == "Chunk là gì?"
    assert [m["content"] for m in db.load_chat_history("u1", "d1", conversation_id=c1, db_path=tdb)] == ["Chunk là gì?"]
    assert db.rename_conversation("u1", c1, "Ôn chunk", db_path=tdb)
    assert not db.rename_conversation("u2", c1, "hack", db_path=tdb)  # sai chủ sở hữu
    assert db.delete_conversation("u1", c2, db_path=tdb)
    assert db.load_chat_history("u1", "d1", conversation_id=c2, db_path=tdb) == []
    assert len(db.list_conversations("u1", "d1", db_path=tdb)) == 1


def test_legacy_chat_rows_migrated(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, learner_id TEXT NOT NULL, doc_id TEXT NOT NULL,"
        " role TEXT NOT NULL, content TEXT NOT NULL, turn_json TEXT DEFAULT '', created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    con.execute("INSERT INTO chat_messages (learner_id, doc_id, role, content) VALUES ('u1','d1','user','cũ')")
    con.commit()
    con.close()
    db.init_db(path)
    db.init_db(path)  # chạy lại không nhân đôi
    convs = db.list_conversations("u1", "d1", db_path=path)
    assert len(convs) == 1 and convs[0]["message_count"] == 1
    assert db.load_chat_history("u1", "d1", conversation_id=convs[0]["id"], db_path=path)[0]["content"] == "cũ"


# ---------- 4. Phản hồi ----------
def test_message_feedback(tdb):
    mid = db.append_chat_message("u1", "d1", "assistant", "ok", turn=_turn(), db_path=tdb)
    assert db.set_message_feedback("u1", mid, 1, db_path=tdb)
    assert db.get_feedback_summary("u1", db_path=tdb) == {"up": 1, "down": 0, "unrated": 0}
    db.set_message_feedback("u1", mid, -1, db_path=tdb)
    assert db.get_feedback_summary("u1", db_path=tdb)["down"] == 1
    assert not db.set_message_feedback("u2", mid, 1, db_path=tdb)
    assert db.load_chat_history("u1", "d1", db_path=tdb)[0]["feedback"] == -1


# ---------- 3. Lịch sử mastery + chuỗi ngày ----------
def test_mastery_timeline_and_streak(tdb):
    db.update_concept_mastery("u1", "d1", "A", is_correct=True, db_path=tdb)
    db.update_concept_mastery("u1", "d1", "B", is_correct=False, db_path=tdb)
    tl = db.get_mastery_timeline("u1", "d1", db_path=tdb)
    assert len(tl) == 1 and set(tl[0]["concepts"]) == {"A", "B"}
    assert tl[0]["overall"] == round(sum(tl[0]["concepts"].values()) / 2, 1)
    st = db.get_study_streak("u1", db_path=tdb)
    assert st["current"] == 1 and st["studied_today"]
    db.reset_learner_profile("u1", "d1", db_path=tdb)
    assert db.get_mastery_timeline("u1", "d1", db_path=tdb) == []


def test_streak_consecutive_days(tdb):
    from datetime import datetime, timedelta, timezone

    with sqlite3.connect(tdb) as con:
        for back in (0, 1, 2, 4):
            ts = (datetime.now(timezone.utc) - timedelta(days=back)).strftime("%Y-%m-%d %H:%M:%S")
            con.execute(
                "INSERT INTO mastery_history (learner_id, doc_id, concept, mastery_score, created_at) VALUES ('u','d','c',50,?)",
                (ts,),
            )
    s = db.get_study_streak("u", db_path=tdb)
    assert s["current"] == 3 and s["longest"] == 3
    # Hôm nay chưa học nhưng hôm qua có: chuỗi vẫn còn.
    s2 = db.get_study_streak("u", db_path=tdb, today=(datetime.now(timezone.utc).astimezone() + timedelta(days=1)).strftime("%Y-%m-%d"))
    assert s2["current"] == 3 and not s2["studied_today"]


# ---------- 15. Số liệu theo người học ----------
def test_learner_activity_summary_is_per_learner(tdb):
    db.append_chat_message("u1", "d1", "user", "a", db_path=tdb)
    db.append_chat_message("u2", "d1", "user", "b", db_path=tdb)
    db.append_chat_message("u2", "d1", "user", "c", db_path=tdb)
    assert db.get_learner_activity_summary("u1", db_path=tdb)["coach_questions"] == 1
    assert db.get_learner_activity_summary("u2", db_path=tdb)["coach_questions"] == 2


def test_bento_strip_has_no_global_ai_calls():
    from core.contracts import LearnerDNA

    dna = LearnerDNA(learner_id="x", name="X")
    html = render_bento_strip_html(dna, {"coverage_pct": 0}, {"streak_days": 3, "coach_questions": 5})
    assert "Lượt gọi AI" not in html and "Chuỗi ngày học" in html


# ---------- 6. Dạng câu hỏi ----------
def _sample_quiz(n=6):
    text, _, _ = load_default_samples()
    return quiz.generate_quiz(text, n=n, demo_mode=True)


def test_fill_blank_and_short_answer_offline_grading():
    q = _sample_quiz()
    fb = quiz.apply_question_types(q, "Điền vào chỗ trống")
    assert all(x.qtype in ("fill_blank", "short_answer") for x in fb.questions)
    blanks = [x for x in fb.questions if x.qtype == "fill_blank"]
    assert blanks and all(quiz.BLANK_MARK in x.blank_prompt for x in blanks)
    ok = quiz.grade(fb, [None] * len(fb.questions), text_answers=[x.answer_text.upper() for x in fb.questions])
    assert ok.score == ok.total
    bad = quiz.grade(fb, [None] * len(fb.questions), text_answers=["không liên quan"] * len(fb.questions), confidences=["Chắc chắn"] * len(fb.questions))
    assert bad.score == 0 and bad.blind_spot_count == bad.total
    empty = quiz.grade(fb, [None] * len(fb.questions))
    assert empty.score == 0 and all(not i.learner_text for i in empty.items)
    mixed = quiz.apply_question_types(q, "Kết hợp cả ba dạng")
    assert {x.qtype for x in mixed.questions} == {"mcq", "fill_blank", "short_answer"}


def test_open_grading_uses_llm_when_live(monkeypatch):
    q = quiz.apply_question_types(_sample_quiz(3), "Tự luận ngắn")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test")
    config.set_runtime_api_key(None)
    calls = {}

    def fake(messages, schema, **kw):
        calls["n"] = calls.get("n", 0) + 1
        return quiz._OpenGrades(grades=[quiz._OpenGrade(reason="ok", is_correct=True) for _ in range(3)])

    monkeypatch.setattr(quiz, "complete_structured", fake)
    res = quiz.grade(q, [None] * 3, text_answers=["a", "b", "c"], demo_mode=False)
    assert calls["n"] == 1 and res.score == 3
    # demo_mode=True thì không gọi LLM.
    quiz.grade(q, [None] * 3, text_answers=["a", "b", "c"], demo_mode=True)
    assert calls["n"] == 1
    # LLM lỗi thì lùi về chấm offline, không văng lỗi.
    monkeypatch.setattr(quiz, "complete_structured", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert quiz.grade(q, [None] * 3, text_answers=["a", "b", "c"], demo_mode=False).total == 3


# ---------- 1. Ôn hôm nay ----------
def test_review_session_collects_due_and_misconceptions(tdb):
    text, _, title = load_default_samples()
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True)
    dna = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=tdb)
    targets = adaptive.collect_review_targets(dna, cmap)
    assert targets and targets[0]["reason"] == "misconception"
    rq, used = adaptive.build_review_session(text, dna, cmap, demo_mode=True)
    assert 3 <= len(rq.questions) <= adaptive.REVIEW_MAX_QUESTIONS and used
    fresh = db.get_learner_dna("nobody", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=tdb)
    assert adaptive.build_review_session(text, fresh, cmap, demo_mode=True)[0].questions == []


# ---------- 2. Trích dẫn ----------
def test_highlight_ranges_and_html_escape():
    text = "Chunk quá lớn làm truy xuất kém.\nOverlap giúp giữ ngữ cảnh <b>quan trọng</b>."
    assert find_highlight_ranges(text, ["Overlap  giúp giữ ngữ cảnh"])
    html = highlight_text_html(text, ["Overlap giúp giữ ngữ cảnh <b>quan trọng</b>."])
    assert '<mark class="sc-hl">' in html and "<b>" not in html and "&lt;b&gt;" in html
    sec = render_doc_section_html({"page": "[Mục 2]", "source": "x.txt", "text": text}, 1, ["Chunk quá lớn"], focused=True)
    assert 'id="sc-sec-2"' in sec and "sc-doc-section focus" in sec


# ---------- 7. Export ----------
def test_export_markdown_and_flashcards(tdb):
    text, _, title = load_default_samples()
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True)
    dna = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=tdb)
    md = export.misconceptions_markdown(dna, title, cmap)
    assert "# Sổ tay lỗi tư duy" in md and "Chunking và Overlap" in md
    cards = export.build_flashcards(dna, cmap)
    assert len(cards) == len(dna.active_misconceptions)
    assert export.flashcards_csv(cards).count("\n") == len(cards)
    assert "Thẻ 1" in export.flashcards_markdown(cards, title)
    assert export.safe_filename("a/b: c", "x.md").endswith("x.md") and "/" not in export.safe_filename("a/b", "x.md")


# ---------- 8. Loader ----------
def _docx(paragraphs):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = ""
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def _pptx(slides):
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i, lines in enumerate(slides, start=1):
            paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in lines)
            z.writestr(f"ppt/slides/slide{i}.xml", f'<?xml version="1.0"?><p:sld xmlns:a="{a}" xmlns:p="p"><p:cSld>{paras}</p:cSld></p:sld>')
    return buf.getvalue()


def test_load_docx_and_pptx():
    out = loader.load_text("a.docx", _docx([("Heading1", "Giới thiệu RAG"), (None, "RAG kết hợp truy xuất và sinh."), (None, "")]))
    assert "Giới thiệu RAG\nRAG kết hợp truy xuất và sinh." in out
    out = loader.load_text("s.PPTX", _pptx([["Tiêu đề", "Ý một"], ["Slide hai"]]))
    assert "[Trang 1]\nTiêu đề\nÝ một" in out and "[Trang 2]\nSlide hai" in out
    combined, secs = loader.load_structured_sections([("s.pptx", _pptx([["Tiêu đề", "Ý một"], ["Slide hai"]]))])
    assert [s["page"] for s in secs] == ["[Trang 1]", "[Trang 2]"]


def test_loader_error_messages():
    with pytest.raises(ValueError, match="không phải file .docx"):
        loader.load_text("x.docx", b"not a zip")
    with pytest.raises(ValueError, match=r"\.docx / \.pptx"):
        loader.load_text("x.doc", b"")
    with pytest.raises(ValueError, match="không có nội dung chữ"):
        loader.load_text("x.docx", _docx([]))
    with pytest.raises(ValueError, match="không hợp lệ"):
        evil = io.BytesIO()
        with zipfile.ZipFile(evil, "w") as z:
            z.writestr("word/document.xml", '<!DOCTYPE x [<!ENTITY a "b">]><w:document xmlns:w="x"/>')
        loader.load_text("x.docx", evil.getvalue())


def test_scanned_pdf_message(monkeypatch):
    class _Page:
        def extract_text(self):
            return ""

    class _Reader:
        def __init__(self, *_a, **_k):
            self.pages = [_Page(), _Page()]

    monkeypatch.setattr(loader, "PdfReader", _Reader)
    with pytest.raises(ValueError, match="ảnh quét"):
        loader.load_text("scan.pdf", b"%PDF")


# ---------- 10-12, 16 ----------
def test_base_url_not_runtime_settable():
    assert not hasattr(config, "set_runtime_base_url")
    assert config.get_effective_base_url() == config.OPENAI_BASE_URL.strip()


def test_session_secret_random_and_persisted(tmp_path, monkeypatch):
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "x" / "a.db"))
    monkeypatch.setattr(config, "_get_setting", lambda name, default="": default)
    config._SESSION_SECRET_CACHE.clear()
    s1 = config.get_session_secret()
    assert len(s1) >= 32 and (tmp_path / "x" / ".session_secret").exists()
    config._SESSION_SECRET_CACHE.clear()
    assert config.get_session_secret() == s1  # đọc lại từ file
    monkeypatch.setattr(config, "_get_setting", lambda name, default="": "tu-moi-truong" if name == "SESSION_SECRET" else default)
    assert config.get_session_secret() == b"tu-moi-truong"


def test_requirements_pin_streamlit():
    import re

    for path in (config.BASE_DIR / "requirements.txt", config.BASE_DIR.parent / "requirements.txt"):
        m = re.search(r"(?m)^streamlit>=(\d+)\.(\d+)", path.read_text(encoding="utf-8"))
        assert m and (int(m[1]), int(m[2])) >= (1, 52), path


def test_eval_does_not_write_results(tmp_path):
    from eval.run_eval import run_offline_eval

    target = config.BASE_DIR / "eval" / "results" / "latest_eval.json"
    before = target.read_bytes()
    run_offline_eval(save=False)
    assert target.read_bytes() == before


# ---------- UI (AppTest) ----------
def _nav(at):
    return [c for c in at.segmented_control if str(c.key).startswith("nav_ctrl_")][0]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "ui.db"))
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    config.set_runtime_api_key(None)
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.text_input("reg_username_input").set_value("feat_user")
    at.text_input("reg_display_name_input").set_value("Feat")
    at.text_input("reg_password_input").set_value("matkhau123")
    at.text_input("reg_password2_input").set_value("matkhau123")
    at.button("btn_register_submit").click().run()
    at.button("btn_load_sample_onboarding").click().run()
    assert not at.exception
    _nav(at).set_value("coach").run()  # đăng ký mới mở bài chẩn đoán ở trang Bản đồ
    yield at
    config.set_runtime_api_key(None)


def test_ui_base_url_input_removed(app):
    assert not [t for t in app.text_input if t.key == "runtime_base_url_input"]


def test_ui_chat_feedback_citation_and_conversations(app):
    app.chat_input[0].set_value("Giải thích Chunking và Overlap trong RAG?").run()
    assert not app.exception
    lid = app.session_state["learner_id"]
    key = f"{lid}::{app.session_state['doc_id']}"
    hist = app.session_state["chat_histories"][key]
    assert hist[-1]["id"] and hist[-1]["turn"].citations
    assert [b for b in app.button if str(b.key).startswith("cite_0_") or str(b.key).startswith("cite_1_")]
    # Trả lời lại thay cặp cuối bằng cặp mới (không nhân đôi).
    n = len(hist)
    app.button("btn_regen").click().run()
    assert not app.exception
    assert len(app.session_state["chat_histories"][key]) == n
    # Giải thích đơn giản hơn thêm một lượt.
    app.button("btn_simpler").click().run()
    assert len(app.session_state["chat_histories"][key]) == n + 2
    # Làm mới chat: tạo cuộc trò chuyện mới, giữ cuộc cũ trong DB.
    convs_before = db.list_conversations(lid, app.session_state["doc_id"])
    app.button("btn_clear_chat").click().run()
    assert app.session_state["chat_histories"][key] == []
    app.chat_input[0].set_value("Embedding là gì?").run()
    convs = db.list_conversations(lid, app.session_state["doc_id"])
    assert len(convs) == len(convs_before) + 1
    old = [c for c in convs if c["id"] == convs_before[0]["id"]][0]
    assert old["message_count"] >= 4


def test_ui_review_flow_records_and_streak(app):
    # Chưa có lịch sử: trang Ôn hôm nay báo chưa có gì cần ôn.
    _nav(app).set_value("review").run()
    assert not app.exception
    lid = app.session_state["learner_id"]
    did = app.session_state["doc_id"]
    db.update_concept_mastery(lid, did, "Chunking và Overlap", is_correct=False)
    db.record_misconception(lid, did, "Chunking và Overlap", "Tưởng chunk càng lớn càng tốt")
    app.run()
    app.button("btn_start_review").click().run()
    assert not app.exception and app.session_state["review_quiz"] is not None
    app.button("btn_submit_review").click().run()
    assert not app.exception and app.session_state["review_result"] is not None
    assert db.recent_quiz_attempts(lid)[0]["difficulty"] == "Ôn tập"
    before = db.get_learner_activity_summary(lid)["quiz_attempts"]
    app.button("btn_submit_review").click().run() if [b for b in app.button if b.key == "btn_submit_review"] else None
    assert db.get_learner_activity_summary(lid)["quiz_attempts"] == before


def test_ui_quiz_one_question_at_a_time_with_open_types(app):
    _nav(app).set_value("practice").run()
    app.selectbox("quiz_type_select").set_value("Kết hợp cả ba dạng").run()
    app.button("btn_gen_quiz").click().run()
    q = app.session_state["quiz"]
    assert {x.qtype for x in q.questions} >= {"mcq", "fill_blank"}
    assert len(app.progress) >= 1
    ver = app.session_state["quiz_version"]
    app.radio(f"q{ver}_0").set_value(q.questions[0].correct_index)
    app.segmented_control(f"conf_{ver}_0").set_value("Chắc chắn")
    app.button("quiz_next").click().run()
    assert app.session_state["quiz_pos"] == 1
    app.text_input(f"qt{ver}_1").set_value(q.questions[1].answer_text)
    app.button("quiz_prev").click().run()
    assert app.session_state["quiz_pos"] == 0
    assert app.radio(f"q{ver}_0").value == q.questions[0].correct_index  # giữ đáp án khi quay lại
    app.button("btn_submit_quiz").click().run()
    assert not app.exception
    res = app.session_state["quiz_result"]
    assert res.items[0].is_correct and res.items[1].is_correct and res.items[1].qtype == "fill_blank"


def test_ui_doc_reader_and_citation_focus(app):
    _nav(app).set_value("doc").run()
    assert not app.exception
    app.session_state["doc_focus_page"] = "[Mục 2]"
    app.session_state["doc_focus_snips"] = [app.session_state["doc_sections"][1]["text"][:60]]
    app.run()
    assert not app.exception
    assert any("sc-doc-section focus" in str(m.value) for m in app.markdown)


def test_ui_switch_document_from_sidebar(app):
    lid = app.session_state["learner_id"]
    custom = (
        "1. Kiến trúc Transformer\nTransformer sử dụng cơ chế Self-Attention để tính mức độ liên quan giữa mọi từ trong câu song song thay vì xử lý tuần tự.\n\n"
        "2. Positional Encoding\nDo Self-Attention không có thứ tự thời gian, Positional Encoding cộng thêm thông tin vị trí vào vector embedding."
    )
    _nav(app).set_value("map").run()
    app.text_area("doc_paste_area").set_value(custom).run()
    app.button("btn_load_paste").click().run()
    new_id = app.session_state["doc_id"]
    assert new_id != db.DEFAULT_DOC_ID
    sel = app.selectbox(f"side_doc_pick_{new_id}")
    assert "Giáo trình mẫu: RAG & OpenAI API" in sel.options
    sel.set_value(db.DEFAULT_DOC_ID).run()
    assert not app.exception and app.session_state["doc_id"] == db.DEFAULT_DOC_ID
    assert any(d["doc_id"] == new_id for d in db.list_learner_documents(lid))


def test_ui_analytics_chart_and_export(app):
    _nav(app).set_value("practice").run()
    app.button("btn_gen_quiz").click().run()
    app.button("btn_submit_quiz").click().run()
    _nav(app).set_value("analytics").run()
    assert not app.exception
    assert db.get_mastery_timeline(app.session_state["learner_id"], app.session_state["doc_id"])
    assert any("Tiến bộ theo thời gian" in str(m.value) for m in app.markdown)
    assert not [m for m in app.metric if m.label == "Tổng lượt gọi"]


def test_signed_cookie_uses_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "c.db"))
    monkeypatch.setattr(config, "_get_setting", lambda name, default="": "bi-mat-1" if name == "SESSION_SECRET" else default)
    import hashlib
    import hmac

    secret = config.get_session_secret()
    # Khóa cũ suy ra từ đường dẫn DB không còn tạo được chữ ký hợp lệ.
    old_key = hashlib.sha256(f"study-coach-session::{config.DB_PATH}".encode()).digest()
    assert secret != old_key
    assert hmac.new(secret, b"u", hashlib.sha256).hexdigest() != hmac.new(old_key, b"u", hashlib.sha256).hexdigest()
