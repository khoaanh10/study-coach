"""Kiểm thử các lỗi lõi A-H (ngân sách phiên, chào hỏi, ngoài tài liệu, Vision demo, chấm ghi chú, Trojan/Incident...).

Không gọi mạng/API thật; mọi test dùng DB tạm (tmp_path).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from core import adaptive, breakthrough, db, llm, vision
from core.coach_graph import run_coach_turn
from core.contracts import (
    ConceptNode,
    DocumentConceptMap,
    IncidentEvaluation,
    TrojanChallenge,
    TrojanStatement,
    VisionCritique,
    VisionStepCheck,
)
from core.loader import (
    _extract_sections_for_file,
    load_default_samples,
    load_sections_from_pasted_text,
)
from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch) -> Path:
    db_file = tmp_path / "core_fixes.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    db.init_db(db_file)
    return db_file


@pytest.fixture
def default_env(temp_db: Path):
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    names = [c.name for c in cmap.concepts]
    return SimpleNamespace(text=text, kb=kb, cmap=cmap, names=names, db=temp_db)


def _turn(env, query: str, learner: str = "me", history: list[dict] | None = None, **kw):
    dna = db.get_learner_dna(learner, db.DEFAULT_DOC_ID, env.names, db_path=env.db)
    return run_coach_turn(
        query,
        kb=env.kb,
        learner_dna=dna,
        concept_map=env.cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        chat_history=history,
        db_path=env.db,
        **kw,
    )


# ----------------------------------------------------------------------------- A
def _insert_call(db_path: Path, cost: float, created_at: str, is_demo: int = 0) -> None:
    db.init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO llm_calls (created_at, model, purpose, tokens_in, tokens_out, latency_ms, cost_usd, is_demo) "
        "VALUES (?, 'm', 'p', 1, 1, 1, ?, ?)",
        (created_at, cost, is_demo),
    )
    conn.commit()
    conn.close()


def test_a_old_calls_do_not_block_budget(temp_db: Path, monkeypatch):
    monkeypatch.setattr(config, "MAX_SESSION_COST_USD", 1.0)
    _insert_call(temp_db, 1.2, "2020-01-01 00:00:00")
    llm._check_budget(temp_db)  # không ném lỗi
    assert db.get_session_cost_usd(db_path=temp_db) == 0.0
    # Telemetry tổng vẫn là tổng trọn đời.
    assert db.get_telemetry_summary(temp_db)["total_cost_usd"] == pytest.approx(1.2)


def test_a_current_session_calls_block_budget(temp_db: Path, monkeypatch):
    monkeypatch.setattr(config, "MAX_SESSION_COST_USD", 1.0)
    _insert_call(temp_db, 1.2, db._now_iso())
    with pytest.raises(llm.LLMError, match="ngân sách"):
        llm._check_budget(temp_db)
    # `since` ghi đè mốc phiên.
    assert db.get_session_cost_usd(db_path=temp_db, since="2999-01-01 00:00:00") == 0.0


# ----------------------------------------------------------------------------- B
@pytest.mark.parametrize("q", ["Giá Bitcoin tăng khi nào?", "Khi nào nên mua cổ phiếu Vingroup?", "They said what about the price?"])
def test_b_substring_greeting_does_not_bypass_scope_gate(default_env, q: str):
    assert _turn(default_env, q).route_taken == "refuse_out_of_doc"


@pytest.mark.parametrize("q", ["Hi", "Xin chào bạn", "hello!"])
def test_b_real_greetings_still_work(default_env, q: str):
    assert _turn(default_env, q).route_taken == "coach_guide"


def _history_after(env, query: str) -> list[dict]:
    t1 = _turn(env, query)
    return [{"role": "user", "content": query}, {"role": "assistant", "content": t1.answer, "turn": t1}]


def test_b_followup_does_not_launder_out_of_doc_question(default_env):
    hist = _history_after(default_env, "RAG là gì?")
    t = _turn(default_env, "Cho mình ví dụ thực tế về Docker container", history=hist)
    assert t.route_taken == "refuse_out_of_doc"


def test_b_pure_followup_keeps_topic_context(default_env):
    hist = _history_after(default_env, "RAG là gì?")
    t = _turn(default_env, "Cho mình ví dụ cụ thể hơn", history=hist)
    assert t.route_taken != "refuse_out_of_doc"


def test_b_request_after_socratic_is_not_graded_as_answer(default_env):
    hist = _history_after(default_env, "Chunking và Overlap hoạt động ra sao?")
    assert hist[1]["turn"].follow_up_question.strip().endswith("?")
    before = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    t = _turn(default_env, "Mình muốn học thêm về embedding và vector", history=hist)
    assert t.route_taken != "socratic_eval"
    after = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    assert [round(c.mastery_score, 1) for c in after.concepts] == [round(c.mastery_score, 1) for c in before.concepts]
    assert len(after.active_misconceptions) == len(before.active_misconceptions)


# ----------------------------------------------------------------------------- C
OUT_OF_DOC = [
    "Hôm nay Hà Nội có mưa không?",
    "Đội tuyển Việt Nam đá trận gần nhất với đội nào?",
    "Giá cà phê xuất khẩu hôm nay là bao nhiêu?",
    "Dân số của Nhật Bản hiện nay là bao nhiêu?",
    "Công thức nấu món bún chả ngon nhất là gì?",
    "Viết Dockerfile để chạy ứng dụng Flask giúp mình",
    "Làm sao scale Kubernetes deployment lên 10 replicas?",
    "Ai là tổng thống Hoa Kỳ hiện tại?",
    "Tỷ giá USD hôm nay là bao nhiêu?",
    "Ngọn núi cao nhất thế giới nằm ở đâu?",
]
IN_DOC = [
    "Tại sao nên để các đoạn chồng lên nhau một phần khi cắt tài liệu?",
    "Vector của hai đoạn có ý nghĩa gần nhau thì sao?",
    "Trích dẫn số trang giúp ích gì cho người dùng khi đọc câu trả lời?",
    "Cần chuẩn bị bộ câu hỏi mẫu có đáp án để làm gì?",
    "Role system, user và assistant khác nhau như thế nào?",
    "Nên lưu API key ở đâu cho an toàn?",
    "Khi nào nên dùng giá trị temperature cao?",
    "Mô hình trả về đúng schema thì lợi gì cho chương trình?",
    "Vì sao chi phí tăng khi prompt dài hơn?",
    "Hạn mức chi tiêu nên được đặt ở đâu?",
]


def test_c_gold_price_is_out_of_doc(default_env):
    _, in_scope = default_env.kb.retrieve("Giá vàng hôm nay là bao nhiêu?")
    assert in_scope is False
    assert _turn(default_env, "Giá vàng hôm nay là bao nhiêu?").route_taken == "refuse_out_of_doc"


def test_c_generalization_out_of_doc(default_env):
    refused = [q for q in OUT_OF_DOC if not default_env.kb.retrieve(q)[1]]
    print(f"OOS bị từ chối: {len(refused)}/{len(OUT_OF_DOC)}; lọt: {set(OUT_OF_DOC) - set(refused)}")
    assert len(refused) >= 8
    routed = [q for q in OUT_OF_DOC if _turn(default_env, q).route_taken == "refuse_out_of_doc"]
    assert len(routed) >= 8


def test_c_generalization_in_doc(default_env):
    accepted = [q for q in IN_DOC if default_env.kb.retrieve(q)[1]]
    print(f"IN được nhận: {len(accepted)}/{len(IN_DOC)}; hụt: {set(IN_DOC) - set(accepted)}")
    assert len(accepted) >= 8


def test_c_retrieve_openai_error_becomes_llm_error(default_env, monkeypatch):
    from openai import AuthenticationError

    err = AuthenticationError.__new__(AuthenticationError)
    err.args = ("bad key sk-secret",)

    def boom(*_a, **_k):
        raise err

    monkeypatch.setattr(default_env.kb.doc_store, "similarity_search_with_score", boom)
    with pytest.raises(llm.LLMError, match="API key không hợp lệ") as ei:
        default_env.kb.retrieve("RAG là gì?")
    assert "sk-secret" not in str(ei.value)


# ----------------------------------------------------------------------------- D
def test_d_demo_vision_user_image_is_honest_and_does_not_touch_profile(default_env):
    me = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    crit, updated = vision.critique_student_image(
        image_bytes=b"not-a-real-image",
        mime_type="image/png",
        note="",
        kb=default_env.kb,
        learner_dna=me,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        sample_id=None,
        db_path=default_env.db,
    )
    assert "không đọc được" in crit.extracted_work
    assert crit.is_demo is True
    assert crit.steps == [] and crit.detected_misconception == ""
    assert "sk-live" not in crit.model_dump_json()
    assert [(c.concept, c.mastery_score) for c in updated.concepts] == [(c.concept, c.mastery_score) for c in me.concepts]
    assert len(updated.active_misconceptions) == len(me.active_misconceptions)


def test_d_demo_vision_sample_keeps_scripted_critique(default_env):
    paths = vision.ensure_sample_vision_images()
    me = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    crit, updated = vision.critique_student_image(
        image_bytes=paths["code-openai-loi"].read_bytes(),
        mime_type="image/png",
        note="",
        kb=default_env.kb,
        learner_dna=me,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        sample_id="code-openai-loi",
        db_path=default_env.db,
    )
    assert crit.is_demo is True and len(crit.steps) == 3 and crit.detected_misconception
    assert len(updated.active_misconceptions) >= 1


def _live_vision(monkeypatch, default_env, critique: VisionCritique, note: str = ""):
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    monkeypatch.setattr(vision, "analyze_image_structured", lambda **_k: critique)
    queries: list[str] = []
    real = default_env.kb.retrieve

    def spy(q, top_k=4):
        queries.append(q)
        return real(q, top_k=top_k)

    monkeypatch.setattr(default_env.kb, "retrieve", spy)
    calls: list[dict] = []
    monkeypatch.setattr(db, "update_concept_mastery", lambda **k: calls.append(k))
    me = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    vision.critique_student_image(
        b"x", "image/png", note, default_env.kb, me, db.DEFAULT_DOC_ID, demo_mode=False, db_path=default_env.db
    )
    return queries, calls, me


def _crit(steps: list[VisionStepCheck], miscon: str = "") -> VisionCritique:
    return VisionCritique(
        extracted_work="x", topic="Chunking và Overlap", overall_assessment="a", steps=steps,
        hint_for_learner="h", detected_misconception=miscon,
    )


def test_d_live_empty_steps_skip_mastery_and_query_uses_concepts(monkeypatch, default_env):
    queries, calls, _ = _live_vision(monkeypatch, default_env, _crit([]))
    assert calls == []
    assert queries and "RAG chunking OpenAI API messages temperature" not in queries[0]
    assert default_env.names[0] in queries[0]


def test_d_live_misconception_is_never_correct(monkeypatch, default_env):
    ok_step = [VisionStepCheck(step_name="s", status="Đúng", feedback="f", page_ref="[Mục 1]")]
    _, calls, _ = _live_vision(monkeypatch, default_env, _crit(ok_step, miscon="Hiểu sai overlap"), note="ghi chú")
    assert len(calls) == 1 and calls[0]["is_correct"] is False


# ----------------------------------------------------------------------------- E
GOOD_NOTES = [
    "Không nên đặt overlap bằng 0 vì ý nằm ở ranh giới hai chunk sẽ bị cắt đứt.",
    "Chunk quá lớn không làm mất ngữ cảnh mà làm lẫn nhiều ý, còn chunk quá nhỏ mới làm mất ngữ cảnh.",
]
BAD_NOTES = [
    "Overlap = 0 là tốt nhất, không cần overlap.",
    "Chunk quá lớn làm mất ngữ cảnh của câu văn.",
]


def _diff(env, note: str):
    return breakthrough._offline_cognitive_diff(note, env.kb, env.cmap, db_path=env.db)


@pytest.mark.parametrize("note", GOOD_NOTES)
def test_e_correct_notes_have_no_contradiction(default_env, note: str):
    rep = _diff(default_env, note)
    assert rep.contradictions == []
    assert rep.alignment_score > 12


@pytest.mark.parametrize("note", BAD_NOTES)
def test_e_wrong_notes_still_caught(default_env, note: str):
    assert len(_diff(default_env, note).contradictions) >= 1


@pytest.mark.parametrize("note", GOOD_NOTES)
def test_e_teach_back_correct_notes_no_misconception(default_env, note: str):
    dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    ev, _ = adaptive.evaluate_teach_back(
        "Chunking và Overlap", note, default_env.kb, dna, db.DEFAULT_DOC_ID, demo_mode=True, db_path=default_env.db
    )
    assert not [m for m in ev.misconceptions_found if "overlap" in m.lower() or "mất ngữ cảnh" in m.lower()]


@pytest.mark.parametrize("note", BAD_NOTES)
def test_e_teach_back_wrong_notes_caught(default_env, note: str):
    dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    ev, _ = adaptive.evaluate_teach_back(
        "Chunking và Overlap", note + " Đây là cách mình hiểu.", default_env.kb, dna,
        db.DEFAULT_DOC_ID, demo_mode=True, db_path=default_env.db,
    )
    assert any("overlap" in m.lower() or "mất ngữ cảnh" in m.lower() for m in ev.misconceptions_found)


def test_e_hardcoded_rag_rules_not_applied_to_other_documents(temp_db: Path):
    text = (
        "NẤU PHỞ BÒ\n\n1. Nước dùng\nHầm xương bò khoảng ba tiếng với hành nướng và gừng nướng để nước dùng trong và ngọt.\n\n"
        "2. Bánh phở\nChần bánh phở qua nước sôi trước khi cho vào tô và xếp thịt bò thái mỏng lên trên."
    )
    _, secs = load_sections_from_pasted_text(text, "pho.txt")
    kb = DualMemoryKnowledgeBase(secs, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = DocumentConceptMap(
        doc_id="pho", doc_title="Phở",
        concepts=[ConceptNode(name="Nước dùng", summary="Hầm xương", page_ref="[Mục 1]", key_terms=["xương"])],
    )
    rep = breakthrough._offline_cognitive_diff(
        "Overlap bằng 0 là tốt nhất, không cần overlap giữa các chunk. Chunk quá lớn làm mất ngữ cảnh.",
        kb, cmap, db_path=temp_db,
    )
    assert not any("overlap" in c.ground_truth.lower() or "gối lên nhau" in c.ground_truth for c in rep.contradictions)
    assert not any(c.page_ref == "[Mục 2]" and "chunk" in c.ground_truth.lower() for c in rep.contradictions)


# ----------------------------------------------------------------------------- F
def test_f_incident_correct_option_is_shuffled_deterministically(default_env):
    gen = lambda i: breakthrough.generate_incident_scenario(  # noqa: E731
        default_env.text, default_env.cmap, default_env.kb, scenario_index=i, demo_mode=True
    )
    scenarios = [gen(i) for i in range(3)]
    assert len({s.correct_option_index for s in scenarios}) > 1
    for i, sc in enumerate(scenarios):
        original = breakthrough._CURRICULUM_INCIDENTS[i]
        assert sc.root_cause_options[sc.correct_option_index] == original.root_cause_options[original.correct_option_index]
        assert sorted(sc.root_cause_options) == sorted(original.root_cause_options)
    assert [s.correct_option_index for s in scenarios] == [gen(i).correct_option_index for i in range(3)]


def _cmap(n: int) -> DocumentConceptMap:
    return DocumentConceptMap(
        doc_id="d", doc_title="T",
        concepts=[ConceptNode(name=f"Khái niệm {i}", summary=f"Tóm tắt {i}.", page_ref=f"[Mục {i}]") for i in range(1, n + 1)],
    )


def _kb(temp_db):
    text = "Tài liệu tùy chỉnh về khái niệm một và khái niệm hai.\n\n1. Mở đầu\nNội dung khái niệm."
    _, secs = load_sections_from_pasted_text(text, "x.txt")
    return text, DualMemoryKnowledgeBase(secs, embeddings=LocalNGramEmbeddings(), fallback_text=text)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
def test_f_dynamic_trojan_always_five_statements_two_traps(temp_db, n: int):
    text, kb = _kb(temp_db)
    ch = breakthrough.generate_trojan_challenge(text, _cmap(n), kb, demo_mode=True)
    assert breakthrough._is_valid_trojan(ch)
    assert len(ch.statements) == 5 and sum(s.is_buggy for s in ch.statements) == 2


def test_f_incident_dynamic_options_valid(temp_db):
    text, kb = _kb(temp_db)
    sc = breakthrough.generate_incident_scenario(text, _cmap(2), kb, demo_mode=True)
    assert breakthrough._is_valid_incident(sc)


def test_f_live_invalid_generated_data_falls_back_offline(temp_db, monkeypatch):
    text, kb = _kb(temp_db)
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    bad_trojan = TrojanChallenge(
        title="t", context_summary="c",
        statements=[TrojanStatement(id=1, topic="a", statement="s", is_buggy=False)] * 2,
    )
    monkeypatch.setattr(breakthrough, "complete_structured", lambda *_a, **_k: bad_trojan)
    ch = breakthrough.generate_trojan_challenge(text, _cmap(3), kb, demo_mode=False)
    assert breakthrough._is_valid_trojan(ch)

    from core.contracts import IncidentScenario

    bad_inc = IncidentScenario(ticket_id="X", severity="SEV-1", title="t", symptom="s", topic="t", page_ref="p",
                               root_cause_options=["a", "b"], correct_option_index=7)
    monkeypatch.setattr(breakthrough, "complete_structured", lambda *_a, **_k: bad_inc)
    sc = breakthrough.generate_incident_scenario(text, _cmap(3), kb, demo_mode=False)
    assert breakthrough._is_valid_incident(sc)


def test_f_live_incident_evaluation_is_clamped_and_synced(default_env, monkeypatch):
    sc = breakthrough.generate_incident_scenario(
        default_env.text, default_env.cmap, default_env.kb, scenario_index=0, demo_mode=True
    )
    wrong = (sc.correct_option_index + 1) % 4
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    monkeypatch.setattr(
        breakthrough, "complete_structured",
        lambda *_a, **_k: IncidentEvaluation(
            resolved_correctly=True, score=250, feedback="f", violated_principle="v", page_ref="p", post_mortem_question="q"
        ),
    )
    me = db.get_learner_dna("me", db.DEFAULT_DOC_ID, default_env.names, db_path=default_env.db)
    ev, _ = breakthrough.evaluate_incident_resolution(
        sc, wrong, "sửa gì đó", ["inspect_logs"], me, db.DEFAULT_DOC_ID, default_env.names,
        demo_mode=False, db_path=default_env.db,
    )
    assert 0 <= ev.score <= 100 and ev.resolved_correctly is False


# ----------------------------------------------------------------------------- G
def test_g_infra_doc_found_in_samples_without_documents_dir(tmp_path: Path, monkeypatch):
    root = tmp_path / "SourceCode"
    (root / "samples").mkdir(parents=True)
    (root / "samples" / "INFRA.md").write_text("# INFRA mẫu", encoding="utf-8")
    monkeypatch.setattr(config, "BASE_DIR", root)
    res = breakthrough.load_project_infra_doc()
    assert res is not None and res[0] == "# INFRA mẫu"


def test_g_infra_doc_search_order(tmp_path: Path, monkeypatch):
    root = tmp_path / "SourceCode"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "INFRA.md").write_text("docs", encoding="utf-8")
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Documents" / "INFRA.md").write_text("parent", encoding="utf-8")
    monkeypatch.setattr(config, "BASE_DIR", root)
    assert breakthrough.load_project_infra_doc()[0] == "docs"
    (root / "docs" / "INFRA.md").unlink()
    assert breakthrough.load_project_infra_doc()[0] == "parent"


# ----------------------------------------------------------------------------- H
def test_h1_tool_exception_is_returned_to_model(temp_db: Path, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test-fake")

    def make_resp(msg):
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="boom", arguments='{"bad_arg": 1}'))
    first = make_resp(SimpleNamespace(content="", tool_calls=[tc]))
    final = make_resp(SimpleNamespace(content="xong", tool_calls=None))
    responses = [first, final]
    sent: list[list[dict]] = []

    def create(**kw):
        sent.append(list(kw["messages"]))
        return responses.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "get_client", lambda: client)

    def boom(x: int) -> str:  # nhận sai tham số -> TypeError
        return str(x)

    answer, traces = llm.complete_with_tools([{"role": "user", "content": "hi"}], [], {"boom": boom}, db_path=temp_db)
    assert answer == "xong"
    assert traces and "Lỗi" in traces[0].summary
    assert any(m.get("role") == "tool" and "Lỗi" in m["content"] for m in sent[-1])


def test_h2_placement_live_filters_invalid_questions(temp_db: Path, monkeypatch):
    from core.contracts import Question, Quiz

    good = Question(question="Q1", options=["a", "b", "c", "d"], explanation="e", correct_index=1)
    bad = Question(question="Q2", options=["a", "b"], explanation="e", correct_index=5)
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    seen: dict = {}

    def fake(messages, schema, **kw):
        seen.update(kw)
        return Quiz(questions=[bad, good, bad])

    monkeypatch.setattr(adaptive, "complete_structured", fake)
    quiz = adaptive.generate_placement_test("văn bản", _cmap(2), demo_mode=False, db_path=temp_db)
    assert [q.question for q in quiz.questions] == ["Q1"]
    assert seen.get("db_path") == temp_db


def test_h3_resolve_only_on_non_guess_correct_answers(temp_db: Path):
    from core.contracts import GradedQuestion, QuizResult

    db.record_misconception("me", db.DEFAULT_DOC_ID, "Chunking", "Nhầm overlap", db_path=temp_db)

    def result(confidence: str, bloom: str) -> QuizResult:
        item = GradedQuestion(
            question="q", options=["a", "b", "c", "d"], chosen_index=0, correct_index=0, is_correct=True,
            explanation="e", topic="Chunking", bloom_level=bloom, confidence=confidence,
        )
        return QuizResult(items=[item], score=100, total=1)

    adaptive.record_quiz_outcome("me", db.DEFAULT_DOC_ID, result("Đoán mò", "Hiểu"), "m", "d", ["Chunking"], db_path=temp_db)
    adaptive.record_quiz_outcome("me", db.DEFAULT_DOC_ID, result("Bình thường", "Nhớ"), "m", "d", ["Chunking"], db_path=temp_db)
    dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, ["Chunking"], db_path=temp_db)
    assert len(dna.active_misconceptions) == 1
    adaptive.record_quiz_outcome("me", db.DEFAULT_DOC_ID, result("Bình thường", "Hiểu"), "m", "d", ["Chunking"], db_path=temp_db)
    dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, ["Chunking"], db_path=temp_db)
    assert len(dna.active_misconceptions) == 0


def test_h4_concept_match_ignores_stopwords():
    order = ["Vector store và truy xuất", "Chunking và Overlap"]
    # Chỉ trùng từ nối "và" thì không được gán nhầm sang khái niệm đầu tiên.
    assert adaptive._match_concept_name("Kiến trúc và hạ tầng", order) == "Kiến trúc và hạ tầng"
    assert adaptive._match_concept_name("Overlap và biên", order) == "Chunking và Overlap"


def test_h5_api_suffix_only_for_the_sample_api_file():
    text = "Tiêu đề\n\n1. Mục một\nNội dung một dài đủ chữ.\n\n2. Mục hai\nNội dung hai dài đủ chữ."
    assert _extract_sections_for_file("rapid-notes.txt", text)[0]["page"] == "[Mục 1]"
    assert _extract_sections_for_file("openai-api-co-ban.txt", text)[0]["page"] == "[Mục 1 · API]"



# ---------------------------------------------------------------------------
# Phản hồi ngắn của người học ("ok", "vâng", "tiếp đi") không được coi là câu hỏi ngoài tài liệu
# ---------------------------------------------------------------------------

def _chat_setup(tmp_path):
    from core import adaptive, db
    from core.loader import load_default_samples
    from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings

    db_path = str(tmp_path / "ack.db")
    combined, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=combined)
    cmap = adaptive.extract_concept_map(combined, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=db_path)
    dna = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=db_path)
    return kb, cmap, dna, db_path


def _ask(kb, cmap, dna, db_path, query, history):
    from core import db
    from core.coach_graph import run_coach_turn

    return run_coach_turn(query, kb, dna, cmap, db.DEFAULT_DOC_ID, demo_mode=True, chat_history=history, db_path=db_path)


def _history_after_first_turn(kb, cmap, dna, db_path):
    first = _ask(kb, cmap, dna, db_path, "Chunk là gì và vì sao không nên để chunk quá nhỏ?", [])
    return [
        {"role": "user", "content": "Chunk là gì"},
        {"role": "assistant", "content": first.answer, "turn": first},
    ], first


@pytest.mark.parametrize("reply", ["ok", "Ok nhé", "oke bạn", "vâng", "dạ", "hiểu rồi", "tiếp đi", "hay quá", "được rồi"])
def test_short_acknowledgement_is_not_out_of_doc(tmp_path, reply):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    history, _ = _history_after_first_turn(kb, cmap, dna, db_path)
    turn = _ask(kb, cmap, dna, db_path, reply, history)
    assert turn.route_taken == "acknowledge"
    assert "không có trong tài liệu" not in turn.answer
    assert turn.follow_up_question  # hỏi lại để kiểm tra mức hiểu thay vì cho qua


def test_acknowledgement_does_not_change_learner_profile(tmp_path):
    from core import db

    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    history, _ = _history_after_first_turn(kb, cmap, dna, db_path)
    before = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=db_path)
    _ask(kb, cmap, dna, db_path, "ok", history)
    after = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=db_path)
    assert [(c.concept, c.attempts, c.mastery_score) for c in before.concepts] == [
        (c.concept, c.attempts, c.mastery_score) for c in after.concepts
    ]
    assert len(before.active_misconceptions) == len(after.active_misconceptions)


def test_acknowledgement_without_history_asks_what_to_study(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    turn = _ask(kb, cmap, dna, db_path, "ok", [])
    assert turn.route_taken == "acknowledge"
    assert "không có trong tài liệu" not in turn.answer


def test_not_understood_explains_again_instead_of_refusing(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    history, _ = _history_after_first_turn(kb, cmap, dna, db_path)
    for reply in ("chưa hiểu", "không hiểu", "chưa rõ"):
        turn = _ask(kb, cmap, dna, db_path, reply, history)
        assert turn.route_taken == "adaptive_explain", reply


def test_out_of_doc_question_after_acknowledgement_flow_is_still_refused(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    history, _ = _history_after_first_turn(kb, cmap, dna, db_path)
    for q in ("Giá vàng hôm nay là bao nhiêu?", "Thủ đô của Pháp là gì?"):
        assert _ask(kb, cmap, dna, db_path, q, history).route_taken == "refuse_out_of_doc", q


def test_longer_real_messages_are_not_swallowed_as_acknowledgement():
    from core.coach_graph import _is_acknowledgement

    assert _is_acknowledgement("ok") == "ack"
    assert _is_acknowledgement("có chứ") == "ack"
    assert _is_acknowledgement("chắc chắn") == "ack"
    assert _is_acknowledgement("chưa hiểu") == "confused"
    for q in ("ok, giải thích embeddings đi", "ok vậy cosine similarity là gì?", "tiếp tục giải thích về chunk nhỏ hơn nữa nhé"):
        assert _is_acknowledgement(q) == "", q


def test_short_yes_no_reply_to_socratic_question_is_not_refused(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    first_turn = _ask(kb, cmap, dna, db_path, "Hãy giảng cho tôi chủ đề 'RAG là gì' theo đúng trình độ hiện tại của tôi.", [])
    history = [
        {"role": "user", "content": "Hãy giảng cho tôi chủ đề 'RAG là gì' theo đúng trình độ hiện tại của tôi."},
        {
            "role": "assistant",
            "content": first_turn.answer,
            "turn": first_turn.model_copy(
                update={
                    "follow_up_question": "Bạn có nghĩ rằng việc sử dụng thông tin từ tài liệu bên ngoài có thể cải thiện độ chính xác của câu trả lời không?",
                }
            ),
        },
    ]
    for reply in ("có chứ", "không ạ"):
        turn = _ask(kb, cmap, dna, db_path, reply, history)
        assert turn.route_taken == "acknowledge", f"{reply} -> {turn.route_taken}"
        assert "ngoài tài liệu" not in turn.answer.lower()
        assert "không có trong tài liệu" not in turn.answer.lower()


def test_llm_intent_router_handles_natural_reply_and_fallbacks(tmp_path, monkeypatch):
    import config
    from core import coach_graph

    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    first_turn = _ask(kb, cmap, dna, db_path, "RAG là gì?", [])
    history = [
        {"role": "user", "content": "RAG là gì?"},
        {
            "role": "assistant",
            "content": first_turn.answer,
            "turn": first_turn.model_copy(
                update={
                    "follow_up_question": "Bạn có nghĩ rằng việc sử dụng tài liệu ngoài giúp giảm bịa thông tin không?",
                }
            ),
        },
    ]

    monkeypatch.setattr(config, "has_live_api_key", lambda: True)

    # 1. Live mode: LLM Intent Router bắt câu trả lời tự nhiên ngoài regex ("mình đoán là có")
    calls: list[dict] = []

    def _fake_structured(messages, schema, **kwargs):
        calls.append({"messages": messages, "schema": schema, **kwargs})
        return coach_graph._IntentRouterOutput(
            intent="confirm_answer",
            matched_topic="RAG là gì",
            reason="Học viên trả lời có/không cho câu hỏi Socratic",
        )

    monkeypatch.setattr(coach_graph, "complete_structured", _fake_structured)
    turn = coach_graph.run_coach_turn(
        kb=kb,
        concept_map=cmap,
        learner_dna=dna,
        query="mình đoán là có",
        doc_id="doc-test",
        chat_history=history,
        db_path=db_path,
        demo_mode=False,
    )
    assert len(calls) == 1
    assert turn.route_taken == "acknowledge"
    assert any(t.tool_name == "classify_intent_llm" for t in turn.tool_traces)

    # 2. Khi complete_structured lỗi -> fallback êm về rule-based, không crash
    def _failing_structured(*args, **kwargs):
        raise RuntimeError("rate limit")

    monkeypatch.setattr(coach_graph, "complete_structured", _failing_structured)
    fallback_turn = coach_graph.run_coach_turn(
        kb=kb,
        concept_map=cmap,
        learner_dna=dna,
        query="có chứ",
        doc_id="doc-test",
        chat_history=history,
        db_path=db_path,
        demo_mode=False,
    )
    assert fallback_turn.route_taken == "acknowledge"

    # 3. Khi demo_mode=True -> không gọi complete_structured
    called_in_demo = False

    def _should_not_call(*args, **kwargs):
        nonlocal called_in_demo
        called_in_demo = True
        raise AssertionError("complete_structured must not be called in demo_mode")

    monkeypatch.setattr(coach_graph, "complete_structured", _should_not_call)
    demo_turn = coach_graph.run_coach_turn(
        kb=kb,
        concept_map=cmap,
        learner_dna=dna,
        query="có chứ",
        doc_id="doc-test",
        chat_history=history,
        db_path=db_path,
        demo_mode=True,
    )
    assert not called_in_demo
    assert demo_turn.route_taken == "acknowledge"



# ---------------------------------------------------------------------------
# "Ôn phần X" phải ra phần ôn tập, không được biến thành chẩn đoán lỗi tư duy hay từ chối
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, expected_topic",
    [
        ("Ôn phần chunking cho mình", "Chunking và Overlap"),
        ("ôn lại phần chunk và overlap", "Chunking và Overlap"),
        ("Mình muốn ôn tập embeddings", "Embeddings và Cosine"),
        ("on tap cau truc loi goi messages giup minh", "Cấu trúc lời gọi & Messages"),
        ("củng cố phần RAG là gì", "RAG là gì"),
    ],
)
def test_review_request_gives_review_not_misconception_diagnosis(tmp_path, query, expected_topic):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    turn = _ask(kb, cmap, dna, db_path, query, [])
    assert turn.route_taken == "adaptive_explain"
    assert f"Ôn nhanh `{expected_topic}`" in turn.answer
    assert "Mình phát hiện trong hồ sơ" not in turn.answer  # không chất vấn lỗi tư duy
    assert "Trước khi đi sâu vào" not in turn.answer  # không lùi sang kiến thức nền
    assert turn.follow_up_question


def test_review_request_without_topic_suggests_weakest_topic(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    turn = _ask(kb, cmap, dna, db_path, "ôn phần nào đó đi", [])
    assert turn.route_taken == "adaptive_explain"
    assert "Ôn nhanh `" in turn.answer
    assert "mình chọn" in turn.answer  # nói rõ vì sao chọn phần này
    assert "không có trong tài liệu" not in turn.answer


def test_review_excerpt_comes_from_the_requested_section(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    turn = _ask(kb, cmap, dna, db_path, "Ôn phần chunking cho mình", [])
    assert "[Mục 2]" in turn.answer and "[Mục 2 · API]" not in turn.answer
    assert "overlap" in turn.answer.lower()


def test_review_request_for_topic_not_in_document_is_refused(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    turn = _ask(kb, cmap, dna, db_path, "Ôn phần Kubernetes cho mình", [])
    assert turn.route_taken == "refuse_out_of_doc"


def test_normal_question_on_topic_with_open_misconception_still_gets_remediation(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    turn = _ask(kb, cmap, dna, db_path, "Vì sao chunk quá nhỏ lại mất ngữ cảnh?", [])
    assert turn.route_taken == "socratic_remediation"


def test_review_request_does_not_change_learner_profile(tmp_path):
    from core import db

    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    names = [c.name for c in cmap.concepts]
    before = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_path)
    _ask(kb, cmap, dna, db_path, "Ôn phần chunking cho mình", [])
    after = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_path)
    assert [(c.concept, c.attempts, c.mastery_score) for c in before.concepts] == [
        (c.concept, c.attempts, c.mastery_score) for c in after.concepts
    ]


def test_review_detection_does_not_trigger_on_english_on():
    from core.coach_graph import _is_review_request

    assert _is_review_request("ôn phần chunking")
    assert _is_review_request("on tap embeddings")
    assert not _is_review_request("rate limit on the api key là gì?")
    assert not _is_review_request("Overlap là gì?")



# ---------------------------------------------------------------------------
# Cổng ngoài-tài-liệu phải giữ nguyên khi dùng embedding "thật" có thang điểm cosine cao hơn embedding cục bộ
# ---------------------------------------------------------------------------

class _HighBaselineEmbeddings:
    """Giả lập embedding OpenAI: mọi vector chia sẻ một thành phần chung nên cosine giữa 2 văn bản bất kỳ đều cao."""

    def __init__(self):
        from core.rag import LocalNGramEmbeddings

        self._local = LocalNGramEmbeddings()
        self._const = [1.0] + [0.0] * (self._local.dim - 1)

    def _mix(self, v):
        import math

        mixed = [0.7 * c + 0.7 * x for c, x in zip(self._const, v)]
        norm = math.sqrt(sum(x * x for x in mixed)) or 1.0
        return [x / norm for x in mixed]

    def embed_query(self, text):
        return self._mix(self._local.embed_query(text))

    def embed_documents(self, texts):
        return [self._mix(self._local.embed_query(x)) for x in texts]

    def __call__(self, text):  # FAISS có thể gọi embedding_function trực tiếp
        return self.embed_query(text)


def _kb_with_high_baseline_embeddings():
    from langchain_core.embeddings import Embeddings

    from core.loader import load_default_samples
    from core.rag import DualMemoryKnowledgeBase

    class _Emb(_HighBaselineEmbeddings, Embeddings):
        pass

    combined, sections, _ = load_default_samples()
    return DualMemoryKnowledgeBase(sections, embeddings=_Emb(), fallback_text=combined)


def test_scope_gate_is_independent_of_embedding_score_scale():
    kb = _kb_with_high_baseline_embeddings()
    assert kb.scope_store is not None  # embedding không phải bản cục bộ -> có cổng riêng
    out_of_doc = [
        "Giá vàng hôm nay là bao nhiêu?",
        "Công thức nấu phở bò gồm những gì?",
        "Làm sao để học tiếng Anh giao tiếp nhanh?",
        "Thủ tục đăng ký tạm trú ở Việt Nam gồm những gì?",
        "Khi nào nên mua cổ phiếu Vingroup?",
        "Ai vô địch World Cup 2022?",
    ]
    for q in out_of_doc:
        chunks, in_scope = kb.retrieve(q, top_k=3)
        assert chunks and chunks[0].score > 0.4, "giả lập phải có điểm cao như embedding thật"
        assert in_scope is False, q
    in_doc = [
        "Cosine similarity dùng để làm gì?",
        "Tham số max tokens dùng để làm gì?",
        "Mô hình có tự nhớ cuộc trò chuyện trước đó không?",
        "Chia nhỏ tài liệu thành đoạn để làm gì?",
    ]
    for q in in_doc:
        assert kb.retrieve(q, top_k=3)[1] is True, q


def test_local_embeddings_do_not_build_a_second_scope_store():
    from core.loader import load_default_samples
    from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings

    combined, sections, _ = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=combined)
    assert kb.scope_store is None



# ---------------------------------------------------------------------------
# Lời chào Live không bịa chủ đề; hỏi nối tiếp sau khi bot lùi về kiến thức nền vẫn đúng chủ đề người học hỏi
# ---------------------------------------------------------------------------

def test_live_prompt_lists_document_concepts_so_the_model_cannot_invent_topics(tmp_path):
    from core.coach_graph import _build_coach_openai_messages, _make_init_state
    from core.prompts import SOCRATIC_COACH_PROMPTS

    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    state = _make_init_state("Xin chào", dna, cmap, "default_curriculum", demo_mode=False)
    msgs = _build_coach_openai_messages(state, "coach_guide", "Hãy chào học viên")
    user_text = msgs[-1]["content"]
    for node in cmap.concepts[:3]:
        assert node.name in user_text
    assert "chỉ được nhắc tới các chủ đề này" in user_text
    assert "không được bịa" in SOCRATIC_COACH_PROMPTS["coach_guide"]


def test_followup_after_prerequisite_bridge_keeps_the_topic_the_learner_asked_about(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    first = _ask(kb, cmap, dna, db_path, "Giải thích embedding là gì", [])
    # Hồ sơ mẫu của Minh yếu "Chunking và Overlap" nên bot lùi về kiến thức nền; chủ đề người học hỏi vẫn là Embeddings.
    assert first.route_taken == "bridge_prerequisite"
    assert first.suggested_quiz_topic == "Chunking và Overlap"
    assert first.topic == "Embeddings và Cosine"
    history = [
        {"role": "user", "content": "Giải thích embedding là gì"},
        {"role": "assistant", "content": first.answer, "turn": first},
    ]
    second = _ask(kb, cmap, dna, db_path, "Cho mình ví dụ thực tế đi", history)
    assert second.route_taken == "adaptive_explain"
    assert second.topic == "Embeddings và Cosine"
    assert "Embeddings và Cosine" in second.answer



def test_thanks_is_a_conversational_reply_not_a_new_greeting(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    history, _ = _history_after_first_turn(kb, cmap, dna, db_path)
    for msg in ("cảm ơn nhé", "Thanks bạn", "cảm ơn nhiều", "Cảm ơn thầy"):
        turn = _ask(kb, cmap, dna, db_path, msg, history)
        assert turn.route_taken == "acknowledge", msg
        assert "Không có gì" in turn.answer
        assert "Chào" not in turn.answer
    # Lời chào thật vẫn là chào hỏi
    assert _ask(kb, cmap, dna, db_path, "xin chào", []).route_taken == "coach_guide"


def test_topic_detection_prefers_words_from_the_concept_name(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    turn = _ask(kb, cmap, dna, db_path, "Mình muốn tìm hiểu về vector store", [])
    assert turn.topic == "Vector store và truy xuất"



def test_next_question_after_socratic_turn_is_acknowledged_with_a_new_question(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    first_turn = _ask(kb, cmap, dna, db_path, "Chunking và Overlap hoạt động ra sao?", [])
    last_follow = first_turn.follow_up_question.strip()
    assert last_follow.endswith("?")
    history = [
        {"role": "user", "content": "Chunking và Overlap hoạt động ra sao?"},
        {"role": "assistant", "content": first_turn.answer, "turn": first_turn},
    ]
    turn = _ask(kb, cmap, dna, db_path, "hỏi câu khác", history)
    assert turn.route_taken == "acknowledge"
    assert "không có trong tài liệu" not in turn.answer.lower()
    assert turn.follow_up_question.strip().endswith("?")
    assert turn.follow_up_question.strip() != last_follow


def test_repeated_yes_after_acknowledge_does_not_repeat_previous_turn(tmp_path):
    kb, cmap, dna, db_path = _chat_setup(tmp_path)
    first_turn = _ask(kb, cmap, dna, db_path, "Chunking và Overlap hoạt động ra sao?", [])
    yes_no_follow = "Bạn có nghĩ rằng việc để các đoạn chồng lên nhau giúp giữ ngữ cảnh không?"
    history = [
        {"role": "user", "content": "Chunking và Overlap hoạt động ra sao?"},
        {
            "role": "assistant",
            "content": first_turn.answer,
            "turn": first_turn.model_copy(update={"follow_up_question": yes_no_follow}),
        },
    ]
    turn1 = _ask(kb, cmap, dna, db_path, "có", history)
    assert turn1.route_taken == "acknowledge"
    history2 = history + [
        {"role": "user", "content": "có"},
        {"role": "assistant", "content": turn1.answer, "turn": turn1},
    ]
    turn2 = _ask(kb, cmap, dna, db_path, "có", history2)
    assert turn2.route_taken == "acknowledge"
    assert turn2.answer != turn1.answer
    assert turn2.follow_up_question != turn1.follow_up_question
