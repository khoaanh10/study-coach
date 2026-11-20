"""Kiểm thử toàn diện cho Study Coach 2.0 (DB, Dual-Memory RAG, LangGraph, Adaptive Quiz, Feynman, Vision)."""
from __future__ import annotations

from pathlib import Path

import pytest
from openai import AuthenticationError, RateLimitError

import config
from core import adaptive, db, llm, vision
from core.coach_graph import run_coach_turn
from core.contracts import Question, Quiz
from core.loader import load_default_samples
from core.quiz import generate_adaptive_quiz, generate_quiz, grade
from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings, compute_coverage_stats


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch) -> Path:
    db_file = tmp_path / "test_study_coach.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    db.init_db(db_file)
    return db_file


def test_db_seed_and_learner_dna_evolution(temp_db: Path):
    learners = db.list_learners(temp_db)
    ids = {x["learner_id"] for x in learners}
    assert {"me", "intern_demo", "senior_demo"}.issubset(ids)

    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, db_path=temp_db)
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, db_path=temp_db)
    assert intern.overall_level == "Mới bắt đầu"
    assert senior.overall_level == "Nâng cao"
    assert len(intern.active_misconceptions) >= 2
    assert senior.mastery_avg > intern.mastery_avg

    # Cập nhật điểm thành thạo cho 'me' và kiểm tra resolve misconception.
    db.record_misconception("me", db.DEFAULT_DOC_ID, "Chunking", "Nhầm overlap = 0", db_path=temp_db)
    db.update_concept_mastery("me", db.DEFAULT_DOC_ID, "Chunking", is_correct=True, bloom_level="Vận dụng", db_path=temp_db)
    resolved_cnt = db.resolve_misconception("me", db.DEFAULT_DOC_ID, "Chunking", db_path=temp_db)
    assert resolved_cnt == 1

    me_dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, db_path=temp_db)
    assert len(me_dna.active_misconceptions) == 0
    assert len(me_dna.resolved_misconceptions) == 1


def test_dual_memory_rag_retrieval_and_out_of_scope(temp_db: Path):
    text, sections, _ = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)

    chunks_cost, in_scope = kb.retrieve("Tìm hiểu về chi phí và giá thành khi dùng embeddings", top_k=3)
    assert in_scope is True
    assert len(chunks_cost) >= 1
    assert any(c.source == "rag-co-ban.txt" for c in chunks_cost)

    _, oos = kb.retrieve("Kỹ thuật LoRA Fine-tuning cập nhật ma trận hạng thấp như thế nào?", top_k=3)
    assert oos is False

    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, db_path=temp_db)
    kb.sync_learner_mistakes(intern)
    mistakes = kb.retrieve_past_mistakes("chunk quá lớn bị mất ngữ cảnh phải không?", top_k=2)
    assert len(mistakes) >= 1
    assert any("Chunking" in m.topic for m in mistakes)

    cov = compute_coverage_stats(intern.concepts)
    assert cov["total"] > 0
    assert cov["misconception"] >= 1


def test_adaptive_quiz_diagnostics_and_custom_text(temp_db: Path):
    text, sections, title = load_default_samples()
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    concept_names = [c.name for c in cmap.concepts]
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, concepts_order=concept_names, db_path=temp_db)

    quiz, strategy = generate_adaptive_quiz(text, intern, n=4, demo_mode=True)
    assert len(quiz.questions) == 4
    assert "Chunking" in strategy

    # Cố tình chọn đáp án sai để kiểm tra bắt bệnh tư duy.
    wrong_picks = [
        (q.correct_index + 1) % 4 if i == 0 else q.correct_index
        for i, q in enumerate(quiz.questions)
    ]
    res = grade(quiz, wrong_picks)
    assert res.score == len(quiz.questions) - 1
    assert res.items[0].diagnosed_misconception != ""
    assert len(res.detected_misconceptions) >= 1

    updated = adaptive.record_quiz_outcome(
        "me", db.DEFAULT_DOC_ID, res, model="Offline-Demo", difficulty="Trung bình", concepts_order=concept_names, db_path=temp_db
    )
    assert len(db.recent_quiz_attempts("me", db_path=temp_db)) == 1
    assert any(c.attempts > 0 for c in updated.concepts)

    # Test sinh quiz Offline Demo trên tài liệu tùy chỉnh mới hoàn toàn.
    custom_text = (
        "1. Mạng nơ-ron tích chập CNN\n"
        "CNN sử dụng các bộ lọc tích chập để tự động trích xuất đặc trưng không gian từ ảnh đầu vào. "
        "Lớp Pooling giúp giảm kích thước bản đồ đặc trưng để hạn chế quá khớp và tiết kiệm tính toán.\n\n"
        "2. Hàm kích hoạt ReLU\n"
        "ReLU biến các giá trị âm thành 0 và giữ nguyên giá trị dương, giúp mạng hội tụ nhanh hơn hàm Sigmoid."
    )
    custom_quiz = generate_quiz(custom_text, n=3, difficulty="Trung bình", demo_mode=True)
    assert len(custom_quiz.questions) >= 2
    assert all(len(q.options) == 4 for q in custom_quiz.questions)


def test_concept_map_placement_teach_back_and_study_plan(temp_db: Path):
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    assert len(cmap.concepts) >= 5

    placement = adaptive.generate_placement_test(text, cmap, demo_mode=True)
    assert len(placement.questions) >= 5

    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=temp_db)
    ev_bad, dna_after_bad = adaptive.evaluate_teach_back(
        concept="Chunking và Overlap",
        learner_answer="Chunk quá lớn làm mất ngữ cảnh nên ta để overlap bằng 0.",
        kb=kb,
        learner_dna=intern,
        doc_id=db.DEFAULT_DOC_ID,
        concepts_order=[c.name for c in cmap.concepts],
        demo_mode=True,
        db_path=temp_db,
    )
    assert ev_bad.understanding_score < 60
    assert len(ev_bad.misconceptions_found) >= 1
    assert ev_bad.socratic_follow_up != ""

    ev_good, _ = adaptive.evaluate_teach_back(
        concept="Chunking và Overlap",
        learner_answer=(
            "Tài liệu dài được chia thành các chunk nhỏ. Chunk quá lớn làm truy xuất kém chính xác vì chứa nhiều ý lẫn lộn, "
            "còn chunk quá nhỏ làm mất ngữ cảnh. Các chunk cần gối lên nhau gọi là overlap để ý ở ranh giới không bị cắt đứt "
            "và lưu kèm metadata tên tài liệu cùng số trang để trích nguồn."
        ),
        kb=kb,
        learner_dna=dna_after_bad,
        doc_id=db.DEFAULT_DOC_ID,
        concepts_order=[c.name for c in cmap.concepts],
        demo_mode=True,
        db_path=temp_db,
    )
    assert ev_good.understanding_score >= 80
    assert ev_good.verdict == "Vững bản chất"

    plan = adaptive.build_study_plan(intern, cmap)
    assert len(plan) == len(cmap.concepts)
    assert plan[0]["rank"] <= plan[-1]["rank"]


def test_langgraph_coach_four_routes_and_adaptation(temp_db: Path):
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    names = [c.name for c in cmap.concepts]

    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=temp_db)
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=temp_db)

    # Nhánh 1: Từ chối ngoài tài liệu
    t_oos = run_coach_turn(
        "Cách cấu hình Kubernetes Pod cho GPU cluster?",
        kb=kb,
        learner_dna=intern,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=temp_db,
    )
    assert t_oos.route_taken == "refuse_out_of_doc"

    # Nhánh 2: Socratic remediation (intern có misconception ở Chunking)
    t_rem = run_coach_turn(
        "Chunking và Overlap hoạt động ra sao?",
        kb=kb,
        learner_dna=intern,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=temp_db,
    )
    assert t_rem.route_taken == "socratic_remediation"
    assert len(t_rem.tool_traces) >= 3

    # Nhánh 3: Bridge prerequisite (intern yếu Embeddings 35% -> hỏi Vector store)
    t_pre = run_coach_turn(
        "Vector store và truy xuất top-k hoạt động thế nào?",
        kb=kb,
        learner_dna=intern,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        focus_topic="Vector store và truy xuất",
        demo_mode=True,
        db_path=temp_db,
    )
    assert t_pre.route_taken == "bridge_prerequisite"
    assert "Embeddings" in t_pre.answer

    # Nhánh 4: Adaptive explain — cùng câu hỏi Chunking nhưng 'me' (Mới bắt đầu) vs 'senior' (Nâng cao) khác hẳn
    me_dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, names, db_path=temp_db)
    t_beg = run_coach_turn(
        "Chunking và Overlap là gì?",
        kb=kb,
        learner_dna=me_dna,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        focus_topic="Chunking và Overlap",
        demo_mode=True,
        db_path=temp_db,
    )
    t_adv = run_coach_turn(
        "Chunking và Overlap là gì?",
        kb=kb,
        learner_dna=senior,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        focus_topic="Chunking và Overlap",
        demo_mode=True,
        db_path=temp_db,
    )
    assert t_beg.route_taken == "adaptive_explain"
    assert t_adv.route_taken == "adaptive_explain"
    assert "Mới bắt đầu" in t_beg.answer
    assert "Nâng cao" in t_adv.answer
    assert t_beg.answer != t_adv.answer


def test_vision_critique_and_sample_images(temp_db: Path):
    paths = vision.ensure_sample_vision_images()
    assert paths["so-do-rag-loi"].exists()
    assert paths["code-openai-loi"].exists()

    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    me_dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=temp_db)

    crit, updated = vision.critique_student_image(
        image_bytes=paths["so-do-rag-loi"].read_bytes(),
        mime_type="image/png",
        note="Chấm sơ đồ RAG",
        kb=kb,
        learner_dna=me_dna,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        sample_id="so-do-rag-loi",
        db_path=temp_db,
    )
    assert len(crit.steps) == 3
    assert any(s.status == "Sai" for s in crit.steps)
    assert len(updated.active_misconceptions) >= 1


def test_llm_retry_and_no_retry_on_auth(temp_db: Path, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)

    calls = {"n": 0}

    class DummyMsg:
        refusal = None
        parsed = Quiz(questions=[Question(question="Q", options=["1", "2", "3", "4"], explanation="E", correct_index=0)])

    class DummyChoice:
        message = DummyMsg()

    class DummyUsage:
        prompt_tokens = 120
        completion_tokens = 80

    class DummyResp:
        choices = [DummyChoice()]
        usage = DummyUsage()

    def flaky_parse(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError.__new__(RateLimitError)
        return DummyResp()

    class DummyCompletions:
        parse = staticmethod(flaky_parse)

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    monkeypatch.setattr(llm, "get_client", lambda: DummyClient())
    res = llm.complete_structured([{"role": "user", "content": "hi"}], Quiz, db_path=temp_db)
    assert calls["n"] == 2
    assert len(res.questions) == 1

    # AuthenticationError không được retry
    auth_calls = {"n": 0}

    def auth_fail(**kwargs):
        auth_calls["n"] += 1
        raise AuthenticationError.__new__(AuthenticationError)

    DummyCompletions.parse = staticmethod(auth_fail)
    with pytest.raises(llm.LLMError, match="API key không hợp lệ"):
        llm.complete_structured([{"role": "user", "content": "hi"}], Quiz, db_path=temp_db)
    assert auth_calls["n"] == 1
