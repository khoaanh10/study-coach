"""Kiểm thử toàn diện 4 tính năng đột phá trong Phòng Lab và Radar Điểm mù Nhận thức."""
from __future__ import annotations

from pathlib import Path

import pytest

import config
from core import adaptive, breakthrough, db
from core.contracts import Question, Quiz
from core.loader import load_default_samples, load_sections_from_pasted_text
from core.quiz import grade
from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch) -> Path:
    db_file = tmp_path / "breakthrough_test.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    db.init_db(db_file)
    return db_file


def test_cognitive_diff_detects_contradictions_and_blind_spots(temp_db: Path):
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    me_dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts], db_path=temp_db)

    rep, updated_dna = breakthrough.run_cognitive_diff(
        learner_note=breakthrough.SAMPLE_NOTES["rag_note_loi"]["content"],
        kb=kb,
        concept_map=cmap,
        learner_dna=me_dna,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=temp_db,
    )
    assert len(rep.contradictions) >= 2
    assert len(rep.blind_spots) >= 2
    assert "BẢN GHI CHÚ ĐÃ HIỆU ĐÍNH" in rep.patched_note
    assert len(updated_dna.active_misconceptions) >= 2

    rep_api, _ = breakthrough.run_cognitive_diff(
        learner_note=breakthrough.SAMPLE_NOTES["api_note_loi"]["content"],
        kb=kb,
        concept_map=cmap,
        learner_dna=updated_dna,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=temp_db,
    )
    assert len(rep_api.contradictions) >= 2


def test_incident_simulator_langgraph_and_evaluation(temp_db: Path):
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    names = [c.name for c in cmap.concepts]
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=temp_db)

    sc = breakthrough.generate_incident_scenario(text, cmap, kb, scenario_index=0, demo_mode=True)
    assert sc.ticket_id == "INC-401"
    assert {c.tool_name for c in sc.clues} == {"inspect_config", "inspect_logs", "inspect_metrics"}

    ev_ok, _ = breakthrough.evaluate_incident_resolution(
        scenario=sc,
        chosen_option_index=sc.correct_option_index,
        learner_fix="Cấu hình lại overlap > 0 và giảm top_k xuống 3-5 để tránh mất ngữ cảnh và tốn token.",
        tools_used=["inspect_config", "inspect_logs", "inspect_metrics"],
        learner_dna=intern,
        doc_id=db.DEFAULT_DOC_ID,
        concepts_order=names,
        demo_mode=True,
        db_path=temp_db,
    )
    assert ev_ok.resolved_correctly is True
    assert ev_ok.investigation_bonus == 15
    assert ev_ok.score >= 85

    # Tài liệu tùy chỉnh vẫn sinh được Ticket sự cố hợp lệ
    custom_txt = (
        "1. Hàng đợi thông điệp Kafka\n"
        "Kafka lưu trữ dữ liệu theo phân vùng partition và cho phép consumer đọc lại theo offset.\n\n"
        "2. Dead Letter Queue\n"
        "Thông điệp lỗi xử lý nhiều lần cần đưa vào DLQ để tránh làm nghẽn consumer group."
    )
    _, custom_secs = load_sections_from_pasted_text(custom_txt, "Kafka Note")
    custom_kb = DualMemoryKnowledgeBase(custom_secs, embeddings=LocalNGramEmbeddings(), fallback_text=custom_txt)
    custom_cmap = adaptive.extract_concept_map(custom_txt, "custom_kafka", "Kafka Note", demo_mode=True, db_path=temp_db)
    custom_sc = breakthrough.generate_incident_scenario(custom_txt, custom_cmap, custom_kb, scenario_index=0, demo_mode=True)
    assert len(custom_sc.clues) == 3
    assert len(custom_sc.root_cause_options) == 4


def test_trojan_bug_hunt_and_scoring(temp_db: Path):
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=temp_db)
    names = [c.name for c in cmap.concepts]
    me_dna = db.get_learner_dna("me", db.DEFAULT_DOC_ID, names, db_path=temp_db)

    ch = breakthrough.generate_trojan_challenge(text, cmap, kb, demo_mode=True)
    assert len(ch.statements) == 5
    buggy_ids = [s.id for s in ch.statements if s.is_buggy]
    assert len(buggy_ids) == 2

    res_perfect, _ = breakthrough.grade_trojan_submission(
        ch, flagged_ids=buggy_ids, learner_dna=me_dna, doc_id=db.DEFAULT_DOC_ID, concepts_order=names, db_path=temp_db
    )
    assert res_perfect["score"] == 100
    assert res_perfect["caught"] == sorted(buggy_ids)

    res_missed, dna_after = breakthrough.grade_trojan_submission(
        ch, flagged_ids=[1], learner_dna=me_dna, doc_id=db.DEFAULT_DOC_ID, concepts_order=names, db_path=temp_db
    )
    assert res_missed["score"] == 0
    assert len(dna_after.active_misconceptions) >= 2


def test_devils_advocate_on_infra_and_curriculum(temp_db: Path):
    loaded = breakthrough.load_project_infra_doc()
    assert loaded is not None
    infra_txt, infra_title = loaded
    _, secs = load_sections_from_pasted_text(infra_txt, infra_title)
    kb = DualMemoryKnowledgeBase(secs, embeddings=LocalNGramEmbeddings(), fallback_text=infra_txt)
    cmap = adaptive.extract_concept_map(infra_txt, "project_infra_doc", infra_title, demo_mode=True, db_path=temp_db)
    senior = db.get_learner_dna("senior_demo", "project_infra_doc", [c.name for c in cmap.concepts], db_path=temp_db)

    aq = breakthrough.generate_devils_advocate(infra_txt, cmap, kb, round_index=0, demo_mode=True)
    assert "MAX_INPUT_CHARS" in aq.question or "12.000" in aq.question

    ev_strong, _ = breakthrough.evaluate_defense(
        aq, aq.sample_strong_answer, kb=kb, learner_dna=senior, doc_id="project_infra_doc", demo_mode=True, db_path=temp_db
    )
    assert ev_strong.defense_score >= 75
    assert ev_strong.verdict == "Thuyết phục"

    ev_weak, _ = breakthrough.evaluate_defense(
        aq, "Em không rõ.", kb=kb, learner_dna=senior, doc_id="project_infra_doc", demo_mode=True, db_path=temp_db
    )
    assert ev_weak.defense_score < 40


def test_confidence_calibration_matrix_and_scoring(temp_db: Path):
    q1 = Question(
        question="Q1", options=["A", "B", "C", "D"], correct_index=0, explanation="E1",
        topic="Chunking và Overlap", bloom_level="Hiểu",
        option_diagnostics=["Chính xác", "Nhầm chunk lớn và nhỏ", "Suy diễn", "Khác"],
    )
    q2 = Question(
        question="Q2", options=["A", "B", "C", "D"], correct_index=1, explanation="E2",
        topic="Embeddings và Cosine", bloom_level="Nhớ",
        option_diagnostics=["Sai", "Chính xác", "Sai", "Sai"],
    )
    quiz = Quiz(questions=[q1, q2])

    # Câu 1 chọn sai nhưng tự tin 'Chắc chắn' -> Điểm mù!
    # Câu 2 chọn đúng nhưng chọn 'Đoán mò' -> Kiến thức giả!
    res = grade(quiz, [1, 1], confidences=["Chắc chắn", "Đoán mò"])
    assert res.score == 1
    assert res.blind_spot_count == 1
    assert res.lucky_guess_count == 1
    assert res.items[0].calibration_tag == "Điểm mù tự tin sai"
    assert res.items[1].calibration_tag == "Đúng do đoán mò (Kiến thức giả)"

    updated = adaptive.record_quiz_outcome(
        "me", db.DEFAULT_DOC_ID, res, model="Offline-Demo", difficulty="Trung bình",
        concepts_order=["Chunking và Overlap", "Embeddings và Cosine"], db_path=temp_db,
    )
    stats = db.get_confidence_stats("me", db.DEFAULT_DOC_ID, db_path=temp_db)
    assert stats["blind_spot"] == 1
    assert stats["lucky_guess"] == 1
    assert any("Điểm mù tự tin cao" in m.description for m in updated.active_misconceptions)
