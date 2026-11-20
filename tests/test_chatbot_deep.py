"""Kiểm thử chuyên sâu Chatbot Socratic (LangGraph + Dual-Memory RAG + Đa lượt hội thoại)."""
from __future__ import annotations

from pathlib import Path

import pytest

import config
from core import adaptive, db, llm
from core.coach_graph import _extract_trailing_question, run_coach_turn, stream_coach_turn
from core.loader import load_default_samples, load_sections_from_pasted_text
from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings


@pytest.fixture
def chat_env(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "chatbot_deep.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    db.init_db(db_file)
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=db_file)
    names = [c.name for c in cmap.concepts]
    return db_file, kb, cmap, names


def test_01_greeting_and_profile_meta_queries(chat_env):
    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    t1 = run_coach_turn("Xin chào", kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, db_path=db_file)
    t2 = run_coach_turn("Bạn là ai, tôi đang yếu phần nào?", kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, db_path=db_file)

    for turn in (t1, t2):
        assert turn.route_taken == "coach_guide"
        assert "Minh — Intern mới học" in turn.answer
        assert "Chunking và Overlap" in turn.answer
        assert turn.follow_up_question != ""


def test_02_document_summary_request(chat_env):
    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    turn = run_coach_turn(
        "Tóm tắt toàn bộ tài liệu này giúp mình",
        kb=kb,
        learner_dna=intern,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=db_file,
    )
    assert turn.route_taken == "coach_guide"
    assert "RAG là gì" in turn.answer
    assert "Chunking và Overlap" in turn.answer
    assert len(turn.citations) >= 3


def test_03_unaccented_vietnamese_queries_grounded(chat_env):
    db_file, kb, cmap, names = chat_env
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    q_cases = [
        ("mo hinh ngon ngu lon va rag la gi", "RAG là gì"),
        ("tai sao phai gui lai lich su hoi thoai", "Mục 2"),
        ("cach danh gia chat luong he thong", "Mục 6"),
    ]
    for q, expected_hint in q_cases:
        turn = run_coach_turn(q, kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, db_path=db_file)
        assert turn.route_taken != "refuse_out_of_doc", f"Từ chối nhầm câu tiếng Việt không dấu: {q}"
        assert expected_hint in turn.answer or any(expected_hint in c for c in turn.citations)


def test_04_out_of_scope_refusal_accented_and_unaccented(chat_env):
    db_file, kb, cmap, names = chat_env
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    oos_queries = [
        "Docker container va Kubernetes pod khac gi nhau",
        "Giá cổ phiếu Bitcoin hôm nay bao nhiêu?",
        "LoRA fine-tuning cap nhat trong so the nao",
    ]
    for q in oos_queries:
        turn = run_coach_turn(q, kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, db_path=db_file)
        assert turn.route_taken == "refuse_out_of_doc", f"Lọt câu ngoài tài liệu: {q}"
        assert turn.citations == []


def test_05_clean_sentence_excerpts_without_midword_cutoff(chat_env):
    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    turn = run_coach_turn(
        "Tại sao đặt overlap bằng 0 khi chia chunk lại nguy hiểm?",
        kb=kb,
        learner_dna=intern,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=db_file,
    )
    assert turn.route_taken == "socratic_remediation"
    assert "ở r..." not in turn.answer
    assert "không bị cắt đứt." in turn.answer


def test_06_multi_turn_contextual_followups(chat_env):
    db_file, kb, cmap, names = chat_env
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    hist: list[dict] = []
    t1 = run_coach_turn("RAG là gì?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=hist, db_path=db_file)
    hist.extend([{"role": "user", "content": "RAG là gì?"}, {"role": "assistant", "content": t1.answer, "turn": t1}])

    t2 = run_coach_turn("Giải thích kỹ hơn đi", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=hist, db_path=db_file)
    hist.extend([{"role": "user", "content": "Giải thích kỹ hơn đi"}, {"role": "assistant", "content": t2.answer, "turn": t2}])

    t3 = run_coach_turn("Cho mình một ví dụ thực tế được không?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=hist, db_path=db_file)

    assert t1.route_taken == "adaptive_explain"
    assert t2.route_taken == "adaptive_explain"
    assert t3.route_taken == "adaptive_explain"
    assert t1.suggested_quiz_topic == "RAG là gì"
    assert t2.suggested_quiz_topic == "RAG là gì"
    assert t3.suggested_quiz_topic == "RAG là gì"
    assert len({t1.answer, t2.answer, t3.answer}) == 3
    assert "Ví dụ" in t3.answer


def test_07_and_08_socratic_answer_eval_and_dna_resolution(chat_env):
    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)
    before_active = len([m for m in intern.active_misconceptions if m.topic == "Chunking và Overlap"])
    assert before_active >= 1

    # Lượt 1: Gia sư bắt bệnh tư duy Chunking và đặt câu hỏi vặn
    t1 = run_coach_turn(
        "Chunking và Overlap hoạt động ra sao?",
        kb=kb,
        learner_dna=intern,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=db_file,
    )
    assert t1.route_taken == "socratic_remediation"
    hist = [{"role": "user", "content": "Chunking và Overlap hoạt động ra sao?"}, {"role": "assistant", "content": t1.answer, "turn": t1}]

    # Kịch bản 7: Học viên trả lời sai -> socratic_eval phát hiện lỗi
    wrong_reply = "Theo mình đặt overlap = 0 là tốt nhất vì đỡ tốn tiền và không cần gửi metadata."
    t_wrong = run_coach_turn(wrong_reply, kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=hist, db_path=db_file)
    assert t_wrong.route_taken == "socratic_eval"
    assert "Đang hiểu lệch" in t_wrong.answer or "15/100" in t_wrong.answer

    # Kịch bản 8: Học viên sửa đúng bản chất -> socratic_eval >= 80 & tự động gỡ misconception trong SQLite
    hist.extend([{"role": "user", "content": wrong_reply}, {"role": "assistant", "content": t_wrong.answer, "turn": t_wrong}])
    good_reply = (
        "Tài liệu dài được chia thành các chunk nhỏ. Chunk quá lớn làm truy xuất kém chính xác vì chứa nhiều ý lẫn lộn, "
        "còn chunk quá nhỏ làm mất ngữ cảnh. Các chunk cần gối lên nhau gọi là overlap để ý ở ranh giới không bị cắt đứt "
        "và lưu metadata tên tài liệu cùng số trang để trích nguồn."
    )
    t_good = run_coach_turn(good_reply, kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=hist, db_path=db_file)
    assert t_good.route_taken == "socratic_eval"
    assert "Vững bản chất" in t_good.answer

    intern_after = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)
    after_active = len([m for m in intern_after.active_misconceptions if m.topic == "Chunking và Overlap"])
    assert after_active == 0
    assert any(m.topic == "Chunking và Overlap" for m in intern_after.resolved_misconceptions)


def test_09_adaptive_tone_across_three_profiles(chat_env):
    db_file, kb, cmap, names = chat_env
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    # Tạo 1 hồ sơ Trung bình
    db.update_concept_mastery("me", db.DEFAULT_DOC_ID, "RAG là gì", score_override=65.0, bloom_level="Hiểu", db_path=db_file)
    db.update_concept_mastery("me", db.DEFAULT_DOC_ID, "Embeddings và Cosine", score_override=60.0, bloom_level="Hiểu", db_path=db_file)
    db.update_concept_mastery("me", db.DEFAULT_DOC_ID, "Vector store và truy xuất", score_override=55.0, bloom_level="Hiểu", db_path=db_file)
    me_mid = db.get_learner_dna("me", db.DEFAULT_DOC_ID, names, db_path=db_file)

    # Dùng lại một DB sạch cho hồ sơ Mới bắt đầu thực sự
    intern_clean = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    t_beg = run_coach_turn("RAG là gì?", kb=kb, learner_dna=intern_clean, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, focus_topic="RAG là gì", demo_mode=True, db_path=db_file)
    t_mid = run_coach_turn("RAG là gì?", kb=kb, learner_dna=me_mid, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, focus_topic="RAG là gì", demo_mode=True, db_path=db_file)
    t_adv = run_coach_turn("RAG là gì?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, focus_topic="RAG là gì", demo_mode=True, db_path=db_file)

    assert len({t_beg.answer, t_mid.answer, t_adv.answer}) == 3
    assert "Mới bắt đầu" in t_beg.answer
    assert "Nâng cao" in t_adv.answer


def test_10_bridge_prerequisite_route(chat_env):
    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    turn = run_coach_turn(
        "Vector store và top-k hoạt động thế nào?",
        kb=kb,
        learner_dna=intern,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        focus_topic="Vector store và truy xuất",
        demo_mode=True,
        db_path=db_file,
    )
    assert turn.route_taken == "bridge_prerequisite"
    assert "Embeddings và Cosine" in turn.answer


def test_11_custom_document_chatbot_grounding(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "custom_chat.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_file))
    db.init_db(db_file)

    raw = (
        "1. Kiến trúc Transformer\n"
        "Transformer sử dụng cơ chế Self-Attention để tính mức độ liên quan giữa mọi từ trong câu song song thay vì xử lý tuần tự như RNN.\n\n"
        "2. Positional Encoding\n"
        "Do Self-Attention xử lý song song và không tự biết thứ tự từ, Positional Encoding cộng vector vị trí vào embedding đầu vào."
    )
    text, secs = load_sections_from_pasted_text(raw, "Transformer Notes")
    kb = DualMemoryKnowledgeBase(secs, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = adaptive.extract_concept_map(text, "doc_tf", "Transformer Notes", demo_mode=True, db_path=db_file)
    dna = db.get_learner_dna("me", "doc_tf", [c.name for c in cmap.concepts], db_path=db_file)

    t_in = run_coach_turn("Positional Encoding để làm gì trong Transformer?", kb=kb, learner_dna=dna, concept_map=cmap, doc_id="doc_tf", demo_mode=True, db_path=db_file)
    t_out = run_coach_turn("Hướng dẫn cấu hình Terraform AWS RDS?", kb=kb, learner_dna=dna, concept_map=cmap, doc_id="doc_tf", demo_mode=True, db_path=db_file)

    assert t_in.route_taken in {"adaptive_explain", "bridge_prerequisite"}
    assert "Positional Encoding" in t_in.answer
    assert t_out.route_taken == "refuse_out_of_doc"


def test_12_live_mode_function_calling_and_trailing_question_split(chat_env, monkeypatch):
    db_file, kb, cmap, names = chat_env
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-live-mock-key")

    class DummyFn:
        name = "lookup_learner_weaknesses"
        arguments = "{}"

    class DummyTC:
        id = "call_1"
        function = DummyFn()

    class DummyMsg1:
        content = ""
        tool_calls = [DummyTC()]

    class DummyMsg2:
        content = (
            "Theo [Mục 1], RAG giúp giảm bịa thông tin bằng cách truy xuất đoạn tài liệu trước khi gọi LLM.\n\n"
            "Trong hệ thống thực tế, bạn sẽ cân bằng tham số top-k thế nào để không bị nhiễu prompt?"
        )
        tool_calls = []

    class DummyUsage:
        prompt_tokens = 150
        completion_tokens = 90

    calls = {"idx": 0}

    class DummyChoice:
        def __init__(self, msg):
            self.message = msg

    class DummyResp:
        def __init__(self, msg):
            self.choices = [DummyChoice(msg)]
            self.usage = DummyUsage()

    def fake_create(**kwargs):
        calls["idx"] += 1
        return DummyResp(DummyMsg1() if calls["idx"] == 1 else DummyMsg2())

    class DummyCompletions:
        create = staticmethod(fake_create)

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    monkeypatch.setattr(llm, "get_client", lambda: DummyClient())

    turn = run_coach_turn(
        "RAG là gì?",
        kb=kb,
        learner_dna=senior,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=False,
        db_path=db_file,
    )
    assert turn.is_demo is False
    assert turn.route_taken == "adaptive_explain"
    assert any(tr.tool_name == "lookup_learner_weaknesses" for tr in turn.tool_traces)
    assert "top-k" in turn.follow_up_question
    assert "câu hỏi" not in turn.follow_up_question.lower() or turn.follow_up_question.endswith("?")


def test_13_stream_coach_turn_live_and_demo(chat_env, monkeypatch):
    db_file, kb, cmap, names = chat_env
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    # 1. Chế độ Demo Offline
    gen_demo, box_demo = stream_coach_turn(
        "RAG là gì?",
        kb=kb,
        learner_dna=senior,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=db_file,
    )
    chunks_demo = list(gen_demo)
    assert len(chunks_demo) >= 2
    assert "turn" in box_demo
    assert box_demo["turn"].answer == "".join(chunks_demo)
    assert box_demo["turn"].route_taken == "adaptive_explain"

    # 2. Chế độ Live API (mock OpenAI stream=True trả về delta chunks)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-live-stream-mock")

    class DummyDelta:
        def __init__(self, content: str):
            self.content = content

    class DummyChoice:
        def __init__(self, content: str):
            self.delta = DummyDelta(content)

    class DummyChunk:
        def __init__(self, content: str):
            self.choices = [DummyChoice(content)]

    def fake_stream_create(**kwargs):
        assert kwargs.get("stream") is True
        return iter(
            [
                DummyChunk("Theo [Mục 1], "),
                DummyChunk("RAG truy xuất các đoạn tài liệu trước khi gọi LLM.\n\n"),
                DummyChunk("Bạn sẽ chọn top-k thế nào cho bài toán này?"),
            ]
        )

    class DummyCompletions:
        create = staticmethod(fake_stream_create)

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    monkeypatch.setattr(llm, "get_client", lambda: DummyClient())

    gen_live, box_live = stream_coach_turn(
        "RAG là gì?",
        kb=kb,
        learner_dna=senior,
        concept_map=cmap,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=False,
        db_path=db_file,
    )
    chunks_live = list(gen_live)
    assert len(chunks_live) == 3
    turn_live = box_live["turn"]
    assert turn_live.is_demo is False
    assert "RAG truy xuất" in turn_live.answer
    assert "top-k" in turn_live.follow_up_question
    assert any(tr.tool_name == "lookup_learner_weaknesses" for tr in turn_live.tool_traces)


@pytest.mark.parametrize(
    ("raw", "expected_body", "expected_q"),
    [
        (
            "Chunking chia tài liệu thành các đoạn nhỏ.\n\n### Câu hỏi kiểm tra:\nBạn có hiểu chunk là gì không?",
            "Chunking chia tài liệu thành các đoạn nhỏ.",
            "Bạn có hiểu chunk là gì không?",
        ),
        (
            "Chunk lớn làm mất độ chính xác.\n\n### Nhắc lại:\nBạn có nhớ vì sao cần overlap không?",
            "Chunk lớn làm mất độ chính xác.",
            "Bạn có nhớ vì sao cần overlap không?",
        ),
        (
            "Overlap giữ ý ở ranh giới chunk.\n\n**Câu hỏi gợi mở:** Vì sao chunk quá nhỏ làm mất ngữ cảnh?",
            "Overlap giữ ý ở ranh giới chunk.",
            "Vì sao chunk quá nhỏ làm mất ngữ cảnh?",
        ),
        (
            "RAG truy xuất trước khi trả lời.\n\n- Bạn thử nêu một ví dụ khác?",
            "RAG truy xuất trước khi trả lời.",
            "Bạn thử nêu một ví dụ khác?",
        ),
    ],
)
def test_14_trailing_question_strips_formatting_labels(raw, expected_body, expected_q):
    body, question = _extract_trailing_question(raw)
    assert body == expected_body
    assert question == expected_q


def test_15_continuation_accept_after_greeting_and_summary(chat_env):
    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    # Luồng 1: Xin chào -> đồng ý học ngay
    t1 = run_coach_turn("Xin chào", kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, db_path=db_file)
    hist1 = [{"role": "user", "content": "Xin chào"}, {"role": "assistant", "content": t1.answer, "turn": t1}]
    t2 = run_coach_turn("Có, bắt đầu ngay đi", kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=hist1, db_path=db_file)
    assert t2.route_taken != "refuse_out_of_doc"
    assert "ok" not in t2.answer.lower()
    assert t2.topic != ""

    # Luồng 2: Tóm tắt tài liệu -> đồng ý đi sâu vào phần được gợi ý
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)
    s1 = run_coach_turn("Tóm tắt tài liệu này", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, db_path=db_file)
    hist2 = [{"role": "user", "content": "Tóm tắt tài liệu này"}, {"role": "assistant", "content": s1.answer, "turn": s1}]
    s2 = run_coach_turn("Ok đi sâu vào phần đó luôn nhé", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=hist2, db_path=db_file)
    assert s2.route_taken == "adaptive_explain"
    assert "không có trong tài liệu" not in s2.answer.lower()


def test_16_coreference_no_infinite_bridge_and_socratic_after_ok(chat_env):
    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)

    # Luồng 3: Hỏi nối tiếp dùng đại từ "nó" -> vẫn neo đúng RAG & [Mục 5] khi thiếu thông tin
    h_rag: list[dict] = []
    r1 = run_coach_turn("RAG là gì?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_rag, db_path=db_file)
    h_rag.extend([{"role": "user", "content": "RAG là gì?"}, {"role": "assistant", "content": r1.answer, "turn": r1}])
    r2 = run_coach_turn("Vậy nó giúp giảm bịa thông tin bằng cách nào?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_rag, db_path=db_file)
    h_rag.extend([{"role": "user", "content": "Vậy nó giúp giảm bịa thông tin bằng cách nào?"}, {"role": "assistant", "content": r2.answer, "turn": r2}])
    r3 = run_coach_turn("Nếu thông tin không có trong tài liệu thì nó làm gì?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_rag, db_path=db_file)
    assert r2.route_taken == "adaptive_explain"
    assert r3.route_taken == "adaptive_explain"
    assert "không có trong tài liệu" in r3.answer.lower() or "Mục 5" in r3.answer

    # Luồng 4: Không lặp vô hạn bridge_prerequisite khi hỏi 3 câu liên tiếp trên Embeddings
    h_emb: list[dict] = []
    e1 = run_coach_turn("Embeddings là gì?", kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_emb, db_path=db_file)
    h_emb.extend([{"role": "user", "content": "Embeddings là gì?"}, {"role": "assistant", "content": e1.answer, "turn": e1}])
    e2 = run_coach_turn("Cosine similarity hoạt động thế nào?", kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_emb, db_path=db_file)
    h_emb.extend([{"role": "user", "content": "Cosine similarity hoạt động thế nào?"}, {"role": "assistant", "content": e2.answer, "turn": e2}])
    e3 = run_coach_turn("Cho mình ví dụ về Embeddings và cosine?", kb=kb, learner_dna=intern, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_emb, db_path=db_file)
    assert e1.route_taken == "bridge_prerequisite"
    assert e2.route_taken == "adaptive_explain"
    assert e3.route_taken == "adaptive_explain"
    assert e2.follow_up_question != e3.follow_up_question

    # Luồng 5: Hỏi 3 role -> trả lời đúng phải được >= 80/100 (đối chiếu đúng [Mục 2 · API])
    h_role: list[dict] = []
    m1 = run_coach_turn("Cấu trúc lời gọi & Messages gồm 3 role nào?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_role, db_path=db_file)
    h_role.extend([{"role": "user", "content": "Cấu trúc lời gọi & Messages gồm 3 role nào?"}, {"role": "assistant", "content": m1.answer, "turn": m1}])
    m2 = run_coach_turn("3 role đó gồm system để đặt vai trò, user là câu hỏi người học và assistant là phản hồi của mô hình.", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_role, db_path=db_file)
    assert m2.route_taken == "socratic_eval"
    assert "Vững bản chất" in m2.answer

    # Luồng 6: Sau acknowledge ("ok") người học trả lời Socratic -> đi vào socratic_eval
    h_ack: list[dict] = []
    a1 = run_coach_turn("Chunking và Overlap là gì?", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_ack, db_path=db_file)
    h_ack.extend([{"role": "user", "content": "Chunking và Overlap là gì?"}, {"role": "assistant", "content": a1.answer, "turn": a1}])
    a2 = run_coach_turn("ok", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_ack, db_path=db_file)
    assert a2.route_taken == "acknowledge"
    assert a2.follow_up_question.endswith("?")
    h_ack.extend([{"role": "user", "content": "ok"}, {"role": "assistant", "content": a2.answer, "turn": a2}])
    a3 = run_coach_turn("Overlap giúp các chunk gối lên nhau để không cắt đứt ý ở ranh giới hai đoạn.", kb=kb, learner_dna=senior, concept_map=cmap, doc_id=db.DEFAULT_DOC_ID, demo_mode=True, chat_history=h_ack, db_path=db_file)
    assert a3.route_taken == "socratic_eval"


def test_17_live_chat_history_preserves_follow_up_question(chat_env):
    from core.coach_graph import _build_coach_openai_messages, _make_init_state
    from core.contracts import CoachTurnResult

    db_file, _, cmap, names = chat_env
    senior = db.get_learner_dna("senior_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)
    prev_turn = CoachTurnResult(
        answer="RAG tìm các đoạn liên quan trước khi trả lời.",
        follow_up_question="Nếu không tìm thấy đoạn phù hợp thì prompt phải chỉ dẫn thế nào?",
        topic="RAG là gì",
        route_taken="adaptive_explain",
        pedagogical_reason="Giải thích theo mức Nâng cao.",
        learner_level_used=senior.overall_level,
    )
    hist = [
        {"role": "user", "content": "RAG là gì?"},
        {"role": "assistant", "content": prev_turn.answer, "turn": prev_turn},
    ]
    state = _make_init_state("Nếu không tìm thấy đoạn phù hợp thì sao?", senior, cmap, db.DEFAULT_DOC_ID, chat_history=hist, demo_mode=False)
    msgs = _build_coach_openai_messages(state, "adaptive_explain_advanced", "Giải thích nối tiếp theo câu hỏi trước")
    assistant_hist = [m["content"] for m in msgs if m["role"] == "assistant"]
    assert assistant_hist and "không tìm thấy đoạn phù hợp" in assistant_hist[0]
    assert "NGỮ CẢNH LƯỢT TRƯỚC" in msgs[-1]["content"]
    assert "RAG là gì" in msgs[-1]["content"]


def test_18_learner_level_progression_no_single_concept_jump_and_placement_sync(chat_env):
    from core.quiz import grade

    db_file, kb, cmap, names = chat_env
    # 1. Trả lời đúng 1 khái niệm duy nhất (85%) không được nhảy cóc từ Mới bắt đầu -> Nâng cao
    db.update_concept_mastery(
        "me", db.DEFAULT_DOC_ID, names[0], is_correct=True, score_override=85.0, bloom_level="Vận dụng", total_concepts=len(names), db_path=db_file
    )
    dna1 = db.get_learner_dna("me", db.DEFAULT_DOC_ID, names, db_path=db_file)
    assert dna1.overall_level == "Trung bình"

    # Hoàn thành thêm >= 2 khái niệm vững bản chất (>=76% TB, <=1 misconception) -> nâng lên Nâng cao
    db.update_concept_mastery("me", db.DEFAULT_DOC_ID, names[1], is_correct=True, score_override=82.0, bloom_level="Vận dụng", total_concepts=len(names), db_path=db_file)
    db.update_concept_mastery("me", db.DEFAULT_DOC_ID, names[2], is_correct=True, score_override=80.0, bloom_level="Vận dụng", total_concepts=len(names), db_path=db_file)
    dna3 = db.get_learner_dna("me", db.DEFAULT_DOC_ID, names, db_path=db_file)
    assert dna3.overall_level == "Nâng cao"

    # 2. Nếu có >= 2 misconception chưa gỡ thì không được gắn nhãn Nâng cao
    db.record_misconception("me", db.DEFAULT_DOC_ID, names[0], "Lỗi hiểu ngược 1", db_path=db_file)
    db.record_misconception("me", db.DEFAULT_DOC_ID, names[1], "Lỗi hiểu ngược 2", db_path=db_file)
    dna_blocked = db.get_learner_dna("me", db.DEFAULT_DOC_ID, names, db_path=db_file)
    assert dna_blocked.overall_level == "Trung bình"

    # Gỡ misconception -> tự đồng bộ lại Nâng cao
    db.resolve_misconception("me", db.DEFAULT_DOC_ID, names[0], db_path=db_file)
    dna_unblocked = db.get_learner_dna("me", db.DEFAULT_DOC_ID, names, db_path=db_file)
    assert dna_unblocked.overall_level == "Nâng cao"

    # 3. Placement Test 3/3 -> recommended_level & DNA overall_level đồng bộ thành Nâng cao
    text, _, _ = load_default_samples()
    placement = adaptive.generate_placement_test(text, cmap, demo_mode=True)
    assert len({adaptive._match_concept_name(q.topic, names) for q in placement.questions}) == 3

    res_perfect = grade(placement, [q.correct_index for q in placement.questions])
    assert res_perfect.recommended_level == "Nâng cao"
    # Dùng learner mới sạch hoàn toàn để kiểm tra đồng bộ sau Placement Test
    dna_placed = adaptive.record_quiz_outcome(
        "new_learner", db.DEFAULT_DOC_ID, res_perfect, model="Offline-Demo", difficulty="Chẩn đoán", concepts_order=names, db_path=db_file
    )
    assert dna_placed.overall_level == "Nâng cao"

    res_mid = grade(placement, [placement.questions[0].correct_index, placement.questions[1].correct_index, (placement.questions[2].correct_index + 1) % 4])
    assert res_mid.recommended_level == "Trung bình"
    dna_mid = adaptive.record_quiz_outcome(
        "mid_learner", db.DEFAULT_DOC_ID, res_mid, model="Offline-Demo", difficulty="Chẩn đoán", concepts_order=names, db_path=db_file
    )
    assert dna_mid.overall_level == "Trung bình"


def test_19_lab_and_socratic_close_progression_loop(chat_env):
    from core import breakthrough

    db_file, kb, cmap, names = chat_env
    intern = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, names, db_path=db_file)
    sc = breakthrough.generate_incident_scenario(load_default_samples()[0], cmap, kb, scenario_index=0, demo_mode=True)
    # Xử lý đúng sự cố Incident (Chunking và Overlap) -> nâng điểm và gỡ misconception cho chủ đề đó
    ev, after_inc = breakthrough.evaluate_incident_resolution(
        scenario=sc,
        chosen_option_index=sc.correct_option_index,
        learner_fix="Thiết lập overlap > 0 để giữ ý ở ranh giới và giảm top_k xuống 3-5 để không đội chi phí token.",
        tools_used=["inspect_config", "inspect_logs", "inspect_metrics"],
        learner_dna=intern,
        doc_id=db.DEFAULT_DOC_ID,
        concepts_order=names,
        demo_mode=True,
        db_path=db_file,
    )
    assert ev.score >= 85
    assert not [m for m in after_inc.active_misconceptions if m.topic == sc.topic]

    # Phản biện Devil's Advocate thuyết phục -> ghi nhận điểm & bloom Đánh giá vào concept_mastery
    aq = breakthrough.generate_devils_advocate(load_default_samples()[0], cmap, kb, round_index=0, demo_mode=True)
    ev_def, after_def = breakthrough.evaluate_defense(
        question=aq,
        learner_defense=aq.sample_strong_answer,
        kb=kb,
        learner_dna=after_inc,
        doc_id=db.DEFAULT_DOC_ID,
        demo_mode=True,
        db_path=db_file,
    )
    assert ev_def.defense_score >= 75
    target = breakthrough._target_topic_for_defense(aq, after_inc)
    c_item = next(c for c in after_def.concepts if c.concept == target)
    assert c_item.bloom_level == "Đánh giá"
