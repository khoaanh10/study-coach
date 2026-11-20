"""Kiểm thử toàn diện Chatbot Socratic trong ``core/coach_graph.py``.

Toàn bộ test chạy offline (demo_mode=True) trừ nhóm Live mode được mock hoàn toàn:
không gọi mạng/API thật và luôn dùng SQLite tạm qua ``tmp_path``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import config
from core import adaptive, coach_graph, db
from core.coach_graph import run_coach_turn, stream_coach_turn
from core.contracts import ConceptNode, DocumentConceptMap
from core.loader import load_default_samples, load_sections_from_pasted_text
from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings


STANDARD_YES_NO_QUESTION = (
    "Bạn có nghĩ rằng việc sử dụng thông tin từ tài liệu bên ngoài "
    "có thể cải thiện độ chính xác của câu trả lời không?"
)
LONG_SOCRATIC_QUESTION = (
    "Bạn có thấy việc trích dẫn nguồn tài liệu quan trọng như thế nào không?"
)

SHORT_YES_NO_REPLIES = ["có", "có chứ", "không ạ", "đúng rồi", "chắc chắn", "tất nhiên"]
AFFIRMATIVE_REPLIES = ["có", "có chứ", "đúng rồi", "chắc chắn", "tất nhiên"]

CORRECT_CHUNK_ANSWER = (
    "Tài liệu dài được chia thành các đoạn nhỏ gọi là chunk. "
    "Chunk quá lớn chứa nhiều ý lẫn lộn nên truy xuất kém chính xác. "
    "Chunk quá nhỏ làm mất ngữ cảnh. Người ta để các chunk gối lên nhau một phần gọi là overlap "
    "để một ý nằm ở ranh giới giữa hai chunk không bị cắt đứt."
)
WRONG_CHUNK_ANSWER = (
    "Overlap bằng 0 là tốt nhất, không cần overlap. "
    "Chunk quá lớn làm mất ngữ cảnh của đoạn văn."
)
OFF_TOPIC_ANSWER = (
    "Hôm nay trời rất đẹp nên mình đi chơi công viên với bạn bè và ăn kem."
)


def _chat_setup(tmp_path, *, learner: str = "intern_demo", filename: str = "chatbot_exhaustive.db"):
    db_path = str(tmp_path / filename)
    combined, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=combined)
    cmap = adaptive.extract_concept_map(combined, db.DEFAULT_DOC_ID, title, demo_mode=True, db_path=db_path)
    names = [c.name for c in cmap.concepts]
    dna = db.get_learner_dna(learner, db.DEFAULT_DOC_ID, names, db_path=db_path)
    return SimpleNamespace(
        kb=kb,
        cmap=cmap,
        dna=dna,
        db_path=db_path,
        names=names,
        doc_id=db.DEFAULT_DOC_ID,
        title=title,
    )


@pytest.fixture
def env(tmp_path):
    return _chat_setup(tmp_path)


def _ask(env, query: str, history: list[dict] | None = None, *, demo_mode: bool = True, **kwargs):
    return run_coach_turn(
        query,
        kb=env.kb,
        learner_dna=env.dna,
        concept_map=env.cmap,
        doc_id=env.doc_id,
        demo_mode=demo_mode,
        chat_history=history or [],
        db_path=env.db_path,
        **kwargs,
    )


def _seed_turn(env, query: str, *, follow_up: str | None = None, route: str | None = None):
    turn = _ask(env, query)
    if follow_up is not None:
        turn = turn.model_copy(update={"follow_up_question": follow_up})
    if route is not None:
        turn = turn.model_copy(update={"route_taken": route})
    history = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": turn.answer, "turn": turn},
    ]
    return turn, history


def _append(history: list[dict], user_text: str, turn) -> list[dict]:
    return [
        *history,
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": turn.answer, "turn": turn},
    ]


def _router_output(intent: str, *, matched_topic: str = "", standalone_query: str = "", reason: str = "mock"):
    return coach_graph._IntentRouterOutput(
        intent=intent,
        standalone_query=standalone_query,
        matched_topic=matched_topic,
        reason=reason,
    )


def _patch_live_router(monkeypatch, intent: str, *, matched_topic: str = "", standalone_query: str = ""):
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    calls: list[dict] = []

    def fake_structured(messages, schema, **kwargs):
        calls.append({"messages": messages, "schema": schema, **kwargs})
        return _router_output(intent, matched_topic=matched_topic, standalone_query=standalone_query)

    monkeypatch.setattr(coach_graph, "complete_structured", fake_structured)
    return calls


# ---------------------------------------------------------------------------
# 1. Phản hồi ngắn Yes/No sau câu hỏi Socratic -> acknowledge, không refuse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reply", SHORT_YES_NO_REPLIES)
def test_short_yes_no_reply_after_socratic_question_routes_acknowledge(env, reply: str):
    _, history = _seed_turn(env, "Chunking và Overlap hoạt động ra sao?", follow_up=STANDARD_YES_NO_QUESTION)
    turn = _ask(env, reply, history)

    assert turn.route_taken == "acknowledge"
    assert "không có trong tài liệu" not in turn.answer.lower()
    assert "ngoài tài liệu" not in turn.answer.lower()


@pytest.mark.parametrize("reply", AFFIRMATIVE_REPLIES)
def test_short_yes_reply_after_long_socratic_question_routes_acknowledge(env, reply: str):
    _, history = _seed_turn(env, "RAG là gì?", follow_up=LONG_SOCRATIC_QUESTION)
    turn = _ask(env, reply, history)

    assert turn.route_taken == "acknowledge"
    assert "không có trong tài liệu" not in turn.answer.lower()


@pytest.mark.parametrize("reply", SHORT_YES_NO_REPLIES)
def test_short_yes_no_reply_after_long_socratic_question_live_router(env, monkeypatch, reply: str):
    _, history = _seed_turn(env, "RAG là gì?", follow_up=LONG_SOCRATIC_QUESTION)
    calls = _patch_live_router(monkeypatch, "confirm_answer", matched_topic="RAG là gì")

    turn = _ask(env, reply, history, demo_mode=False)

    assert turn.route_taken == "acknowledge"
    assert "không có trong tài liệu" not in turn.answer.lower()
    assert len(calls) == 1
    assert reply in calls[0]["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 2. Lặp phản hồi ngắn nhiều lượt -> không lặp nguyên văn lượt trước
# ---------------------------------------------------------------------------

def test_repeated_short_replies_do_not_repeat_previous_turn(env):
    _, history = _seed_turn(env, "Chunking và Overlap hoạt động ra sao?", follow_up=STANDARD_YES_NO_QUESTION)

    turn1 = _ask(env, "có", history)
    assert turn1.route_taken == "acknowledge"
    history1 = _append(history, "có", turn1)

    turn2 = _ask(env, "có", history1)
    assert turn2.route_taken == "acknowledge"
    assert turn2.answer != turn1.answer
    assert turn2.follow_up_question != turn1.follow_up_question
    history2 = _append(history1, "có", turn2)

    turn3 = _ask(env, "ok", history2)
    assert turn3.route_taken == "acknowledge"
    assert turn3.answer != turn2.answer
    assert turn3.follow_up_question != turn2.follow_up_question


# ---------------------------------------------------------------------------
# 3. Yêu cầu điều hướng phiên học -> acknowledge + câu hỏi mới
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    ["hỏi câu khác", "hỏi câu khác?", "đổi câu đi", "bỏ qua", "mình không biết", "chịu"],
)
def test_navigation_requests_route_acknowledge_with_new_question(env, query: str):
    _, history = _seed_turn(env, "Chunking và Overlap hoạt động ra sao?")
    last_follow = history[-1]["turn"].follow_up_question.strip()

    turn = _ask(env, query, history)

    assert turn.route_taken == "acknowledge", f"{query} -> {turn.route_taken}"
    assert "không có trong tài liệu" not in turn.answer.lower()
    assert turn.follow_up_question.strip().endswith("?")
    assert turn.follow_up_question.strip() != last_follow


# ---------------------------------------------------------------------------
# 4. Câu hỏi ngoài tài liệu xen giữa vẫn giữ mạch RAG
# ---------------------------------------------------------------------------

def test_out_of_doc_interleaved_keeps_rag_thread(env):
    rag_turn, history = _seed_turn(env, "RAG là gì?")
    assert "RAG" in rag_turn.topic

    out_turn = _ask(env, "Giá vàng hôm nay bao nhiêu?", history)
    assert out_turn.route_taken == "refuse_out_of_doc"
    history2 = _append(history, "Giá vàng hôm nay bao nhiêu?", out_turn)

    follow_turn = _ask(env, "cho ví dụ đi", history2)

    assert follow_turn.route_taken == "adaptive_explain"
    assert follow_turn.topic == rag_turn.topic
    assert "RAG" in follow_turn.answer
    assert "ví dụ" in follow_turn.answer.lower()
    assert "không có trong tài liệu" not in follow_turn.answer.lower()


# ---------------------------------------------------------------------------
# 5. Đại từ quy chiếu & follow-up đa lượt -> bám đúng chủ đề đang bàn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "nó hoạt động thế nào",
        "cái đó khác gì",
        "giải thích kỹ hơn",
        "chưa hiểu",
        "cho ví dụ thực tế",
    ],
)
def test_coreference_and_elaboration_followups_stick_to_topic(env, query: str):
    first_turn, history = _seed_turn(env, "RAG là gì?")

    turn = _ask(env, query, history)

    assert turn.route_taken != "refuse_out_of_doc", f"{query} -> {turn.route_taken}"
    assert turn.topic == first_turn.topic
    assert first_turn.topic in turn.answer or "RAG" in turn.answer


# ---------------------------------------------------------------------------
# 6. Chấm Socratic: câu sai/lạc đề điểm thấp, câu đúng ý chính điểm cao
# ---------------------------------------------------------------------------

def test_socratic_scoring_low_for_wrong_and_high_for_correct(tmp_path):
    env = _chat_setup(tmp_path)
    common = dict(
        concept_map=env.cmap,
        doc_id=env.doc_id,
        concepts_order=env.names,
        demo_mode=True,
        db_path=env.db_path,
    )

    ev_off, _ = adaptive.evaluate_teach_back(
        "Chunking và Overlap", OFF_TOPIC_ANSWER, env.kb, env.dna, **common
    )
    ev_wrong, after_wrong = adaptive.evaluate_teach_back(
        "Chunking và Overlap", WRONG_CHUNK_ANSWER, env.kb, env.dna, **common
    )
    ev_correct, after_correct = adaptive.evaluate_teach_back(
        "Chunking và Overlap", CORRECT_CHUNK_ANSWER, env.kb, after_wrong, **common
    )

    assert ev_off.understanding_score < 40
    assert ev_wrong.understanding_score < 50
    assert ev_correct.understanding_score >= 80
    assert ev_correct.understanding_score > ev_wrong.understanding_score
    assert ev_wrong.misconceptions_found
    assert not ev_correct.misconceptions_found
    assert len(after_correct.active_misconceptions) < len(after_wrong.active_misconceptions)


def test_socratic_answer_routes_to_socratic_eval(env):
    _, history = _seed_turn(env, "Chunking và Overlap hoạt động ra sao?")

    turn = _ask(env, WRONG_CHUNK_ANSWER, history)

    assert turn.route_taken == "socratic_eval"
    assert "Chấm câu trả lời Socratic" in turn.answer
    assert "overlap" in turn.answer.lower()
    assert turn.follow_up_question.strip().endswith("?")


# ---------------------------------------------------------------------------
# 7. Ôn tập, tóm tắt, chào hỏi, điểm yếu và điều hướng "tiếp đi"
# ---------------------------------------------------------------------------

def test_review_request_with_explicit_topic(env):
    turn = _ask(env, "ôn phần chunking cho mình", [])

    assert turn.route_taken == "adaptive_explain"
    assert "Ôn nhanh `Chunking và Overlap`" in turn.answer
    assert "không có trong tài liệu" not in turn.answer.lower()
    assert turn.follow_up_question


def test_review_request_without_topic_suggests_one(env):
    turn = _ask(env, "ôn phần nào đó đi", [])

    assert turn.route_taken == "adaptive_explain"
    assert "Ôn nhanh `" in turn.answer
    assert "mình chọn" in turn.answer
    assert "không có trong tài liệu" not in turn.answer.lower()


def test_review_request_for_unknown_topic_is_refused(env):
    turn = _ask(env, "ôn phần Kubernetes cho mình", [])

    assert turn.route_taken == "refuse_out_of_doc"
    assert "không có trong tài liệu" in turn.answer.lower()


def test_summary_greeting_and_weakness_are_guided(env):
    summary = _ask(env, "tóm tắt tài liệu", [])
    assert summary.route_taken == "coach_guide"
    assert "Tóm tắt cấu trúc tài liệu" in summary.answer

    greeting = _ask(env, "xin chào", [])
    assert greeting.route_taken == "coach_guide"
    assert "Chào" in greeting.answer

    weakness = _ask(env, "điểm yếu của tôi là gì", [])
    assert weakness.route_taken == "coach_guide"
    assert "Ưu tiên" in weakness.answer


@pytest.mark.parametrize("seed_query", ["xin chào", "tóm tắt tài liệu"])
def test_tiep_di_after_guide_continues_the_offered_topic(env, seed_query: str):
    first_turn, history = _seed_turn(env, seed_query)

    turn = _ask(env, "tiếp đi", history)

    assert turn.route_taken == "adaptive_explain"
    assert turn.topic in env.names
    assert "không có trong tài liệu" not in turn.answer.lower()
    assert turn.follow_up_question.strip().endswith("?")


# ---------------------------------------------------------------------------
# 8. Live mode: LLM Intent Router, stream không double-inspect, fallback an toàn
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reply, intent",
    [
        ("mình đoán là có", "confirm_answer"),
        ("cũng tùy trường hợp", "confirm_answer"),
        ("đổi sang câu dễ hơn đi", "next_question"),
    ],
)
def test_live_intent_router_catches_natural_replies(env, monkeypatch, reply: str, intent: str):
    _, history = _seed_turn(env, "RAG là gì?", follow_up=STANDARD_YES_NO_QUESTION)
    calls = _patch_live_router(monkeypatch, intent, matched_topic="RAG là gì")

    turn = _ask(env, reply, history, demo_mode=False)

    assert turn.route_taken == "acknowledge"
    assert len(calls) == 1
    assert calls[0]["schema"] is coach_graph._IntentRouterOutput
    assert reply in calls[0]["messages"][-1]["content"]
    assert any(t.tool_name == "classify_intent_llm" for t in turn.tool_traces)


def test_live_adaptive_explain_calls_llm_with_tools(env, monkeypatch):
    calls = _patch_live_router(monkeypatch, "standard_qa", matched_topic="RAG là gì")
    tool_calls: list[dict] = []

    def fake_tools(messages, tools_schema, tool_dispatch, **kwargs):
        tool_calls.append({"messages": messages, "tools": tools_schema, **kwargs})
        return ("Câu trả lời live về RAG.\n\nBạn thấy RAG giúp gì?", [])

    monkeypatch.setattr(coach_graph, "complete_with_tools", fake_tools)

    turn = _ask(env, "RAG là gì?", [], demo_mode=False)

    assert calls
    assert tool_calls
    assert "Câu trả lời live về RAG." in turn.answer
    assert turn.follow_up_question.strip().endswith("?")


@pytest.mark.parametrize(
    "query, intent, expected_route",
    [
        ("có", "confirm_answer", "acknowledge"),
        ("Giá vàng hôm nay bao nhiêu?", "standard_qa", "refuse_out_of_doc"),
    ],
)
def test_stream_live_acknowledge_and_refuse_inspect_only_once(env, monkeypatch, query: str, intent: str, expected_route: str):
    _, history = _seed_turn(env, "RAG là gì?", follow_up=STANDARD_YES_NO_QUESTION)
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    calls: list[dict] = []

    def fake_structured(messages, schema, **kwargs):
        calls.append({"messages": messages, "schema": schema, **kwargs})
        return coach_graph._IntentRouterOutput(
            intent=intent,
            in_scope=(expected_route != "refuse_out_of_doc"),
            matched_topic="" if expected_route == "refuse_out_of_doc" else "RAG là gì",
            reason="mock live router",
        )

    monkeypatch.setattr(coach_graph, "complete_structured", fake_structured)

    stream, result_box = stream_coach_turn(
        query,
        kb=env.kb,
        learner_dna=env.dna,
        concept_map=env.cmap,
        doc_id=env.doc_id,
        demo_mode=False,
        chat_history=history,
        db_path=env.db_path,
    )
    streamed = "".join(stream)
    turn = result_box["turn"]

    assert turn.route_taken == expected_route
    assert streamed.strip()
    assert len(calls) == 1, f"complete_structured bị gọi {len(calls)} lần (double-inspect)"


def test_live_router_error_falls_back_to_rules_without_crashing(env, monkeypatch):
    _, history = _seed_turn(env, "RAG là gì?", follow_up=STANDARD_YES_NO_QUESTION)
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)

    def failing_structured(*args, **kwargs):
        raise RuntimeError("rate limit")

    monkeypatch.setattr(coach_graph, "complete_structured", failing_structured)

    turn = _ask(env, "có chứ", history, demo_mode=False)

    assert turn.route_taken == "acknowledge"
    assert not any(t.tool_name == "classify_intent_llm" for t in turn.tool_traces)
    assert "không có trong tài liệu" not in turn.answer.lower()


# ---------------------------------------------------------------------------
# 9. Edge cases đầu vào và tài liệu custom không phải RAG
# ---------------------------------------------------------------------------

def test_unaccented_vietnamese_question_is_not_refused(env):
    turn = _ask(env, "chunking va overlap la gi", [])

    assert turn.route_taken != "refuse_out_of_doc"
    assert turn.topic in env.names or "chunk" in turn.answer.lower()


def test_punctuation_only_input_is_handled_gracefully(env):
    turn = _ask(env, "???", [])

    assert turn.route_taken in {"refuse_out_of_doc", "acknowledge"}
    assert turn.answer.strip()
    if turn.route_taken == "refuse_out_of_doc":
        assert "không có trong tài liệu" in turn.answer.lower()


def test_long_valid_question_stays_in_scope(env):
    query = (
        "Giải thích chi tiết cách chunking và overlap phối hợp với embeddings và vector store "
        "để truy xuất đúng đoạn tài liệu trong RAG."
    )

    turn = _ask(env, query, [])

    assert turn.route_taken != "refuse_out_of_doc"
    assert turn.answer.strip()
    assert turn.topic in env.names


def test_custom_sql_document_is_answered_without_rag_hardcoded_rules(tmp_path):
    db_path = str(tmp_path / "sql_index.db")
    db.init_db(db_path)
    text = (
        "SQL INDEX\n\n"
        "1. Khái niệm\n"
        "Index trong SQL là cấu trúc dữ liệu phụ giúp tăng tốc truy vấn trên bảng lớn.\n\n"
        "2. Cơ chế hoạt động\n"
        "Khi tạo index trên một cột, hệ quản trị cơ sở dữ liệu xây dựng cây B-tree để tìm bản ghi "
        "nhanh hơn thay vì quét toàn bảng.\n\n"
        "3. Khi nào nên dùng\n"
        "Nên tạo index cho cột thường xuyên lọc hoặc nối bảng, nhưng tránh tạo quá nhiều index "
        "vì làm chậm thao tác ghi."
    )
    _, sections = load_sections_from_pasted_text(text, "sql-index.txt")
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=text)
    cmap = DocumentConceptMap(
        doc_id="sql_doc",
        doc_title="SQL Index",
        concepts=[
            ConceptNode(
                name="SQL Index",
                summary="Cây B-tree giúp tìm bản ghi nhanh hơn quét toàn bảng.",
                page_ref="[Mục 2]",
                key_terms=["index", "B-tree", "truy vấn", "bảng"],
            )
        ],
    )
    dna = db.get_learner_dna("sql_learner", "sql_doc", ["SQL Index"], db_path=db_path)

    turn = run_coach_turn(
        "SQL index hoạt động như thế nào?",
        kb=kb,
        learner_dna=dna,
        concept_map=cmap,
        doc_id="sql_doc",
        demo_mode=True,
        db_path=db_path,
    )

    assert turn.route_taken != "refuse_out_of_doc"
    assert turn.topic == "SQL Index"
    assert turn.answer.strip()
    assert "overlap" not in turn.answer.lower()


def test_ko_biet_giai_thich_stays_in_topic_not_refused(env, monkeypatch):
    _, history = _seed_turn(
        env,
        "Chunking và Overlap là gì?",
        follow_up="Tại sao chúng ta cần sử dụng overlap khi chia nhỏ tài liệu thành các chunk?",
    )
    # Demo/offline mode
    demo_turn = _ask(env, "ko biết, giải thích", history)
    assert demo_turn.route_taken != "refuse_out_of_doc"
    assert demo_turn.route_taken == "adaptive_explain"
    assert "chunk" in demo_turn.topic.lower() or "overlap" in demo_turn.topic.lower()

    # Live mode: LLM Intent Router classifies as followup_elaborate
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    monkeypatch.setattr(
        coach_graph,
        "complete_structured",
        lambda *a, **kw: coach_graph._IntentRouterOutput(
            intent="followup_elaborate",
            matched_topic="Chunking và Overlap",
            reason="Học viên không biết câu hỏi vừa rồi và nhờ giải thích lại",
        ),
    )
    monkeypatch.setattr(
        coach_graph,
        "complete_with_tools",
        lambda *a, **kw: (
            "Overlap giúp các đoạn chunk gối lên nhau để không cắt đứt ý quan trọng ở ranh giới [Mục 2].\n\nBạn thử lấy 1 ví dụ về overlap nhé?",
            [],
        ),
    )
    live_turn = run_coach_turn(
        query="ko biết, giải thích",
        kb=env.kb,
        learner_dna=env.dna,
        concept_map=env.cmap,
        doc_id=env.doc_id,
        demo_mode=False,
        chat_history=history,
        db_path=env.db_path,
    )
    assert live_turn.route_taken == "adaptive_explain"
    assert "ngoài tài liệu" not in live_turn.answer.lower()


# ---------------------------------------------------------------------------
# 9. Colloquial / Teencode / Topic-continuity Stress-Test
# ---------------------------------------------------------------------------

def test_hello_fen_then_bat_dau_keeps_offered_topic(env):
    """Gợi ý học 'RAG là gì' kết thúc bằng '!' (follow_up rỗng) -> 'bắt đầu' phải vào đúng chủ đề đó."""
    assert "RAG là gì" in env.names
    base = _ask(env, "hello fen")
    turn = base.model_copy(
        update=dict(
            answer="Chào bạn! Mình gợi ý bạn học chủ đề 'RAG là gì' trước. Hãy bắt đầu với chủ đề đó nhé!",
            follow_up_question="",
            topic="Tổng quan tài liệu",
            route_taken="coach_guide",
        )
    )
    history = [
        dict(role="user", content="hello fen"),
        dict(role="assistant", content=turn.answer, turn=turn),
    ]

    result = _ask(env, "bắt đầu", history)

    assert result.route_taken == "adaptive_explain"
    assert result.topic == "RAG là gì"
    assert result.topic != "Chunking và Overlap"


COLLOQUIAL_CASES = [
    ("ko biết, giải thích", "followup_elaborate"),
    ("k hiểu lắm, nói lại dễ hiểu đi", "followup_elaborate"),
    ("ý là sao z", "followup_elaborate"),
    ("chịu luôn, giải thích giùm", "followup_elaborate"),
    ("cho 1 vd đời thường đi", "followup_example"),
    ("ờ chắc v", "confirm_answer"),
    ("đổi câu dễ hơn coi", "next_question"),
]
SOCRATIC_FOLLOW_UP = "Tại sao chúng ta cần sử dụng overlap khi chia nhỏ tài liệu thành các chunk?"


@pytest.mark.parametrize("reply, live_intent", COLLOQUIAL_CASES)
def test_colloquial_replies_demo_mode_not_refused(env, reply: str, live_intent: str):
    _, history = _seed_turn(env, "Chunking và Overlap là gì?", follow_up=SOCRATIC_FOLLOW_UP)

    turn = _ask(env, reply, history)

    assert turn.route_taken != "refuse_out_of_doc"
    assert "ngoài tài liệu" not in turn.answer.lower()


@pytest.mark.parametrize("reply, live_intent", COLLOQUIAL_CASES)
def test_colloquial_replies_live_mode_not_refused(env, monkeypatch, reply: str, live_intent: str):
    _, history = _seed_turn(env, "Chunking và Overlap là gì?", follow_up=SOCRATIC_FOLLOW_UP)
    _patch_live_router(monkeypatch, live_intent, matched_topic="Chunking và Overlap")
    monkeypatch.setattr(
        coach_graph,
        "complete_with_tools",
        lambda *a, **kw: ("Overlap giúp các chunk gối lên nhau để không mất ý [Mục 2].\n\nBạn thử nêu lại bằng lời của mình nhé?", []),
    )

    turn = _ask(env, reply, history, demo_mode=False)

    assert turn.route_taken != "refuse_out_of_doc"
    assert "ngoài tài liệu" not in turn.answer.lower()


def test_colloquial_example_request_for_out_of_doc_term_is_still_refused(env):
    _, history = _seed_turn(env, "Chunking và Overlap là gì?", follow_up=SOCRATIC_FOLLOW_UP)

    turn = _ask(env, "cho 1 vd về Kubernetes đi", history)

    assert turn.route_taken == "refuse_out_of_doc"


def test_llm_router_misclassifying_ko_biet_giai_thich_as_standard_qa_is_rescued(env, monkeypatch):
    _, history = _seed_turn(
        env,
        "RAG là gì?",
        follow_up="Bạn có thể cho mình biết, tại sao việc chia tài liệu thành các chunk lại quan trọng trong quá trình truy xuất thông tin?",
    )
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    monkeypatch.setattr(
        coach_graph,
        "complete_structured",
        lambda *a, **kw: coach_graph._IntentRouterOutput(
            intent="standard_qa",
            standalone_query="",
            matched_topic="",
            reason="misclassified by small model",
        ),
    )
    monkeypatch.setattr(
        coach_graph,
        "complete_with_tools",
        lambda *a, **kw: (
            "Chia tài liệu thành chunk giúp truy xuất đúng đoạn liên quan thay vì nhồi cả tài liệu dài [Mục 2].\n\nBạn thử tóm tắt lại lợi ích của chunk bằng 1 câu nhé?",
            [],
        ),
    )
    turn = run_coach_turn(
        query="ko biết, giải thích",
        kb=env.kb,
        learner_dna=env.dna,
        concept_map=env.cmap,
        doc_id=env.doc_id,
        demo_mode=False,
        chat_history=history,
        db_path=env.db_path,
    )
    assert turn.route_taken == "adaptive_explain"
    assert "ngoài tài liệu" not in turn.answer.lower()


def test_creative_short_socratic_example_is_evaluated_not_refused(env, monkeypatch):
    _, history = _seed_turn(
        env,
        "RAG là gì?",
        follow_up="Bạn có thể cho mình biết một ví dụ về một kỹ năng mà bạn nghĩ Claude có thể học không?",
    )
    monkeypatch.setattr(config, "has_live_api_key", lambda: True)
    monkeypatch.setattr(
        coach_graph,
        "complete_structured",
        lambda *a, **kw: coach_graph._IntentRouterOutput(
            intent="socratic_answer",
            matched_topic="RAG là gì",
            reason="Học viên đưa ra ví dụ ngắn trả lời câu hỏi Socratic",
        ),
    )
    monkeypatch.setattr(
        coach_graph.adaptive,
        "evaluate_teach_back",
        lambda **kw: (
            coach_graph.adaptive.TeachBackEvaluation(
                topic=kw["concept"],
                understanding_score=80,
                verdict="Hiểu đúng trọng tâm",
                correct_points=["Đưa ra ví dụ thực tế hợp lý về quy trình làm video."],
                missing_points=["Chưa nêu rõ cấu trúc thư mục SKILL.md và scripts/."],
                detected_misconceptions=[],
                socratic_follow_up="Theo bạn, skill làm video sẽ cần những gì trong SKILL.md và thư mục scripts/?",
            ),
            kw["learner_dna"],
        ),
    )
    turn = run_coach_turn(
        query="skill làm video",
        kb=env.kb,
        learner_dna=env.dna,
        concept_map=env.cmap,
        doc_id=env.doc_id,
        demo_mode=False,
        chat_history=history,
        db_path=env.db_path,
    )
    assert turn.route_taken == "socratic_eval"
    assert "ngoài tài liệu" not in turn.answer.lower()
