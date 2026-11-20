"""Kiểm thử các lỗi giao diện đã sửa trong app.py: key runtime, độ khó quiz, lỗi LLM, nộp lặp, lỗi nhỏ."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import openai
import pytest
from streamlit.testing.v1 import AppTest

import config
from core import llm

APP = str(config.BASE_DIR / "app.py")


def _goto(at: AppTest, tab: str) -> AppTest:
    """Đổi tab qua st.segmented_control (key động nav_ctrl_*)."""
    ctrl = [c for c in at.segmented_control if str(c.key).startswith("nav_ctrl_")][0]
    ctrl.set_value(tab).run()
    return at


def _login_and_load_sample(at: AppTest, name: str = "Học viên Test") -> AppTest:
    """Đăng nhập thẳng bằng tên rồi nạp giáo trình mẫu RAG ở màn nạp tài liệu."""
    at.text_input("login_name_input").set_value(name).run()
    at.button("btn_login_direct").click().run()
    assert not at.exception
    at.button("btn_load_sample_onboarding").click().run()
    assert not at.exception
    assert at.session_state["has_document"] is True
    return at


@pytest.fixture
def make_app(tmp_path: Path, monkeypatch):
    """Tạo AppTest dùng DB tạm, chưa có API key (Offline), đã đăng nhập + nạp giáo trình mẫu."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "ui_fix.db"))
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    config.set_runtime_api_key(None)

    def _make() -> AppTest:
        at = AppTest.from_file(APP, default_timeout=90)
        at.run()
        assert not at.exception
        return _login_and_load_sample(at)

    yield _make
    config.set_runtime_api_key(None)


def _scalar(sql: str, *args):
    with sqlite3.connect(config.DB_PATH) as con:
        return con.execute(sql, args).fetchone()[0]


def _all_text(at: AppTest) -> str:
    return " ".join(str(e.value) for e in at.error)


# ---------- Lỗi 1: dán key không làm sập app ----------
def test_runtime_key_paste_does_not_crash(make_app):
    at = make_app()
    assert config.has_live_api_key() is False

    at.text_input("runtime_key_input").set_value("sk-test").run()
    assert not at.exception
    assert config.get_effective_api_key() == "sk-test"

    at.text_input("runtime_key_input").set_value("").run()
    assert not at.exception
    assert config.has_live_api_key() is False


# ---------- Lỗi 2: độ khó ghi đúng vào lịch sử ----------
def test_quiz_difficulty_recorded_in_history(make_app):
    at = make_app()
    _goto(at, "practice")
    at.selectbox("quiz_diff_select").set_value("Khó").run()
    at.button("btn_gen_quiz").click().run()
    assert not at.exception
    assert at.session_state["quiz_difficulty_used"] == "Khó"
    at.button("btn_submit_quiz").click().run()
    assert not at.exception
    assert _scalar("SELECT difficulty FROM quiz_attempts ORDER BY id DESC LIMIT 1") == "Khó"


# ---------- Lỗi 3: lỗi LLM ở chế độ Live không hiện traceback ----------
def _dead_client(**kw):
    """OpenAI client thật nhưng transport giả: mọi request báo ConnectError ngay (không ra mạng, không chờ)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused (giả lập)", request=request)

    return openai.OpenAI(
        api_key=kw.get("api_key") or "sk-fake",
        base_url="http://127.0.0.1:9",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture
def live_app(make_app, monkeypatch):
    """App đã đăng nhập + nạp giáo trình mẫu, sau đó bật key giả để mọi lời gọi LLM đi qua client chết."""
    at = make_app()
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(config, "OPENAI_BASE_URL", "http://127.0.0.1:9")
    # Client giả: mọi lời gọi ném APIConnectionError ngay (không ra mạng, không chờ timeout/retry của Windows).
    monkeypatch.setattr(llm, "OpenAI", _dead_client)
    at.run()
    assert not at.exception
    assert config.has_live_api_key() is True
    return at


def _assert_friendly_error(at: AppTest) -> None:
    assert not at.exception
    assert "Không kết nối được" in _all_text(at)


def test_live_llm_error_placement(live_app):
    at = live_app
    _goto(at, "map")
    at.button("btn_start_placement").click().run()
    _assert_friendly_error(at)
    assert at.session_state["placement_quiz"] is None


def _boom(*_a, **_k):
    raise llm.LLMError("Không kết nối được tới máy chủ AI (giả lập).")


def test_lab_generation_errors_are_friendly(make_app, monkeypatch):
    """core/breakthrough tự fallback khi Live lỗi, nên giả lập LLMError lọt ra để kiểm nhánh bắt lỗi của UI."""
    from core import breakthrough

    at = make_app()
    monkeypatch.setattr(breakthrough, "generate_incident_scenario", _boom)
    monkeypatch.setattr(breakthrough, "generate_trojan_challenge", _boom)
    monkeypatch.setattr(breakthrough, "generate_devils_advocate", _boom)
    _goto(at, "lab")
    at.segmented_control("lab_mode").set_value("incident").run()
    _assert_friendly_error(at)
    assert at.session_state["incident_scenario"] is None

    at.segmented_control("lab_mode").set_value("trojan_adv").run()
    _assert_friendly_error(at)
    assert at.session_state["trojan_challenge"] is None
    assert at.session_state["adv_question"] is None


def test_lab_evaluation_errors_are_friendly_and_not_recorded(make_app, monkeypatch):
    from core import breakthrough

    at = make_app()
    _goto(at, "lab")
    at.segmented_control("lab_mode").set_value("incident").run()
    sc = at.session_state["incident_scenario"]
    at.radio(f"inc_pick_{sc.ticket_id}").set_value(0).run()
    monkeypatch.setattr(breakthrough, "evaluate_incident_resolution", _boom)
    monkeypatch.setattr(breakthrough, "evaluate_defense", _boom)
    at.button("btn_solve_incident").click().run()
    _assert_friendly_error(at)
    assert at.session_state["incident_eval"] is None

    at.segmented_control("lab_mode").set_value("trojan_adv").run()
    at.button("btn_fill_adv_sample").click().run()
    at.button("btn_eval_adv").click().run()
    _assert_friendly_error(at)
    assert at.session_state["adv_eval"] is None


def test_live_llm_error_coach_pending(live_app):
    at = live_app
    lid = at.session_state["learner_id"]
    _goto(at, "coach")
    assert not at.exception
    # Yêu cầu chờ (pending_coach) lỗi: không có tin nhắn user mồ côi, có nút thử lại.
    at.session_state["pending_coach"] = {"query": "Giải thích Chunking?", "focus_topic": None, "remediation_context": None}
    at.run()
    _assert_friendly_error(at)
    assert at.session_state["chat_histories"].get(lid, []) == []
    assert at.session_state["coach_failed"]["pending"]["query"] == "Giải thích Chunking?"
    assert at.button("btn_retry_coach") is not None


def test_live_llm_error_practice(live_app):
    at = live_app
    _goto(at, "practice")
    at.button("btn_gen_quiz").click().run()
    _assert_friendly_error(at)
    assert at.session_state["quiz"] is None


# ---------- Lỗi 4: bấm nộp lặp chỉ ghi DB một lần ----------
def test_trojan_resubmit_writes_once(make_app):
    at = make_app()
    _goto(at, "lab")
    at.segmented_control("lab_mode").set_value("trojan_adv").run()
    ch = at.session_state["trojan_challenge"]
    buggy = [s.id for s in ch.statements if s.is_buggy]
    assert buggy
    # Chỉ tick một nửa để có lỗi bị bỏ sót => có ghi lỗi tư duy/mastery.
    first_chk = [c for c in at.checkbox if str(c.key).startswith("trojan_chk_")][0]
    first_chk.set_value(True)
    before = (_scalar("SELECT COALESCE(SUM(attempts),0) FROM concept_mastery"),
              _scalar("SELECT COALESCE(SUM(occurrences),0) FROM misconceptions"))
    at.button("btn_grade_trojan").click().run()
    assert not at.exception
    after1 = (_scalar("SELECT COALESCE(SUM(attempts),0) FROM concept_mastery"),
              _scalar("SELECT COALESCE(SUM(occurrences),0) FROM misconceptions"))
    assert after1 != before
    for _ in range(2):
        at.button("btn_grade_trojan").click().run()
        assert not at.exception
    after3 = (_scalar("SELECT COALESCE(SUM(attempts),0) FROM concept_mastery"),
              _scalar("SELECT COALESCE(SUM(occurrences),0) FROM misconceptions"))
    assert after3 == after1
    assert at.session_state["trojan_result"] is not None


def test_teach_back_resubmit_writes_once(make_app):
    at = make_app()
    _goto(at, "practice")
    at.radio("practice_mode").set_value("Vấn đáp ngược Feynman (Teach-Back)").run()
    at.button("btn_fill_feynman").click().run()
    sql = ("SELECT COALESCE(SUM(attempts),0) + (SELECT COALESCE(SUM(occurrences),0) FROM misconceptions) "
           "FROM concept_mastery")
    before = _scalar(sql)
    at.button("btn_eval_feynman").click().run()
    assert not at.exception
    after1 = _scalar(sql)
    assert after1 > before
    for _ in range(2):
        at.button("btn_eval_feynman").click().run()
        assert not at.exception
    assert _scalar(sql) == after1
    first_eval = at.session_state["teach_back_eval"]
    assert first_eval is not None

    # Đổi đầu vào thì được chấm lại (token đổi).
    at.button("btn_fill_feynman_good").click().run()
    at.button("btn_eval_feynman").click().run()
    assert _scalar(sql) > after1


def test_incident_and_adv_resubmit_writes_once(make_app):
    at = make_app()
    _goto(at, "lab")
    at.segmented_control("lab_mode").set_value("incident").run()
    sc = at.session_state["incident_scenario"]
    wrong = (sc.correct_option_index + 1) % len(sc.root_cause_options)
    at.radio(f"inc_pick_{sc.ticket_id}").set_value(wrong).run()
    sql = ("SELECT COALESCE(SUM(attempts),0) + (SELECT COALESCE(SUM(occurrences),0) FROM misconceptions) "
           "FROM concept_mastery")
    at.button("btn_solve_incident").click().run()
    a1 = _scalar(sql)
    at.button("btn_solve_incident").click().run()
    at.button("btn_solve_incident").click().run()
    assert not at.exception
    assert _scalar(sql) == a1

    at.segmented_control("lab_mode").set_value("trojan_adv").run()
    at.button("btn_fill_adv_sample").click().run()
    at.button("btn_eval_adv").click().run()
    b1 = _scalar(sql)
    at.button("btn_eval_adv").click().run()
    at.button("btn_eval_adv").click().run()
    assert not at.exception
    assert _scalar(sql) == b1


# ---------- Lỗi 5: các lỗi nhỏ ----------
def test_account_switch_clears_previous_results(make_app):
    """Đăng xuất rồi đăng nhập tài khoản khác: kết quả phiên trước và tài liệu không được rò sang."""
    at = make_app()
    _goto(at, "lab")
    at.text_area("diff_note_input").set_value(
        "Chunk qua lon lam mat ngu canh, con overlap nen bang 0 de tiet kiem bo nho."
    ).run()
    at.button("btn_run_diff").click().run()
    assert at.session_state["diff_report"] is not None
    at.button("btn_logout").click().run()
    assert not at.exception
    assert at.session_state["has_document"] is False
    at.text_input("login_name_input").set_value("Tài khoản khác").run()
    at.button("btn_login_direct").click().run()
    assert not at.exception
    for k in ("diff_report", "incident_eval", "trojan_result", "adv_eval", "teach_back_eval", "quiz_result"):
        assert at.session_state[k] is None, k


def test_document_switch_clears_chat_history(make_app):
    """Hội thoại của tài liệu cũ không được mang sang tài liệu mới (gây lệch chủ đề khi hỏi nối tiếp)."""
    at = make_app()
    lid = at.session_state["learner_id"]
    chat_key = f"{lid}::{at.session_state['doc_id']}"
    _goto(at, "coach")
    at.chat_input[0].set_value("RAG là gì?").run()
    assert not at.exception
    assert len(at.session_state["chat_histories"][chat_key]) == 2

    _goto(at, "map")
    custom_doc = (
        "1. Kiến trúc Transformer\n"
        "Transformer sử dụng cơ chế Self-Attention để tính mức độ liên quan giữa mọi từ trong câu song song thay vì xử lý tuần tự như RNN.\n\n"
        "2. Positional Encoding\n"
        "Do Self-Attention xử lý song song và không tự biết thứ tự từ, Positional Encoding cộng vector vị trí vào embedding đầu vào."
    )
    at.text_area("doc_paste_area").set_value(custom_doc).run()
    at.button("btn_load_paste").click().run()
    assert not at.exception
    assert at.session_state["chat_histories"] == {}
    assert at.session_state["pending_coach"] is None
    assert at.session_state["coach_failed"] is None

    _goto(at, "coach")
    assert at.session_state["chat_histories"].get(chat_key, []) == []


def test_reset_progress_clears_session_results(make_app):
    at = make_app()
    _goto(at, "lab")
    at.text_area("diff_note_input").set_value(
        "Chunk qua lon lam mat ngu canh, con overlap nen bang 0 de tiet kiem bo nho."
    ).run()
    at.button("btn_run_diff").click().run()
    assert at.session_state["diff_report"] is not None
    at.button("btn_reset_one").click().run()
    assert not at.exception
    assert at.session_state["diff_report"] is None


def test_trojan_checkbox_key_unique_per_version(make_app):
    at = make_app()
    _goto(at, "lab")
    at.segmented_control("lab_mode").set_value("trojan_adv").run()
    keys1 = {c.key for c in at.checkbox if str(c.key).startswith("trojan_chk_")}
    at.button("btn_new_trojan").click().run()
    assert not at.exception
    keys2 = {c.key for c in at.checkbox if str(c.key).startswith("trojan_chk_")}
    assert keys1 and keys2 and not (keys1 & keys2)


def test_vision_sample_button_changes_image_and_clears_critique(make_app):
    at = make_app()
    _goto(at, "coach")
    at.button("btn_run_vision").click().run()
    assert at.session_state["vision_critique"] is not None
    at.button("btn_vis_s2").click().run()
    assert not at.exception
    assert at.session_state["vision_sample_id"] == "code-openai-loi"
    assert at.session_state["vision_critique"] is None
    assert at.session_state["vision_force_sample"] is True


def test_diff_note_editable_after_file_upload_logic(make_app):
    """Ghi chú tay được giữ qua các lượt rerun (không bị ghi đè khi không có file mới)."""
    at = make_app()
    _goto(at, "lab")
    at.text_area("diff_note_input").set_value("Ghi chú tự sửa tay dài hơn hai mươi lăm ký tự.").run()
    at.run()
    assert at.text_area("diff_note_input").value.startswith("Ghi chú tự sửa tay")


def test_eval_json_missing_keys_does_not_crash(make_app):
    at = make_app()
    at.session_state["eval_summary"] = {"in_scope_recall": 1.0}
    _goto(at, "analytics")
    assert not at.exception
