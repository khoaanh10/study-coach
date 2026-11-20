"""Kiểm thử giao diện Streamlit bằng AppTest: Đăng nhập -> Nạp tài liệu -> Workspace Full-width."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import config

APP = str(config.BASE_DIR / "app.py")


def _nav_ctrl(at: AppTest):
    found = [c for c in at.segmented_control if str(c.key).startswith("nav_ctrl_")]
    return found[0] if found else None


def _goto(at: AppTest, tab: str) -> AppTest:
    """Đổi tab qua st.segmented_control (key động nav_ctrl_*, đổi mỗi khi app gọi _set_nav)."""
    _nav_ctrl(at).set_value(tab).run()
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


def _new_app(tmp_path: Path, monkeypatch, db_name: str) -> AppTest:
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / db_name))
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    config.set_runtime_api_key(None)
    at = AppTest.from_file(APP, default_timeout=20)
    at.run()
    assert not at.exception
    return at


def test_login_screen_flow(tmp_path: Path, monkeypatch):
    at = _new_app(tmp_path, monkeypatch, "ui_login.db")
    # Chưa đăng nhập: chỉ có màn login.
    assert not at.session_state["logged_in_learner_id"]
    assert at.text_input("login_name_input") is not None
    assert _nav_ctrl(at) is None

    # Tên rỗng bị cảnh báo, vẫn ở màn login.
    at.button("btn_login_direct").click().run()
    assert not at.exception
    assert any("họ và tên" in str(w.value) for w in at.warning)
    assert not at.session_state["logged_in_learner_id"]

    # Đăng nhập thành công -> màn nạp tài liệu (chưa có RAG).
    at.text_input("login_name_input").set_value("Nguyễn Test").run()
    at.button("btn_login_direct").click().run()
    assert not at.exception
    assert at.session_state["logged_in_learner_id"]
    assert at.session_state["has_document"] is False
    assert at.session_state["kb"] is None
    assert at.button("btn_load_sample_onboarding") is not None
    assert _nav_ctrl(at) is None


def test_logout_clears_document_state(tmp_path: Path, monkeypatch):
    at = _new_app(tmp_path, monkeypatch, "ui_logout.db")
    _login_and_load_sample(at)
    assert at.session_state["doc_id"] == "default_curriculum"
    assert at.session_state["kb"] is not None
    assert at.session_state["concept_map"] is not None

    at.button("btn_logout").click().run()
    assert not at.exception
    assert not at.session_state["logged_in_learner_id"]
    assert at.session_state["has_document"] is False
    assert not at.session_state["doc_id"]
    assert at.session_state["kb"] is None
    assert at.session_state["concept_map"] is None
    assert at.text_input("login_name_input") is not None


def test_nav_segmented_control_and_chat_to_quiz_button(tmp_path: Path, monkeypatch):
    at = _new_app(tmp_path, monkeypatch, "ui_nav.db")
    _login_and_load_sample(at)
    for tab in ("map", "practice", "lab", "analytics", "coach"):
        _goto(at, tab)
        assert not at.exception
        assert at.session_state["active_nav"] == tab

    # Hỏi một câu rồi bấm nút chuyển tab sang Quiz từ tin nhắn gia sư (nếu có gợi ý chủ đề).
    at.chat_input[0].set_value("Giải thích Chunking trong RAG?").run()
    assert not at.exception
    btns = [b for b in at.button if str(b.key).startswith("chat_to_quiz_")]
    if btns:
        btns[-1].click().run()
        assert not at.exception
        assert at.session_state["active_nav"] == "practice"
        _goto(at, "coach")
        assert at.session_state["active_nav"] == "coach"


def test_app_full_navigation_and_interactions(tmp_path: Path, monkeypatch):
    at = _new_app(tmp_path, monkeypatch, "ui_test.db")
    _login_and_load_sample(at)
    lid = at.session_state["learner_id"]

    # 1. Mở bài Chẩn đoán đầu vào ở Tab Bản đồ
    _goto(at, "map")
    at.button("btn_start_placement").click().run()
    assert not at.exception
    assert at.session_state["placement_quiz"] is not None
    at.button("btn_close_placement").click().run()
    assert not at.exception

    # 2. Tab Gia sư Socratic & Vision
    _goto(at, "coach")
    assert not at.exception
    chat_key = f"{lid}::{at.session_state['doc_id']}"
    at.button("qp_chunk").click().run()
    assert not at.exception
    assert len(at.session_state["chat_histories"][chat_key]) >= 2

    # Gõ câu hỏi tự do vào Chatbot và kiểm tra trích dẫn
    at.chat_input[0].set_value("Giải thích Embeddings và Cosine similarity trong RAG?").run()
    assert not at.exception
    hist = at.session_state["chat_histories"][chat_key]
    assert len(hist) >= 4
    assert hist[-1]["turn"].route_taken == "adaptive_explain"
    assert len(hist[-1]["turn"].citations) >= 1

    at.button("qp_oos").click().run()
    assert not at.exception
    assert at.session_state["chat_histories"][chat_key][-1]["turn"].route_taken == "refuse_out_of_doc"

    # Thử chấm ảnh Vision mẫu
    at.button("btn_run_vision").click().run()
    assert not at.exception
    assert at.session_state["vision_critique"] is not None

    # 3. Tab Quiz thích ứng & Vấn đáp Feynman
    _goto(at, "practice")
    assert not at.exception
    at.button("btn_gen_quiz").click().run()
    assert not at.exception
    assert at.session_state["quiz"] is not None

    at.button("btn_submit_quiz").click().run()
    assert not at.exception
    assert at.session_state["quiz_result"] is not None

    at.radio("practice_mode").set_value("Vấn đáp ngược Feynman (Teach-Back)").run()
    assert not at.exception
    at.selectbox[-1].set_value("Chunking và Overlap").run()
    at.button("btn_fill_feynman").click().run()
    at.button("btn_eval_feynman").click().run()
    assert not at.exception
    assert at.session_state["teach_back_eval"].understanding_score < 60

    at.button("btn_fill_feynman_good").click().run()
    at.button("btn_eval_feynman").click().run()
    assert not at.exception
    assert at.session_state["teach_back_eval"].understanding_score >= 80

    # 4. Tab Lộ trình & Số liệu và chạy Eval 20 câu
    _goto(at, "analytics")
    assert not at.exception
    at.button("btn_run_eval_ui").click().run()
    assert not at.exception
    assert at.session_state["eval_summary"]["overall_retrieval_pass_rate"] == 100.0


def test_cross_tab_remediation_and_custom_doc_paste(tmp_path: Path, monkeypatch):
    at = _new_app(tmp_path, monkeypatch, "ui_cross_test.db")
    _login_and_load_sample(at)
    lid = at.session_state["learner_id"]

    # Tạo quiz ở Tab Practice, chọn sai câu 1 rồi bấm 'Nhờ Gia sư gỡ rối câu này'
    _goto(at, "practice")
    at.button("btn_gen_quiz").click().run()
    quiz = at.session_state["quiz"]
    wrong_choice = (quiz.questions[0].correct_index + 1) % 4
    ver = at.session_state["quiz_version"]
    at.radio(f"q{ver}_0").set_value(wrong_choice).run()
    at.button("btn_submit_quiz").click().run()
    assert not at.exception
    assert at.session_state["quiz_result"] is not None

    at.button("rem_coach_1").click().run()
    assert not at.exception
    assert at.session_state["active_nav"] == "coach"
    chat_key = f"{lid}::{at.session_state['doc_id']}"
    last_msg = at.session_state["chat_histories"][chat_key][-1]
    assert last_msg["turn"].route_taken == "socratic_remediation"

    # Nạp văn bản dán tay mới rồi khôi phục giáo trình mẫu
    _goto(at, "map")
    custom_doc = (
        "1. Kiến trúc Transformer\n"
        "Transformer sử dụng cơ chế Self-Attention để tính mức độ liên quan giữa mọi từ trong câu song song thay vì xử lý tuần tự như RNN.\n\n"
        "2. Positional Encoding\n"
        "Do Self-Attention không có thứ tự thời gian, Positional Encoding cộng thêm thông tin vị trí vào vector embedding của từng token."
    )
    at.text_area("doc_paste_area").set_value(custom_doc).run()
    at.button("btn_load_paste").click().run()
    assert not at.exception
    assert len(at.session_state["concept_map"].concepts) >= 2

    at.button("btn_restore_sample_doc").click().run()
    assert not at.exception
    assert at.session_state["doc_id"] == "default_curriculum"


def test_breakthrough_lab_and_confidence_quiz(tmp_path: Path, monkeypatch):
    at = _new_app(tmp_path, monkeypatch, "ui_lab_test.db")
    _login_and_load_sample(at)

    # 1. Phòng Lab Đột Phá -> Git Diff cho Não bộ
    _goto(at, "lab")
    assert not at.exception
    from core.breakthrough import SAMPLE_NOTES
    at.text_area("diff_note_input").set_value(SAMPLE_NOTES["rag_note_loi"]["content"]).run()
    at.button("btn_run_diff").click().run()
    assert not at.exception
    assert at.session_state["diff_report"] is not None
    assert len(at.session_state["diff_report"].contradictions) >= 2

    # 2. Ca Trực Cứu Hỏa Production & chạy cả 3 tool điều tra
    at.segmented_control("lab_mode").set_value("incident").run()
    assert not at.exception
    at.button("btn_inc_t1").click().run()
    at.button("btn_inc_t2").click().run()
    at.button("btn_inc_t3").click().run()
    sc = at.session_state["incident_scenario"]
    at.radio(f"inc_pick_{sc.ticket_id}").set_value(sc.correct_option_index).run()
    at.button("btn_solve_incident").click().run()
    assert not at.exception
    assert at.session_state["incident_eval"].resolved_correctly is True
    assert at.session_state["incident_eval"].investigation_bonus == 15

    # 3. Săn Lỗi Cài Bẫy & Hội đồng Phản biện
    at.segmented_control("lab_mode").set_value("trojan_adv").run()
    assert not at.exception
    ch = at.session_state["trojan_challenge"]
    tv = at.session_state["trojan_version"]
    sfx = f"_v{tv}" if tv else ""
    for s in ch.statements:
        if s.is_buggy:
            at.checkbox(f"trojan_chk_{ch.title[:10]}_{s.id}{sfx}").set_value(True)
    at.button("btn_grade_trojan").click().run()
    assert not at.exception
    assert at.session_state["trojan_result"]["score"] == 100

    at.button("btn_fill_adv_sample").click().run()
    at.button("btn_eval_adv").click().run()
    assert not at.exception
    assert at.session_state["adv_eval"].defense_score >= 75

    # Nạp INFRA.md của nhóm để luyện Q&A
    at.button("btn_load_infra_adv").click().run()
    assert not at.exception
    assert at.session_state["doc_id"] == "project_infra_doc"

    # 4. Quiz với thanh chọn mức tự tin -> Radar Điểm mù
    _goto(at, "practice")
    at.button("btn_gen_quiz").click().run()
    ver = at.session_state["quiz_version"]
    quiz = at.session_state["quiz"]
    wrong_choice = (quiz.questions[0].correct_index + 1) % 4
    at.radio(f"q{ver}_0").set_value(wrong_choice).run()
    at.segmented_control(f"conf_{ver}_0").set_value("Chắc chắn").run()
    at.button("btn_submit_quiz").click().run()
    assert not at.exception
    assert at.session_state["quiz_result"].blind_spot_count == 1

    _goto(at, "analytics")
    assert not at.exception
