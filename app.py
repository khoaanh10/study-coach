"""Study Coach 2.0 — Gia sư AI Thích ứng theo Năng lực Cá nhân. Chạy: python -m streamlit run app.py"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st
from openai import OpenAIError

import config
from core import adaptive, breakthrough, db, vision
from core.coach_graph import run_coach_turn, stream_coach_turn
from core.contracts import (
    CoachTurnResult,
    ConceptMastery,
    DocumentConceptMap,
    GradedQuestion,
    LearnerDNA,
    Quiz,
    QuizResult,
)
from core.llm import LLMError
from core.loader import (
    load_default_samples,
    load_sections_from_pasted_text,
    load_structured_sections,
)
from core.quiz import _looks_meaningless, generate_adaptive_quiz, grade
from core.rag import DualMemoryKnowledgeBase, build_embeddings, compute_coverage_stats
from core.ui_theme import (
    clean_html,
    esc,
    icon_svg,
    inject_theme_css,
    render_bento_strip_html,
    render_citation_pills_html,
    render_concept_card_html,
    render_confidence_matrix_html,
    render_git_diff_card_html,
    render_incident_ticket_html,
    render_mastery_chart_html,
    render_brand_lockup_html,
    render_misconception_card_html,
    render_next_step_callout_html,
    render_route_banner_html,
)

# Mọi lỗi gọi LLM đều được bắt và hiện thông báo thân thiện thay vì traceback.
_LLM_ERRORS = (LLMError, OpenAIError)

st.set_page_config(page_title="Study Coach — Gia sư AI Thích ứng", layout="wide")
st.markdown(inject_theme_css(), unsafe_allow_html=True)

NAV_LABELS = {
    "map": ":material/account_tree: Bản đồ năng lực",
    "coach": ":material/psychology: Gia sư Socratic & Vision",
    "practice": ":material/quiz: Quiz thích ứng & Vấn đáp",
    "lab": ":material/science: Phòng Lab Đột Phá",
    "analytics": ":material/analytics: Lộ trình & Số liệu",
}


@st.cache_resource(show_spinner=False)
def _init_app_once() -> bool:
    db.init_db()
    vision.ensure_sample_vision_images()
    return True


@st.cache_resource(show_spinner=False)
def _load_default_kb_and_map(use_live_embeddings: bool):
    text, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(
        sections, embeddings=build_embeddings(demo_mode=not use_live_embeddings), fallback_text=text
    )
    cmap = adaptive.extract_concept_map(text, db.DEFAULT_DOC_ID, title, demo_mode=True)
    return text, sections, title, kb, cmap


def _bootstrap_state() -> None:
    _init_app_once()

    if "doc_text" not in st.session_state:
        # Không tự nạp giáo trình mẫu: người học tự upload/dán tài liệu (hoặc bấm "Dùng thử giáo trình mẫu").
        st.session_state.doc_id = ""
        st.session_state.doc_title = ""
        st.session_state.doc_text = ""
        st.session_state.doc_sections = []
        st.session_state.kb = None
        st.session_state.concept_map = None
        st.session_state.has_document = False
        st.session_state.pending_placement_after_doc = False

    defaults = {
        "active_nav": "coach",
        "learner_id": "",
        "selected_model": config.DEFAULT_MODEL,
        "chat_histories": {},
        "pending_coach": None,
        "ab_comparison": None,
        "quiz": None,
        "quiz_strategy": "",
        "quiz_result": None,
        "quiz_recorded": False,
        "quiz_version": 0,
        "quiz_focus_select": "Tự động chọn chủ đề yếu nhất",
        "quiz_diff_select": "Tự động theo năng lực",
        "quiz_difficulty_used": "Tự động",
        "quiz_focus_used": "Tự động",
        "trojan_version": 0,
        "submission_results": {},
        "diff_note_sig": None,
        "vision_force_sample": False,
        "vision_img_sig": None,
        "vision_up_sig": None,
        "coach_failed": None,
        "placement_quiz": None,
        "placement_result": None,
        "practice_mode": "Quiz trắc nghiệm thích ứng",
        "teach_back_eval": None,
        "teach_back_draft": "",
        "feynman_concept": "",
        "vision_sample_id": "so-do-rag-loi",
        "vision_critique": None,
        "eval_summary": None,
        "lab_mode": "diff",
        "diff_note_input": "",
        "diff_report": None,
        "incident_idx": 0,
        "incident_scenario": None,
        "incident_tools_used": [],
        "incident_eval": None,
        "trojan_idx": 0,
        "trojan_challenge": None,
        "trojan_result": None,
        "adv_idx": 0,
        "adv_question": None,
        "adv_defense_input": "",
        "adv_eval": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _submission_token(kind: str, *parts: object) -> str:
    """Token chống nộp lặp: loại thao tác + hồ sơ + tài liệu + nội dung đầu vào đã nộp."""
    raw = "\x1f".join(str(p) for p in (kind, st.session_state.learner_id, st.session_state.doc_id, *parts))
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _cached_submission(token: str) -> tuple[bool, object]:
    """Trả (đã_xử_lý, kết_quả_cũ) cho token; nếu đã có thì KHÔNG được chấm/ghi DB lại."""
    store: dict = st.session_state.setdefault("submission_results", {})
    return (token in store, store.get(token))


def _remember_submission(token: str, result: object) -> None:
    st.session_state.setdefault("submission_results", {})[token] = result


def _reset_learner_session_results() -> None:
    """Xóa kết quả của hồ sơ trước để hồ sơ mới không thấy dữ liệu cũ."""
    for k in (
        "quiz", "quiz_result", "placement_quiz", "placement_result", "teach_back_eval",
        "vision_critique", "diff_report", "incident_eval", "trojan_result", "adv_eval",
        "pending_coach", "coach_failed",
    ):
        st.session_state[k] = None
    st.session_state.quiz_recorded = False
    st.session_state.incident_tools_used = []
    st.session_state.teach_back_draft = ""
    st.session_state.adv_defense_input = ""
    st.session_state.trojan_version = st.session_state.get("trojan_version", 0) + 1
    st.session_state.submission_results = {}


def _on_runtime_key_change() -> None:
    key = (st.session_state.get("runtime_key_input") or "").strip()
    st.session_state["session_api_key"] = key
    config.set_runtime_api_key(key or None)


def _is_demo_mode() -> bool:
    """Chỉ còn AI Portal Live: chưa có key thì coi như chưa sẵn sàng (không có chế độ Demo trên UI)."""
    return not config.has_live_api_key()


def _current_dna() -> LearnerDNA:
    cmap: DocumentConceptMap | None = st.session_state.get("concept_map")
    order = [c.name for c in cmap.concepts] if cmap is not None else None
    return db.get_learner_dna(
        learner_id=st.session_state.learner_id,
        doc_id=st.session_state.get("doc_id") or db.DEFAULT_DOC_ID,
        concepts_order=order,
    )


def _set_nav(tab: str) -> None:
    st.session_state["active_nav"] = tab
    st.session_state["_nav_ver"] = st.session_state.get("_nav_ver", 0) + 1


def _jump_to_coach_topic(topic: str) -> None:
    _set_nav("coach")
    st.session_state.pending_coach = {
        "query": f"Hãy giảng cho tôi chủ đề '{topic}' theo đúng trình độ hiện tại của tôi.",
        "focus_topic": topic,
        "remediation_context": None,
    }


def _jump_to_coach_remediation(item: GradedQuestion) -> None:
    chosen_txt = item.options[item.chosen_index] if item.chosen_index is not None else "Bỏ trống"
    correct_txt = item.options[item.correct_index]
    _set_nav("coach")
    st.session_state.pending_coach = {
        "query": f"Tại sao câu '{item.question}' lại chọn '{correct_txt}' chứ không phải '{chosen_txt}'?",
        "focus_topic": item.topic,
        "remediation_context": (
            f"Câu hỏi: {item.question} | Học viên chọn: {chosen_txt} | "
            f"Đáp án đúng: {correct_txt} | Lỗi chẩn đoán: {item.diagnosed_misconception}"
        ),
    }


def _jump_to_quiz_topic(topic: str) -> None:
    _set_nav("practice")
    st.session_state.practice_mode = "Quiz trắc nghiệm thích ứng"
    st.session_state.quiz_focus_select = topic
    st.session_state.quiz = None
    st.session_state.quiz_result = None
    st.session_state.quiz_recorded = False


def _jump_to_feynman_topic(topic: str) -> None:
    _set_nav("practice")
    st.session_state.practice_mode = "Vấn đáp ngược Feynman (Teach-Back)"
    st.session_state.feynman_concept = topic
    st.session_state.teach_back_eval = None


def _fill_feynman_sample(concept: str, good: bool) -> None:
    if good:
        if "chunk" in concept.lower():
            st.session_state.teach_back_draft = (
                "Tài liệu dài được chia thành các đoạn nhỏ gọi là chunk. Chunk quá lớn làm truy xuất kém chính xác vì "
                "chứa nhiều ý lẫn lộn, còn chunk quá nhỏ làm mất ngữ cảnh. Vì vậy cần để các chunk gối lên nhau gọi là "
                "overlap để ý ở ranh giới không bị cắt đứt, đồng thời lưu kèm metadata tên tài liệu và số trang để trích nguồn."
            )
        else:
            st.session_state.teach_back_draft = (
                f"Khái niệm {concept} giúp hệ thống xử lý dữ liệu bám sát tài liệu gốc, tách thông tin theo cấu trúc rõ ràng, "
                "lưu kèm ngữ cảnh và kiểm chứng nguồn trích dẫn để giảm thiểu sai lệch hoặc bịa đặt thông tin."
            )
    else:
        st.session_state.teach_back_draft = (
            f"Theo em hiểu về {concept}, ta chia văn bản thành đoạn lớn cho đỡ tốn chi phí, "
            "còn chunk quá lớn thì làm mất ngữ cảnh nên cần đặt overlap bằng 0 để tiết kiệm bộ nhớ."
        )


def _retry_quiz() -> None:
    st.session_state.quiz_result = None
    st.session_state.quiz_recorded = False
    st.session_state.quiz_version = st.session_state.get("quiz_version", 0) + 1


def _new_quiz() -> None:
    st.session_state.quiz = None
    st.session_state.quiz_result = None
    st.session_state.quiz_recorded = False


def _reset_doc_scoped_inputs(cmap: DocumentConceptMap | None) -> None:
    st.session_state.diff_note_input = ""
    st.session_state.teach_back_draft = ""
    st.session_state.adv_defense_input = ""
    st.session_state.diff_note_sig = None
    st.session_state.vision_img_sig = None
    st.session_state.vision_critique = None
    st.session_state.pending_coach = None
    st.session_state.coach_failed = None
    st.session_state.feynman_concept = cmap.concepts[0].name if cmap is not None and cmap.concepts else ""


def _apply_new_document(text: str, sections: list[dict], title: str, doc_id: str) -> None:
    demo = _is_demo_mode()
    kb = _build_cached_doc_kb(doc_id, not demo, sections, text)
    cmap = adaptive.extract_concept_map(
        text,
        doc_id=doc_id,
        doc_title=title,
        model=st.session_state.selected_model,
        demo_mode=demo,
    )
    if st.session_state.get("learner_id") and doc_id != db.DEFAULT_DOC_ID:
        db.save_learner_document(st.session_state.learner_id, doc_id, title, text, sections, cmap)
    st.session_state.doc_id = doc_id
    st.session_state.doc_title = title
    st.session_state.doc_text = text
    st.session_state.doc_sections = sections
    st.session_state.kb = kb
    st.session_state.concept_map = cmap
    # Hội thoại cũ thuộc tài liệu cũ: giữ lại sẽ khiến Gia sư bám chủ đề không còn trong tài liệu mới.
    st.session_state.chat_histories = {}
    _reset_learner_session_results()
    _reset_doc_scoped_inputs(cmap)
    st.session_state.quiz = None
    st.session_state.quiz_result = None
    st.session_state.placement_quiz = None
    st.session_state.placement_result = None
    st.session_state.teach_back_eval = None
    st.session_state.ab_comparison = None
    st.session_state.diff_report = None
    st.session_state.incident_scenario = None
    st.session_state.incident_tools_used = []
    st.session_state.incident_eval = None
    st.session_state.trojan_challenge = None
    st.session_state.trojan_result = None
    st.session_state.adv_question = None
    st.session_state.adv_eval = None
    st.session_state.quiz_focus_select = "Tự động chọn chủ đề yếu nhất"
    st.session_state.trojan_version = st.session_state.get("trojan_version", 0) + 1
    st.session_state.submission_results = {}


@st.cache_resource(show_spinner=False)
def _build_cached_doc_kb(doc_id: str, live: bool, _sections: list[dict], _doc_text: str) -> DualMemoryKnowledgeBase:
    """Cache FAISS index theo (doc_id, live) để đăng nhập lại không phải gọi lại OpenAI Embeddings."""
    return DualMemoryKnowledgeBase(
        _sections,
        embeddings=build_embeddings(demo_mode=not live),
        fallback_text=_doc_text,
    )


def _restore_saved_document(learner_id: str, doc_id: str) -> bool:
    """Mở lại tài liệu đã lưu của profile (dùng cache FAISS, không gọi LLM dựng lại concept map)."""
    saved = db.load_learner_document(learner_id, doc_id)
    if saved is None:
        return False
    live = config.has_live_api_key() and not _is_demo_mode()
    kb = _build_cached_doc_kb(doc_id, live, saved["sections"], saved["doc_text"])
    st.session_state.doc_id = doc_id
    st.session_state.doc_title = saved["title"]
    st.session_state.doc_text = saved["doc_text"]
    st.session_state.doc_sections = saved["sections"]
    st.session_state.kb = kb
    st.session_state.concept_map = saved["concept_map"]
    st.session_state.has_document = True
    st.session_state.chat_histories = {}
    _reset_learner_session_results()
    _reset_doc_scoped_inputs(saved["concept_map"])
    kb.sync_learner_mistakes(db.get_learner_dna(learner_id, doc_id))
    return True


LEVEL_OPTIONS = ["Mới bắt đầu", "Trung cấp", "Nâng cao"]


@st.cache_resource(show_spinner=False)
def _session_signing_key() -> bytes:
    return hashlib.sha256(f"study-coach-session::{config.DB_PATH}".encode("utf-8")).digest()


def _make_session_token(learner_id: str) -> str:
    import hmac

    sig = hmac.new(_session_signing_key(), learner_id.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    return f"{learner_id}.{sig}"


def _verify_session_token(token: str) -> str:
    import hmac

    if not token or "." not in token:
        return ""
    lid, sig = token.rsplit(".", 1)
    expected = hmac.new(_session_signing_key(), lid.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    if not hmac.compare_digest(sig, expected):
        return ""
    if not db.get_user_display_name(lid):
        return ""
    return lid


def _logout() -> None:
    st.session_state["logged_in_learner_id"] = ""
    st.session_state["learner_id"] = ""
    st.session_state["_logged_out"] = True
    st.session_state["active_nav"] = "coach"
    _reset_learner_session_results()
    st.session_state.doc_id = ""
    st.session_state.doc_title = ""
    st.session_state.doc_text = ""
    st.session_state.doc_sections = []
    st.session_state.kb = None
    st.session_state.concept_map = None
    st.session_state.has_document = False
    st.session_state.pending_placement_after_doc = False
    st.session_state.chat_histories = {}
    try:
        st.query_params.pop("session", None)
    except Exception:
        pass


def _do_login(learner_id: str, start_placement: bool = False) -> None:
    st.session_state["logged_in_learner_id"] = learner_id
    st.session_state["learner_id"] = learner_id
    st.session_state["_logged_out"] = False
    _reset_learner_session_results()
    st.session_state["pending_placement_after_doc"] = bool(start_placement)
    st.session_state["active_nav"] = "coach"
    try:
        st.query_params.pop("session", None)
    except Exception:
        pass
    if not st.session_state.get("has_document"):
        saved_docs = db.list_learner_documents(learner_id)
        if saved_docs:
            _restore_saved_document(learner_id, saved_docs[0]["doc_id"])
    if start_placement and st.session_state.get("has_document") and st.session_state.get("concept_map") is not None:
        st.session_state["pending_placement_after_doc"] = False
        _open_placement_if_new()


def _open_placement_if_new() -> bool:
    """Mở bài test đầu vào nếu hồ sơ chưa có lịch sử ở tài liệu hiện tại. Trả True nếu đã mở."""
    dna = db.get_learner_dna(st.session_state.learner_id, st.session_state.doc_id)
    if any(c.attempts > 0 for c in dna.concepts):
        return False
    st.session_state["active_nav"] = "map"
    st.session_state.placement_quiz = adaptive.generate_placement_test(
        text=st.session_state.doc_text,
        concept_map=st.session_state.concept_map,
        demo_mode=_is_demo_mode(),
        model=st.session_state.selected_model,
    )
    return True


def _finish_document_load() -> None:
    """Sau khi nạp tài liệu: đánh dấu đã có tài liệu, mở test đầu vào nếu đang chờ, rồi rerun."""
    st.session_state["has_document"] = True
    st.session_state["active_nav"] = "coach"
    if st.session_state.pop("pending_placement_after_doc", False):
        try:
            _open_placement_if_new()
        except _LLM_ERRORS as e:
            st.session_state["active_nav"] = "coach"
            st.toast(str(e))
    st.rerun()


def _close_document() -> None:
    st.session_state.doc_id = ""
    st.session_state.doc_title = ""
    st.session_state.doc_text = ""
    st.session_state.doc_sections = []
    st.session_state.kb = None
    st.session_state.concept_map = None
    st.session_state.has_document = False
    st.session_state.pending_placement_after_doc = False
    st.session_state.placement_quiz = None
    st.session_state.placement_result = None
    st.session_state.quiz = None
    st.session_state.quiz_result = None
    st.session_state.pending_coach = None
    st.session_state.active_nav = "coach"


def _load_sample_document() -> None:
    text, secs, title, kb, cmap = _load_default_kb_and_map(config.has_live_api_key())
    st.session_state.doc_id = db.DEFAULT_DOC_ID
    st.session_state.doc_title = title
    st.session_state.doc_text = text
    st.session_state.doc_sections = secs
    st.session_state.kb = kb
    st.session_state.concept_map = cmap
    st.session_state.chat_histories = {}


def _render_login_screen() -> None:
    """Màn hình đăng nhập tự do: người học tự nhập tên, trình độ ban đầu đánh giá qua bài test đầu vào."""
    _, mid, _ = st.columns([1.2, 2.2, 1.2])
    with mid:
        st.markdown(
            '<div class="sc-login-card">'
            + render_brand_lockup_html("Gia sư AI cá nhân hóa theo năng lực của bạn")
            + "<p>Đăng nhập bằng tài khoản của bạn hoặc đăng ký mới. Trình độ ban đầu được đánh giá qua bài kiểm tra đầu vào.</p></div>",
            unsafe_allow_html=True,
        )
        tab_login, tab_register = st.tabs(["Đăng nhập", "Đăng ký"])
        api_key = ""
        with tab_login:
            with st.form("form_login", border=True):
                username = st.text_input("Tên đăng nhập", key="login_username_input")
                password = st.text_input("Mật khẩu", type="password", key="login_password_input")
                if not config.has_live_api_key():
                    api_key = st.text_input(
                        "AI Portal API Key",
                        type="password",
                        key="login_api_key_input",
                        placeholder="Dán key AI Portal để chạy Live...",
                    )
                submitted_login = st.form_submit_button(
                    "Đăng nhập", key="btn_login_submit", type="primary", width="stretch"
                )
        with tab_register:
            with st.form("form_register", border=True):
                reg_username = st.text_input("Tên đăng nhập", key="reg_username_input")
                reg_display = st.text_input(
                    "Họ và tên hiển thị", key="reg_display_name_input", placeholder="Ví dụ: Nguyễn Khoa Anh"
                )
                reg_password = st.text_input("Mật khẩu", type="password", key="reg_password_input")
                reg_password2 = st.text_input("Nhập lại mật khẩu", type="password", key="reg_password2_input")
                submitted_register = st.form_submit_button(
                    "Tạo tài khoản & Đánh giá đầu vào", key="btn_register_submit", type="primary", width="stretch"
                )

        def _apply_api_key() -> None:
            if (api_key or "").strip():
                config.set_runtime_api_key(api_key.strip())
                st.session_state["session_api_key"] = api_key.strip()

        if submitted_login:
            if not (username or "").strip() or not password:
                st.warning("Hãy nhập tên đăng nhập và mật khẩu.")
            else:
                with st.spinner("🔐 Đang xác thực tài khoản và tải không gian học tập..."):
                    lid = db.authenticate_user(username.strip(), password)
                    if lid:
                        _apply_api_key()
                        _do_login(lid, start_placement=False)
                if not lid:
                    st.error("Sai tên đăng nhập hoặc mật khẩu.")
                else:
                    st.rerun()
        elif submitted_register:
            if not (reg_username or "").strip() or not reg_password:
                st.warning("Hãy nhập tên đăng nhập và mật khẩu.")
            elif reg_password != reg_password2:
                st.error("Mật khẩu nhập lại không khớp.")
            else:
                try:
                    with st.spinner("✨ Đang tạo tài khoản và khởi tạo không gian học tập..."):
                        lid = db.register_user(
                            reg_username.strip(), reg_password, display_name=(reg_display or "").strip()
                        )
                        _apply_api_key()
                        _do_login(lid, start_placement=True)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()


def _toggle_dashboard_strip() -> None:
    st.session_state.show_dashboard_strip = not st.session_state.get("show_dashboard_strip", True)


def _toggle_chat_focus() -> None:
    st.session_state.chat_focus_mode = not st.session_state.get("chat_focus_mode", False)


def _reset_current_profile_cb() -> None:
    db.reset_learner_profile(st.session_state.learner_id, st.session_state.doc_id)
    _reset_learner_session_results()


def _render_header(dna: LearnerDNA) -> None:
    """Header 1 hàng: brand · tài khoản · báo cáo học tập · cài đặt."""
    c_brand, c_acc, c_rep, c_cfg = st.columns([5.8, 2.3, 2.1, 1.4], vertical_alignment="center")
    with c_brand:
        st.markdown(
            render_brand_lockup_html("Gia sư AI hiểu năng lực của từng người học"),
            unsafe_allow_html=True,
        )
    with c_acc:
        with st.popover(f"👤 {dna.name}", width="stretch"):
            st.markdown(f"**{dna.name}**")
            st.caption(f"Trình độ hiện tại: {dna.overall_level}")
            st.button(
                "🚪 Đăng xuất",
                key="btn_logout",
                on_click=_logout,
                width="stretch",
            )
            if st.button(
                "Làm mới tiến độ",
                icon=":material/restart_alt:",
                key="btn_reset_one",
                on_click=_reset_current_profile_cb,
                width="stretch",
            ):
                st.toast("Đã đặt lại tiến độ của tài khoản hiện tại.")
    with c_rep:
        dash_open = st.session_state.get("show_dashboard_strip", True)
        st.button(
            "✖ Đóng hồ sơ" if dash_open else "📊 Hồ sơ & Cài đặt",
            key="btn_toggle_dashboard",
            on_click=_toggle_dashboard_strip,
            width="stretch",
        )
    with c_cfg:
        with st.popover("⚙️ Cài đặt", width="stretch"):
            st.markdown("**AI Portal Live**")
            st.text_input(
                "API Key",
                type="password",
                placeholder="Dán key AI Portal tại đây...",
                key="runtime_key_input",
                on_change=_on_runtime_key_change,
            )
            st.text_input(
                "Base URL",
                value=config.get_effective_base_url(),
                key="runtime_base_url_input",
                on_change=lambda: config.set_runtime_base_url(st.session_state.get("runtime_base_url_input")),
            )
            st.selectbox("Model", options=config.MODEL_CHOICES, key="selected_model")
            st.caption(
                f"Trạng thái: {'● Live AI Portal' if config.has_live_api_key() else '○ Chưa có API key'} · "
                f"Ngân sách phiên: ${db.get_session_cost_usd():.4f} / ${config.MAX_SESSION_COST_USD:.2f}"
            )
            st.divider()
            st.markdown("**Quản lý tài liệu**")
            _render_doc_manager()


def _render_tab_map(dna: LearnerDNA, coverage: dict) -> None:
    cmap: DocumentConceptMap = st.session_state.concept_map
    mastery_by_name = {c.concept: c for c in dna.concepts}
    plan = adaptive.build_study_plan(dna, cmap)

    # Banner & Chẩn đoán đầu vào 60s
    c_intro, c_act = st.columns([7.5, 4.5], vertical_alignment="center")
    with c_intro:
        st.markdown('<p class="sc-section-eyebrow">Bản đồ nhiệt kiến thức cá nhân hóa</p>', unsafe_allow_html=True)
        st.write(
            f"Tài liệu **{cmap.doc_title}** gồm **{len(cmap.concepts)} khái niệm cốt lõi** có liên kết tiên quyết. "
            f"Hiện đã làm chủ **{coverage['mastered']}/{coverage['total']}** mục."
        )
    n_pl = adaptive.placement_question_count(len(cmap.concepts), len(st.session_state.doc_text or ""))
    with c_act:
        if st.button(
            f"Chẩn đoán trình độ đầu vào ({n_pl} câu)",
            icon=":material/fact_check:",
            type="primary",
            key="btn_start_placement",
            width="stretch",
        ):
            try:
                with st.spinner(f"Đang tạo bài chẩn đoán {n_pl} câu phủ đều các chủ đề (thang Bloom)..."):
                    st.session_state.placement_quiz = adaptive.generate_placement_test(
                        text=st.session_state.doc_text,
                        concept_map=cmap,
                        model=st.session_state.selected_model,
                        demo_mode=_is_demo_mode(),
                    )
                    st.session_state.placement_result = None
            except _LLM_ERRORS as e:
                st.error(str(e))

    if plan:
        st.markdown(render_next_step_callout_html(dna, plan[0]), unsafe_allow_html=True)

    # Placement test panel
    p_quiz: Quiz | None = st.session_state.get("placement_quiz")
    if p_quiz and p_quiz.questions:
        with st.container(border=True):
            st.markdown("#### Bài Chẩn đoán Năng lực Đầu vào (Thang Bloom: Nhớ → Hiểu → Vận dụng)")
            if st.session_state.get("placement_result") is None:
                with st.form("placement_quiz_form", border=False):
                    p_picks = []
                    for idx, pq in enumerate(p_quiz.questions):
                        st.markdown(f"**Câu {idx + 1} [{pq.bloom_level} · {pq.topic}].** {pq.question}")
                        pick = st.radio(
                            f"placement_{idx}",
                            options=range(len(pq.options)),
                            format_func=lambda k, opts=pq.options: f"{'ABCD'[k]}. {opts[k]}",
                            index=None,
                            key=f"pl_q_{idx}",
                            label_visibility="collapsed",
                        )
                        p_picks.append(pick)
                    submitted_pl = st.form_submit_button(
                        "Chấm bài & Cập nhật hồ sơ",
                        icon=":material/check:",
                        type="primary",
                        key="btn_submit_placement",
                        width="stretch",
                    )
                if submitted_pl:
                    res = grade(p_quiz, p_picks)
                    adaptive.record_quiz_outcome(
                        learner_id=dna.learner_id,
                        doc_id=st.session_state.doc_id,
                        result=res,
                        model=st.session_state.selected_model,
                        difficulty="Chẩn đoán",
                        concepts_order=[c.name for c in cmap.concepts],
                    )
                    st.session_state.placement_result = res
                    st.rerun()
                if st.button(
                    "Bỏ qua bài kiểm tra (tính 0%)",
                    icon=":material/skip_next:",
                    key="btn_close_placement",
                    width="stretch",
                    help="Không làm bài đầu vào: hồ sơ giữ ở mức 0% (Mới bắt đầu), bạn có thể làm lại bất cứ lúc nào.",
                ):
                    st.session_state.placement_quiz = None
                    st.toast("Đã bỏ qua bài kiểm tra đầu vào, hồ sơ giữ ở mức 0%.")
                    st.rerun()
            else:
                res: QuizResult = st.session_state.placement_result
                st.success(
                    f"Kết quả chẩn đoán: **{res.score}/{res.total}** · Đề xuất trình độ khởi điểm: **{res.recommended_level}**. "
                    "Bản đồ nhiệt bên dưới đã được cập nhật."
                )
                if res.detected_misconceptions:
                    for m in res.detected_misconceptions:
                        st.markdown(render_misconception_card_html("Phát hiện lỗi tư duy", m), unsafe_allow_html=True)
                if st.button("Hoàn tất & Ẩn bảng chẩn đoán", key="btn_done_placement"):
                    st.session_state.placement_quiz = None
                    st.session_state.placement_result = None
                    st.rerun()

    # Grid Bản đồ nhiệt
    cols = st.columns(2, gap="medium")
    for idx, node in enumerate(cmap.concepts):
        m = mastery_by_name.get(node.name) or ConceptMastery(concept=node.name)
        with cols[idx % 2]:
            st.markdown(render_concept_card_html(node, m), unsafe_allow_html=True)
            if m.attempts > 0:
                _sc = m.mastery_score
                if 65 <= _sc <= 69:
                    st.caption("Trần mức Nhớ (68%) · làm Quiz Trung bình/Khó để vượt")
                elif 80 <= _sc <= 85:
                    st.caption("Trần mức Hiểu (84%) · làm Quiz Khó hoặc Lab để lên 100%")
                elif 93 <= _sc <= 97:
                    st.caption("Trần mức Vận dụng (96%) · làm Quiz Khó hoặc Lab để chạm 100%")
            b1, b2 = st.columns(2)
            with b1:
                st.button(
                    "Học với Gia sư",
                    icon=":material/psychology:",
                    key=f"map_coach_{idx}",
                    on_click=_jump_to_coach_topic,
                    args=(node.name,),
                    width="stretch",
                )
            with b2:
                st.button(
                    "Luyện Quiz mục này",
                    icon=":material/quiz:",
                    key=f"map_quiz_{idx}",
                    on_click=_jump_to_quiz_topic,
                    args=(node.name,),
                    width="stretch",
                )


def _render_document_onboarding(dna: LearnerDNA) -> None:
    """Màn nạp tài liệu đầu tiên: hiển thị khi chưa có tài liệu nào."""
    _render_header(dna)
    _, mid, _ = st.columns([0.9, 3.0, 0.9])
    with mid:
        st.markdown("### 📚 Nạp tài liệu học tập của bạn")
        st.caption(
            "Tải lên hoặc dán tài liệu bạn muốn học. Gia sư sẽ dựng bản đồ năng lực, "
            "bài kiểm tra và hội thoại bám sát chính tài liệu đó."
        )
        _render_doc_manager(onboarding=True)
        st.divider()
        if st.button("✨ Dùng thử giáo trình mẫu (RAG Cơ bản)", key="btn_load_sample_onboarding", width="stretch"):
            with st.status("✨ Đang chuẩn bị giáo trình mẫu...", expanded=True) as status:
                st.write("📖 Đọc nội dung giáo trình & lập bản đồ tri thức...")
                _load_sample_document()
                status.update(label="✅ Giáo trình mẫu đã sẵn sàng!", state="complete", expanded=False)
            _finish_document_load()


def _render_doc_manager(onboarding: bool = False) -> None:
    """Nạp tài liệu (upload / dán text / giáo trình mẫu). Dùng cho popover Cài đặt và màn nạp tài liệu đầu tiên."""
    sfx = "_onb" if onboarding else ""
    saved_docs = db.list_learner_documents(st.session_state.learner_id) if st.session_state.get("learner_id") else []
    if saved_docs:
        st.markdown("**📚 Tài liệu đã lưu của bạn**")
        for sd in saved_docs:
            is_current = st.session_state.get("has_document") and st.session_state.get("doc_id") == sd["doc_id"]
            if st.button(
                f"{'✅ ' if is_current else '📄 '}{sd['title']}",
                key=f"btn_open_saved_{sd['doc_id']}{sfx}",
                width="stretch",
                disabled=bool(is_current),
            ):
                if _restore_saved_document(st.session_state.learner_id, sd["doc_id"]):
                    _finish_document_load()
                else:
                    st.error("Không mở lại được tài liệu này.")
    with st.container():
        t_file, t_paste = st.tabs(["📤 Tải tệp (PDF / TXT / MD)", "📋 Dán nội dung"])
        with t_file:
            uploaded = st.file_uploader(
                "Chọn một hoặc nhiều file tài liệu học tập",
                type=["pdf", "txt", "md"],
                accept_multiple_files=True,
                key="doc_uploader" + sfx,
            )
            if uploaded:
                st.caption(f"Đã chọn {len(uploaded)} file. Giới hạn xử lý quiz {config.MAX_INPUT_CHARS:,} ký tự; toàn bộ nội dung được index tự động.")
            if st.button("Nạp file & Dựng Bản đồ Năng lực", icon=":material/auto_awesome:", type="primary", key="btn_load_files" + sfx, width="stretch"):
                if not uploaded:
                    st.warning("Hãy chọn ít nhất 1 file PDF, TXT hoặc MD trước.")
                else:
                    try:
                        with st.status("✨ Đang phân tích tài liệu của bạn...", expanded=True) as status:
                            st.write("📄 Đọc nội dung & tách cấu trúc các mục...")
                            file_tuples = [
                                (f.name[:-3] + ".txt" if f.name.lower().endswith(".md") else f.name, f.getvalue())
                                for f in uploaded
                            ]
                            combined, secs = load_structured_sections(file_tuples)
                            if len(combined.strip()) < 100 or _looks_meaningless(combined):
                                raise ValueError("Nội dung tài liệu quá ngắn hoặc không chứa thông tin học tập rõ ràng.")
                            new_id = "doc_" + hashlib.sha256(combined[:2000].encode("utf-8")).hexdigest()[:10]
                            title = ", ".join(f.name for f in uploaded[:2]) + (f" (+{len(uploaded) - 2})" if len(uploaded) > 2 else "")
                            st.write("🧠 Xây dựng Bản đồ tri thức & chuẩn bị Gia sư AI...")
                            _apply_new_document(combined, secs, title, new_id)
                            status.update(label=f"✅ Đã nạp '{title}' thành công!", state="complete", expanded=False)
                        st.toast(f"Đã nạp tài liệu '{title}' và lập bản đồ khái niệm.")
                        _finish_document_load()
                    except (ValueError, *_LLM_ERRORS) as e:
                        st.error(str(e))
        with t_paste:
            pasted = st.text_area(
                "Dán nội dung bài học / ghi chú",
                height=190,
                placeholder="Dán nội dung tài liệu (tối thiểu 100 ký tự)...",
                key="doc_paste_area" + sfx,
            )
            if pasted:
                st.caption(f"{len(pasted):,} ký tự")
            col_p1, col_p2 = st.columns([2, 1]) if not onboarding else (st.container(), None)
            with col_p1:
                if st.button("Nạp văn bản & Lập bản đồ", icon=":material/auto_awesome:", type="primary", key="btn_load_paste" + sfx, width="stretch"):
                    if len(pasted.strip()) < 100 or _looks_meaningless(pasted):
                        st.error("Văn bản quá ngắn hoặc không có nội dung kiến thức rõ ràng (cần tối thiểu 100 ký tự có nghĩa).")
                    else:
                        try:
                            with st.status("✨ Đang phân tích nội dung bài học...", expanded=True) as status:
                                st.write("📄 Tách mục & xây dựng Bản đồ tri thức...")
                                combined, secs = load_sections_from_pasted_text(pasted, "Tài liệu dán tay")
                                new_id = "paste_" + hashlib.sha256(combined[:2000].encode("utf-8")).hexdigest()[:10]
                                _apply_new_document(combined, secs, "Tài liệu văn bản cá nhân", new_id)
                                status.update(label="✅ Đã nạp văn bản thành công!", state="complete", expanded=False)
                            st.toast("Đã nạp văn bản mới và lập bản đồ khái niệm.")
                            _finish_document_load()
                        except _LLM_ERRORS as e:
                            st.error(str(e))
            if col_p2 is not None:
                with col_p2:
                    if st.button("Dùng lại giáo trình mẫu", icon=":material/restore:", key="btn_restore_sample_doc", width="stretch"):
                        try:
                            text, secs, title = load_default_samples()
                            _apply_new_document(text, secs, title, db.DEFAULT_DOC_ID)
                            _finish_document_load()
                        except _LLM_ERRORS as e:
                            st.error(str(e))
    if not onboarding and st.session_state.get("has_document"):
        if st.button("Đóng tài liệu hiện tại", icon=":material/close:", key="btn_close_doc", width="stretch"):
            _close_document()
            st.rerun()


def _chat_key(dna: LearnerDNA) -> str:
    return f"{dna.learner_id}::{st.session_state.doc_id}"


def _get_chat_history(dna: LearnerDNA) -> list:
    histories: dict = st.session_state.chat_histories
    key = _chat_key(dna)
    if key not in histories:
        histories[key] = db.load_chat_history(dna.learner_id, st.session_state.doc_id)
    return histories[key]


def _execute_coach_query(
    query: str,
    dna: LearnerDNA,
    focus_topic: str | None = None,
    remediation_context: str | None = None,
) -> None:
    history = _get_chat_history(dna)

    # Chỉ ghi vào lịch sử SAU khi thành công để không để lại tin nhắn người dùng mồ côi.
    turn = run_coach_turn(
        query=query,
        kb=st.session_state.kb,
        learner_dna=dna,
        concept_map=st.session_state.concept_map,
        doc_id=st.session_state.doc_id,
        focus_topic=focus_topic,
        remediation_context=remediation_context,
        model=st.session_state.selected_model,
        demo_mode=_is_demo_mode(),
        chat_history=list(history),
    )
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": turn.answer, "turn": turn})
    db.append_chat_message(dna.learner_id, st.session_state.doc_id, "user", query)
    db.append_chat_message(dna.learner_id, st.session_state.doc_id, "assistant", turn.answer, turn=turn)


def _fill_diff_note(note_key: str) -> None:
    st.session_state.diff_note_input = breakthrough.SAMPLE_NOTES[note_key]["content"]
    st.session_state.diff_report = None


def _fill_adv_sample(sample_answer: str) -> None:
    st.session_state.adv_defense_input = sample_answer


def _render_tab_lab(dna: LearnerDNA) -> None:
    cmap: DocumentConceptMap = st.session_state.concept_map
    concept_names = [c.name for c in cmap.concepts]

    lab_labels = {
        "diff": ":material/difference: Git Diff cho Não bộ",
        "incident": ":material/local_fire_department: Ca Trực Cứu Hỏa Production",
        "trojan_adv": ":material/bug_report: Săn Lỗi & Phản Biện Q&A",
    }
    lab_mode = st.segmented_control(
        "Chế độ Phòng Lab Đột Phá",
        options=list(lab_labels.keys()),
        format_func=lambda k: lab_labels[k],
        selection_mode="single",
        key="lab_mode",
        label_visibility="collapsed",
        width="stretch",
    )
    if not lab_mode:
        lab_mode = "diff"

    if lab_mode == "diff":
        st.markdown('<p class="sc-section-eyebrow">Git Diff cho Não bộ · Soi mâu thuẫn giữa Ghi chú cá nhân &amp; Giáo trình gốc</p>', unsafe_allow_html=True)
        c_in, c_out = st.columns([5.5, 6.5], gap="large")
        with c_in:
            with st.container(border=True):
                st.caption("Khác với NotebookLM gộp chung mọi file, tính năng này tách bạch Ground Truth (Giáo trình) và My Notes (Ghi chép cá nhân) để phát hiện chỗ hiểu ngược và điểm mù.")
                if st.session_state.doc_id == db.DEFAULT_DOC_ID:
                    sn1, sn2 = st.columns(2)
                    with sn1:
                        st.button(
                            "Mẫu Note RAG lỗi",
                            icon=":material/description:",
                            key="btn_note_rag",
                            on_click=_fill_diff_note,
                            args=("rag_note_loi",),
                            width="stretch",
                        )
                    with sn2:
                        st.button(
                            "Mẫu Note API lỗi",
                            icon=":material/description:",
                            key="btn_note_api",
                            on_click=_fill_diff_note,
                            args=("api_note_loi",),
                            width="stretch",
                        )
                note_file = st.file_uploader("Hoặc tải file ghi chú cá nhân (.txt / .md)", type=["txt", "md"], key="diff_note_uploader")
                if note_file is not None:
                    note_bytes = note_file.getvalue()
                    note_sig = (note_file.name, len(note_bytes))
                    # Chỉ nạp khi là file mới để người dùng vẫn sửa tay được nội dung sau đó.
                    if st.session_state.get("diff_note_sig") != note_sig:
                        st.session_state.diff_note_sig = note_sig
                        st.session_state.diff_note_input = note_bytes.decode("utf-8", errors="replace")
                else:
                    st.session_state.diff_note_sig = None

                note_txt = st.text_area(
                    "Nội dung ghi chú / tóm tắt của bạn",
                    key="diff_note_input",
                    height=220,
                    placeholder="Dán ghi chú ôn tập của bạn vào đây để AI đối chiếu với tài liệu nguồn...",
                )
                if st.button(
                    "Soi mâu thuẫn với Giáo trình gốc",
                    icon=":material/difference:",
                    type="primary",
                    key="btn_run_diff",
                    width="stretch",
                ):
                    if len(note_txt.strip()) < 25:
                        st.warning("Hãy nhập ít nhất 1–2 câu ghi chú (tối thiểu 25 ký tự) để đối chiếu.")
                    else:
                        tok = _submission_token("diff", note_txt)
                        done, old = _cached_submission(tok)
                        if done:
                            # Đã chấm đúng ghi chú này rồi: không gọi lại/ghi DB lần nữa.
                            st.session_state.diff_report = old
                            st.rerun()
                        try:
                            with st.spinner("Đang đối chiếu từng nhận định qua FAISS & phát hiện điểm mù..."):
                                report, _ = breakthrough.run_cognitive_diff(
                                    learner_note=note_txt,
                                    kb=st.session_state.kb,
                                    concept_map=cmap,
                                    learner_dna=dna,
                                    doc_id=st.session_state.doc_id,
                                    model=st.session_state.selected_model,
                                    demo_mode=_is_demo_mode(),
                                )
                            _remember_submission(tok, report)
                            st.session_state.diff_report = report
                            st.toast(f"Hoàn tất Cognitive Diff · Khớp {report.alignment_score}/100.")
                            st.rerun()
                        except _LLM_ERRORS as e:
                            st.error(str(e))

        with c_out:
            rep = st.session_state.get("diff_report")
            with st.container(border=True):
                st.markdown("#### Kết quả Đối chiếu (Cognitive Diff)")
                if not rep:
                    st.info("Bấm nút bên trái để chạy đối chiếu ghi chú cá nhân với giáo trình gốc.")
                else:
                    st.markdown(
                        f'<div class="sc-score-hero">{rep.alignment_score}<small>/100 độ khớp</small></div>',
                        unsafe_allow_html=True,
                    )
                    st.write(rep.summary)
                    if rep.contradictions:
                        st.markdown("**1. Hiểu ngược / Mâu thuẫn cần sửa ngay (đã cập nhật vào Learner DNA):**")
                        for c_item in rep.contradictions:
                            st.markdown(render_git_diff_card_html(c_item), unsafe_allow_html=True)
                    if rep.blind_spots:
                        with st.expander(f"2. Điểm mù bỏ sót ({len(rep.blind_spots)} khái niệm chưa có trong ghi chú)", icon=":material/insights:", expanded=True):
                            for b in rep.blind_spots:
                                st.markdown(f"- **{b.topic}** `{b.page_ref}` — {b.importance_reason}")
                    if rep.verified_points:
                        with st.expander(f"3. Điểm ghi đúng chuẩn giáo trình ({len(rep.verified_points)} mục)", icon=":material/check:", expanded=False):
                            for v in rep.verified_points:
                                st.markdown(f"- :green[{v.learner_claim}] `{v.page_ref}`")
                    if rep.patched_note:
                        with st.expander("Bản ghi chú đã hiệu đính sẵn sàng ôn thi", icon=":material/auto_awesome:", expanded=False):
                            st.code(rep.patched_note, language="markdown")

    elif lab_mode == "incident":
        st.markdown('<p class="sc-section-eyebrow">Ca Trực Cứu Hỏa Production · Biến tài liệu lý thuyết thành Ticket sự cố điều phối qua LangGraph</p>', unsafe_allow_html=True)
        if st.session_state.incident_scenario is None:
            try:
                with st.spinner("Đang dựng ca sự cố từ tài liệu..."):
                    st.session_state.incident_scenario = breakthrough.generate_incident_scenario(
                        doc_text=st.session_state.doc_text,
                        concept_map=cmap,
                        kb=st.session_state.kb,
                        scenario_index=st.session_state.incident_idx,
                        model=st.session_state.selected_model,
                        demo_mode=_is_demo_mode(),
                    )
            except _LLM_ERRORS as e:
                st.error(str(e))
                return
        sc = st.session_state.incident_scenario

        h_left, h_right = st.columns([8, 4], vertical_alignment="center")
        with h_left:
            st.caption("Hãy đóng vai Kỹ sư trực sự cố: chạy các công cụ điều tra bên dưới để thu thập manh mối trước khi chốt nguyên nhân gốc.")
        with h_right:
            if st.button("Đổi ca sự cố khác", icon=":material/refresh:", key="btn_next_incident", width="stretch"):
                try:
                    next_idx = st.session_state.incident_idx + 1
                    new_sc = breakthrough.generate_incident_scenario(
                        doc_text=st.session_state.doc_text,
                        concept_map=cmap,
                        kb=st.session_state.kb,
                        scenario_index=next_idx,
                        model=st.session_state.selected_model,
                        demo_mode=_is_demo_mode(),
                    )
                    # Chỉ đổi state khi sinh thành công để giữ trạng thái nhất quán.
                    st.session_state.incident_idx = next_idx
                    st.session_state.incident_scenario = new_sc
                    st.session_state.incident_tools_used = []
                    st.session_state.incident_eval = None
                    st.rerun()
                except _LLM_ERRORS as e:
                    st.error(str(e))

        st.markdown(render_incident_ticket_html(sc), unsafe_allow_html=True)

        # 3 Công cụ điều tra sự cố
        t_cols = st.columns(3)
        tool_meta = [
            ("inspect_config", "Kiểm tra Config", ":material/tune:", "btn_inc_t1"),
            ("inspect_logs", "Xem Log truy xuất", ":material/terminal:", "btn_inc_t2"),
            ("inspect_metrics", "Đo Chi phí & Độ trễ", ":material/speed:", "btn_inc_t3"),
        ]
        used_list: list[str] = st.session_state.incident_tools_used
        for col_obj, (t_id, t_lbl, t_ic, t_key) in zip(t_cols, tool_meta):
            with col_obj:
                if st.button(t_lbl, icon=t_ic, key=t_key, width="stretch"):
                    if t_id not in used_list:
                        used_list.append(t_id)
                        st.session_state.incident_tools_used = used_list

        if used_list:
            clue_map = {c.tool_name: c for c in sc.clues}
            for t_id in used_list:
                clue = clue_map.get(t_id)
                if clue:
                    st.code(f"[{clue.title}]\n{clue.output}", language="text")

        col_fix, col_res = st.columns([6.5, 5.5], gap="large")
        with col_fix:
            with st.container(border=True):
                st.markdown("#### Chẩn đoán Nguyên nhân gốc & Phương án vá")
                pick_idx = st.radio(
                    "Chọn nguyên nhân gốc gây ra sự cố",
                    options=range(len(sc.root_cause_options)),
                    format_func=lambda k: sc.root_cause_options[k],
                    index=None,
                    key=f"inc_pick_{sc.ticket_id}",
                )
                fix_txt = st.text_input(
                    "Đề xuất chỉnh cấu hình / mã nguồn ngắn gọn",
                    placeholder="Ví dụ: Bật overlap > 0, giảm top_k xuống 3-5, gửi lại messages...",
                    key=f"inc_fix_{sc.ticket_id}",
                )
                if st.button("Xác nhận nguyên nhân & Vá sự cố", icon=":material/done_all:", type="primary", key="btn_solve_incident", width="stretch"):
                    if pick_idx is None:
                        st.warning("Hãy chọn một phương án nguyên nhân gốc trước khi xác nhận.")
                    else:
                        tok = _submission_token(
                            "incident", st.session_state.incident_idx, sc.ticket_id, pick_idx, fix_txt,
                            ",".join(sorted(st.session_state.incident_tools_used)),
                        )
                        done, old = _cached_submission(tok)
                        if done:
                            st.session_state.incident_eval = old
                            st.rerun()
                        try:
                            ev, _ = breakthrough.evaluate_incident_resolution(
                                scenario=sc,
                                chosen_option_index=pick_idx,
                                learner_fix=fix_txt,
                                tools_used=st.session_state.incident_tools_used,
                                learner_dna=dna,
                                doc_id=st.session_state.doc_id,
                                concepts_order=concept_names,
                                model=st.session_state.selected_model,
                                demo_mode=_is_demo_mode(),
                            )
                        except _LLM_ERRORS as e:
                            st.error(str(e))
                        else:
                            _remember_submission(tok, ev)
                            st.session_state.incident_eval = ev
                            st.toast(f"Kết quả xử lý {sc.ticket_id}: {ev.score}/100.")
                            st.rerun()

        with col_res:
            ev = st.session_state.get("incident_eval")
            with st.container(border=True):
                st.markdown("#### Đánh giá Post-Mortem")
                if not ev:
                    st.info("Chạy tối đa 3 công cụ điều tra bên trên để nhận +15 điểm thưởng quy trình, sau đó chọn nguyên nhân gốc.")
                else:
                    st.markdown(
                        f'<div class="sc-score-hero">{ev.score}<small>/100</small></div>',
                        unsafe_allow_html=True,
                    )
                    if ev.resolved_correctly:
                        st.success(ev.feedback)
                    else:
                        st.error(ev.feedback)
                    st.info(f"**Nguyên lý tài liệu ({ev.page_ref}):** {ev.violated_principle}")
                    st.warning(f"**Câu hỏi Post-Mortem:** {ev.post_mortem_question}")

    else:
        st.markdown('<p class="sc-section-eyebrow">Rèn phản xạ chống tin mù quáng AI &amp; Luyện bảo vệ Đồ án trước Hội đồng</p>', unsafe_allow_html=True)
        col_tr, col_adv = st.columns([6, 6], gap="large")
        with col_tr:
            with st.container(border=True):
                _render_trojan_panel(dna, cmap, concept_names)
        with col_adv:
            with st.container(border=True):
                _render_adv_panel(dna)


def _render_trojan_panel(dna: LearnerDNA, cmap: DocumentConceptMap, concept_names: list[str]) -> None:
    st.markdown("#### Săn Lỗi Cài Bẫy (Spot-the-Bug)")
    if st.session_state.trojan_challenge is None:
        try:
            with st.spinner("Đang cài bẫy các nhận định sai..."):
                st.session_state.trojan_challenge = breakthrough.generate_trojan_challenge(
                    doc_text=st.session_state.doc_text,
                    concept_map=cmap,
                    kb=st.session_state.kb,
                    challenge_index=st.session_state.trojan_idx,
                    model=st.session_state.selected_model,
                    demo_mode=_is_demo_mode(),
                )
        except _LLM_ERRORS as e:
            st.error(str(e))
            return
    ch = st.session_state.trojan_challenge
    st.caption(ch.context_summary)

    flagged_ids: list[int] = []
    ver = st.session_state.get("trojan_version", 0)
    for stmt in ch.statements:
        if st.checkbox(
            f"Nhận định #{stmt.id} ({stmt.topic}): {stmt.statement}",
            key=f"trojan_chk_{ch.title[:10]}_{stmt.id}" + (f"_v{ver}" if ver else ""),
        ):
            flagged_ids.append(stmt.id)

    tc1, tc2 = st.columns([6, 4])
    with tc1:
        if st.button("Kiểm tra mắt tinh Tech Lead", icon=":material/fact_check:", type="primary", key="btn_grade_trojan", width="stretch"):
            tok = _submission_token(
                "trojan", ver, st.session_state.trojan_idx, ch.title, ",".join(map(str, sorted(flagged_ids))),
            )
            done, old = _cached_submission(tok)
            if done:
                # Nộp lặp cùng một bài: giữ kết quả cũ, không cập nhật mastery/lỗi tư duy lần nữa.
                st.session_state.trojan_result = old
                st.rerun()
            res, _ = breakthrough.grade_trojan_submission(
                challenge=ch,
                flagged_ids=flagged_ids,
                learner_dna=dna,
                doc_id=st.session_state.doc_id,
                concepts_order=concept_names,
            )
            _remember_submission(tok, res)
            st.session_state.trojan_result = res
            st.toast(f"Kết quả săn lỗi: {res['score']}/100.")
            st.rerun()
    with tc2:
        if st.button("Tạo bộ mới", icon=":material/refresh:", key="btn_new_trojan", width="stretch"):
            st.session_state.trojan_idx += 1
            st.session_state.trojan_version = ver + 1
            st.session_state.trojan_challenge = None
            st.session_state.trojan_result = None
            st.rerun()

    tr_res = st.session_state.get("trojan_result")
    if tr_res:
        st.markdown(f"**Điểm Audit:** `{tr_res['score']}/100` · Bắt trúng `{len(tr_res['caught'])}/{tr_res['total_bugs']}` lỗi · Báo nhầm `{len(tr_res['false_alarms'])}`")
        for stmt in ch.statements:
            if stmt.is_buggy:
                caught_tag = "Đã bắt trúng" if stmt.id in tr_res["caught"] else "Bị lừa bỏ sót"
                st.error(f"**Nhận định #{stmt.id} ({caught_tag}) {stmt.page_ref}:** {stmt.bug_explanation}")


def _render_adv_panel(dna: LearnerDNA) -> None:
    st.markdown("#### Hội đồng Phản biện Q&A (Devil's Advocate)")
    ab1, ab2 = st.columns(2)
    with ab1:
        if st.session_state.doc_id == db.DEFAULT_DOC_ID and st.button("Nạp INFRA.md của nhóm để luyện Q&A", icon=":material/gavel:", key="btn_load_infra_adv", width="stretch"):
            infra_loaded = breakthrough.load_project_infra_doc()
            if infra_loaded:
                inf_txt, inf_title = infra_loaded
                try:
                    secs = load_sections_from_pasted_text(inf_txt, inf_title)[1]
                    _apply_new_document(inf_txt, secs, inf_title, "project_infra_doc")
                except _LLM_ERRORS as e:
                    st.error(str(e))
                else:
                    st.toast("Đã nạp Documents/INFRA.md để luyện bảo vệ kiến trúc Hackathon.")
                    st.rerun()
    with ab2:
        if st.button("Đổi câu phản biện khác", icon=":material/refresh:", key="btn_next_adv", width="stretch"):
            st.session_state.adv_idx += 1
            st.session_state.adv_question = None
            st.session_state.adv_eval = None
            st.rerun()

    if st.session_state.adv_question is None:
        try:
            with st.spinner("Đang chuẩn bị câu chất vấn..."):
                st.session_state.adv_question = breakthrough.generate_devils_advocate(
                    doc_text=st.session_state.doc_text,
                    concept_map=st.session_state.concept_map,
                    kb=st.session_state.kb,
                    round_index=st.session_state.adv_idx,
                    model=st.session_state.selected_model,
                    demo_mode=_is_demo_mode(),
                )
        except _LLM_ERRORS as e:
            st.error(str(e))
            return
    aq = st.session_state.adv_question
    st.warning(f"**{aq.persona}:** {aq.question}")
    if aq.conflicting_sections:
        st.caption("Các phần tài liệu bị chất vấn: " + " ↔ ".join(aq.conflicting_sections))

    st.button(
        "Điền thử câu trả lời bảo vệ mẫu",
        icon=":material/edit_note:",
        key="btn_fill_adv_sample",
        on_click=_fill_adv_sample,
        args=(aq.sample_strong_answer,),
    )
    def_ans = st.text_area(
        "Câu trả lời bảo vệ của bạn",
        key="adv_defense_input",
        height=130,
        placeholder="Trình bày lập luận bảo vệ thiết kế hoặc giải pháp bám theo tài liệu...",
    )
    if st.button("Nộp phần bảo vệ Q&A", icon=":material/balance:", type="primary", key="btn_eval_adv", width="stretch"):
        if len(def_ans.strip()) < 15:
            st.warning("Hãy viết câu trả lời đầy đủ hơn (tối thiểu 15 ký tự).")
        else:
            tok = _submission_token("adv", st.session_state.adv_idx, aq.question, def_ans)
            done, old = _cached_submission(tok)
            if done:
                st.session_state.adv_eval = old
                st.rerun()
            try:
                ev_new, _ = breakthrough.evaluate_defense(
                    question=aq,
                    learner_defense=def_ans,
                    kb=st.session_state.kb,
                    learner_dna=dna,
                    doc_id=st.session_state.doc_id,
                    model=st.session_state.selected_model,
                    demo_mode=_is_demo_mode(),
                )
            except _LLM_ERRORS as e:
                st.error(str(e))
            else:
                _remember_submission(tok, ev_new)
                st.session_state.adv_eval = ev_new
                st.rerun()

    ev_adv = st.session_state.get("adv_eval")
    if ev_adv:
        st.markdown(f"**Điểm phản biện:** `{ev_adv.defense_score}/100` · Kết luận: `{ev_adv.verdict}`")
        for s_pt in ev_adv.strengths:
            st.markdown(f"- :green[{s_pt}]")
        for g_pt in ev_adv.gaps_or_contradictions:
            st.markdown(f"- :red[{g_pt}]")
        st.info(f"**Câu hỏi xoáy tiếp theo:** {ev_adv.next_hard_follow_up}")


def _render_tab_coach(dna: LearnerDNA) -> None:
    histories: dict = st.session_state.chat_histories
    history: list[dict] = _get_chat_history(dna)
    pending = st.session_state.pop("pending_coach", None)

    failed = st.session_state.get("coach_failed")
    if failed:
        st.error(failed["error"])
        if st.button("Thử lại yêu cầu vừa lỗi", icon=":material/refresh:", key="btn_retry_coach"):
            st.session_state.pending_coach = failed["pending"]
            st.session_state.coach_failed = None
            st.rerun()

    doc_title = st.session_state.get("doc_title", "Tài liệu")
    cmap_ctx = st.session_state.get("concept_map")
    topics_preview = ", ".join(c.name for c in cmap_ctx.concepts[:2]) if cmap_ctx and cmap_ctx.concepts else "chưa có chủ đề"
    focus_mode = bool(st.session_state.get("chat_focus_mode", False))
    chat_height = 760 if focus_mode else 560
    weak_topic = (
        dna.active_misconceptions[0].topic
        if dna.active_misconceptions
        else (dna.concepts[0].concept if dna.concepts else "Chunking và Overlap")
    )

    with st.expander(
        f"💡 Bối cảnh học tập hiện tại: {doc_title} · {topics_preview} · Trình độ: {dna.overall_level} ({len(dna.active_misconceptions)} lỗi đang mở)",
        expanded=False,
    ):
        st.markdown(f"**{dna.name}** · Trình độ `{dna.overall_level}`")
        if dna.active_misconceptions:
            for m in dna.active_misconceptions[:2]:
                st.markdown(
                    render_misconception_card_html(m.topic, m.description, m.occurrences, False),
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Chưa ghi nhận lỗi tư duy đang mở — Gia sư sẽ tự đẩy độ sâu lên mức Vận dụng / Phân tích.")

    if _is_demo_mode():
        st.warning("Chưa có AI Portal API key — hãy nhập key trong ⚙️ Cài đặt để Gia sư trả lời bằng AI Portal Live.")

    with st.container(border=True):
        ch_head, ch_focus, ch_reset = st.columns([7.4, 1.8, 2.2], vertical_alignment="center")
        with ch_head:
            status_html = (
                '<span style="display:inline-flex;align-items:center;gap:4px;background:#e6f6ec;color:#15803d;border:1px solid #bbf7d0;padding:1px 8px;border-radius:999px;font-size:0.74rem;font-weight:700;">● Trực tuyến</span>'
                if not _is_demo_mode()
                else '<span style="display:inline-flex;align-items:center;gap:4px;background:#fef3c7;color:#b45309;border:1px solid #fde68a;padding:1px 8px;border-radius:999px;font-size:0.74rem;font-weight:700;">● Chưa kết nối</span>'
            )
            st.markdown(
                clean_html(
                    f"""
                    <div class="sc-chat-card-header" style="display:flex;align-items:center;justify-content:flex-start;gap:0.75rem;background:transparent;padding:0.15rem 0.2rem;box-shadow:none;margin:0;">
                      <div class="sc-chat-avatar-badge" style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#14532d 0%,#166534 100%);color:#ffffff;display:inline-flex;align-items:center;justify-content:center;font-size:1.2rem;box-shadow:0 3px 8px rgba(20,83,45,0.22);flex-shrink:0;">🤖</div>
                      <div>
                        <div style="display:flex;align-items:center;gap:0.5rem;font-weight:800;font-size:1.02rem;color:#0f2e1d;">
                          <span>Study Coach AI</span> {status_html}
                        </div>
                        <div style="font-size:0.77rem;color:#4b6e58;margin-top:1px;">Gia sư Socratic thích ứng theo tài liệu của bạn</div>
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )
        with ch_focus:
            st.button(
                "🗕 Thu nhỏ" if focus_mode else "⛶ Phóng to",
                key="btn_chat_focus",
                on_click=_toggle_chat_focus,
                width="stretch",
            )
        with ch_reset:
            def _clear_chat_cb() -> None:
                histories[_chat_key(dna)] = []
                db.clear_chat_history(dna.learner_id, st.session_state.doc_id)

            if st.button(
                "🔄 Làm mới chat",
                key="btn_clear_chat",
                on_click=_clear_chat_cb,
                width="stretch",
            ):
                st.toast("Đã bắt đầu lại cuộc trò chuyện.")

        st.markdown(
            '<hr style="margin:0.4rem 0 0.55rem 0;border:none;border-top:1.5px solid #e2efe7;">',
            unsafe_allow_html=True,
        )
        chat_box = st.container(height=chat_height, border=False)
        with chat_box:
            if not history:
                st.info(
                    f"Gia sư đã nạp hồ sơ **{dna.name}** (trình độ **{dna.overall_level}**). "
                    "Hãy chọn một gợi ý bên dưới hoặc đặt câu hỏi về tài liệu ở thanh nhập liệu."
                )
            for idx, msg in enumerate(history):
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        turn: CoachTurnResult | None = msg.get("turn")
                        if turn:
                            st.markdown(render_route_banner_html(turn), unsafe_allow_html=True)
                            st.markdown(turn.answer)
                            if turn.citations:
                                st.markdown(render_citation_pills_html(turn.citations), unsafe_allow_html=True)
                            if turn.follow_up_question:
                                st.markdown(
                                    clean_html(
                                        f"""
                                        <div style="background:#eef8f2;border-left:3.5px solid #16a34a;padding:0.65rem 0.9rem;border-radius:0 8px 8px 0;margin:0.65rem 0 0.85rem 0;font-size:0.86rem;color:#133822;">
                                          <b>💡 Câu hỏi Socratic dành cho bạn:</b> {esc(turn.follow_up_question)}
                                        </div>
                                        """
                                    ),
                                    unsafe_allow_html=True,
                                )
                            if turn.suggested_quiz_topic:
                                st.button(
                                    f"Luyện ngay Quiz chủ đề '{turn.suggested_quiz_topic}'",
                                    icon=":material/quiz:",
                                    type="primary" if idx == len(history) - 1 else "secondary",
                                    key=f"chat_to_quiz_{idx}",
                                    on_click=_jump_to_quiz_topic,
                                    args=(turn.suggested_quiz_topic,),
                                )
                            if turn.tool_traces or turn.retrieved_chunks:
                                with st.expander(
                                    "🔍 Chi tiết suy luận & Nguồn tài liệu đối chiếu",
                                    icon=":material/manage_search:",
                                    expanded=False,
                                ):
                                    if turn.tool_traces:
                                        st.markdown(
                                            clean_html(
                                                f"""
                                                <div class="sc-trace-box">
                                                  <b>Lý do chọn nhánh:</b> {esc(turn.pedagogical_reason)}
                                                </div>
                                                """
                                            ),
                                            unsafe_allow_html=True,
                                        )
                                        for tr in turn.tool_traces:
                                            st.markdown(f"- `{tr.tool_name}` — {tr.summary}")
                                    if turn.retrieved_chunks:
                                        if turn.tool_traces:
                                            st.divider()
                                        for ch in turn.retrieved_chunks:
                                            st.markdown(f"**{ch.page}** · `{ch.source}` (độ khớp `{ch.score:.2f}`)\n\n> {ch.content}")
                        else:
                            st.markdown(msg["content"])

        is_sample_doc = st.session_state.doc_id == db.DEFAULT_DOC_ID
        _cmap = st.session_state.concept_map
        first_topic = _cmap.concepts[0].name if (_cmap and _cmap.concepts) else "khái niệm chính"
        short_first = first_topic if len(first_topic) <= 24 else first_topic[:22] + "…"
        prompt_clicked: str | None = None
        st.markdown(
            '<hr style="margin:0.35rem 0 0.45rem 0;border:none;border-top:1px solid #e2efe7;">',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sc-quick-label">⚡ Gợi ý hỏi nhanh:</div>', unsafe_allow_html=True)
        qb1, qb2, qb3 = st.columns(3)
        with qb1:
            btn1_label = "📖 Giải thích: Chunking" if is_sample_doc else f"📖 Giải thích: {short_first}"
            btn1_prompt = (
                "Giải thích Chunking và Overlap trong RAG là gì?"
                if is_sample_doc
                else f"Hãy giảng cho tôi chủ đề '{first_topic}' theo đúng trình độ hiện tại của tôi."
            )
            if st.button(btn1_label, key="qp_chunk", help=f"Hỏi về {first_topic}", width="stretch"):
                prompt_clicked = btn1_prompt
        with qb2:
            if st.button("🎯 Gỡ lỗi phần yếu nhất", key="qp_weak", help=f"Xoáy vào chủ đề yếu: {weak_topic}", width="stretch"):
                prompt_clicked = f"Tôi thường bị nhầm phần {weak_topic}, hãy giảng và kiểm tra lại giúp tôi."
        with qb3:
            if st.button("🛡️ Thử hỏi ngoài tài liệu", key="qp_oos", help="Kiểm tra khả năng từ chối bịa đặt ngoài nguồn", width="stretch"):
                prompt_clicked = "Giá vàng hôm nay trên thị trường quốc tế là bao nhiêu?"

        with st.expander(
            "📎 Đính kèm ảnh bài tập / Sơ đồ (Vision AI)",
            expanded=bool(st.session_state.get("vision_critique")),
        ):
            samples = vision.list_sample_vision_images() if is_sample_doc else []
            if is_sample_doc and samples:
                sc1, sc2 = st.columns(2)
                with sc1:
                    if st.button("Mẫu Sơ đồ RAG", icon=":material/schema:", key="btn_vis_s1", width="stretch"):
                        st.session_state.vision_sample_id = "so-do-rag-loi"
                        st.session_state.vision_force_sample = True
                        st.session_state.vision_critique = None
                with sc2:
                    if st.button("Mẫu Code API lỗi", icon=":material/code:", key="btn_vis_s2", width="stretch"):
                        st.session_state.vision_sample_id = "code-openai-loi"
                        st.session_state.vision_force_sample = True
                        st.session_state.vision_critique = None

            uploaded_img = st.file_uploader(
                "Hoặc tải ảnh sơ đồ / bài giải của bạn" if is_sample_doc else "Tải ảnh sơ đồ / bài giải của bạn",
                type=["png", "jpg", "jpeg"],
                key="vision_uploader",
            )
            up_sig = (uploaded_img.name, uploaded_img.size) if uploaded_img is not None else None
            if up_sig != st.session_state.get("vision_up_sig"):
                # Ảnh tải lên mới (hoặc bị gỡ) thì bỏ cờ ưu tiên ảnh mẫu.
                st.session_state.vision_up_sig = up_sig
                st.session_state.vision_force_sample = False
            # Khi vừa bấm nút mẫu thì ưu tiên mẫu, bỏ qua ảnh đã tải lên trước đó.
            if uploaded_img is not None and not (is_sample_doc and st.session_state.get("vision_force_sample")):
                img_bytes = uploaded_img.getvalue()
                mime = uploaded_img.type or "image/png"
                sample_id = None
                img_caption = f"Ảnh của bạn: {uploaded_img.name}"
            elif is_sample_doc and samples:
                cur_sample = next((x for x in samples if x["id"] == st.session_state.vision_sample_id), samples[0])
                img_bytes = Path(cur_sample["path"]).read_bytes()
                mime = "image/png"
                sample_id = cur_sample["id"]
                img_caption = cur_sample["title"]
            else:
                img_bytes, mime, sample_id, img_caption = None, "image/png", None, ""

            img_sig = None
            if img_bytes is None:
                st.caption("Tải lên ảnh bài làm hoặc sơ đồ (PNG/JPG) để Gia sư đối chiếu trực tiếp với tài liệu của bạn.")
            else:
                st.image(img_bytes, caption=img_caption, width="stretch")
                img_sig = hashlib.sha256(img_bytes).hexdigest()
                if st.session_state.get("vision_img_sig") != img_sig:
                    # Ảnh đổi thì nhận xét cũ không còn đúng.
                    if st.session_state.get("vision_img_sig") is not None:
                        st.session_state.vision_critique = None
                    st.session_state.vision_img_sig = img_sig
            vis_note = st.text_input(
                "Ghi chú kèm theo ảnh",
                value="Nhờ Gia sư soi từng bước xem em thiết kế đúng tài liệu chưa ạ?",
                key="vision_note_input",
            )
            if st.button("Chấm bài nháp theo tài liệu", icon=":material/add_a_photo:", type="primary", key="btn_run_vision", width="stretch", disabled=img_bytes is None):
                tok = _submission_token("vision", img_sig, vis_note, sample_id)
                done, old = _cached_submission(tok)
                if done:
                    st.session_state.vision_critique = old
                    st.rerun()
                try:
                    with st.spinner("Đang soi từng bước bài nháp đối chiếu với tài liệu nguồn..."):
                        critique, _ = vision.critique_student_image(
                            image_bytes=img_bytes,
                            mime_type=mime,
                            note=vis_note,
                            kb=st.session_state.kb,
                            learner_dna=dna,
                            doc_id=st.session_state.doc_id,
                            model=st.session_state.selected_model,
                            demo_mode=_is_demo_mode(),
                            sample_id=sample_id,
                        )
                    _remember_submission(tok, critique)
                    st.session_state.vision_critique = critique
                    st.toast("Đã chấm bài nháp và cập nhật lỗi tư duy vào hồ sơ.")
                except _LLM_ERRORS as e:
                    st.error(str(e))

            crit = st.session_state.get("vision_critique")
            if crit:
                st.markdown(f"**Nhận xét tổng quát ({crit.topic}):** {crit.overall_assessment}")
                for st_step in crit.steps:
                    badge_cls = "mastered" if st_step.status == "Đúng" else ("learning" if st_step.status == "Cần chỉnh" else "misconception")
                    ic_name = "check-circle" if st_step.status == "Đúng" else ("clock" if st_step.status == "Cần chỉnh" else "alert-triangle")
                    st.markdown(
                        clean_html(
                            f"""
                            <div style="border:1px solid var(--border);border-radius:8px;padding:0.55rem 0.7rem;margin:0.35rem 0;background:var(--bg-canvas);">
                              <div style="display:flex;justify-content:space-between;align-items:center;gap:0.4rem;">
                                <b style="font-size:0.86rem;">{esc(st_step.step_name)}</b>
                                <span class="sc-status-badge {badge_cls}">{icon_svg(ic_name, 13)} {esc(st_step.status)}</span>
                              </div>
                              <div style="font-size:0.83rem;color:var(--ink-secondary);margin-top:4px;">{esc(st_step.feedback)} <span class="sc-mono-chip">{esc(st_step.page_ref)}</span></div>
                            </div>
                            """
                        ),
                        unsafe_allow_html=True,
                    )
                st.info(f"**Gợi ý sửa vừa tầm ({dna.overall_level}):** {crit.hint_for_learner}")

    user_msg = st.chat_input("Hỏi bất cứ điều gì về tài liệu hoặc trả lời câu hỏi vặn của Gia sư...")
    active_query = (pending["query"] if pending else None) or prompt_clicked or user_msg
    if active_query:
        try:
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(active_query)
                with st.chat_message("assistant"):
                    gen, result_box = stream_coach_turn(
                        query=active_query,
                        kb=st.session_state.kb,
                        learner_dna=dna,
                        concept_map=st.session_state.concept_map,
                        doc_id=st.session_state.doc_id,
                        focus_topic=pending.get("focus_topic") if pending else None,
                        remediation_context=pending.get("remediation_context") if pending else None,
                        model=st.session_state.selected_model,
                        demo_mode=_is_demo_mode(),
                        chat_history=history,
                    )
                    st.write_stream(gen)
            if "turn" in result_box:
                turn = result_box["turn"]
                history.append({"role": "user", "content": active_query})
                history.append({"role": "assistant", "content": turn.answer, "turn": turn})
                db.append_chat_message(dna.learner_id, st.session_state.doc_id, "user", active_query)
                db.append_chat_message(
                    dna.learner_id, st.session_state.doc_id, "assistant", turn.answer, turn=turn
                )
            st.session_state.coach_failed = None
            st.rerun()
        except _LLM_ERRORS as e:
            if pending:
                st.session_state.coach_failed = {"error": str(e), "pending": pending}
                st.rerun()
            else:
                st.error(str(e))


def _render_tab_practice(dna: LearnerDNA) -> None:
    cmap: DocumentConceptMap = st.session_state.concept_map
    concept_names = [c.name for c in cmap.concepts]

    mode = st.radio(
        "Chế độ luyện tập",
        options=["Quiz trắc nghiệm thích ứng", "Vấn đáp ngược Feynman (Teach-Back)"],
        horizontal=True,
        key="practice_mode",
        label_visibility="collapsed",
    )

    if mode == "Quiz trắc nghiệm thích ứng":
        quiz: Quiz | None = st.session_state.get("quiz")
        result: QuizResult | None = st.session_state.get("quiz_result")

        if quiz is None:
            col_cfg, col_tip = st.columns([7, 5], gap="large")
            with col_cfg:
                with st.container(border=True):
                    st.markdown("#### Thiết lập Quiz Thích ứng theo Learner DNA")
                    topic_opts = ["Tự động chọn chủ đề yếu nhất"] + concept_names
                    if st.session_state.quiz_focus_select not in topic_opts:
                        st.session_state.quiz_focus_select = topic_opts[0]
                    focus_choice = st.selectbox("Chủ đề ưu tiên", options=topic_opts, key="quiz_focus_select")
                    diff_opts = ["Tự động theo năng lực"] + config.DIFFICULTIES
                    diff_choice = st.selectbox("Độ khó", options=diff_opts, key="quiz_diff_select")
                    n_q = st.slider("Số câu hỏi", config.MIN_QUESTIONS, config.MAX_QUESTIONS, 5)
                    if st.button("Tạo Quiz Thích ứng", icon=":material/play_arrow:", type="primary", key="btn_gen_quiz", width="stretch"):
                        try:
                            focus_val = None if focus_choice == "Tự động chọn chủ đề yếu nhất" else focus_choice
                            diff_val = None if diff_choice == "Tự động theo năng lực" else diff_choice
                            with st.spinner("Đang sinh câu hỏi bám nguồn và gài bẫy chẩn đoán lỗi tư duy..."):
                                generated, strategy = generate_adaptive_quiz(
                                    text=st.session_state.doc_text,
                                    learner_dna=dna,
                                    n=n_q,
                                    focus_topic=focus_val,
                                    difficulty_override=diff_val,
                                    model=st.session_state.selected_model,
                                    demo_mode=_is_demo_mode(),
                                )
                            st.session_state.quiz = generated
                            # Lưu độ khó/chủ đề thực sự dùng vào khóa không phải widget (selectbox sẽ không được vẽ khi đang làm bài).
                            st.session_state.quiz_difficulty_used = diff_val or "Tự động"
                            st.session_state.quiz_focus_used = focus_val or "Tự động"
                            st.session_state.quiz_strategy = strategy
                            st.session_state.quiz_result = None
                            st.session_state.quiz_recorded = False
                            st.session_state.quiz_version = st.session_state.get("quiz_version", 0) + 1
                            st.rerun()
                        except _LLM_ERRORS as e:
                            st.error(str(e))
            with col_tip:
                with st.container(border=True):
                    st.markdown("#### Cách Quiz này hiểu bạn")
                    st.markdown(
                        f"- **Hồ sơ hiện tại:** {dna.name} (`{dna.overall_level}` · TB `{dna.mastery_avg:.0f}%`)\n"
                        "- **Chẩn đoán đáp án nhiễu:** Mỗi phương án sai được gắn sẵn một nhãn *lỗi tư duy*; chọn nhầm đáp án nào, hệ thống bắt đúng bệnh đó.\n"
                        "- **Kiểm soát bịa đặt:** Mọi câu hỏi đều phải trích nguyên văn `evidence` từ tài liệu."
                    )
        elif result is None:
            if st.session_state.quiz_strategy:
                st.info(st.session_state.quiz_strategy)
            version = st.session_state.get("quiz_version", 0)
            total = len(quiz.questions)
            picks: list[int | None] = []
            confs: list[str] = []
            with st.form(f"adaptive_quiz_form_{version}", border=False):
                col_q, col_side = st.columns([8, 3.5], gap="large")
                with col_q:
                    grid = st.columns(2, gap="medium")
                    for i, q in enumerate(quiz.questions):
                        with grid[i % 2].container(border=True):
                            st.caption(f"{q.topic} · Mức Bloom: {q.bloom_level} · {q.page_ref or '[Tài liệu]'}")
                            st.markdown(f"**Câu {i + 1}/{total}.** {q.question}")
                            choice = st.radio(
                                f"Câu {i + 1}",
                                options=range(len(q.options)),
                                format_func=lambda k, opts=q.options: f"{'ABCD'[k]}. {opts[k]}",
                                index=None,
                                key=f"q{version}_{i}",
                                label_visibility="collapsed",
                            )
                            conf = st.segmented_control(
                                "Độ tự tin",
                                options=["Đoán mò", "Bình thường", "Chắc chắn"],
                                default="Bình thường",
                                key=f"conf_{version}_{i}",
                            )
                            picks.append(choice)
                            confs.append(conf or "Bình thường")
                with col_side:
                    with st.container(border=True):
                        st.markdown(f"**Bài kiểm tra gồm {total} câu**")
                        st.caption("Chọn đáp án và mức tự tin cho từng câu rồi bấm Nộp bài (không làm mới trang khi chọn). Câu bỏ trống tính sai khi nộp.")
                    submitted_quiz = st.form_submit_button(
                        "Nộp bài & Chẩn đoán",
                        icon=":material/done_all:",
                        type="primary",
                        key="btn_submit_quiz",
                        width="stretch",
                    )
            if submitted_quiz:
                res = grade(quiz, picks, confidences=confs)
                st.session_state.quiz_result = res
                if not st.session_state.quiz_recorded:
                    adaptive.record_quiz_outcome(
                        learner_id=dna.learner_id,
                        doc_id=st.session_state.doc_id,
                        result=res,
                        model=st.session_state.selected_model if not _is_demo_mode() else "Offline-Demo",
                        difficulty=st.session_state.get("quiz_difficulty_used") or "Tự động",
                        concepts_order=concept_names,
                    )
                    st.session_state.quiz_recorded = True
                st.rerun()
            st.button("Tạo quiz khác", on_click=_new_quiz, key="btn_abandon_quiz")
        else:
            wrong = sum(1 for it in result.items if not it.is_correct and it.chosen_index is not None)
            skipped = sum(1 for it in result.items if it.chosen_index is None)
            ratio = result.score / result.total if result.total else 0.0
            col_side, col_rev = st.columns([3.5, 8], gap="large")
            with col_side:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="sc-score-hero">{result.score}<small>/{result.total}</small></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(ratio, text=f"Tỉ lệ đúng {ratio:.0%}")
                    chips = "".join(
                        f'<span class="{"on" if it.is_correct else ("bad" if it.chosen_index is not None else "")}">{i}</span>'
                        for i, it in enumerate(result.items, start=1)
                    )
                    st.markdown(f'<div class="sc-chips">{chips}</div>', unsafe_allow_html=True)
                    st.caption(f"Đúng {result.score} · Sai {wrong} · Bỏ trống {skipped}")
                if result.blind_spot_count:
                    st.error(f"Cảnh báo {result.blind_spot_count} Điểm mù nguy hiểm (Tự tin chắc chắn nhưng trả lời Sai).")
                if result.lucky_guess_count:
                    st.info(f"Có {result.lucky_guess_count} câu Đúng do đoán mò — hệ thống vẫn giữ lịch nhắc ôn lại.")
                if result.detected_misconceptions:
                    st.warning(f"Phát hiện {len(result.detected_misconceptions)} lỗi tư duy đã được ghi vào hồ sơ Learner DNA.")
                only_wrong = st.toggle("Chỉ xem câu sai", value=False) if result.score < result.total else False
                st.button("Tạo quiz mới", icon=":material/add:", type="primary", on_click=_new_quiz, key="btn_res_new", width="stretch")
                st.button("Làm lại quiz này", icon=":material/replay:", on_click=_retry_quiz, key="btn_res_retry", width="stretch")

            with col_rev:
                shown = [(i, it) for i, it in enumerate(result.items, start=1) if not (only_wrong and it.is_correct)]
                grid = st.columns(2, gap="medium")
                for pos, (i, it) in enumerate(shown):
                    badge = "Đúng" if it.is_correct else ("Bỏ trống" if it.chosen_index is None else "Sai")
                    with grid[pos % 2].container(border=True):
                        cal_suffix = f" · `{it.calibration_tag}`" if it.calibration_tag else ""
                        st.markdown(f"**Câu {i}.** {it.question} &nbsp; `{badge}`{cal_suffix}")
                        for k, opt in enumerate(it.options):
                            line = f"{'ABCD'[k]}. {opt}"
                            if k == it.correct_index:
                                st.markdown(f":green[**✔ {line}**]")
                            elif k == it.chosen_index:
                                st.markdown(f":red[✘ ~~{line}~~ (bạn chọn)]")
                            else:
                                st.markdown(f"&nbsp;&nbsp;&nbsp;{line}")
                        if it.explanation:
                            st.info(f"{it.explanation} ({it.page_ref or 'Tài liệu'})")
                        if it.evidence:
                            st.caption(f'Bằng chứng nguồn: "{it.evidence}"')
                        if not it.is_correct:
                            if it.chosen_index is None:
                                st.warning(f"Chưa trả lời câu này ({it.topic}) — hệ thống tính sai để kiểm tra lại phần kiến thức này.")
                            elif it.diagnosed_misconception:
                                st.markdown(
                                    render_misconception_card_html(
                                        f"Chẩn đoán lỗi ({it.topic})",
                                        it.diagnosed_misconception,
                                    ),
                                    unsafe_allow_html=True,
                                )
                            st.button(
                                "Nhờ Gia sư gỡ rối câu này",
                                icon=":material/psychology:",
                                key=f"rem_coach_{i}",
                                on_click=_jump_to_coach_remediation,
                                args=(it,),
                                width="stretch",
                            )
    else:
        col_f_in, col_f_out = st.columns([6, 6], gap="large")
        with col_f_in:
            with st.container(border=True):
                st.markdown("#### Vấn đáp ngược Feynman (Teach-Back)")
                st.caption(
                    "Đóng vai người giảng: tự giải thích lại khái niệm dưới đây bằng lời của chính bạn. "
                    "Hệ thống đối chiếu đoạn gốc qua FAISS để bắt đúng chỗ hiểu thiếu hoặc hiểu lệch."
                )
                target_opts = concept_names or ["Chunking và Overlap"]
                default_idx = 0
                if st.session_state.get("feynman_concept") in target_opts:
                    default_idx = target_opts.index(st.session_state["feynman_concept"])
                chosen_concept = st.selectbox("Khái niệm muốn kiểm tra bản chất", options=target_opts, index=default_idx)

                if st.session_state.doc_id == db.DEFAULT_DOC_ID:
                    fs1, fs2 = st.columns(2)
                    with fs1:
                        st.button(
                            "Mẫu có lỗ hổng",
                            icon=":material/edit_note:",
                            key="btn_fill_feynman",
                            on_click=_fill_feynman_sample,
                            args=(chosen_concept, False),
                            width="stretch",
                        )
                    with fs2:
                        st.button(
                            "Mẫu vững bản chất",
                            icon=":material/verified:",
                            key="btn_fill_feynman_good",
                            on_click=_fill_feynman_sample,
                            args=(chosen_concept, True),
                            width="stretch",
                        )

                ans_text = st.text_area(
                    "Lời giảng của bạn",
                    key="teach_back_draft",
                    height=170,
                    placeholder="Viết ngắn gọn 2–4 câu theo cách bạn hiểu...",
                )
                if st.button(
                    "Nộp bài giảng & Chấm bản chất",
                    icon=":material/record_voice_over:",
                    type="primary",
                    key="btn_eval_feynman",
                    width="stretch",
                ):
                    if len(ans_text.strip()) < 15:
                        st.warning("Hãy viết ít nhất một câu giải thích đầy đủ hơn (tối thiểu 15 ký tự).")
                    else:
                        tok = _submission_token("teachback", chosen_concept, ans_text)
                        done, old = _cached_submission(tok)
                        if done:
                            st.session_state.teach_back_eval = old
                            st.rerun()
                        try:
                            with st.spinner("Đang đối chiếu câu giải thích với tài liệu gốc qua FAISS..."):
                                ev, _ = adaptive.evaluate_teach_back(
                                    concept=chosen_concept,
                                    learner_answer=ans_text,
                                    kb=st.session_state.kb,
                                    learner_dna=dna,
                                    doc_id=st.session_state.doc_id,
                                    concepts_order=concept_names,
                                    model=st.session_state.selected_model,
                                    demo_mode=_is_demo_mode(),
                                )
                            _remember_submission(tok, ev)
                            st.session_state.teach_back_eval = ev
                            st.toast(f"Đã chấm Vấn đáp Feynman ({ev.understanding_score}/100) và cập nhật Learner DNA.")
                            st.rerun()
                        except _LLM_ERRORS as e:
                            st.error(str(e))

        with col_f_out:
            ev = st.session_state.get("teach_back_eval")
            with st.container(border=True):
                st.markdown("#### Kết quả chấm bản chất")
                if not ev:
                    st.info("Chưa có bài nộp. Hãy chọn một khái niệm bên trái và giải thích bằng lời của bạn.")
                else:
                    st.markdown(
                        f'<div class="sc-score-hero">{ev.understanding_score}<small>/100</small></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Kết luận:** `{ev.verdict}` · Chủ đề `{ev.topic}`")
                    if ev.citations:
                        st.markdown(render_citation_pills_html(ev.citations), unsafe_allow_html=True)
                    if ev.correct_points:
                        st.markdown("**Ý bạn nói đúng:**")
                        for pt in ev.correct_points:
                            st.markdown(f"- :green[{pt}]")
                    if ev.missing_points:
                        st.markdown("**Ý quan trọng còn thiếu trong tài liệu:**")
                        for pt in ev.missing_points:
                            st.markdown(f"- :orange[{pt}]")
                    if ev.misconceptions_found:
                        for m in ev.misconceptions_found:
                            st.markdown(render_misconception_card_html(ev.topic, m), unsafe_allow_html=True)
                    st.info(f"**Câu hỏi vặn tiếp theo:** {ev.socratic_follow_up}")
                    st.button(
                        "Thảo luận tiếp câu hỏi này với Gia sư",
                        icon=":material/psychology:",
                        key="feynman_to_coach",
                        on_click=_jump_to_coach_topic,
                        args=(ev.topic,),
                        width="stretch",
                    )


def _render_tab_analytics(dna: LearnerDNA) -> None:
    cmap: DocumentConceptMap = st.session_state.concept_map
    plan = adaptive.build_study_plan(dna, cmap)

    st.markdown('<p class="sc-section-eyebrow">Phân bố năng lực thành thạo &amp; Radar Ảo tưởng Hiểu bài</p>', unsafe_allow_html=True)
    c_chart, c_radar = st.columns([6.5, 5.5], gap="large")
    with c_chart:
        st.markdown(render_mastery_chart_html(dna.concepts), unsafe_allow_html=True)
    with c_radar:
        conf_stats = db.get_confidence_stats(dna.learner_id, st.session_state.doc_id)
        st.markdown(render_confidence_matrix_html(conf_stats), unsafe_allow_html=True)

    st.markdown('<p class="sc-section-eyebrow">Lộ trình ôn tập ưu tiên hôm nay</p>', unsafe_allow_html=True)
    for idx, item in enumerate(plan):
        with st.container(border=True):
            c1, c2, c3 = st.columns([6.5, 2.5, 3.0], vertical_alignment="center")
            with c1:
                st.markdown(f"**{idx + 1}. [{item['priority_label']}] {item['concept']}** `{item['page_ref']}`")
                st.caption(item["reason"])
            with c2:
                st.metric("Thành thạo", f"{item['mastery_score']:.0f}%")
            with c3:
                if item["rank"] in (1, 2):
                    btn_lbl, btn_ic, cb = "Học với Gia sư", ":material/psychology:", _jump_to_coach_topic
                elif item["rank"] == 3:
                    btn_lbl, btn_ic, cb = "Vấn đáp Feynman", ":material/record_voice_over:", _jump_to_feynman_topic
                else:
                    btn_lbl, btn_ic, cb = "Luyện Quiz ngay", ":material/quiz:", _jump_to_quiz_topic
                st.button(
                    btn_lbl,
                    icon=btn_ic,
                    key=f"plan_act_{idx}",
                    on_click=cb,
                    args=(item["concept"],),
                    width="stretch",
                )

    c_misc, c_hist = st.columns([6, 6], gap="large")
    with c_misc:
        st.markdown('<p class="sc-section-eyebrow">Sổ tay Lỗi tư duy (SQLite)</p>', unsafe_allow_html=True)
        with st.container(border=True):
            if not dna.active_misconceptions and not dna.resolved_misconceptions:
                st.caption("Chưa có lỗi tư duy nào được ghi nhận cho hồ sơ này.")
            for m in dna.active_misconceptions:
                st.markdown(render_misconception_card_html(m.topic, m.description, m.occurrences, False), unsafe_allow_html=True)
            for m in dna.resolved_misconceptions:
                st.markdown(render_misconception_card_html(m.topic, m.description, m.occurrences, True), unsafe_allow_html=True)

    with c_hist:
        st.markdown('<p class="sc-section-eyebrow">Lịch sử làm bài gần nhất (SQLite)</p>', unsafe_allow_html=True)
        with st.container(border=True):
            attempts = db.recent_quiz_attempts(dna.learner_id, limit=8)
            if not attempts:
                st.caption("Chưa có lượt nộp quiz nào được lưu.")
            else:
                st.dataframe(
                    [
                        {
                            "Thời điểm": a["created_at"][5:16],
                            "Điểm": f"{a['score']}/{a['total']}",
                            "Độ khó": a["difficulty"],
                            "Model": a["model"],
                            "Chủ đề yếu": ", ".join(a["weak_topics"]) or "Không",
                        }
                        for a in attempts
                    ],
                    hide_index=True,
                )

    st.markdown('<p class="sc-section-eyebrow" style="margin-top:0.8rem;">Dashboard Kỹ thuật & Bộ đánh giá 20 câu (Giám khảo / Mentor)</p>', unsafe_allow_html=True)
    col_tel, col_ev = st.columns([6, 6], gap="large")
    with col_tel:
        with st.container(border=True):
            st.markdown("#### Telemetry Token, Độ trễ & Chi phí")
            tel = db.get_telemetry_summary(limit=8)
            m1, m2, m3 = st.columns(3)
            m1.metric("Tổng lượt gọi", tel["total_calls"])
            m2.metric("Độ trễ TB", f"{tel['avg_latency_ms']:.0f} ms")
            m3.metric("Tổng chi phí", f"${tel['total_cost_usd']:.4f}")
            if tel["by_model"]:
                st.dataframe(
                    [
                        {
                            "Model / Engine": row["model"],
                            "Số lượt": row["calls"],
                            "Độ trễ TB (ms)": round(row["avg_latency_ms"], 1),
                            "Tokens": row["total_tokens"],
                        }
                        for row in tel["by_model"]
                    ],
                    hide_index=True,
                )

    def _eval_panel() -> None:
        with st.container(border=True):
            st.markdown("#### Bộ 20 câu Eval chuẩn (`eval/questions.json`)")
            eval_path = config.BASE_DIR / "eval" / "results" / "latest_eval.json"
            if st.button("Chạy kiểm thử bộ 20 câu ngay", icon=":material/play_circle:", key="btn_run_eval_ui"):
                from eval.run_eval import run_offline_eval
                st.session_state.eval_summary = run_offline_eval()
            ev_data = st.session_state.get("eval_summary")
            if ev_data is None and eval_path.exists():
                try:
                    ev_data = json.loads(eval_path.read_text(encoding="utf-8"))
                except Exception:
                    ev_data = None
            if ev_data:
                e1, e2, e3 = st.columns(3)
                e1.metric("Nhận diện trong TL", ev_data.get("in_scope_recall", "—"))
                e2.metric("Trích đúng nguồn", ev_data.get("source_citation_accuracy", "—"))
                e3.metric("Chặn ngoài TL", ev_data.get("out_of_scope_refusal", "—"))
                st.caption(
                    f"Tỉ lệ đạt tổng: **{ev_data.get('overall_retrieval_pass_rate', '—')}%** · "
                    f"Độ trễ truy xuất TB: **{ev_data.get('avg_retrieval_latency_ms', '—')} ms** · "
                    f"Lỗi khớp bằng chứng: **{len(ev_data.get('evidence_errors') or [])}**"
                )
                if ev_data.get("baseline_comparison"):
                    st.caption("Đối chiếu Trước / Sau nâng cấp RAG (bộ 20 câu chuẩn):")
                    st.dataframe(
                        [
                            {
                                "Phiên bản": (
                                    "Trước (Chunk 1400, Overlap 0)"
                                    if "Trước" in str(row.get("name", ""))
                                    else "Sau (Dual-Memory RAG)"
                                ),
                                "Trong TL": row.get("in_scope_recall", "—"),
                                "Nguồn": row.get("source_citation_accuracy", "—"),
                                "Chặn OOS": row.get("out_of_scope_refusal", "—"),
                                "Đạt": f"{row.get('overall_pass_rate', '—')}%",
                            }
                            for row in ev_data["baseline_comparison"]
                        ],
                        hide_index=True,
                        width="stretch",
                    )
            else:
                st.caption("Bấm nút phía trên để chạy kiểm chứng trực tiếp bộ 20 câu hỏi đánh giá.")

    def _progress_panel() -> None:
        with st.container(border=True):
            st.markdown("#### Thống kê tiến độ tài liệu của bạn")
            studied = [c for c in dna.concepts if c.attempts > 0]
            mastered = [c for c in dna.concepts if c.status == "mastered"]
            total_att = sum(c.attempts for c in dna.concepts)
            total_ok = sum(c.correct for c in dna.concepts)
            p1, p2, p3 = st.columns(3)
            p1.metric("Đã luyện", f"{len(studied)}/{len(dna.concepts)}")
            p2.metric("Làm chủ", len(mastered))
            p3.metric("Tỉ lệ đúng", f"{(100 * total_ok / total_att):.0f}%" if total_att else "—")
            st.caption(f"Tài liệu: **{st.session_state.doc_title}** · {total_att} lượt làm bài đã ghi nhận.")

    with col_ev:
        if st.session_state.doc_id == db.DEFAULT_DOC_ID:
            _eval_panel()
        else:
            _progress_panel()

    if st.session_state.doc_id == db.DEFAULT_DOC_ID:
        with st.expander("Đối chiếu 5 lớp công nghệ khóa học trong hệ thống hiện tại", icon=":material/layers:", expanded=False):
            st.dataframe(
                [
                    {"Tuần": "Tuần 1", "Công nghệ": "OpenAI API & Pydantic Structured Output", "Áp dụng trong Study Coach": "Quiz có evidence nguyên văn & option_diagnostics bắt lỗi tư duy (`core/quiz.py`, `core/contracts.py`)"},
                    {"Tuần": "Tuần 2", "Công nghệ": "Function Calling, Streaming, Telemetry SQLite", "Áp dụng trong Study Coach": "Tool đọc/ghi Learner DNA, đếm token/chi phí & Retry backoff (`core/llm.py`, `core/db.py`)"},
                    {"Tuần": "Tuần 3", "Công nghệ": "Embeddings & Semantic Search (FAISS)", "Áp dụng trong Study Coach": "Vector index tài liệu + Bản đồ nhiệt độ phủ + Chấm Vấn đáp Feynman (`core/rag.py`, `core/adaptive.py`)"},
                    {"Tuần": "Tuần 4", "Công nghệ": "LangChain Dual-Memory RAG", "Áp dụng trong Study Coach": "Tách chunk có overlap & metadata + Ký ức kép tài liệu và lịch sử lỗi sai (`core/rag.py`)"},
                    {"Tuần": "Tuần 5", "Công nghệ": "LangGraph Agent & Vision", "Áp dụng trong Study Coach": "Đồ thị rẽ 4 nhánh sư phạm theo trình độ + Vision chấm bài nháp từng bước (`core/coach_graph.py`, `core/vision.py`)"},
                ],
                hide_index=True,
            )


def main() -> None:
    # Áp lại key của phiên hiện tại (config giữ key là biến toàn tiến trình) để tránh rò key giữa các phiên.
    config.set_runtime_api_key(st.session_state.get("session_api_key") or None)
    try:
        _main_body()
    except _LLM_ERRORS as e:
        # Lưới an toàn: lời gọi LLM sót vẫn không hiện traceback.
        st.error(str(e))


def _main_body() -> None:
    _bootstrap_state()
    st.session_state.setdefault("chat_focus_mode", False)
    st.session_state.setdefault("show_dashboard_strip", True)
    st.session_state.setdefault("logged_in_learner_id", "")
    st.session_state.setdefault("_logged_out", False)
    if not st.session_state["logged_in_learner_id"] and not st.session_state.get("_logged_out"):
        try:
            cookie_tok = str(st.context.cookies.get("sc_session", "") or st.query_params.get("session", "") or "")
            restored_lid = _verify_session_token(cookie_tok)
            if restored_lid:
                _do_login(restored_lid, start_placement=False)
        except Exception:
            pass
    try:
        if "session" in st.query_params:
            st.query_params.pop("session", None)
    except Exception:
        pass
    login_slot = st.empty()
    if not st.session_state["logged_in_learner_id"]:
        if st.session_state.get("_logged_out"):
            st.html(
                '<script>document.cookie = "sc_session=; path=/; max-age=0; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax";</script>',
                unsafe_allow_javascript=True,
            )
        with login_slot.container():
            _render_login_screen()
        return
    login_slot.empty()
    st.session_state["learner_id"] = st.session_state["logged_in_learner_id"]
    if db.get_user_display_name(st.session_state["learner_id"]):
        tok = _make_session_token(st.session_state["learner_id"])
        st.html(
            f'<script>document.cookie = "sc_session={tok}; path=/; max-age=604800; SameSite=Lax";</script>',
            unsafe_allow_javascript=True,
        )
    telemetry = db.get_telemetry_summary(limit=5)
    in_focus = st.session_state.get("active_nav") == "coach" and bool(st.session_state.chat_focus_mode)

    dna = _current_dna()
    if not st.session_state.get("has_document") or st.session_state.get("kb") is None:
        _render_document_onboarding(dna)
        return
    _render_header(dna)
    coverage = compute_coverage_stats(dna.concepts)
    if not in_focus and st.session_state.show_dashboard_strip:
        st.markdown(render_bento_strip_html(dna, coverage, telemetry), unsafe_allow_html=True)

    cur_nav = st.session_state.get("active_nav") or "coach"
    if cur_nav not in NAV_LABELS:
        cur_nav = "coach"
    nav_ver = st.session_state.get("_nav_ver", 0)
    picked_nav = st.segmented_control(
        "Điều hướng",
        options=list(NAV_LABELS.keys()),
        default=cur_nav,
        format_func=lambda k: NAV_LABELS[k],
        selection_mode="single",
        key=f"nav_ctrl_{nav_ver}",
        label_visibility="collapsed",
        width="stretch",
    )
    active_nav = picked_nav or cur_nav
    st.session_state["active_nav"] = active_nav

    if active_nav == "map":
        _render_tab_map(dna, coverage)
    elif active_nav == "lab":
        _render_tab_lab(dna)
    elif active_nav == "coach":
        _render_tab_coach(dna)
    elif active_nav == "practice":
        _render_tab_practice(dna)
    else:
        _render_tab_analytics(dna)


if __name__ == "__main__":
    main()
