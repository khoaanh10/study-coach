"""Study Coach - Tuần 1: Quiz Maker. Chạy: streamlit run app.py"""
import streamlit as st

import config
from core.llm import LLMError
from core.loader import load_text
from core.quiz import generate_quiz, grade

st.set_page_config(page_title="Study Coach - Quiz Maker", layout="centered")

st.markdown(
    """
    <style>
    .block-container {padding-top: 3.5rem; max-width: 760px;}
    .masthead {border-bottom: 2px solid #1f2421; padding-bottom: .9rem; margin-bottom: 1.4rem;}
    .masthead .kicker {font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: #6b6f6a;}
    .masthead h1 {font-family: Georgia, "Times New Roman", serif; font-weight: 700;
                  font-size: 2.3rem; margin: .15rem 0 .3rem 0; padding: 0;}
    .masthead p {margin: 0; color: #4a4f4b;}
    .step {font-size: .75rem; letter-spacing: .12em; text-transform: uppercase;
           color: #6b6f6a; margin: 1rem 0 .3rem 0;}
    div[data-testid="stMetric"] {border-left: 3px solid #1b5e4b; padding-left: .9rem;}
    div[data-testid="stMetricValue"] {font-family: Georgia, "Times New Roman", serif;}
    .stButton button {border-radius: 6px;}
    section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stExpandSidebarButton"] {display: none;}
    .loading {margin-top: .8rem;}
    .loading .msg {font-weight: 600; color: #1b5e4b; margin-bottom: .5rem;}
    .loading .msg i {font-style: normal; animation: sc-dot 1.2s infinite both;}
    .loading .msg i:nth-child(2) {animation-delay: .2s;}
    .loading .msg i:nth-child(3) {animation-delay: .4s;}
    .loading .bar {height: 3px; background: #e3ded2; border-radius: 2px; overflow: hidden; margin-bottom: 1rem;}
    .loading .bar span {display: block; width: 35%; height: 100%; background: #1b5e4b;
                        border-radius: 2px; animation: sc-slide 1.3s ease-in-out infinite;}
    .loading .sk {border: 1px solid #e3ded2; border-radius: 8px; padding: 1rem; margin-bottom: .7rem;
                  animation: sc-in .4s ease both;}
    .loading .sk:nth-of-type(4) {animation-delay: .15s;}
    .loading .sk div {height: .7rem; border-radius: 4px; margin: .55rem 0;
                      background: linear-gradient(90deg, #ebe6da 25%, #f6f3ea 50%, #ebe6da 75%);
                      background-size: 200% 100%; animation: sc-shimmer 1.4s linear infinite;}
    .loading .sk div:first-child {height: .9rem; width: 70%;}
    @keyframes sc-dot {0%, 80%, 100% {opacity: .15;} 40% {opacity: 1;}}
    @keyframes sc-slide {0% {transform: translateX(-110%);} 100% {transform: translateX(300%);}}
    @keyframes sc-shimmer {0% {background-position: 200% 0;} 100% {background-position: -200% 0;}}
    @keyframes sc-in {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
    </style>
    <div class="masthead">
      <div class="kicker">Study Coach / Tuần 1</div>
      <h1>Quiz Maker</h1>
      <p>Dán văn bản hoặc upload PDF/TXT. Câu hỏi chỉ được ra từ nội dung đó.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def _retry():
    """Làm lại cùng bộ câu hỏi: xóa kết quả và đổi key của các ô chọn."""
    st.session_state.result = None
    st.session_state.quiz_version = st.session_state.get("quiz_version", 0) + 1


def _new_quiz():
    st.session_state.quiz = None
    st.session_state.result = None


if not config.OPENAI_API_KEY:
    st.warning("Chưa có OPENAI_API_KEY trong file .env")

quiz = st.session_state.get("quiz")
result = st.session_state.get("result")

# ---------- Bước 1: tài liệu ----------
if not quiz:
    st.markdown('<p class="step">Bước 1 · Chọn tài liệu</p>', unsafe_allow_html=True)
    tab_paste, tab_file = st.tabs(["Dán văn bản", "Upload PDF/TXT"])
    with tab_paste:
        pasted = st.text_area(
            "Văn bản", height=240, placeholder="Dán nội dung cần ôn tập...", label_visibility="collapsed"
        )
        if pasted:
            over = len(pasted) > config.MAX_INPUT_CHARS
            st.caption(
                f"{len(pasted):,} ký tự"
                + (f" · chỉ dùng {config.MAX_INPUT_CHARS:,} ký tự đầu" if over else "")
            )
    with tab_file:
        uploaded = st.file_uploader(
            "Chọn một hoặc nhiều file", type=["pdf", "txt"], accept_multiple_files=True
        )
        if len(uploaded) > 1:
            st.caption(
                f"Độ dài tối đa {config.MAX_INPUT_CHARS:,} ký tự được chia đều cho {len(uploaded)} file."
            )

    st.markdown('<p class="step">Tùy chọn</p>', unsafe_allow_html=True)
    with st.container(border=True):
        col_diff, col_n, col_model = st.columns([5, 2, 3])
        difficulty = col_diff.radio("Độ khó", config.DIFFICULTIES, index=1, horizontal=True)
        n = col_n.number_input("Số câu", config.MIN_QUESTIONS, config.MAX_QUESTIONS, 5, step=1)
        model = col_model.selectbox(
            "Model", config.MODEL_CHOICES, help="Model mạnh hơn ra câu hỏi tốt hơn nhưng tốn hơn."
        )

    if st.button("Tạo quiz", type="primary", use_container_width=True):
        loading = st.empty()
        try:
            if uploaded:
                per_file = config.MAX_INPUT_CHARS // len(uploaded)
                parts = [
                    f"=== {f.name} ===\n{load_text(f.name, f.getvalue())[:per_file]}" for f in uploaded
                ]
                source = "\n\n".join(parts)
            else:
                source = pasted
            skeleton = '<div class="sk">' + '<div style="width:{}%"></div>' * 5 + "</div>"
            loading.markdown(
                '<div class="loading">'
                f'<div class="msg">Đang tạo {n} câu hỏi mức {difficulty.lower()}<i>.</i><i>.</i><i>.</i></div>'
                '<div class="bar"><span></span></div>'
                + skeleton.format(70, 45, 55, 40, 50)
                + skeleton.format(60, 50, 38, 52, 44)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.session_state.quiz = generate_quiz(source, n, difficulty, model)
            st.session_state.result = None
            st.session_state.quiz_version = st.session_state.get("quiz_version", 0) + 1
            st.rerun()
        except (LLMError, ValueError) as e:
            loading.empty()
            st.error(str(e))

# ---------- Bước 2: làm bài ----------
elif not result:
    version = st.session_state.get("quiz_version", 0)
    total = len(quiz.questions)
    st.markdown('<p class="step">Bước 2 · Làm bài</p>', unsafe_allow_html=True)
    progress_slot = st.empty()

    picks = []
    for i, q in enumerate(quiz.questions):
        with st.container(border=True):
            st.markdown(f"**Câu {i + 1}/{total}.** {q.question}")
            choice = st.radio(
                f"Câu {i + 1}",
                options=range(len(q.options)),
                format_func=lambda k, opts=q.options: f"{'ABCD'[k]}. {opts[k]}",
                index=None,
                key=f"q{version}_{i}",
                label_visibility="collapsed",
            )
            picks.append(choice)

    answered = sum(p is not None for p in picks)
    progress_slot.progress(answered / total, text=f"Đã trả lời {answered}/{total} câu")

    col_submit, col_new = st.columns([3, 1])
    if col_submit.button("Nộp bài", type="primary", use_container_width=True):
        st.session_state.result = grade(quiz, picks)
        st.rerun()
    col_new.button("Quiz mới", on_click=_new_quiz, use_container_width=True)
    if answered < total:
        st.caption("Câu chưa chọn sẽ tính là sai khi nộp.")

# ---------- Bước 3: kết quả ----------
else:
    st.markdown('<p class="step">Bước 3 · Kết quả</p>', unsafe_allow_html=True)
    wrong = sum(1 for it in result.items if not it.is_correct and it.chosen_index is not None)
    skipped = sum(1 for it in result.items if it.chosen_index is None)
    ratio = result.score / result.total if result.total else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Điểm", f"{result.score}/{result.total}")
    c2.metric("Tỉ lệ đúng", f"{ratio:.0%}")
    c3.metric("Sai / Bỏ trống", f"{wrong} / {skipped}")
    st.progress(ratio)

    if ratio == 1:
        st.success("Hoàn hảo! Bạn nắm chắc phần này rồi.")
    elif ratio >= 0.7:
        st.success("Khá tốt. Xem lại các câu sai bên dưới để chắc hơn.")
    elif ratio >= 0.4:
        st.warning("Tạm ổn. Nên đọc lại tài liệu rồi làm lại.")
    else:
        st.error("Cần ôn lại phần này. Đọc kỹ giải thích từng câu bên dưới.")

    only_wrong = st.toggle("Chỉ xem câu sai", value=False) if result.score < result.total else False

    for i, it in enumerate(result.items, start=1):
        if only_wrong and it.is_correct:
            continue
        badge = "Đúng" if it.is_correct else ("Bỏ trống" if it.chosen_index is None else "Sai")
        with st.container(border=True):
            st.markdown(f"**Câu {i}.** {it.question} &nbsp; `{badge}`")
            for k, opt in enumerate(it.options):
                line = f"{'ABCD'[k]}. {opt}"
                if k == it.correct_index:
                    st.markdown(f":green[**✔ {line}**]")
                elif k == it.chosen_index:
                    st.markdown(f":red[✘ ~~{line}~~ (bạn chọn)]")
                else:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{line}")
            if it.explanation:
                st.info(it.explanation)

    col_retry, col_new = st.columns(2)
    col_retry.button("Làm lại quiz này", on_click=_retry, use_container_width=True)
    col_new.button("Tạo quiz mới", type="primary", on_click=_new_quiz, use_container_width=True)
