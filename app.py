"""Study Coach - Tuần 1: Quiz Maker. Chạy: streamlit run app.py"""
import streamlit as st

import config
from core.llm import LLMError
from core.loader import load_text
from core.quiz import generate_quiz, grade

st.set_page_config(page_title="Study Coach - Quiz Maker", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding: 3.2rem 3rem 3rem 3rem; max-width: 100%;}
    .masthead {display: flex; justify-content: space-between; align-items: flex-end; gap: 2rem;
               border-bottom: 2px solid #1f2421; padding-bottom: .9rem; margin-bottom: 1.4rem;}
    .masthead .kicker {font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; color: #6b6f6a;}
    .masthead h1 {font-family: Georgia, "Times New Roman", serif; font-weight: 700;
                  font-size: 2.3rem; margin: .15rem 0 .3rem 0; padding: 0;}
    .masthead p {margin: 0; color: #4a4f4b;}
    .steps {display: flex; gap: .5rem; flex: none;}
    .steps span {font-size: .8rem; padding: .35rem .8rem; border-radius: 999px;
                 border: 1px solid #cfc9bb; color: #6b6f6a; white-space: nowrap;}
    .steps span.on {background: #1b5e4b; border-color: #1b5e4b; color: #fff; font-weight: 600;}
    .steps span.done {border-color: #1b5e4b; color: #1b5e4b;}
    .step {font-size: .75rem; letter-spacing: .12em; text-transform: uppercase;
           color: #6b6f6a; margin: 0 0 .4rem 0;}
    .hint {color: #4a4f4b; font-size: .9rem; line-height: 1.6;}
    .hint b {color: #1f2421;}
    .score {font-family: Georgia, "Times New Roman", serif; font-size: 4rem; line-height: 1; color: #1f2421;
            margin-bottom: .8rem;}
    .score small {font-size: 1.6rem; color: #6b6f6a;}
    .chips {display: flex; flex-wrap: wrap; gap: .4rem; margin: .6rem 0 .2rem 0;}
    .chips span {width: 2rem; height: 2rem; display: flex; align-items: center; justify-content: center;
                 border-radius: 6px; border: 1px solid #cfc9bb; font-size: .85rem; color: #6b6f6a;}
    .chips span.on {background: #1b5e4b; border-color: #1b5e4b; color: #fff;}
    .chips span.bad {background: #b3261e; border-color: #b3261e; color: #fff;}
    /* Cột có .side-panel bám theo màn hình khi cuộn danh sách câu hỏi. */
    div[data-testid="stColumn"]:has(.side-panel) {position: sticky; top: 3.5rem; align-self: flex-start;}
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
    .loading .grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: .8rem;}
    .loading .sk {border: 1px solid #e3ded2; border-radius: 8px; padding: 1rem; animation: sc-in .4s ease both;}
    .loading .sk div {height: .7rem; border-radius: 4px; margin: .55rem 0;
                      background: linear-gradient(90deg, #ebe6da 25%, #f6f3ea 50%, #ebe6da 75%);
                      background-size: 200% 100%; animation: sc-shimmer 1.4s linear infinite;}
    .loading .sk div:first-child {height: .9rem;}
    @keyframes sc-dot {0%, 80%, 100% {opacity: .15;} 40% {opacity: 1;}}
    @keyframes sc-slide {0% {transform: translateX(-110%);} 100% {transform: translateX(300%);}}
    @keyframes sc-shimmer {0% {background-position: 200% 0;} 100% {background-position: -200% 0;}}
    @keyframes sc-in {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
    @media (max-width: 900px) {
        .block-container {padding: 3.2rem 1rem 2rem 1rem;}
        .masthead {flex-direction: column; align-items: flex-start;}
    }
    </style>
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


quiz = st.session_state.get("quiz")
result = st.session_state.get("result")
current = 1 if not quiz else (2 if not result else 3)

steps_html = "".join(
    f'<span class="{"on" if k == current else ("done" if k < current else "")}">{k}. {name}</span>'
    for k, name in enumerate(["Chọn tài liệu", "Làm bài", "Kết quả"], start=1)
)
st.markdown(
    f"""
    <div class="masthead">
      <div>
        <div class="kicker">Study Coach / Tuần 1</div>
        <h1>Quiz Maker</h1>
        <p>Dán văn bản hoặc upload PDF/TXT. Câu hỏi chỉ được ra từ nội dung đó.</p>
      </div>
      <div class="steps">{steps_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not config.OPENAI_API_KEY:
    st.warning("Chưa có OPENAI_API_KEY trong file .env")

# ---------- Bước 1: tài liệu ----------
if current == 1:
    col_doc, col_opt = st.columns([7, 3], gap="large")

    with col_doc:
        st.markdown('<p class="step">Tài liệu</p>', unsafe_allow_html=True)
        tab_paste, tab_file = st.tabs(["Dán văn bản", "Upload PDF/TXT"])
        with tab_paste:
            pasted = st.text_area(
                "Văn bản", height=420, placeholder="Dán nội dung cần ôn tập...", label_visibility="collapsed"
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

    with col_opt:
        st.markdown('<p class="step">Tùy chọn</p>', unsafe_allow_html=True)
        with st.container(border=True):
            difficulty = st.radio("Độ khó", config.DIFFICULTIES, index=1, horizontal=True)
            n = st.number_input("Số câu", config.MIN_QUESTIONS, config.MAX_QUESTIONS, 5, step=1)
            model = st.selectbox(
                "Model", config.MODEL_CHOICES, help="Model mạnh hơn ra câu hỏi tốt hơn nhưng tốn hơn."
            )
        create = st.button("Tạo quiz", type="primary", use_container_width=True)
        st.markdown(
            f"""
            <div class="hint">
              <b>Lưu ý</b><br>
              · Văn bản cần ít nhất 100 ký tự và có nội dung để học.<br>
              · Mỗi quiz dùng tối đa {config.MAX_INPUT_CHARS:,} ký tự đầu.<br>
              · Mỗi câu hỏi được soát lại với tài liệu trước khi hiện ra, nên cần chờ khoảng 20 giây.
            </div>
            """,
            unsafe_allow_html=True,
        )

    loading = st.empty()
    if create:
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
            widths = [(70, 45, 55, 40, 50), (60, 50, 38, 52, 44), (75, 42, 48, 36, 54), (64, 52, 40, 46, 38)]
            loading.markdown(
                '<div class="loading">'
                f'<div class="msg">Đang tạo {n} câu hỏi mức {difficulty.lower()}<i>.</i><i>.</i><i>.</i></div>'
                '<div class="bar"><span></span></div><div class="grid">'
                + "".join(skeleton.format(*w) for w in widths)
                + "</div></div>",
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
elif current == 2:
    version = st.session_state.get("quiz_version", 0)
    total = len(quiz.questions)
    col_questions, col_side = st.columns([8, 3], gap="large")

    picks = []
    with col_questions:
        st.markdown('<p class="step">Câu hỏi</p>', unsafe_allow_html=True)
        grid = st.columns(2, gap="medium")
        for i, q in enumerate(quiz.questions):
            with grid[i % 2].container(border=True):
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
    with col_side:
        st.markdown('<span class="side-panel"></span><p class="step">Tiến độ</p>', unsafe_allow_html=True)
        with st.container(border=True):
            st.progress(answered / total, text=f"Đã trả lời {answered}/{total} câu")
            chips = "".join(
                f'<span class="{"on" if p is not None else ""}">{i + 1}</span>' for i, p in enumerate(picks)
            )
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
            if answered < total:
                st.caption("Câu chưa chọn sẽ tính là sai khi nộp.")
        if st.button("Nộp bài", type="primary", use_container_width=True):
            st.session_state.result = grade(quiz, picks)
            st.rerun()
        st.button("Quiz mới", on_click=_new_quiz, use_container_width=True)

# ---------- Bước 3: kết quả ----------
else:
    wrong = sum(1 for it in result.items if not it.is_correct and it.chosen_index is not None)
    skipped = sum(1 for it in result.items if it.chosen_index is None)
    ratio = result.score / result.total if result.total else 0
    col_side, col_review = st.columns([3, 8], gap="large")

    with col_side:
        st.markdown('<span class="side-panel"></span><p class="step">Kết quả</p>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f'<div class="score">{result.score}<small>/{result.total}</small></div>', unsafe_allow_html=True
            )
            st.progress(ratio, text=f"Tỉ lệ đúng {ratio:.0%}")
            chips = "".join(
                f'<span class="{"on" if it.is_correct else ("bad" if it.chosen_index is not None else "")}">{i}</span>'
                for i, it in enumerate(result.items, start=1)
            )
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
            st.caption(f"Đúng {result.score} · Sai {wrong} · Bỏ trống {skipped}")
        if ratio == 1:
            st.success("Hoàn hảo! Bạn nắm chắc phần này rồi.")
        elif ratio >= 0.7:
            st.success("Khá tốt. Xem lại các câu sai để chắc hơn.")
        elif ratio >= 0.4:
            st.warning("Tạm ổn. Nên đọc lại tài liệu rồi làm lại.")
        else:
            st.error("Cần ôn lại phần này. Đọc kỹ giải thích từng câu.")
        only_wrong = st.toggle("Chỉ xem câu sai", value=False) if result.score < result.total else False
        st.button("Tạo quiz mới", type="primary", on_click=_new_quiz, use_container_width=True)
        st.button("Làm lại quiz này", on_click=_retry, use_container_width=True)

    with col_review:
        st.markdown('<p class="step">Xem lại từng câu</p>', unsafe_allow_html=True)
        shown = [(i, it) for i, it in enumerate(result.items, start=1) if not (only_wrong and it.is_correct)]
        grid = st.columns(2, gap="medium")
        for pos, (i, it) in enumerate(shown):
            badge = "Đúng" if it.is_correct else ("Bỏ trống" if it.chosen_index is None else "Sai")
            with grid[pos % 2].container(border=True):
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
