"""Hệ thống Design Tokens & Components chuẩn UI/UX Pro Max cho Study Coach."""
from __future__ import annotations

import html
import re
from pathlib import Path

from core.contracts import (
    CoachTurnResult,
    ConceptMastery,
    ConceptNode,
    DiffContradiction,
    IncidentScenario,
    LearnerDNA,
)

_SVG_PATHS: dict[str, str] = {
    "book": '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "circle-dashed": '<path d="M10.1 2.18a9.93 9.93 0 0 1 3.8 0"/><path d="M17.6 3.71a9.95 9.95 0 0 1 2.69 2.7"/><path d="M21.82 10.1a9.93 9.93 0 0 1 0 3.8"/><path d="M20.29 17.6a9.95 9.95 0 0 1-2.7 2.69"/><path d="M13.9 21.82a9.94 9.94 0 0 1-3.8 0"/><path d="M6.4 20.29a9.95 9.95 0 0 1-2.69-2.7"/><path d="M2.18 13.9a9.93 9.93 0 0 1 0-3.8"/><path d="M3.71 6.4a9.95 9.95 0 0 1 2.7-2.69"/>',
    "route": '<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>',
    "quote": '<path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z"/><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 1 0 1 1 1z"/>',
    "user": '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "git-compare": '<circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M13 6h3a2 2 0 0 1 2 2v7"/><path d="M11 18H8a2 2 0 0 1-2-2V9"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
    "shield-alert": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/>',
    "search-check": '<path d="m8 11 2 2 4-4"/><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
}


def clean_html(raw: str) -> str:
    """Loại bỏ thụt lề đầu dòng và dòng trống để Markdown không hiểu nhầm khối HTML thành code block."""
    return "\n".join(line.lstrip() for line in raw.strip().splitlines() if line.strip())


def esc(text: str) -> str:
    return html.escape(str(text or ""), quote=True)


def icon_svg(name: str, size: int = 16) -> str:
    body = _SVG_PATHS.get(name, _SVG_PATHS["target"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'style="flex:none;vertical-align:middle">{body}</svg>'
    )


_CSS_PATH = Path(__file__).with_name("ui_theme.css")
_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700'
    '&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
)


def inject_theme_css() -> str:
    """Trả về chuỗi <link> font + <style> đọc từ core/ui_theme.css (design tokens & component styles)."""
    css = _CSS_PATH.read_text(encoding="utf-8")
    return _FONT_LINKS + "\n<style>\n" + clean_html(css) + "\n</style>"


def render_brand_lockup_html(subtitle: str = "Gia sư hiểu năng lực từng người học trên tài liệu riêng") -> str:
    return clean_html(
        f"""
        <div class="sc-brand-lockup">
          <div class="sc-logo-box">{icon_svg("brain", 22)}</div>
          <div>
            <div class="sc-title">Study Coach</div>
            <p class="sc-subtitle">{esc(subtitle)}</p>
          </div>
        </div>
        """
    )


def render_masthead_html(doc_title: str, is_live: bool, model_name: str) -> str:
    """Giữ lại để tương thích ngược: lockup thương hiệu kèm tên tài liệu và trạng thái kết nối."""
    mode_txt = f"AI Portal Live ({model_name})" if is_live else "Chưa kết nối AI Portal"
    pill_cls = "sc-live-pill" if is_live else "sc-live-pill off"
    return clean_html(
        f"""
        <div class="sc-panel" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.75rem;">
          {render_brand_lockup_html()}
          <div class="sc-meta-row" style="margin:0;">
            <span class="sc-prereq-chip">{esc(doc_title)}</span>
            <span class="{pill_cls}">{esc(mode_txt)}</span>
          </div>
        </div>
        """
    )


def render_login_hero_html() -> str:
    """Khối giới thiệu bên trái màn đăng nhập."""
    items = [
        ("layers", "Bản đồ năng lực theo từng khái niệm", "Biết chính xác phần nào đã vững, phần nào còn hổng."),
        ("quote", "Gia sư Socratic bám sát tài liệu", "Mọi câu trả lời đều trích nguồn, không bịa ngoài tài liệu."),
        ("target", "Quiz thích ứng bắt đúng lỗi tư duy", "Mỗi đáp án sai chỉ ra bạn đang hiểu lệch ở đâu."),
    ]
    items_html = "".join(
        f'<div class="sc-hero-item"><div class="sc-hero-item-icon">{icon_svg(ic, 17)}</div>'
        f"<div><b>{esc(title)}</b><span>{esc(desc)}</span></div></div>"
        for ic, title, desc in items
    )
    return clean_html(
        f"""
        <div class="sc-hero">
          {render_brand_lockup_html("Adaptive Personal AI Tutor · AIA19 Group 1")}
          <div>
            <div class="sc-hero-headline">Học đúng chỗ mình còn yếu, trên <em>chính tài liệu của bạn</em>.</div>
            <p class="sc-hero-lead">Tải giáo trình lên, làm bài chẩn đoán vài phút, và để gia sư AI dẫn bạn đi từ chỗ đang hiểu lệch tới chỗ làm chủ.</p>
          </div>
          <div class="sc-hero-list">{items_html}</div>
        </div>
        """
    )


def render_auth_head_html() -> str:
    return clean_html(
        """
        <div class="sc-auth-head">
          <h2>Bắt đầu phiên học</h2>
          <p>Đăng nhập bằng tài khoản của bạn hoặc đăng ký mới. Trình độ ban đầu được đánh giá qua bài kiểm tra đầu vào.</p>
        </div>
        """
    )


def render_sidebar_status_html(doc_title: str, level: str, coverage_pct: float, streak_days: int | None = None) -> str:
    """Khối 'đang học' trong thanh dọc: tên tài liệu, trình độ, % hoàn thành và chuỗi ngày học."""
    pct = max(0.0, min(100.0, coverage_pct))
    streak_html = ""
    if streak_days is not None:
        streak_txt = f"Chuỗi {int(streak_days)} ngày học" if streak_days else "Chưa có chuỗi ngày học"
        streak_html = f'<div class="sc-side-streak">{icon_svg("flame", 14)}<span>{esc(streak_txt)}</span></div>'
    return clean_html(
        f"""
        <div class="sc-side-status">
          <div class="sc-side-label">Đang học</div>
          <div class="sc-side-doc">{icon_svg("book", 15)}<span>{esc(doc_title)}</span></div>
          <div class="sc-side-level"><span>{esc(level)}</span><b>{pct:.0f}%</b></div>
          <div class="sc-progress-track"><div class="sc-progress-fill" style="width:{pct:.0f}%;"></div></div>
          {streak_html}
        </div>
        """
    )


def render_streak_html(streak: dict, due_count: int, misc_count: int) -> str:
    """Thẻ chuỗi ngày học + số mục cần ôn hôm nay (trang Ôn hôm nay và đầu trang Bản đồ)."""
    cur = int(streak.get("current", 0) or 0)
    longest = int(streak.get("longest", 0) or 0)
    if streak.get("studied_today"):
        hint = "Hôm nay bạn đã học. Giữ nhịp này nhé."
    elif cur:
        hint = "Hôm nay chưa học: làm một phiên ôn để nối chuỗi."
    else:
        hint = "Làm một phiên ôn để bắt đầu chuỗi ngày học."
    return clean_html(
        f"""
        <div class="sc-streak">
          <div class="sc-streak-main">
            <div class="sc-streak-icon{' on' if cur else ''}">{icon_svg("flame", 22)}</div>
            <div>
              <div class="sc-streak-num">{cur}<small>ngày liên tục</small></div>
              <div class="sc-streak-hint">{esc(hint)} Kỷ lục: {longest} ngày.</div>
            </div>
          </div>
          <div class="sc-streak-stats">
            <div><b>{int(due_count)}</b><span>khái niệm đến hạn ôn</span></div>
            <div><b>{int(misc_count)}</b><span>lỗi tư duy đang mở</span></div>
          </div>
        </div>
        """
    )


def render_review_targets_html(targets: list[dict]) -> str:
    """Danh sách mục sẽ được ôn trong phiên hôm nay."""
    if not targets:
        return ""
    rows = []
    for t in targets:
        is_misc = t.get("reason") == "misconception"
        badge = (
            '<span class="sc-status-badge misconception">Lỗi tư duy</span>'
            if is_misc
            else '<span class="sc-status-badge learning">Đến hạn ôn</span>'
        )
        rows.append(
            f'<div class="sc-review-row"><div><b>{esc(t.get("topic", ""))}</b>'
            f'<span>{esc(t.get("detail", ""))}</span></div>'
            f'<div class="sc-review-meta">{badge}<i>{float(t.get("mastery_score", 0) or 0):.0f}%</i></div></div>'
        )
    return clean_html(f'<div class="sc-review-list">{"".join(rows)}</div>')


# ---------- Trình đọc tài liệu & đánh dấu đoạn được trích dẫn ----------
def section_anchor(index: int) -> str:
    return f"sc-sec-{index + 1}"


def find_highlight_ranges(text: str, snippets: list[str]) -> list[tuple[int, int]]:
    """Tìm vị trí các đoạn trích (chunk Gia sư đã dùng) trong văn bản một mục.

    So khớp bỏ qua khác biệt khoảng trắng; đoạn trích không khớp nguyên khối thì thử từng câu của nó.
    Trả các khoảng [start, end) đã gộp, sắp theo thứ tự."""
    ranges: list[tuple[int, int]] = []

    def _find(piece: str) -> tuple[int, int] | None:
        words = piece.split()
        if not words:
            return None
        pattern = r"\s+".join(re.escape(w) for w in words)
        m = re.search(pattern, text)
        return (m.start(), m.end()) if m else None

    for snip in snippets or []:
        snip = (snip or "").strip()
        if len(snip) < 8:
            continue
        hit = _find(snip)
        if hit:
            ranges.append(hit)
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n+", snip):
            if len(sent.strip()) >= 20:
                hit = _find(sent.strip())
                if hit:
                    ranges.append(hit)
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def highlight_text_html(text: str, snippets: list[str] | None = None, query: str = "") -> str:
    """Escape văn bản, bọc <mark> quanh đoạn được trích dẫn (và từ khóa tìm kiếm nếu có)."""
    ranges = find_highlight_ranges(text, snippets or [])
    q = (query or "").strip()
    if q and len(q) >= 2:
        ranges += [(m.start(), m.end()) for m in re.finditer(re.escape(q), text, flags=re.IGNORECASE)]
        ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        ranges = merged
    out: list[str] = []
    pos = 0
    for start, end in ranges:
        out.append(esc(text[pos:start]))
        out.append(f'<mark class="sc-hl">{esc(text[start:end])}</mark>')
        pos = end
    out.append(esc(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def render_doc_section_html(
    section: dict,
    index: int,
    snippets: list[str] | None = None,
    focused: bool = False,
    query: str = "",
    with_anchor: bool = True,
) -> str:
    """Một mục của tài liệu: nhãn [Trang/Mục n] (trùng với nhãn trích dẫn), nguồn và nội dung."""
    text = str(section.get("text", ""))
    # Mục được trích mà không khớp đoạn con nào: đánh dấu cả mục (qua class focus) thay vì để trống.
    body = highlight_text_html(text, snippets, query)
    anchor = f' id="{section_anchor(index)}"' if with_anchor else ""
    cls = "sc-doc-section focus" if focused else "sc-doc-section"
    return clean_html(
        f"""
        <div class="{cls}"{anchor}>
          <div class="sc-doc-section-head">
            <span class="sc-mono-chip">{icon_svg("quote", 12)} {esc(section.get("page", ""))}</span>
            <span class="sc-doc-source">{esc(section.get("source", ""))}</span>
          </div>
          <div class="sc-doc-section-body">{body}</div>
        </div>
        """
    )


def render_doc_toc_html(sections: list[dict], indexes: list[int] | None = None) -> str:
    """Mục lục của tài liệu: mỗi dòng là một liên kết neo tới đúng mục trong trang."""
    idxs = indexes if indexes is not None else list(range(len(sections)))
    links = "".join(
        f'<a href="#{section_anchor(i)}"><span class="sc-mono-chip">{esc(sections[i].get("page", ""))}</span>'
        f'<span>{esc(str(sections[i].get("topic_hint", ""))[:60])}</span></a>'
        for i in idxs
    )
    return clean_html(f'<nav class="sc-doc-toc" aria-label="Mục lục tài liệu">{links}</nav>')


def render_page_head_html(title: str, lead: str) -> str:
    return clean_html(f'<div class="sc-page-head"><h2>{esc(title)}</h2><p>{esc(lead)}</p></div>')


def render_chat_header_html(is_live: bool) -> str:
    pill = (
        '<span class="sc-live-pill">Trực tuyến</span>'
        if is_live
        else '<span class="sc-live-pill off">Chưa kết nối</span>'
    )
    return clean_html(
        f"""
        <div class="sc-chat-card-header">
          <div class="sc-chat-avatar-badge">{icon_svg("brain", 22)}</div>
          <div>
            <div class="sc-chat-name"><span>Study Coach AI</span> {pill}</div>
            <div class="sc-chat-sub">Gia sư Socratic thích ứng theo tài liệu của bạn</div>
          </div>
        </div>
        """
    )


def render_chat_empty_html(name: str, level: str) -> str:
    return clean_html(
        f"""
        <div class="sc-empty">
          <div class="sc-empty-icon">{icon_svg("brain", 28)}</div>
          <div class="sc-empty-title">Chào {esc(name)}, hôm nay học gì?</div>
          <div class="sc-empty-text">Gia sư đã nạp hồ sơ của bạn (trình độ <b>{esc(level)}</b>). Chọn một gợi ý bên dưới hoặc đặt câu hỏi về tài liệu ở thanh nhập liệu.</div>
        </div>
        """
    )


def render_coach_context_html(doc_title: str, topics: str, name: str, level: str, open_errors: int) -> str:
    """Thẻ bối cảnh học tập ở cột phải của màn Gia sư."""
    err_cls = "misconception" if open_errors else "mastered"
    err_txt = f"{open_errors} lỗi đang mở" if open_errors else "Không có lỗi đang mở"
    return clean_html(
        f"""
        <div class="sc-ctx">
          <div class="sc-ctx-title">Bối cảnh học tập</div>
          <div class="sc-ctx-row">{icon_svg("book", 15)}<div><span>Tài liệu</span><b>{esc(doc_title)}</b></div></div>
          <div class="sc-ctx-row">{icon_svg("layers", 15)}<div><span>Chủ đề mở đầu</span><b>{esc(topics)}</b></div></div>
          <div class="sc-ctx-row">{icon_svg("user", 15)}<div><span>Người học</span><b>{esc(name)} · {esc(level)}</b></div></div>
          <div class="sc-meta-row"><span class="sc-status-badge {err_cls}">{esc(err_txt)}</span></div>
        </div>
        """
    )


def render_typing_html(label: str = "Gia sư đang suy nghĩ") -> str:
    """Chỉ báo đang soạn câu trả lời (ba chấm nhấp nháy) trong bong bóng của Gia sư."""
    return clean_html(
        f"""
        <div class="sc-typing" role="status" aria-live="polite">
          <span class="sc-typing-dots"><i></i><i></i><i></i></span>
          <span>{esc(label)}…</span>
        </div>
        """
    )


def render_socratic_question_html(question: str) -> str:
    return clean_html(
        f"""
        <div class="sc-socratic">
          {icon_svg("quote", 15)}
          <div><b>Câu hỏi Socratic dành cho bạn:</b> {esc(question)}</div>
        </div>
        """
    )


def render_vision_step_html(step_name: str, status: str, feedback: str, page_ref: str) -> str:
    badge_cls = "mastered" if status == "Đúng" else ("learning" if status == "Cần chỉnh" else "misconception")
    ic_name = "check-circle" if status == "Đúng" else ("clock" if status == "Cần chỉnh" else "alert-triangle")
    return clean_html(
        f"""
        <div class="sc-step-card">
          <div class="sc-step-head">
            <span>{esc(step_name)}</span>
            <span class="sc-status-badge {badge_cls}">{icon_svg(ic_name, 13)} {esc(status)}</span>
          </div>
          <div class="sc-step-body">{esc(feedback)} <span class="sc-mono-chip">{esc(page_ref)}</span></div>
        </div>
        """
    )


def render_bento_strip_html(dna: LearnerDNA, coverage: dict, activity: dict | None = None) -> str:
    """Dải KPI của RIÊNG người học. `activity`: {"streak_days", "coach_questions"} (số liệu theo người học;
    số lượt gọi AI toàn hệ thống đã chuyển sang bảng kỹ thuật ở trang Số liệu)."""
    activity = activity or {}
    misc = len(dna.active_misconceptions)
    streak_days = int(activity.get("streak_days", 0) or 0)
    coach_q = int(activity.get("coach_questions", 0) or 0)
    attempts = sum(c.attempts for c in dna.concepts)
    cov_pct = max(0.0, min(100.0, float(coverage.get("coverage_pct", 0) or 0)))

    def card(icon: str, label: str, value: str, meta: str, alert: bool = False, extra: str = "") -> str:
        cls = " alert" if alert else ""
        return (
            f'<div class="sc-kpi"><div class="sc-kpi-top"><span class="sc-kpi-label">{label}</span>'
            f'<span class="sc-kpi-icon{cls}">{icon_svg(icon, 16)}</span></div>'
            f'<div class="sc-kpi-value">{value}</div><div class="sc-kpi-meta">{meta}</div>{extra}</div>'
        )

    progress = (
        f'<div class="sc-progress-track"><div class="sc-progress-fill" style="width:{cov_pct:.0f}%;"></div></div>'
    )
    return clean_html(
        '<div class="sc-bento-grid">'
        + card("layers", "Trình độ", esc(dna.overall_level), f"đã hoàn thành {cov_pct:.0f}% khóa học", extra=progress)
        + card("check-circle", "Độ chính xác", f"{dna.mastery_avg:.0f}<small>%</small>", "điểm nắm vững trung bình")
        + card(
            "alert-triangle",
            "Lỗi tư duy đang mở",
            str(misc),
            "cần ôn tập thêm" if misc else "chưa ghi nhận lỗi nào",
            alert=misc > 0,
        )
        + card("book", "Đã luyện tập", f"{attempts}<small>câu</small>", "tổng lượt trả lời")
        + card("flame", "Chuỗi ngày học", f"{streak_days}<small>ngày</small>", f"đã hỏi Gia sư {coach_q} câu")
        + "</div>"
    )


def _status_meta(status: str) -> tuple[str, str, str]:
    return {
        "mastered": ("Đã làm chủ", "check-circle", "#107A4D"),
        "learning": ("Cần ôn lại", "clock", "#B25E09"),
        "misconception": ("Đang hiểu lệch", "alert-triangle", "#C0291C"),
    }.get(status, ("Chưa học", "circle-dashed", "#525B6B"))


def render_concept_card_html(node: ConceptNode, mastery: ConceptMastery) -> str:
    label, ic, color = _status_meta(mastery.status)
    prereq_html = (
        f'<span class="sc-prereq-chip">Tiên quyết: {esc(", ".join(node.prerequisites))}</span>'
        if node.prerequisites
        else '<span class="sc-prereq-chip">Kiến thức nền tảng gốc</span>'
    )
    due_html = '<span class="sc-status-badge learning">Đến hạn ôn</span>' if mastery.due_for_review else ""
    pct = max(4.0 if mastery.attempts > 0 else 0.0, min(100.0, mastery.mastery_score))
    return clean_html(
        f"""
        <div class="sc-concept-card st-{esc(mastery.status)}">
          <div class="sc-card-top">
            <h4 class="sc-card-title">{esc(node.name)}</h4>
            <span class="sc-status-badge {esc(mastery.status)}">{icon_svg(ic, 14)} {esc(label)} · {mastery.mastery_score:.0f}%</span>
          </div>
          <p class="sc-card-desc">{esc(node.summary)}</p>
          <div class="sc-progress-track">
            <div class="sc-progress-fill" style="width:{pct:.0f}%;background:{color};"></div>
          </div>
          <div class="sc-meta-row">
            <span class="sc-mono-chip">{esc(node.page_ref or '[Tài liệu]')}</span>
            <span class="sc-prereq-chip">Bloom: <b>{esc(mastery.bloom_level)}</b></span>
            {prereq_html}
            {due_html}
          </div>
        </div>
        """
    )


def render_route_banner_html(turn: CoachTurnResult) -> str:
    route_meta = {
        "adaptive_explain": ("Giảng thích ứng theo trình độ", "adaptive", "brain"),
        "socratic_remediation": ("Gỡ rối Socratic (Bắt bệnh tư duy)", "remediation", "alert-triangle"),
        "bridge_prerequisite": ("Vá lỗ hổng kiến thức nền tảng", "prereq", "route"),
        "refuse_out_of_doc": ("Từ chối ngoài tài liệu (Chống bịa)", "refuse", "shield-check"),
        "coach_guide": ("Điều phối lộ trình & Hồ sơ cá nhân", "guide", "target"),
        "acknowledge": ("Xác nhận ngắn: kiểm tra mức hiểu", "guide", "check-circle"),
        "socratic_eval": ("Chấm phản hồi Socratic & Cập nhật DNA", "eval", "check-circle"),
    }.get(turn.route_taken, (turn.route_taken, "adaptive", "route"))
    label, cls, ic = route_meta
    return clean_html(
        f"""
        <div class="sc-route-banner">
          <div style="display:flex;align-items:center;gap:0.55rem;flex-wrap:wrap;">
            <span class="sc-route-pill {cls}">{icon_svg(ic, 14)} {esc(label)}</span>
            <span>Trình độ áp dụng: <b>{esc(turn.learner_level_used)}</b></span>
          </div>
          <div class="sc-route-meta">{turn.latency_ms} ms · {len(turn.tool_traces)} tools</div>
        </div>
        """
    )


def render_citation_pills_html(citations: list[str]) -> str:
    if not citations:
        return ""
    chips = "".join(f'<span class="sc-mono-chip">{icon_svg("quote", 12)} {esc(c)}</span>' for c in citations)
    return clean_html(f'<div class="sc-meta-row" style="margin:0.35rem 0 0.5rem 0;">{chips}</div>')


def render_misconception_card_html(
    topic: str,
    description: str,
    occurrences: int = 1,
    resolved: bool = False,
) -> str:
    cls = "sc-misc-box resolved" if resolved else "sc-misc-box"
    ic = "check-circle" if resolved else "alert-triangle"
    state_txt = "Đã khắc phục" if resolved else f"Lặp lại {occurrences} lần"
    return clean_html(
        f"""
        <div class="{cls}">
          <div style="padding-top:2px;">{icon_svg(ic, 17)}</div>
          <div>
            <div style="font-weight:700;">{esc(topic)} · <span style="font-weight:500;">{esc(state_txt)}</span></div>
            <div style="color:var(--ink);margin-top:2px;">{esc(description)}</div>
          </div>
        </div>
        """
    )


def render_next_step_callout_html(dna: LearnerDNA, top_plan: dict) -> str:
    """Hiển thị khuyến nghị bước học tiếp theo ngay trên đầu Bản đồ Năng lực."""
    return clean_html(
        f"""
        <div class="sc-callout">
          <div class="sc-callout-icon">{icon_svg("target", 18)}</div>
          <div>
            <div class="sc-callout-kicker">Bước tiếp theo AI đề xuất cho {esc(dna.name)} · {esc(top_plan['priority_label'])}</div>
            <div class="sc-callout-title">
              {esc(top_plan['concept'])} <span class="sc-mono-chip">{esc(top_plan.get('page_ref') or '[Tài liệu]')}</span>
            </div>
            <div class="sc-callout-body">{esc(top_plan['reason'])}</div>
          </div>
        </div>
        """
    )


def render_mastery_chart_html(concepts: list[ConceptMastery]) -> str:
    """Biểu đồ thanh ngang trực quan phân bố điểm thành thạo từng khái niệm (kèm nhãn trạng thái)."""
    if not concepts:
        return ""
    rows_html = []
    for c in concepts:
        _label, ic, color = _status_meta(c.status)
        pct = max(2.0 if c.attempts > 0 else 0.0, min(100.0, c.mastery_score))
        rows_html.append(
            f'<div class="sc-chart-row">'
            f'<div class="sc-chart-name">{esc(c.concept)}</div>'
            f'<div class="sc-chart-track"><div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:999px;"></div></div>'
            f'<div class="sc-chart-val" style="color:{color};">{icon_svg(ic, 13)} {c.mastery_score:.0f}% · {esc(c.bloom_level)}</div>'
            f"</div>"
        )
    inner = "".join(rows_html)
    return clean_html(
        f"""
        <div class="sc-panel">
          <div class="sc-panel-title">Mức thành thạo theo từng khái niệm &amp; cấp độ Bloom</div>
          {inner}
        </div>
        """
    )


def render_git_diff_card_html(item: DiffContradiction) -> str:
    """Thẻ đối lập kiểu git diff giữa ghi chú cá nhân và tài liệu gốc."""
    return clean_html(
        f"""
        <div class="sc-diff-card">
          <div class="sc-diff-head">
            <span style="display:inline-flex;align-items:center;gap:0.45rem;">{icon_svg("git-compare", 16)} Mâu thuẫn ở chủ đề: {esc(item.topic)}</span>
            <span style="display:inline-flex;align-items:center;gap:0.4rem;">
              <span class="sc-mono-chip">{esc(item.page_ref)}</span>
              <span class="sc-status-badge misconception">Mức {esc(item.severity)}</span>
            </span>
          </div>
          <div class="sc-diff-del">- Ghi chú của bạn: {esc(item.learner_claim)}</div>
          <div class="sc-diff-add">+ Giáo trình gốc ({esc(item.page_ref)}): {esc(item.ground_truth)}</div>
          <div class="sc-diff-note"><b>Giải thích:</b> {esc(item.explanation)}</div>
        </div>
        """
    )


def render_incident_ticket_html(scenario: IncidentScenario) -> str:
    """Thẻ Ticket sự cố Production chuyên nghiệp."""
    sev_cls = "misconception" if scenario.severity == "SEV-1" else "learning"
    return clean_html(
        f"""
        <div class="sc-ticket">
          <div class="sc-ticket-head">
            <span class="sc-ticket-id">{icon_svg("flame", 16)} TICKET SỰ CỐ PRODUCTION · {esc(scenario.ticket_id)}</span>
            <span style="display:inline-flex;align-items:center;gap:0.4rem;">
              <span class="sc-status-badge {sev_cls}">{esc(scenario.severity)}</span>
              <span class="sc-prereq-chip">Liên quan: <b>{esc(scenario.topic)}</b></span>
            </span>
          </div>
          <div class="sc-ticket-title">{esc(scenario.title)}</div>
          <div class="sc-ticket-body">{esc(scenario.symptom)}</div>
        </div>
        """
    )


def render_confidence_matrix_html(stats: dict) -> str:
    """Ma trận 2x2 Radar Ảo tưởng Hiểu bài (Confidence Calibration / Dunning-Kruger)."""
    cells = [
        ("Vững thực chất", "Trả lời Đúng + Tự tin cao/vừa", stats.get("solid_correct", 0), "check-circle", "mastered"),
        ("Đúng do đoán mò", "Kiến thức giả · Chọn 'Đoán mò' — cần ôn lại", stats.get("lucky_guess", 0), "clock", "learning"),
        ("Điểm mù tự tin sai", "Nguy hiểm nhất · Tự tin chắc chắn nhưng Sai", stats.get("blind_spot", 0), "shield-alert", "misconception"),
        ("Biết mình chưa biết", "Trả lời Sai/Bỏ trống + Tự nhận chưa chắc", stats.get("aware_gap", 0), "search-check", "unseen"),
    ]
    cards = []
    for title, desc, val, ic, tone in cells:
        cards.append(
            f'<div class="sc-matrix-cell" style="background:var(--{tone}-bg);border-color:var(--{tone}-bd);">'
            f'<div class="sc-matrix-head" style="color:var(--{tone}-fg);">'
            f'<span style="display:inline-flex;align-items:center;gap:0.4rem;">{icon_svg(ic, 16)} {esc(title)}</span>'
            f'<span class="sc-matrix-num">{val}</span>'
            f"</div>"
            f'<div class="sc-matrix-desc">{esc(desc)}</div>'
            f"</div>"
        )
    inner = "".join(cards)
    return clean_html(
        f"""
        <div class="sc-panel">
          <div class="sc-panel-title">Radar ảo tưởng hiểu bài (độ lệch giữa tự tin và thực lực)</div>
          <div class="sc-matrix-grid">{inner}</div>
        </div>
        """
    )
