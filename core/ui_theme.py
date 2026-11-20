"""Hệ thống Design Tokens & Components chuẩn UI/UX Pro Max cho Study Coach."""
from __future__ import annotations

import html

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


def inject_theme_css() -> str:
    """Trả về chuỗi <style> hoàn chỉnh với design tokens chuẩn UI/UX Pro Max."""
    return clean_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,600;6..72,700&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg-canvas: #f3f9f6;
            --bg-gradient: linear-gradient(180deg, #eef7f2 0%, #f7fbf9 58%, #f3f9f6 100%);
            --bg-surface: #FFFFFF;
            --bg-subtle: #EDF6F0;
            --bg-mint: #e6f4ec;
            --border: #cfe3d7;
            --border-strong: #b8d8c6;
            --ink: #10241a;
            --ink-secondary: #2f4638;
            --ink-muted: #5b7263;
            --brand: #166534;
            --brand-deep: #1b5e3c;
            --brand-hover: #14532d;
            --brand-soft: #e6f4ec;
            --brand-line: #cfe3d7;
            --card-shadow: 0 2px 10px rgba(22, 101, 52, 0.05);
            --card-shadow-hover: 0 6px 18px rgba(22, 101, 52, 0.10);
            --citation: #1D4ED8;
            --citation-soft: #EFF6FF;
            --mastered-fg: #15803d;
            --mastered-bg: #ECFDF5;
            --mastered-bd: #A7F3D0;
            --learning-fg: #B45309;
            --learning-bg: #FFFBEB;
            --learning-bd: #FDE68A;
            --misconception-fg: #B91C1C;
            --misconception-bg: #FEF2F2;
            --misconception-bd: #FECACA;
            --unseen-fg: #475569;
            --unseen-bg: #F1F5F9;
            --unseen-bd: #CBD5E1;
            --radius-md: 14px;
            --radius-lg: 18px;
        }
        html, body, [class*="css"] {
            font-family: "Be Vietnam Pro", system-ui, -apple-system, sans-serif;
            color: var(--ink);
            line-height: 1.55;
            -webkit-font-smoothing: antialiased;
        }
        .stApp {
            background: var(--bg-gradient);
            background-attachment: fixed;
            min-height: 100vh;
        }
        .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding: 2.8rem 2rem 1.2rem 2rem !important;
        }
        [data-stale="true"],
        div[data-stale="true"],
        .stElementContainer[data-stale="true"],
        div[data-testid="stAppViewBlockContainer"] [data-stale="true"] {
            opacity: 1 !important;
            filter: none !important;
            transition: none !important;
        }
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"],
        button[data-testid="stExpandSidebarButton"],
        .stAppDeployButton,
        #MainMenu, footer {
            display: none !important;
        }
        header[data-testid="stHeader"] {
            background: transparent;
            height: 2rem;
        }
        /* Touch target >= 44px & clear focus rings */
        .stButton > button, .stPopover > button {
            min-height: 2.6rem;
            border-radius: 12px !important;
            font-weight: 600 !important;
            border: 1px solid var(--border) !important;
            background: var(--bg-surface) !important;
            color: var(--ink-secondary) !important;
            box-shadow: none !important;
            transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease, box-shadow 160ms ease;
        }
        .stButton > button:hover, .stPopover > button:hover {
            background: var(--brand-soft) !important;
            border-color: var(--brand-line) !important;
            color: var(--brand-hover) !important;
        }
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background: var(--brand) !important;
            border-color: var(--brand) !important;
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"] *,
        .stButton > button[data-testid="stBaseButton-primary"] *,
        div[data-testid="stChatMessage"] button[data-testid="stBaseButton-primary"] *,
        div[data-testid="stChatMessage"] button[kind="primary"] * {
            color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: var(--brand-hover) !important;
            border-color: var(--brand-hover) !important;
            box-shadow: 0 4px 12px rgba(22, 101, 52, 0.18) !important;
        }
        .stButton > button:focus-visible, .stPopover > button:focus-visible {
            outline: 2px solid var(--brand) !important;
            outline-offset: 2px !important;
        }
        div[data-testid="stSegmentedControl"] {
            margin: 0.3rem 0 0.7rem 0;
        }
        div[data-testid="stSegmentedControl"] button {
            min-height: 2.7rem !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }
        /* Containers / expanders dùng ngôn ngữ card sạch */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--radius-md) !important;
            border-color: var(--border) !important;
            background: var(--bg-surface);
            box-shadow: var(--card-shadow);
        }
        div[data-testid="stExpander"] details {
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            background: var(--bg-surface);
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600;
            color: var(--brand-deep);
        }
        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--border);
            background: var(--brand-soft);
            color: var(--brand-deep);
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition: none !important;
                animation: none !important;
            }
        }
        /* Masthead — compact botanical header */
        .sc-masthead {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            padding: 0.7rem 1rem;
            border: 1px solid var(--brand-line);
            border-radius: var(--radius-lg);
            background: linear-gradient(135deg, #ffffff 0%, #f2faf5 100%);
            box-shadow: var(--card-shadow);
            margin-bottom: 0.7rem;
        }
        .sc-brand-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }
        .sc-logo-box {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: linear-gradient(160deg, #1b5e3c 0%, #166534 100%);
            color: #FFFFFF;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex: none;
            box-shadow: 0 2px 8px rgba(22, 101, 52, 0.22);
        }
        .sc-kicker {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--brand);
        }
        .sc-title {
            font-family: "Newsreader", Georgia, serif;
            font-size: 1.5rem;
            font-weight: 700;
            line-height: 1.1;
            color: var(--brand-deep);
            margin: 0.04rem 0 0.08rem 0;
        }
        .sc-subtitle {
            font-size: 0.82rem;
            color: var(--ink-muted);
            margin: 0;
            line-height: 1.4;
        }
        .sc-badges {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
        }
        .sc-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid var(--border);
            background: var(--bg-surface);
            color: var(--ink-secondary);
        }
        .sc-pill.live {
            background: var(--mastered-bg);
            border-color: var(--mastered-bd);
            color: var(--mastered-fg);
        }
        .sc-pill.demo {
            background: var(--citation-soft);
            border-color: #BFDBFE;
            color: var(--citation);
        }
        /* Bento Summary Grid — compact horizontal KPI strip */
        .sc-bento-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 0.65rem;
            margin: 0.25rem 0 0.7rem 0;
        }
        @media (max-width: 960px) {
            .sc-bento-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .block-container {
                padding: 0.9rem 0.9rem 1.4rem 0.9rem !important;
            }
        }
        .sc-bento-card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.62rem 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            box-shadow: var(--card-shadow);
            transition: box-shadow 160ms ease, border-color 160ms ease, transform 160ms ease;
        }
        .sc-bento-card:hover {
            box-shadow: var(--card-shadow-hover);
            border-color: var(--brand-line);
        }
        .sc-bento-card.accent {
            border-left: 3px solid var(--brand);
        }
        .sc-bento-card.alert {
            border-left: 3px solid var(--misconception-fg);
        }
        .sc-bento-label {
            display: inline-flex;
            align-items: center;
            gap: 0.34rem;
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--ink-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .sc-bento-value {
            font-family: "Newsreader", Georgia, serif;
            font-size: 1.34rem;
            font-weight: 700;
            color: var(--brand-deep);
            line-height: 1.15;
        }
        .sc-bento-meta {
            font-size: 0.76rem;
            color: var(--ink-secondary);
            line-height: 1.35;
        }
        /* Concept Heatmap Cards */
        .sc-concept-card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            margin-bottom: 0.55rem;
            box-shadow: var(--card-shadow);
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }
        .sc-concept-card:hover {
            border-color: var(--brand-line);
            box-shadow: var(--card-shadow-hover);
            transform: translateY(-1px);
        }
        .sc-concept-card.st-misconception {
            border-left: 4px solid var(--misconception-fg);
        }
        .sc-concept-card.st-mastered {
            border-left: 4px solid var(--mastered-fg);
        }
        .sc-concept-card.st-learning {
            border-left: 4px solid var(--learning-fg);
        }
        .sc-concept-card.st-unseen {
            border-left: 4px solid var(--unseen-fg);
        }
        .sc-card-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.6rem;
            margin-bottom: 0.45rem;
        }
        .sc-card-title {
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--ink);
            margin: 0;
        }
        .sc-status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.32rem;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            white-space: nowrap;
            border: 1px solid transparent;
        }
        .sc-status-badge.mastered {
            color: var(--mastered-fg);
            background: var(--mastered-bg);
            border-color: var(--mastered-bd);
        }
        .sc-status-badge.learning {
            color: var(--learning-fg);
            background: var(--learning-bg);
            border-color: var(--learning-bd);
        }
        .sc-status-badge.misconception {
            color: var(--misconception-fg);
            background: var(--misconception-bg);
            border-color: var(--misconception-bd);
        }
        .sc-status-badge.unseen {
            color: var(--unseen-fg);
            background: var(--unseen-bg);
            border-color: var(--unseen-bd);
        }
        .sc-progress-track {
            width: 100%;
            height: 7px;
            border-radius: 999px;
            background: var(--bg-subtle);
            overflow: hidden;
            margin: 0.45rem 0;
        }
        .sc-progress-fill {
            height: 100%;
            border-radius: 999px;
        }
        .sc-meta-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            margin-top: 0.45rem;
        }
        .sc-mono-chip {
            font-family: "JetBrains Mono", monospace;
            font-size: 0.76rem;
            font-weight: 500;
            padding: 0.16rem 0.5rem;
            border-radius: 6px;
            background: var(--citation-soft);
            color: var(--citation);
            border: 1px solid #DBEAFE;
        }
        .sc-prereq-chip {
            font-size: 0.78rem;
            padding: 0.16rem 0.52rem;
            border-radius: 6px;
            background: var(--bg-subtle);
            color: var(--ink-secondary);
        }
        /* Coach & Citation & Diagnostic Boxes */
        .sc-route-banner {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.55rem;
            padding: 0.55rem 0.85rem;
            border-radius: 12px;
            background: var(--brand-soft);
            border: 1px solid var(--brand-line);
            margin-bottom: 0.5rem;
            font-size: 0.84rem;
        }
        .sc-route-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
        }
        .sc-route-pill.adaptive { background: var(--brand-soft); color: var(--brand); }
        .sc-route-pill.remediation { background: var(--misconception-bg); color: var(--misconception-fg); }
        .sc-route-pill.prereq { background: var(--learning-bg); color: var(--learning-fg); }
        .sc-route-pill.refuse { background: var(--unseen-bg); color: var(--unseen-fg); }
        .sc-route-pill.guide { background: var(--citation-soft); color: var(--citation); }
        .sc-route-pill.eval { background: var(--mastered-bg); color: var(--mastered-fg); }
        .sc-misc-box {
            display: flex;
            align-items: flex-start;
            gap: 0.65rem;
            padding: 0.78rem 0.95rem;
            border-radius: 10px;
            background: var(--misconception-bg);
            border: 1px solid var(--misconception-bd);
            color: var(--misconception-fg);
            margin: 0.45rem 0;
            font-size: 0.88rem;
        }
        .sc-misc-box.resolved {
            background: var(--mastered-bg);
            border-color: var(--mastered-bd);
            color: var(--mastered-fg);
        }
        .sc-section-eyebrow {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin: 0.2rem 0 0.45rem 0;
        }
        .sc-score-hero {
            font-family: "Newsreader", Georgia, serif;
            font-size: 3.1rem;
            font-weight: 700;
            line-height: 1;
            color: var(--ink);
            margin: 0.2rem 0 0.5rem 0;
        }
        .sc-score-hero small {
            font-size: 1.35rem;
            color: var(--ink-muted);
        }
        .sc-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.55rem 0;
        }
        .sc-chips span {
            width: 2rem;
            height: 2rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            border: 1px solid var(--border-strong);
            font-size: 0.84rem;
            font-weight: 600;
            color: var(--ink-secondary);
            background: var(--bg-surface);
        }
        .sc-chips span.on {
            background: var(--brand);
            border-color: var(--brand);
            color: #FFFFFF;
        }
        .sc-chips span.bad {
            background: var(--misconception-fg);
            border-color: var(--misconception-fg);
            color: #FFFFFF;
        }
        .sc-matrix-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.85rem;
        }
        @media (max-width: 760px) {
            .sc-matrix-grid {
                grid-template-columns: 1fr;
            }
        }
        .sc-diff-card {
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 0.75rem;
        }
        .sc-diff-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            padding: 0.55rem 0.9rem;
            background: var(--bg-subtle);
            border-bottom: 1px solid var(--border);
            font-weight: 700;
            font-size: 0.88rem;
        }
        .sc-diff-del {
            background: #FEF2F2;
            color: #991B1B;
            padding: 0.55rem 0.9rem;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.83rem;
            border-bottom: 1px solid #FECACA;
        }
        .sc-diff-add {
            background: #ECFDF5;
            color: #065F46;
            padding: 0.55rem 0.9rem;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.83rem;
            border-bottom: 1px solid #A7F3D0;
        }
        .sc-diff-note {
            padding: 0.55rem 0.9rem;
            font-size: 0.85rem;
            color: var(--ink-secondary);
        }
        /* ===== Full-width Chatbot Workspace ===== */
        .sc-chat-ws-header {
            display: flex;
            justify-content: flex-start;
            align-items: center;
            gap: 0.65rem;
            background: linear-gradient(135deg, #ffffff 0%, #f0f9f4 100%);
            border: 1px solid #d5e6dc;
            border-radius: 14px;
            padding: 10px 16px;
            margin-bottom: 8px;
            box-shadow: var(--card-shadow);
        }
        .sc-chat-ws-id {
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
            min-width: 0;
        }
        .sc-chat-avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            flex: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(160deg, #1b5e3c 0%, #166534 100%);
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(22, 101, 52, 0.22);
        }
        .sc-chat-name {
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--brand-deep);
            line-height: 1.2;
        }
        .sc-chat-status {
            display: inline-flex;
            align-items: center;
            gap: 0.34rem;
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--brand);
        }
        .sc-chat-status .sc-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #22c55e;
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.16);
        }
        .sc-context-pill {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.45rem;
            background: #e9f5ee;
            border: 1px solid #cce3d6;
            border-radius: 12px;
            padding: 8px 14px;
            margin-bottom: 10px;
            font-size: 0.82rem;
            color: #165231;
        }
        .sc-context-pill b {
            color: #14532d;
        }
        .sc-context-pill .sc-context-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.14rem 0.55rem;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid #cce3d6;
            font-weight: 600;
            color: #165231;
        }
        /* ===== st.chat_message bubbles ===== */
        div[data-testid="stChatMessage"] {
            border-radius: 16px !important;
            padding: 0.85rem 1rem !important;
            margin-bottom: 0.65rem;
            border: 1px solid #dcebe2;
            background: var(--bg-surface);
            box-shadow: var(--card-shadow);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]),
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: #e7f6ec;
            border-color: #cfe8da;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]),
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            background: var(--bg-surface);
            border-color: #dcebe2;
        }
        div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
        div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
            width: 34px !important;
            height: 34px !important;
            border-radius: 50% !important;
            background: linear-gradient(160deg, #1b5e3c 0%, #166534 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 2px 6px rgba(22, 101, 52, 0.20);
        }
        div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
            background: linear-gradient(160deg, #2f7d55 0%, #1b5e3c 100%) !important;
        }
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li {
            color: var(--ink);
            font-size: 0.93rem;
            line-height: 1.62;
        }
        div[data-testid="stChatMessage"] a {
            color: var(--brand);
            text-decoration: underline;
            text-underline-offset: 2px;
        }
        /* Chat input */
        div[data-testid="stChatInput"] {
            border-radius: 14px !important;
            border: 1px solid var(--border) !important;
            background: var(--bg-surface) !important;
            box-shadow: var(--card-shadow);
        }
        div[data-testid="stChatInput"]:focus-within {
            border-color: var(--brand) !important;
            box-shadow: 0 0 0 3px rgba(22, 101, 52, 0.10) !important;
        }
        /* Markdown / code block trong chat */
        div[data-testid="stChatMessage"] div[data-testid="stCodeBlock"] {
            border: 1px solid #dcebe2 !important;
            border-radius: 12px !important;
            overflow: hidden;
            background: #f6fbf8 !important;
        }
        div[data-testid="stChatMessage"] pre,
        div[data-testid="stChatMessage"] div[data-testid="stCodeBlock"] pre {
            background: #f6fbf8 !important;
            border-radius: 12px !important;
            padding: 0.8rem 0.95rem !important;
            margin: 0 !important;
            overflow-x: auto;
        }
        div[data-testid="stChatMessage"] pre code,
        div[data-testid="stChatMessage"] code {
            font-family: "JetBrains Mono", monospace !important;
            font-size: 0.82rem !important;
            line-height: 1.6 !important;
            color: var(--ink) !important;
        }
        div[data-testid="stChatMessage"] :not(pre) > code {
            background: var(--brand-soft) !important;
            color: #14532d !important;
            border: 1px solid var(--brand-line);
            border-radius: 6px;
            padding: 0.08rem 0.38rem;
        }
        div[data-testid="stChatMessage"] table {
            border-collapse: collapse;
            width: 100%;
            margin: 0.5rem 0;
            font-size: 0.86rem;
        }
        div[data-testid="stChatMessage"] th {
            background: var(--brand-soft);
            color: #14532d;
            font-weight: 700;
            text-align: left;
        }
        div[data-testid="stChatMessage"] th,
        div[data-testid="stChatMessage"] td {
            border: 1px solid var(--border);
            padding: 0.4rem 0.6rem;
        }
        div[data-testid="stChatMessage"] blockquote {
            border-left: 3px solid var(--brand-line);
            background: #f6fbf8;
            margin: 0.5rem 0;
            padding: 0.45rem 0.85rem;
            border-radius: 0 10px 10px 0;
            color: var(--ink-secondary);
        }
        /* ===== Responsive: laptop & tablet ===== */
        @media (max-width: 1280px) {
            .block-container {
                padding: 0.85rem 1.1rem 1.25rem 1.1rem !important;
                max-width: 100% !important;
            }
            .sc-bento-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 0.55rem;
            }
            .sc-bento-card {
                padding: 0.55rem 0.7rem;
            }
            .sc-masthead {
                gap: 0.55rem;
                padding: 0.6rem 0.8rem;
            }
            .sc-title {
                font-size: 1.34rem;
            }
            .sc-chat-ws-header {
                padding: 10px 14px;
            }
            div[data-testid="stChatMessage"] {
                padding: 0.7rem 0.8rem !important;
            }
        }
        @media (max-width: 900px) {
            .sc-bento-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .sc-chat-ws-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.5rem;
            }
            .sc-matrix-grid {
                grid-template-columns: 1fr;
            }
        }
        /* ===== Layout mẫu AI Learning Workspace ===== */
        .sc-brand-lockup { display:flex; align-items:center; gap:0.7rem; }
        .sc-brand-lockup .sc-logo-box { width:42px; height:42px; background:#166534; }
        .sc-brand-lockup .sc-kicker { font-size:0.64rem; }
        .sc-brand-lockup .sc-title { font-size:1.35rem; margin:0; }
        .sc-brand-lockup .sc-subtitle { font-size:0.78rem; }
        .sc-bento-grid { grid-template-columns: repeat(5, minmax(0, 1fr)); gap:0.7rem; }
        .sc-bento-card, .sc-bento-card.accent, .sc-bento-card.alert {
            flex-direction:row; align-items:center; gap:0.7rem;
            background:#ffffff; border:1px solid #dcebe2; border-left:1px solid #dcebe2;
            border-radius:14px; padding:0.7rem 0.85rem; box-shadow:none;
        }
        .sc-kpi-icon {
            width:38px; height:38px; flex:none; border-radius:50%;
            background:#e6f4ec; color:#166534;
            display:inline-flex; align-items:center; justify-content:center;
        }
        .sc-kpi-icon.alert { background:#fdecec; color:#b91c1c; }
        .sc-kpi-body { display:flex; flex-direction:column; min-width:0; }
        .sc-bento-value { font-size:1.3rem; }
        div[data-testid="stSegmentedControl"] [role="radiogroup"],
        div[data-testid="stSegmentedControl"] > div {
            width:100%; background:#ffffff; border:1px solid #dcebe2; border-radius:999px; padding:4px; gap:4px;
        }
        div[data-testid="stSegmentedControl"] button {
            flex:1 1 0; border-radius:999px !important; border:none !important;
            background:transparent !important; color:#166534 !important;
        }
        button[data-testid="stBaseButton-segmented_controlActive"],
        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
        div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
            background:#166534 !important; color:#ffffff !important; border-color:#166534 !important;
        }
        button[data-testid="stBaseButton-segmented_controlActive"] *,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] * {
            color:#ffffff !important;
        }
        div[data-testid="stBottom"],
        div[data-testid="stBottom"] > div,
        div[data-testid="stBottomBlockContainer"] {
            background: #f3f9f6 !important;
        }
        div[data-testid="stBottomBlockContainer"] {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 0.5rem !important;
            padding-bottom: 0.9rem !important;
        }
        div[data-testid="stChatInput"] {
            max-width: 100% !important;
            width: 100% !important;
            background: #ffffff !important;
            border: 1px solid #cfe3d7 !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 10px rgba(22, 101, 52, 0.06) !important;
        }
        div[data-testid="stSpinner"] > div {
            display: flex !important;
            align-items: center !important;
            gap: 0.7rem !important;
            background: linear-gradient(135deg, #ffffff 0%, #f0f9f4 100%) !important;
            border: 1px solid #cfe3d7 !important;
            border-radius: 14px !important;
            padding: 12px 16px !important;
            margin: 8px 0 !important;
            box-shadow: 0 2px 10px rgba(22, 101, 52, 0.06) !important;
            color: #165231 !important;
            font-weight: 500 !important;
        }
        div[data-testid="stSpinner"] svg,
        div[data-testid="stStatusWidget"] svg {
            color: #166534 !important;
        }
        div[data-testid="stStatusWidget"] {
            background: linear-gradient(135deg, #ffffff 0%, #f2faf5 100%) !important;
            border: 1px solid #cfe3d7 !important;
            border-radius: 14px !important;
            box-shadow: 0 3px 12px rgba(22, 101, 52, 0.07) !important;
        }
        .sc-login-card {
            width:100%; box-sizing:border-box; margin:4vh auto 1rem auto; padding:1.6rem 1.8rem;
            background:#ffffff; border:1.5px solid #166534; border-radius:18px;
            box-shadow:0 8px 28px rgba(22,101,52,0.10); text-align:center;
        }
        .sc-login-card .sc-brand-lockup { justify-content:center; text-align:left; }
        .sc-login-card p { color:var(--ink-secondary); font-size:0.88rem; margin:0.6rem 0 0 0; }
        @media (max-width: 960px) { .sc-bento-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
        /* ===== 5-Tier Visual Hierarchy ===== */
        /* Tier 1: Top navigation segmented bar — sage track, dark active pill */
        div[data-testid="stSegmentedControl"],
        div[data-testid="stButtonGroup"] {
            background: #dceee3 !important;
            border: 1.5px solid #b7d7c4 !important;
            border-radius: 14px !important;
            padding: 5px 6px !important;
            box-shadow: inset 0 1px 2px rgba(15, 46, 29, 0.06) !important;
        }
        div[data-testid="stSegmentedControl"] [role="radiogroup"],
        div[data-testid="stSegmentedControl"] > div,
        div[data-testid="stButtonGroup"] > div {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            gap: 5px !important;
            width: 100% !important;
        }
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stButtonGroup"] button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #23533a !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            min-height: 36px !important;
        }
        button[kind="segmented_controlActive"],
        button[data-testid="stBaseButton-segmented_controlActive"],
        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
        div[data-testid="stButtonGroup"] button[aria-checked="true"],
        div[data-testid="stButtonGroup"] button[aria-pressed="true"] {
            background: linear-gradient(135deg, #14532d 0%, #166534 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 3px 10px rgba(20, 83, 45, 0.28) !important;
        }
        button[kind="segmented_controlActive"] p,
        button[kind="segmented_controlActive"] span,
        button[data-testid="stBaseButton-segmented_controlActive"] p,
        button[data-testid="stBaseButton-segmented_controlActive"] span,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
        div[data-testid="stSegmentedControl"] button[aria-checked="true"] span,
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] p,
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"] span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        /* Tier 2: Secondary vs primary action buttons */
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: #f0f8f3 !important;
            color: #166534 !important;
            border: 1px solid #b6d9c4 !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            min-height: 34px !important;
            padding: 0.28rem 0.85rem !important;
            box-shadow: 0 1px 2px rgba(22, 101, 52, 0.05) !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background: #dcf0e4 !important;
            border-color: #16a34a !important;
            color: #0f3d22 !important;
            transform: translateY(-1px);
        }
        div[data-testid="stButton"] > button[kind="primary"] {
            border-radius: 999px !important;
            background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
            color: #ffffff !important;
            min-height: 36px !important;
            font-weight: 700 !important;
            box-shadow: 0 3px 10px rgba(22, 163, 74, 0.25) !important;
        }
        /* Tier 3: Expanders — soft tinted collapsible drawers */
        div[data-testid="stExpander"],
        div[data-testid="stExpander"] details {
            background: #ebf6ef !important;
            border: 1px solid #c6e2d1 !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }
        div[data-testid="stExpander"] summary {
            padding: 0.45rem 0.85rem !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: #1e4d33 !important;
        }
        /* Tier 4: Chat messages — distinct user vs assistant surfaces */
        div[data-testid="stChatMessage"] {
            border-radius: 14px !important;
            padding: 0.85rem 1.1rem !important;
            margin-bottom: 0.65rem !important;
            background: #ffffff !important;
            border: 1px solid #dcebe2 !important;
            box-shadow: 0 2px 6px rgba(15, 46, 29, 0.03) !important;
        }
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            background: #e3f5ea !important;
            border: 1px solid #bce2ca !important;
        }
        div[data-testid="stChatMessage"] div[data-testid="stExpander"] {
            background: #f5faf7 !important;
            border: 1px dashed #c5dfd0 !important;
            border-radius: 8px !important;
            margin-top: 0.35rem !important;
        }
        /* Tier 5: Unified chat card classes */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff !important;
            border: 1.5px solid #cde2d5 !important;
            border-radius: 18px !important;
            box-shadow: 0 6px 22px rgba(15, 46, 29, 0.06) !important;
        }
        .sc-chat-card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.15rem 0.2rem;
        }
        .sc-chat-avatar-badge {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: linear-gradient(135deg, #14532d 0%, #166534 100%);
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            box-shadow: 0 3px 8px rgba(20, 83, 45, 0.22);
            flex-shrink: 0;
        }
        .sc-quick-label {
            font-size: 0.75rem;
            font-weight: 700;
            color: #3b634c;
            letter-spacing: 0.03em;
            margin: 0.25rem 0 0.3rem 0.15rem;
        }
        .sc-trace-box {
            font-size: 0.82rem;
            color: var(--ink-secondary);
            line-height: 1.5;
        }
        </style>
        """
    )


def render_masthead_html(doc_title: str, is_live: bool, model_name: str) -> str:
    mode_cls = "live" if is_live else "demo"
    mode_icon = "shield-check" if is_live else "cpu"
    mode_txt = f"AI Portal Live ({model_name})" if is_live else "Chế độ Demo Offline (Bám nguồn thật)"
    return clean_html(
        f"""
        <div class="sc-masthead">
          <div class="sc-brand-row">
            <div class="sc-logo-box">{icon_svg("brain", 24)}</div>
            <div>
              <div class="sc-kicker">AIA19_Group1 · Adaptive Personal AI Tutor</div>
              <div class="sc-title">Study Coach</div>
              <p class="sc-subtitle">Gia sư hiểu năng lực từng người học trên tài liệu riêng · 5 lớp công nghệ LangChain &amp; LangGraph</p>
            </div>
          </div>
          <div class="sc-badges">
            <span class="sc-pill">{icon_svg("book", 15)} {esc(doc_title)}</span>
            <span class="sc-pill {mode_cls}">{icon_svg(mode_icon, 15)} {esc(mode_txt)}</span>
          </div>
        </div>
        """
    )


def render_brand_lockup_html(subtitle: str = "Gia sư hiểu năng lực từng người học trên tài liệu riêng") -> str:
    return clean_html(
        f"""
        <div class="sc-brand-lockup">
          <div class="sc-logo-box">{icon_svg("brain", 24)}</div>
          <div>
            <div class="sc-kicker">AI19_GROUP1 · ADAPTIVE PERSONAL AI TUTOR</div>
            <div class="sc-title">Study Coach</div>
            <p class="sc-subtitle">{esc(subtitle)}</p>
          </div>
        </div>
        """
    )


def render_bento_strip_html(dna: LearnerDNA, coverage: dict, telemetry: dict) -> str:
    misc = len(dna.active_misconceptions)
    calls = telemetry.get("total_calls", 0)
    attempts = sum(c.attempts for c in dna.concepts)

    def card(icon: str, label: str, value: str, meta: str, alert: bool = False) -> str:
        cls = " alert" if alert else ""
        return (
            f'<div class="sc-bento-card"><div class="sc-kpi-icon{cls}">{icon_svg(icon, 18)}</div>'
            f'<div class="sc-kpi-body"><div class="sc-bento-label">{label}</div>'
            f'<div class="sc-bento-value">{value}</div><div class="sc-bento-meta">{meta}</div></div></div>'
        )

    return clean_html(
        '<div class="sc-bento-grid">'
        + card("layers", "Tiến độ học tập", esc(dna.overall_level),
               f"đã hoàn thành {coverage.get('coverage_pct', 0):.0f}% khóa học")
        + card("check-circle", "Độ chính xác", f"{dna.mastery_avg:.0f}%", "điểm nắm vững trung bình")
        + card("alert-triangle", "Lỗi thường gặp", str(misc), "cần ôn tập thêm", alert=misc > 0)
        + card("cpu", "Phiên học", f"{calls} lượt", "AI Portal Live")
        + card("book", "Điểm danh hiệu", f"{attempts} câu", "đã luyện tập")
        + "</div>"
    )


def _status_meta(status: str) -> tuple[str, str, str]:
    return {
        "mastered": ("Đã làm chủ", "check-circle", "#15803d"),
        "learning": ("Cần ôn lại", "clock", "#B45309"),
        "misconception": ("Đang hiểu lệch", "alert-triangle", "#B91C1C"),
    }.get(status, ("Chưa học", "circle-dashed", "#475569"))


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
          <p style="margin:0 0 0.35rem 0;font-size:0.88rem;color:var(--ink-secondary);">{esc(node.summary)}</p>
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
            <span style="color:var(--ink-secondary);">Trình độ áp dụng: <b>{esc(turn.learner_level_used)}</b></span>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--ink-muted);">
            {turn.latency_ms} ms · {len(turn.tool_traces)} tools
          </div>
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
            <div style="font-weight:700;">{esc(topic)} · <span style="font-weight:600;">{esc(state_txt)}</span></div>
            <div style="color:var(--ink);margin-top:2px;">{esc(description)}</div>
          </div>
        </div>
        """
    )


def render_next_step_callout_html(dna: LearnerDNA, top_plan: dict) -> str:
    """Hiển thị khuyến nghị bước học tiếp theo ngay trên đầu Bản đồ Năng lực."""
    return clean_html(
        f"""
        <div style="background:var(--brand-soft);border:1px solid var(--brand-line);border-left:4px solid var(--brand);border-radius:14px;padding:0.8rem 1rem;margin:0.4rem 0 0.7rem 0;display:flex;align-items:flex-start;gap:0.7rem;box-shadow:var(--card-shadow);">
          <div style="color:var(--brand);padding-top:2px;">{icon_svg("target", 18)}</div>
          <div>
            <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--brand);">
              Bước tiếp theo AI đề xuất cho {esc(dna.name)}
            </div>
            <div style="font-size:0.95rem;font-weight:700;color:var(--ink);margin-top:2px;">
              [{esc(top_plan['priority_label'])}] {esc(top_plan['concept'])} <span class="sc-mono-chip">{esc(top_plan.get('page_ref') or '[Tài liệu]')}</span>
            </div>
            <div style="font-size:0.86rem;color:var(--ink-secondary);margin-top:2px;">{esc(top_plan['reason'])}</div>
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
        label, ic, color = _status_meta(c.status)
        pct = max(2.0 if c.attempts > 0 else 0.0, min(100.0, c.mastery_score))
        rows_html.append(
            f'<div style="display:grid;grid-template-columns:minmax(160px,2.1fr) minmax(110px,3.5fr) 148px;gap:0.75rem;align-items:center;padding:0.42rem 0;border-bottom:1px solid var(--border);">'
            f'<div style="font-size:0.86rem;font-weight:600;color:var(--ink);">{esc(c.concept)}</div>'
            f'<div style="background:var(--bg-subtle);height:10px;border-radius:999px;overflow:hidden;"><div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:999px;"></div></div>'
            f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:0.35rem;white-space:nowrap;font-size:0.8rem;font-weight:700;color:{color};font-family:\'JetBrains Mono\',monospace;">{icon_svg(ic, 13)} {c.mastery_score:.0f}% · {esc(c.bloom_level)}</div>'
            f'</div>'
        )
    inner = "".join(rows_html)
    return clean_html(
        f"""
        <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:14px;padding:0.9rem 1.05rem;margin-bottom:0.75rem;box-shadow:var(--card-shadow);">
          <div style="font-size:0.9rem;font-weight:700;color:var(--brand-deep);margin-bottom:0.45rem;">Biểu đồ năng lực thành thạo theo từng khái niệm &amp; cấp độ Bloom</div>
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
        <div style="background:var(--bg-surface);border:1px solid var(--border-strong);border-left:5px solid var(--misconception-fg);border-radius:14px;padding:0.9rem 1.1rem;margin:0.4rem 0 0.7rem 0;box-shadow:var(--card-shadow);">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.4rem;">
            <span style="display:inline-flex;align-items:center;gap:0.45rem;font-weight:700;font-size:0.92rem;color:var(--misconception-fg);">
              {icon_svg("flame", 17)} TICKET SỰ CỐ PRODUCTION · {esc(scenario.ticket_id)}
            </span>
            <span style="display:inline-flex;align-items:center;gap:0.4rem;">
              <span class="sc-status-badge {sev_cls}">{esc(scenario.severity)}</span>
              <span class="sc-prereq-chip">Liên quan: <b>{esc(scenario.topic)}</b></span>
            </span>
          </div>
          <div style="font-size:1.08rem;font-weight:700;color:var(--ink);margin-bottom:0.3rem;">{esc(scenario.title)}</div>
          <div style="font-size:0.9rem;color:var(--ink-secondary);">{esc(scenario.symptom)}</div>
        </div>
        """
    )


def render_confidence_matrix_html(stats: dict) -> str:
    """Ma trận 2x2 Radar Ảo tưởng Hiểu bài (Confidence Calibration / Dunning-Kruger)."""
    cells = [
        ("Vững thực chất", "Trả lời Đúng + Tự tin cao/vừa", stats.get("solid_correct", 0), "check-circle", "#15803d", "#ECFDF5", "#A7F3D0"),
        ("Đúng do đoán mò", "Kiến thức giả · Chọn 'Đoán mò' — cần ôn lại", stats.get("lucky_guess", 0), "clock", "#B45309", "#FFFBEB", "#FDE68A"),
        ("Điểm mù tự tin sai", "Nguy hiểm nhất · Tự tin chắc chắn nhưng Sai", stats.get("blind_spot", 0), "shield-alert", "#B91C1C", "#FEF2F2", "#FECACA"),
        ("Biết mình chưa biết", "Trả lời Sai/Bỏ trống + Tự nhận chưa chắc", stats.get("aware_gap", 0), "search-check", "#475569", "#F1F5F9", "#CBD5E1"),
    ]
    cards = []
    for title, desc, val, ic, fg, bg, bd in cells:
        cards.append(
            f'<div style="background:{bg};border:1px solid {bd};border-radius:14px;padding:0.8rem 0.95rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:0.6rem;color:{fg};font-weight:700;font-size:0.86rem;">'
            f'<span style="display:inline-flex;align-items:center;gap:0.4rem;">{icon_svg(ic, 16)} {esc(title)}</span>'
            f'<span style="font-family:\'Newsreader\',serif;font-size:1.45rem;white-space:nowrap;">{val}</span>'
            f'</div>'
            f'<div style="font-size:0.8rem;color:var(--ink-secondary);margin-top:4px;">{esc(desc)}</div>'
            f'</div>'
        )
    inner = "".join(cards)
    return clean_html(
        f"""
        <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:14px;padding:0.9rem 1.05rem;margin-bottom:0.75rem;box-shadow:var(--card-shadow);">
          <div style="font-size:0.92rem;font-weight:700;color:var(--brand-deep);margin-bottom:0.55rem;">Radar Ảo tưởng Hiểu bài (Đo độ lệch giữa Tự tin và Thực lực)</div>
          <div class="sc-matrix-grid">{inner}</div>
        </div>
        """
    )
