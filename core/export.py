"""Xuất dữ liệu học tập ra file tải về: sổ tay lỗi tư duy, ghi chú đã hiệu đính (Cognitive Diff) và flashcard.

Chỉ sinh Markdown / CSV bằng thư viện chuẩn (không thêm phụ thuộc nặng để xuất PDF)."""
from __future__ import annotations

import csv
import re
from datetime import datetime
from io import StringIO

from core.contracts import CognitiveDiffReport, DocumentConceptMap, LearnerDNA, MisconceptionRecord


def _today() -> str:
    return datetime.now().strftime("%d/%m/%Y")


def safe_filename(title: str, suffix: str) -> str:
    """Tên file an toàn cho nút tải về (bỏ ký tự lạ, giữ chữ có dấu)."""
    stem = re.sub(r"[^\w\-]+", "-", (title or "tai-lieu").strip(), flags=re.UNICODE).strip("-")[:50] or "tai-lieu"
    return f"{stem}-{suffix}"


def _concept_lookup(concept_map: DocumentConceptMap | None) -> dict[str, tuple[str, str]]:
    if concept_map is None:
        return {}
    return {c.name.lower(): (c.summary, c.page_ref) for c in concept_map.concepts}


def _clean_desc(text: str) -> str:
    return re.sub(r"^\[Điểm mù tự tin cao\]\s*", "", (text or "").strip())


def misconceptions_markdown(
    dna: LearnerDNA,
    doc_title: str,
    concept_map: DocumentConceptMap | None = None,
) -> str:
    """Sổ tay lỗi tư duy dạng Markdown: lỗi đang mở, lỗi đã khắc phục, kèm kiến thức đúng theo tài liệu."""
    lookup = _concept_lookup(concept_map)
    lines = [
        "# Sổ tay lỗi tư duy",
        "",
        f"- **Người học:** {dna.name} ({dna.overall_level})",
        f"- **Tài liệu:** {doc_title}",
        f"- **Ngày xuất:** {_today()}",
        f"- **Đang mở:** {len(dna.active_misconceptions)} · **Đã khắc phục:** {len(dna.resolved_misconceptions)}",
        "",
    ]

    def _section(title: str, records: list[MisconceptionRecord], empty: str) -> None:
        lines.extend([f"## {title}", ""])
        if not records:
            lines.extend([empty, ""])
            return
        for i, m in enumerate(records, start=1):
            summary, page_ref = lookup.get(m.topic.lower(), ("", ""))
            lines.append(f"### {i}. {m.topic}")
            lines.append("")
            lines.append(f"- **Hiểu lệch:** {_clean_desc(m.description)}")
            if "[Điểm mù tự tin cao]" in (m.description or ""):
                lines.append("- **Mức độ:** Điểm mù (trả lời sai nhưng rất tự tin)")
            lines.append(f"- **Số lần lặp lại:** {m.occurrences}")
            if m.last_seen:
                lines.append(f"- **Lần gần nhất:** {m.last_seen[:10]}")
            if summary:
                lines.append(f"- **Theo tài liệu {page_ref}:** {summary}".replace("  ", " "))
            lines.append("")

    _section("Lỗi đang mở", dna.active_misconceptions, "_Không có lỗi tư duy nào đang mở._")
    _section("Lỗi đã khắc phục", dna.resolved_misconceptions, "_Chưa có lỗi nào được đánh dấu đã khắc phục._")
    return "\n".join(lines).rstrip() + "\n"


def build_flashcards(
    dna: LearnerDNA,
    concept_map: DocumentConceptMap | None = None,
    include_resolved: bool = False,
) -> list[dict]:
    """Biến mỗi lỗi tư duy thành một thẻ: mặt trước nêu nhận định sai để tự sửa, mặt sau là kiến thức đúng."""
    lookup = _concept_lookup(concept_map)
    records = list(dna.active_misconceptions) + (list(dna.resolved_misconceptions) if include_resolved else [])
    cards: list[dict] = []
    for m in records:
        summary, page_ref = lookup.get(m.topic.lower(), ("", ""))
        desc = _clean_desc(m.description)
        front = f"[{m.topic}] Nhận định sau sai ở đâu? «{desc}»"
        back = summary or f"Xem lại phần '{m.topic}' trong tài liệu để tự sửa nhận định này."
        if page_ref:
            back = f"{back} (Nguồn: {page_ref})"
        cards.append({"front": front, "back": back, "topic": m.topic})
    return cards


def flashcards_csv(cards: list[dict]) -> str:
    """CSV hai cột Mặt trước, Mặt sau (thêm cột thẻ tag): nhập thẳng được vào Anki / Quizlet."""
    buf = StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for c in cards:
        writer.writerow([c["front"], c["back"], re.sub(r"\s+", "_", c.get("topic", "")).strip("_")])
    return buf.getvalue()


def flashcards_markdown(cards: list[dict], doc_title: str) -> str:
    lines = [f"# Flashcard lỗi tư duy — {doc_title}", "", f"_Xuất ngày {_today()} · {len(cards)} thẻ_", ""]
    for i, c in enumerate(cards, start=1):
        lines.extend([f"## Thẻ {i}", "", f"**Hỏi:** {c['front']}", "", f"**Đáp:** {c['back']}", ""])
    return "\n".join(lines).rstrip() + "\n"


def diff_note_markdown(report: CognitiveDiffReport, doc_title: str) -> str:
    """Ghi chú đã hiệu đính sau Cognitive Diff kèm danh sách chỗ đã sửa, dạng Markdown."""
    lines = [
        "# Ghi chú đã hiệu đính (Cognitive Diff)",
        "",
        f"- **Tài liệu gốc:** {doc_title}",
        f"- **Ngày xuất:** {_today()}",
        f"- **Độ khớp trước khi sửa:** {report.alignment_score}/100",
        "",
        "## Nội dung ghi chú đã sửa",
        "",
        (report.patched_note or "_Chưa có bản ghi chú hiệu đính._").strip(),
        "",
    ]
    if report.contradictions:
        lines.extend(["## Các chỗ đã sửa so với ghi chú ban đầu", ""])
        for c in report.contradictions:
            lines.append(f"- **{c.topic}** {c.page_ref}")
            lines.append(f"  - Ghi chú cũ: {c.learner_claim}")
            lines.append(f"  - Theo tài liệu: {c.ground_truth}")
        lines.append("")
    if report.blind_spots:
        lines.extend(["## Phần còn thiếu nên bổ sung", ""])
        for b in report.blind_spots:
            lines.append(f"- **{b.topic}** {b.page_ref} — {b.importance_reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
