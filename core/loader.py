"""Đọc văn bản từ file upload (PDF hoặc TXT) và giữ cấu trúc [Trang n] / [Mục n] cho RAG."""
from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

import config

_PAGE_SPLIT_RE = re.compile(r"(\[Trang\s+\d+\])")
_NUMBERED_HEADING_RE = re.compile(r"(?m)^(\d+\.\s+[^\n]{2,80})$")


def load_text(filename: str, data: bytes) -> str:
    """Trả về văn bản. PDF có đánh dấu [Trang n] để sau này trích nguồn."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(data))
            pages = list(reader.pages)
        except Exception as e:
            raise ValueError(
                f"Không đọc được file {filename}: file PDF bị hỏng hoặc có mật khẩu."
            ) from e
        parts = []
        for i, page in enumerate(pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"[Trang {i}]\n{text}")
        if not parts:
            raise ValueError(
                f"File {filename} không có lớp chữ (có thể là PDF scan hoặc ảnh chụp). "
                "Hãy dùng PDF có chữ chọn được, hoặc dán văn bản trực tiếp."
            )
        return "\n\n".join(parts)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="replace")
    raise ValueError("Chỉ hỗ trợ file .pdf hoặc .txt")


def _extract_sections_for_file(filename: str, text: str) -> list[dict]:
    """Tách văn bản của một file thành các đoạn kèm nhãn trang/mục để trích nguồn chính xác."""
    sections: list[dict] = []
    if "[Trang " in text:
        tokens = _PAGE_SPLIT_RE.split(text)
        current_page = "[Trang 1]"
        for tok in tokens:
            tok_s = tok.strip()
            if not tok_s:
                continue
            if tok_s.startswith("[Trang ") and tok_s.endswith("]"):
                current_page = tok_s
                continue
            first_line = tok_s.splitlines()[0].strip()[:60]
            sections.append(
                {
                    "source": filename,
                    "page": current_page,
                    "topic_hint": first_line,
                    "text": tok_s,
                }
            )
        if sections:
            return sections

    # File TXT hoặc văn bản dán: tách theo các mục đánh số (1. ..., 2. ...).
    # Chỉ gắn hậu tố " · API" cho đúng file mẫu OpenAI API (không đoán theo tên file bất kỳ chứa "api").
    suffix = " · API" if filename.lower() == "openai-api-co-ban.txt" else ""
    parts = _NUMBERED_HEADING_RE.split(text.strip())
    if len(parts) >= 3:
        intro = parts[0].strip()
        idx = 1
        sec_num = 1
        while idx < len(parts):
            heading = parts[idx].strip()
            body = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
            m_num = re.match(r"^(\d+)\.", heading)
            num_label = m_num.group(1) if m_num else str(sec_num)
            combined = f"{heading}\n{body}".strip()
            if intro and sec_num == 1:
                combined = f"{intro}\n{combined}"
            sections.append(
                {
                    "source": filename,
                    "page": f"[Mục {num_label}{suffix}]",
                    "topic_hint": re.sub(r"^\d+\.\s*", "", heading).strip(),
                    "text": combined,
                }
            )
            sec_num += 1
            idx += 2
        return sections

    # Fallback theo đoạn văn bản.
    paras = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
    for i, p in enumerate(paras or [text.strip()], start=1):
        sections.append(
            {
                "source": filename,
                "page": f"[Đoạn {i}{suffix}]",
                "topic_hint": p.splitlines()[0][:55],
                "text": p,
            }
        )
    return sections


def load_structured_sections(files: list[tuple[str, bytes]]) -> tuple[str, list[dict]]:
    """Đọc nhiều file PDF/TXT và trả về (văn bản tổng hợp, danh sách section có metadata trang/mục)."""
    all_sections: list[dict] = []
    per_file_texts: list[str] = []
    for fname, raw in files:
        raw_text = load_text(fname, raw)
        secs = _extract_sections_for_file(fname, raw_text)
        all_sections.extend(secs)
        formatted = "\n\n".join(f"{s['page']} {s['text']}" if not s["text"].startswith("[Trang ") else f"{s['page']}\n{s['text']}" for s in secs)
        per_file_texts.append(f"=== {fname} ===\n{formatted}")
    return "\n\n".join(per_file_texts), all_sections


def load_sections_from_pasted_text(text: str, title: str = "Văn bản dán") -> tuple[str, list[dict]]:
    secs = _extract_sections_for_file(title, text)
    return text.strip(), secs


def load_default_samples() -> tuple[str, list[dict], str]:
    """Nạp sẵn bộ giáo trình mẫu trong samples/ để người dùng dùng được ngay khi mở app."""
    samples_dir = config.BASE_DIR / "samples"
    files: list[tuple[str, bytes]] = []
    for fname in ("rag-co-ban.txt", "openai-api-co-ban.txt"):
        fpath = samples_dir / fname
        if fpath.exists():
            files.append((fname, fpath.read_bytes()))
    if not files:
        return "", [], "Chưa có tài liệu"
    combined, sections = load_structured_sections(files)
    return combined, sections, "Giáo trình mẫu: RAG Cơ bản & OpenAI API"
