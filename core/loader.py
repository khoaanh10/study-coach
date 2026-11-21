"""Đọc văn bản từ file upload (PDF, DOCX, PPTX, TXT, MD) và giữ cấu trúc [Trang n] / [Mục n] cho RAG."""
from __future__ import annotations

import posixpath
import re
import zipfile
from io import BytesIO
from xml.etree import ElementTree

from pypdf import PdfReader

import config

_PAGE_SPLIT_RE = re.compile(r"(\[Trang\s+\d+\])")
_NUMBERED_HEADING_RE = re.compile(r"(?m)^(\d+\.\s+[^\n]{2,80})$")


SUPPORTED_EXTENSIONS = ("pdf", "docx", "pptx", "txt", "md")
_MAX_XML_BYTES = 40 * 1024 * 1024  # chặn file nén "bom" trong DOCX/PPTX
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_P_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
_R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _open_office_zip(filename: str, data: bytes, kind: str) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ValueError(
            f"Không đọc được file {filename}: không phải file {kind} hợp lệ (file hỏng, có mật khẩu, "
            f"hoặc là định dạng cũ .doc/.ppt). Hãy lưu lại dưới dạng {kind} rồi tải lên."
        ) from e


def _read_office_xml(zf: zipfile.ZipFile, member: str, filename: str) -> ElementTree.Element | None:
    try:
        info = zf.getinfo(member)
    except KeyError:
        return None
    if info.file_size > _MAX_XML_BYTES:
        raise ValueError(f"File {filename} quá lớn để xử lý (một phần nội dung vượt 40 MB sau khi giải nén).")
    raw = zf.read(member)
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        # Office Open XML không dùng DTD; chặn để tránh tấn công mở rộng thực thể XML.
        raise ValueError(f"File {filename} chứa nội dung XML không hợp lệ.")
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as e:
        raise ValueError(f"Không đọc được file {filename}: nội dung bên trong bị hỏng.") from e


def _docx_paragraph_text(par: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in par.iter():
        if node.tag == f"{_W_NS}t":
            parts.append(node.text or "")
        elif node.tag == f"{_W_NS}tab":
            parts.append(" ")
        elif node.tag in (f"{_W_NS}br", f"{_W_NS}cr"):
            parts.append("\n")
    return "".join(parts).strip()


def _load_docx(filename: str, data: bytes) -> str:
    """Đọc văn bản Word (.docx) bằng thư viện chuẩn: mỗi đoạn một khối, tiêu đề (Heading) mở đầu một mục."""
    with _open_office_zip(filename, data, ".docx") as zf:
        root = _read_office_xml(zf, "word/document.xml", filename)
    if root is None:
        raise ValueError(f"File {filename} không phải tài liệu Word (.docx) hợp lệ.")
    blocks: list[str] = []
    for par in root.iter(f"{_W_NS}p"):
        text = _docx_paragraph_text(par)
        if not text:
            continue
        style = par.find(f"{_W_NS}pPr/{_W_NS}pStyle")
        style_val = (style.get(f"{_W_NS}val", "") if style is not None else "").lower()
        is_heading = style_val.startswith(("heading", "title", "tiu")) or bool(re.match(r"^(h|heading)\d$", style_val))
        if is_heading:
            # Tiêu đề gắn liền đoạn ngay sau nó để mỗi mục có tên chủ đề ở dòng đầu.
            blocks.append("\x00" + text)
        else:
            blocks.append(text)
    if not blocks:
        raise ValueError(
            f"File {filename} không có nội dung chữ (có thể chỉ gồm hình ảnh). "
            "Hãy dùng file có văn bản, hoặc dán nội dung trực tiếp."
        )
    out: list[str] = []
    pending_heading = ""
    for b in blocks:
        if b.startswith("\x00"):
            if pending_heading:
                out.append(pending_heading)
            pending_heading = b[1:]
        elif pending_heading:
            out.append(f"{pending_heading}\n{b}")
            pending_heading = ""
        else:
            out.append(b)
    if pending_heading:
        out.append(pending_heading)
    return "\n\n".join(out)


def _pptx_slide_order(zf: zipfile.ZipFile, filename: str) -> list[str]:
    """Danh sách file slide theo đúng thứ tự trình chiếu (đọc presentation.xml), dự phòng theo số trong tên file."""
    names = set(zf.namelist())
    ordered: list[str] = []
    pres = _read_office_xml(zf, "ppt/presentation.xml", filename)
    rels = _read_office_xml(zf, "ppt/_rels/presentation.xml.rels", filename)
    if pres is not None and rels is not None:
        targets = {r.get("Id"): r.get("Target", "") for r in rels}
        for sld in pres.iter(f"{_P_NS}sldId"):
            target = targets.get(sld.get(f"{_R_NS}id"), "")
            if not target:
                continue
            path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("ppt", target))
            if path in names:
                ordered.append(path)
    if not ordered:
        slides = [n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        ordered = sorted(slides, key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
    return ordered


def _load_pptx(filename: str, data: bytes) -> str:
    """Đọc slide PowerPoint (.pptx): mỗi slide thành một [Trang n] để trích nguồn theo số slide."""
    parts: list[str] = []
    with _open_office_zip(filename, data, ".pptx") as zf:
        slide_paths = _pptx_slide_order(zf, filename)
        if not slide_paths:
            raise ValueError(f"File {filename} không phải bản trình chiếu PowerPoint (.pptx) hợp lệ.")
        for i, path in enumerate(slide_paths, start=1):
            root = _read_office_xml(zf, path, filename)
            if root is None:
                continue
            lines = []
            for par in root.iter(f"{_A_NS}p"):
                line = "".join(t.text or "" for t in par.iter(f"{_A_NS}t")).strip()
                if line:
                    lines.append(line)
            if lines:
                parts.append(f"[Trang {i}]\n" + "\n".join(lines))
    if not parts:
        raise ValueError(
            f"File {filename} không có chữ trên slide nào (có thể toàn hình ảnh). "
            "Hãy dùng bản trình chiếu có văn bản, hoặc dán nội dung trực tiếp."
        )
    return "\n\n".join(parts)


def load_text(filename: str, data: bytes) -> str:
    """Trả về văn bản. PDF/PPTX có đánh dấu [Trang n] để sau này trích nguồn."""
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
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""
            if text:
                parts.append(f"[Trang {i}]\n{text}")
        if not parts:
            raise ValueError(
                f"File {filename} là PDF dạng ảnh quét (scan): cả {len(pages)} trang đều không có lớp chữ "
                "nên không trích được nội dung. Hãy chạy nhận dạng chữ (OCR) cho file trước, "
                "dùng bản PDF có chữ bôi đen/chọn được, hoặc dán văn bản trực tiếp ở thẻ «Dán nội dung»."
            )
        return "\n\n".join(parts)
    if name.endswith(".docx"):
        return _load_docx(filename, data)
    if name.endswith(".pptx"):
        return _load_pptx(filename, data)
    if name.endswith((".doc", ".ppt")):
        raise ValueError(
            f"File {filename} là định dạng Office cũ (.doc/.ppt). Hãy mở và lưu lại thành .docx / .pptx rồi tải lên."
        )
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    raise ValueError("Chỉ hỗ trợ file .pdf, .docx, .pptx, .txt hoặc .md")


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
