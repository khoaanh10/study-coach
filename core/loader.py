"""Đọc văn bản từ file upload (PDF hoặc TXT)."""
from io import BytesIO

from pypdf import PdfReader


def load_text(filename: str, data: bytes) -> str:
    """Trả về văn bản. PDF có đánh dấu [Trang n] để sau này trích nguồn."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        parts = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"[Trang {i}]\n{text}")
        return "\n\n".join(parts)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="replace")
    raise ValueError("Chỉ hỗ trợ file .pdf hoặc .txt")
