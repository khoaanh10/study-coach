"""Đọc văn bản từ file upload (PDF hoặc TXT)."""
from io import BytesIO

from pypdf import PdfReader


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
