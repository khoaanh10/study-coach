"""Chạy các ca kiểm thử ở mức hàm (có gọi API thật) và ghi log vào docs/evidence/.

Chạy: python scripts/collect_evidence.py
"""
import io
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "evidence"
OUT.mkdir(parents=True, exist_ok=True)

import config  # noqa: E402
from core.loader import load_text  # noqa: E402
from core.quiz import generate_quiz  # noqa: E402


def record(tc: str, title: str, fn):
    buf = io.StringIO()
    buf.write(f"{tc}: {title}\nThời điểm chạy: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
    start = time.time()
    try:
        fn(buf)
    except Exception as e:  # ghi lại đúng lỗi nhận được
        buf.write(f"NGOẠI LỆ: {type(e).__name__}: {e}\n")
    buf.write(f"\nThời gian: {time.time() - start:.1f}s\n")
    (OUT / f"{tc}.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(tc, "xong")


def show_quiz(buf, quiz):
    buf.write(f"Số câu nhận được: {len(quiz.questions)}\n")
    for i, q in enumerate(quiz.questions, 1):
        buf.write(f"\nCâu {i}. {q.question}\n")
        for k, o in enumerate(q.options):
            mark = "*" if k == q.correct_index else " "
            buf.write(f"  {mark} {'ABCD'[k]}. {o}\n")
        buf.write(f"  Giải thích: {q.explanation}\n")


RAG = (ROOT / "samples" / "rag-co-ban.txt").read_bytes()
API = (ROOT / "samples" / "openai-api-co-ban.txt").read_bytes()


def tc04(buf):
    buf.write(f"Model: {config.STRONG_MODEL}\n")
    show_quiz(buf, generate_quiz(RAG.decode("utf-8"), 3, "Trung bình", config.STRONG_MODEL))


def tc07(buf):
    text = load_text("rag-co-ban.txt", RAG)
    buf.write(f"load_text trả về {len(text)} ký tự\n")
    show_quiz(buf, generate_quiz(text, 3, "Trung bình"))


def tc08(buf):
    pdf = OUT / "sample-text.pdf"
    text = load_text(pdf.name, pdf.read_bytes())
    buf.write(f"File: {pdf.name}\nload_text trả về {len(text)} ký tự, 80 ký tự đầu: {text[:80]!r}\n")
    show_quiz(buf, generate_quiz(text, 3, "Trung bình"))


def tc09(buf):
    files = [("rag-co-ban.txt", RAG), ("openai-api-co-ban.txt", API)]
    per_file = config.MAX_INPUT_CHARS // len(files)
    source = "\n\n".join(f"=== {n} ===\n{load_text(n, d)[:per_file]}" for n, d in files)
    buf.write(f"Ghép 2 file, mỗi file tối đa {per_file} ký tự, tổng {len(source)} ký tự\n")
    show_quiz(buf, generate_quiz(source, 6, "Trung bình"))


def tc10(buf):
    buf.write("Gọi load_text('bai.docx', ...)\n")
    load_text("bai.docx", b"abc")


def tc11(buf):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(width=595, height=842)
    b = io.BytesIO()
    w.write(b)
    text = load_text("scan.pdf", b.getvalue())
    buf.write(f"PDF 1 trang không có lớp chữ. load_text trả về: {text!r}\n")
    show_quiz(buf, generate_quiz(text, 3, "Trung bình"))


def tc12(buf):
    long_text = (RAG.decode("utf-8") + "\n") * 8
    buf.write(f"Đầu vào {len(long_text)} ký tự, giới hạn {config.MAX_INPUT_CHARS}\n")
    show_quiz(buf, generate_quiz(long_text, 3, "Trung bình"))


def run_with_env(buf, env_extra):
    code = (
        "from core.quiz import generate_quiz\n"
        "try:\n"
        "    generate_quiz(open('samples/rag-co-ban.txt', encoding='utf-8').read(), 3, 'Trung bình')\n"
        "    print('KHÔNG có lỗi')\n"
        "except Exception as e:\n"
        "    print(type(e).__name__ + ': ' + str(e))\n"
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **env_extra}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    buf.write(r.stdout + r.stderr[-500:])


def tc17(buf):
    buf.write("Chạy với OPENAI_API_KEY rỗng\n")
    run_with_env(buf, {"OPENAI_API_KEY": ""})


def tc18(buf):
    buf.write("Chạy với OPENAI_API_KEY sai\n")
    run_with_env(buf, {"OPENAI_API_KEY": "sk-key-sai-de-kiem-thu"})


def ut(buf):
    r = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=ROOT, capture_output=True, text=True)
    buf.write(r.stdout[-3000:])


if __name__ == "__main__":
    record("UT-pytest", "Test tự động", ut)
    record("TC-04", "Đổi model (mức hàm)", tc04)
    record("TC-07", "File TXT (mức hàm)", tc07)
    record("TC-08", "File PDF có chữ (mức hàm)", tc08)
    record("TC-09", "Nhiều file (mức hàm)", tc09)
    record("TC-10", "File sai định dạng (mức hàm)", tc10)
    record("TC-11", "PDF không có lớp chữ (mức hàm)", tc11)
    record("TC-12", "Văn bản rất dài (mức hàm)", tc12)
    record("TC-17", "Thiếu API key", tc17)
    record("TC-18", "Sai API key", tc18)
