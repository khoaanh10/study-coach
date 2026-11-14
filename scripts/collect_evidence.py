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


def tc13(buf):
    text = RAG.decode("utf-8")
    for level in ("Dễ", "Khó"):
        buf.write(f"--- Mức {level} ---" + chr(10))
        for q in generate_quiz(text, 5, level).questions:
            buf.write(f"- {q.question}" + chr(10))


SHORT = (
    "RAG (Retrieval-Augmented Generation) là kỹ thuật cho mô hình ngôn ngữ trả lời dựa trên tài liệu bên ngoài. "
    "Hệ thống trước tiên tìm các đoạn tài liệu liên quan đến câu hỏi, sau đó đưa các đoạn đó vào prompt để mô hình "
    "trả lời. Tài liệu dài được chia thành các đoạn nhỏ gọi là chunk. Chunk quá lớn làm việc truy xuất kém chính xác, "
    "còn chunk quá nhỏ làm mất ngữ cảnh. Các chunk thường gối lên nhau một phần, gọi là overlap. Embedding là cách "
    "biến một đoạn văn bản thành một vector số, sao cho các đoạn có ý nghĩa gần nhau thì vector gần nhau. "
    "Độ gần thường được đo bằng cosine similarity."
)


def bug01(buf):
    """Hồi quy BUG-01: 10 lần tạo quiz trên đoạn văn từng gây lỗi, đếm câu gán nhầm hậu quả."""
    bad = 0
    for i in range(10):
        quiz = generate_quiz(SHORT, 5, "Trung bình")
        buf.write(f"Lần {i + 1}: {len(quiz.questions)} câu" + chr(10))
        for q in quiz.questions:
            ans = q.options[q.correct_index]
            wrong = "lớn" in q.question and "nhỏ" not in q.question and "ngữ cảnh" in ans
            bad += wrong
            buf.write(f"  {'SAI ' if wrong else ''}{q.question} => {ans}" + chr(10))
    buf.write(f"{chr(10)}Số câu gán 'mất ngữ cảnh' cho chunk quá lớn: {bad}/10 lần chạy" + chr(10))


NONSENSE = {
    "Gõ bừa một chuỗi lặp": "asdasdasd " * 20,
    "Gõ bừa bàn phím": "asdf qwer uiop zxcv hjkl bnm fgh tyu rew poi lkj mnb vcx dsa ewq iuy tre oiu kjh gfd nbv cxz yui opa",
    "Từ thật xếp lộn xộn": "bàn xanh chạy hôm qua ngủ cái con mèo số bảy trời ghế uống nhanh lá cây điện thoại bay tím và nhưng hoặc cửa sổ ăn sách đỏ chậm mưa bút hát núi cười lạnh xe muối giấy nhảy",
    "Lorem ipsum": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    "Tán gẫu không có thông tin": "hôm nay trời đẹp quá ha ha ha tôi không biết viết gì cả nên cứ gõ đại vài chữ cho đủ dài thôi chứ thật ra chẳng có nội dung gì hết á trời ơi buồn ngủ ghê luôn đó nha mọi người ơi",
    "Ghi chú học tập viết tắt (phải ra quiz)": "ghi chú bài hôm nay: embedding = biến text thành vector, vector gần nhau thì nghĩa gần nhau. đo bằng cosine. chunk to quá thì tìm kém chính xác, nhỏ quá mất ngữ cảnh. overlap để khỏi cắt ý. nhớ lưu metadata tên file + số trang",
    "Tài liệu tiếng Anh (phải ra quiz)": "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water. It takes place in the chloroplasts, which contain chlorophyll. Oxygen is released as a by-product.",
}


def tc19(buf):
    for name, text in NONSENSE.items():
        try:
            quiz = generate_quiz(text, 3, "Trung bình")
            buf.write(f"{name}: RA QUIZ {len(quiz.questions)} câu, ví dụ: {quiz.questions[0].question}" + chr(10))
        except Exception as e:
            buf.write(f"{name}: TỪ CHỐI ({e})" + chr(10))


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
    record("TC-13", "Độ khó Dễ so với Khó (mức hàm)", tc13)
    record("BUG-01", "Hồi quy đáp án lệch văn bản nguồn", bug01)
    record("TC-19", "Văn bản vô nghĩa hoặc không có kiến thức (mức hàm)", tc19)
    record("TC-17", "Thiếu API key", tc17)
    record("TC-18", "Sai API key", tc18)
