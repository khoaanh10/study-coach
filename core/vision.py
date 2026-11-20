"""Chấm bài làm / sơ đồ viết tay hoặc ảnh chụp màn hình code bằng Vision, đối chiếu RAG và trình độ học viên."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config
from core import db
from core.contracts import LearnerDNA, VisionCritique
from core.demo_provider import demo_critique_image
from core.llm import analyze_image_structured
from core.prompts import VISION_CRITIQUE_PROMPT
from core.rag import DualMemoryKnowledgeBase

VISION_SAMPLES_DIR = config.BASE_DIR / "samples" / "vision"

SAMPLE_METADATA = {
    "so-do-rag-loi": {
        "filename": "so-do-rag-loi.png",
        "title": "Sơ đồ Pipeline RAG học viên tự phác thảo",
        "caption": "Sơ đồ có lỗi đảo ngược thứ tự truy xuất và cấu hình Chunk = 50.000, Overlap = 0.",
    },
    "code-openai-loi": {
        "filename": "code-openai-loi.png",
        "title": "Ảnh chụp đoạn code gọi OpenAI API",
        "caption": "Code mắc lỗi hardcode API key, thiếu lịch sử messages và đặt temperature = 1.8.",
    },
}


def _load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_rag_diagram_sample(dest: Path) -> None:
    img = Image.new("RGB", (980, 560), "#F7F7F4")
    draw = ImageDraw.Draw(img)
    f_title = _load_font(26, bold=True)
    f_box = _load_font(20, bold=True)
    f_sub = _load_font(17, bold=False)

    draw.rectangle([20, 20, 960, 540], outline="#DDD9CE", width=3, fill="#FFFFFF")
    draw.text((48, 42), "BÀI LÀM NHÁP: THIẾT KẾ LUỒNG HỎI ĐÁP RAG", fill="#141B2D", font=f_title)
    draw.text((48, 78), "Học viên vẽ sơ đồ và thông số cấu hình cho giáo trình PDF 40 trang", fill="#526071", font=f_sub)

    # Box 1
    draw.rounded_rectangle([50, 140, 440, 270], radius=14, fill="#FEF2F2", outline="#B91C1C", width=3)
    draw.text((70, 160), "1. Chunking (Tách tài liệu)", fill="#141B2D", font=f_box)
    draw.text((70, 195), "- chunk_size = 50.000 ky tu (cho do ton)", fill="#334155", font=f_sub)
    draw.text((70, 224), "- overlap = 0 (khong can goi dau)", fill="#B91C1C", font=f_sub)

    # Box 2
    draw.rounded_rectangle([520, 140, 930, 270], radius=14, fill="#EFF6FF", outline="#1D4ED8", width=3)
    draw.text((540, 160), "2. Luồng xử lý khi User hỏi", fill="#141B2D", font=f_box)
    draw.text((540, 195), "Cau hoi -> Gui thang LLM tra loi truoc", fill="#B91C1C", font=f_sub)
    draw.text((540, 224), "-> Sau do moi tim trong Vector Store", fill="#334155", font=f_sub)

    # Arrow between Box 1 & Box 2
    draw.line([440, 205, 515, 205], fill="#141B2D", width=4)
    draw.polygon([(520, 205), (506, 196), (506, 214)], fill="#141B2D")

    # Box 3
    draw.rounded_rectangle([50, 330, 930, 480], radius=14, fill="#ECFDF5", outline="#0F766E", width=3)
    draw.text((70, 355), "3. Truy xuất & Nguồn dữ liệu", fill="#141B2D", font=f_box)
    draw.text((70, 392), "- Luu vector trong FAISS, chon top_k = 3 doan gan nhat", fill="#334155", font=f_sub)
    draw.text((70, 424), "- Moi chunk luu kem metadata: ten file + so trang [Trang n]", fill="#0F766E", font=f_sub)

    img.save(dest, format="PNG")


def _draw_code_sample(dest: Path) -> None:
    img = Image.new("RGB", (980, 560), "#141B2D")
    draw = ImageDraw.Draw(img)
    f_title = _load_font(24, bold=True)
    f_code = _load_font(20, bold=False)
    f_note = _load_font(18, bold=False)

    draw.rectangle([20, 20, 960, 540], outline="#2E3A5C", width=3, fill="#1E2840")
    draw.text((48, 42), "ẢNH CHỤP CODE PYTHON: TẠO CHATBOT HỘI THOẠI NHIỀU LƯỢT", fill="#F7F7F4", font=f_title)
    draw.line([48, 86, 930, 86], fill="#2E3A5C", width=2)

    lines = [
        ("1  from openai import OpenAI", "#BCC5DB"),
        ("2", "#BCC5DB"),
        ('3  client = OpenAI(api_key="sk-live-hardcoded-998877")', "#FCA5A5"),
        ("4", "#BCC5DB"),
        ("5  def hoi_tiep(cau_hoi_moi):", "#93ABFF"),
        ("6      # Khong can gui lich su vi OpenAI tu nho ngu canh", "#FCD34D"),
        ('7      resp = client.chat.completions.create(', "#F7F7F4"),
        ('8          model="GPT-4o-mini",', "#F7F7F4"),
        ('9          messages=[{"role": "user", "content": cau_hoi_moi}],', "#FCA5A5"),
        ("10         temperature=1.8,  # dung cho trich xuat JSON on dinh", "#FCA5A5"),
        ("11     )", "#F7F7F4"),
        ("12     return resp.choices[0].message.content", "#F7F7F4"),
    ]
    y = 112
    for text_line, color in lines:
        draw.text((54, y), text_line, fill=color, font=f_code)
        y += 32

    draw.text((54, 502), "Ghi chú học viên: Em chạy thấy bot không nhớ câu hỏi trước đó là sao ạ?", fill="#BCC5DB", font=f_note)
    img.save(dest, format="PNG")


def ensure_sample_vision_images() -> dict[str, Path]:
    """Đảm bảo 2 ảnh bài làm mẫu có sẵn trong samples/vision/."""
    VISION_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rag_path = VISION_SAMPLES_DIR / SAMPLE_METADATA["so-do-rag-loi"]["filename"]
    code_path = VISION_SAMPLES_DIR / SAMPLE_METADATA["code-openai-loi"]["filename"]
    if not rag_path.exists():
        _draw_rag_diagram_sample(rag_path)
    if not code_path.exists():
        _draw_code_sample(code_path)
    return {"so-do-rag-loi": rag_path, "code-openai-loi": code_path}


def list_sample_vision_images() -> list[dict]:
    paths = ensure_sample_vision_images()
    items = []
    for key, meta in SAMPLE_METADATA.items():
        items.append({**meta, "id": key, "path": paths[key]})
    return items


def critique_student_image(
    image_bytes: bytes,
    mime_type: str,
    note: str,
    kb: DualMemoryKnowledgeBase,
    learner_dna: LearnerDNA,
    doc_id: str,
    model: str | None = None,
    demo_mode: bool = False,
    sample_id: str | None = None,
    db_path: str | Path | None = None,
) -> tuple[VisionCritique, LearnerDNA]:
    """Phân tích ảnh bài nháp học viên, đối chiếu đoạn nguồn RAG và cập nhật Learner DNA."""
    if demo_mode or not config.has_live_api_key():
        critique = demo_critique_image(
            image_bytes=image_bytes,
            note=note,
            learner_dna=learner_dna,
            sample_id=sample_id,
            db_path=db_path,
        )
    else:
        # Không có ghi chú: dùng tên khái niệm của hồ sơ/tài liệu hiện tại làm truy vấn (không dùng chuỗi cứng của giáo trình mẫu).
        rag_query = note.strip() or " ".join(c.concept for c in learner_dna.concepts[:3])
        chunks = kb.retrieve(rag_query, top_k=4)[0] if rag_query else []
        context = "\n\n".join(f"{c.page}: {c.content}" for c in chunks)
        user_prompt = (
            f"TRÌNH ĐỘ HỌC VIÊN: {learner_dna.overall_level} (Điểm TB: {learner_dna.mastery_avg:.0f}%)\n"
            f"GHI CHÚ / CÂU HỎI HỌC VIÊN: {note or 'Chấm giúp em bài làm trong ảnh này.'}\n\n"
            f"TÀI LIỆU NGUỒN ĐỐI CHIẾU:\n{context}"
        )
        critique = analyze_image_structured(
            image_bytes=image_bytes,
            mime_type=mime_type,
            user_prompt=user_prompt,
            system_prompt=VISION_CRITIQUE_PROMPT,
            schema=VisionCritique,
            model=model or config.DEFAULT_MODEL,
            purpose="vision_critique",
            db_path=db_path,
        )

    # Chỉ ghi hồ sơ khi thật sự có nội dung đã chấm: Demo Offline với ảnh tải lên và kết quả không có bước nào thì bỏ qua.
    graded = bool(critique.steps) and not (critique.is_demo and not sample_id)
    if graded:
        if critique.detected_misconception:
            db.record_misconception(
                learner_id=learner_dna.learner_id,
                doc_id=doc_id,
                topic=critique.topic,
                description=critique.detected_misconception,
                db_path=db_path,
            )
        has_wrong = any(s.status == "Sai" for s in critique.steps)
        # Có lỗi tư duy được ghi nhận thì không được cộng điểm như làm đúng.
        is_correct = not has_wrong and not critique.detected_misconception
        db.update_concept_mastery(
            learner_id=learner_dna.learner_id,
            doc_id=doc_id,
            concept=critique.topic,
            is_correct=is_correct,
            bloom_level="Vận dụng",
            total_concepts=len(learner_dna.concepts) or None,
            db_path=db_path,
        )
        if is_correct:
            db.resolve_misconception(
                learner_id=learner_dna.learner_id,
                doc_id=doc_id,
                topic=critique.topic,
                db_path=db_path,
            )
    updated_dna = db.get_learner_dna(
        learner_id=learner_dna.learner_id,
        doc_id=doc_id,
        concepts_order=[c.concept for c in learner_dna.concepts],
        db_path=db_path,
    )
    return critique, updated_dna
