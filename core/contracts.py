"""Hợp đồng dữ liệu dùng chung giữa các module (UI, quiz, DB, RAG, LangGraph Coach, Vision)."""
from pydantic import BaseModel, Field, field_validator


class Question(BaseModel):
    question: str
    options: list[str]  # đúng 4 đáp án
    # evidence đứng trước correct_index để model tìm bằng chứng rồi mới chốt đáp án.
    evidence: str = ""  # câu trích nguyên văn từ văn bản nguồn
    explanation: str
    correct_index: int  # 0..3
    topic: str = "Kiến thức chung"
    bloom_level: str = "Hiểu"
    page_ref: str = ""
    # Chẩn đoán lỗi tư duy ứng với từng lựa chọn (đủ 4 phần tử A..D; đáp án đúng ghi "Chính xác").
    option_diagnostics: list[str] = Field(default_factory=list)
    # Dạng câu hỏi: mcq (trắc nghiệm) | fill_blank (điền vào chỗ trống) | short_answer (tự luận ngắn).
    qtype: str = "mcq"
    # fill_blank: câu trích có chỗ trống "_____"; answer_text: đáp án mẫu cho câu dạng mở.
    blank_prompt: str = ""
    answer_text: str = ""


class Quiz(BaseModel):
    questions: list[Question]


class GradedQuestion(BaseModel):
    question: str
    options: list[str]
    chosen_index: int | None
    correct_index: int
    is_correct: bool
    explanation: str
    topic: str = "Kiến thức chung"
    bloom_level: str = "Hiểu"
    evidence: str = ""
    page_ref: str = ""
    diagnosed_misconception: str = ""
    confidence: str = "Bình thường"
    calibration_tag: str = ""
    qtype: str = "mcq"
    learner_text: str = ""  # câu trả lời tự gõ (dạng điền chỗ trống / tự luận)
    expected_text: str = ""  # đáp án mẫu của câu dạng mở
    grader_feedback: str = ""  # nhận xét của người chấm (LLM hoặc bộ chấm offline)


class QuizResult(BaseModel):
    score: int
    total: int
    items: list[GradedQuestion]
    weak_topics: list[str] = Field(default_factory=list)
    detected_misconceptions: list[str] = Field(default_factory=list)
    recommended_level: str = "Trung bình"
    blind_spot_count: int = 0
    lucky_guess_count: int = 0


class ConceptNode(BaseModel):
    name: str
    summary: str
    page_ref: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)


class DocumentConceptMap(BaseModel):
    doc_id: str
    doc_title: str
    concepts: list[ConceptNode] = Field(default_factory=list)


class ConceptMastery(BaseModel):
    concept: str
    mastery_score: float = 0.0  # 0..100
    bloom_level: str = "Nhớ"
    attempts: int = 0
    correct: int = 0
    last_reviewed: str = ""
    next_review_due: str = ""
    due_for_review: bool = False
    status: str = "unseen"  # unseen | misconception | learning | mastered


class MisconceptionRecord(BaseModel):
    id: int = 0
    topic: str
    description: str
    occurrences: int = 1
    resolved: bool = False
    last_seen: str = ""


class LearnerDNA(BaseModel):
    learner_id: str
    name: str
    role_label: str = ""
    overall_level: str = "Mới bắt đầu"  # Mới bắt đầu | Trung bình | Nâng cao
    mastery_avg: float = 0.0
    concepts: list[ConceptMastery] = Field(default_factory=list)
    active_misconceptions: list[MisconceptionRecord] = Field(default_factory=list)
    resolved_misconceptions: list[MisconceptionRecord] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    content: str
    source: str = ""
    page: str = ""
    topic: str = ""
    score: float = 0.0
    memory_type: str = "document"  # document | mistake


class TeachBackEvaluation(BaseModel):
    topic: str
    understanding_score: int  # 0..100
    verdict: str  # Vững bản chất | Hiểu một phần | Đang hiểu lệch
    correct_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    misconceptions_found: list[str] = Field(default_factory=list)
    socratic_follow_up: str
    citations: list[str] = Field(default_factory=list)


class VisionStepCheck(BaseModel):
    step_name: str
    status: str  # Đúng | Cần chỉnh | Sai
    feedback: str
    page_ref: str = ""


class VisionCritique(BaseModel):
    extracted_work: str
    topic: str
    overall_assessment: str
    steps: list[VisionStepCheck] = Field(default_factory=list)
    hint_for_learner: str
    detected_misconception: str = ""
    citations: list[str] = Field(default_factory=list)
    is_demo: bool = False  # True: kết quả Demo Offline (UI có thể gắn nhãn "demo")


class ToolCallTrace(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    summary: str = ""


class CoachTurnResult(BaseModel):
    answer: str
    route_taken: str
    pedagogical_reason: str
    learner_level_used: str
    citations: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    tool_traces: list[ToolCallTrace] = Field(default_factory=list)
    follow_up_question: str = ""
    suggested_quiz_topic: str = ""
    # Chủ đề thực sự của lượt này (khác suggested_quiz_topic khi bot lùi về kiến thức nền) để hỏi nối tiếp đúng mạch.
    topic: str = ""
    latency_ms: int = 0
    cost_usd: float = 0.0
    is_demo: bool = False

    @field_validator("retrieved_chunks", mode="before")
    @classmethod
    def _coerce_stale_chunks(cls, v):
        """Streamlit reload module -> kb lưu trong session_state còn sinh RetrievedChunk của class cũ.
        Chuyển mọi BaseModel 'ngoại lai' về dict để pydantic dựng lại bằng class hiện tại."""
        if isinstance(v, (list, tuple)):
            return [
                x.model_dump() if isinstance(x, BaseModel) and not isinstance(x, RetrievedChunk) else x
                for x in v
            ]
        return v


class DiffContradiction(BaseModel):
    topic: str
    learner_claim: str
    ground_truth: str
    page_ref: str
    severity: str = "Cao"
    explanation: str


class DiffBlindSpot(BaseModel):
    topic: str
    page_ref: str
    importance_reason: str


class DiffVerifiedItem(BaseModel):
    learner_claim: str
    page_ref: str
    note: str = ""


class CognitiveDiffReport(BaseModel):
    alignment_score: int  # 0..100
    summary: str
    contradictions: list[DiffContradiction] = Field(default_factory=list)
    blind_spots: list[DiffBlindSpot] = Field(default_factory=list)
    verified_points: list[DiffVerifiedItem] = Field(default_factory=list)
    patched_note: str = ""


class IncidentClue(BaseModel):
    tool_name: str  # inspect_config | inspect_logs | inspect_metrics
    title: str
    output: str


class IncidentScenario(BaseModel):
    ticket_id: str
    severity: str  # SEV-1 | SEV-2
    title: str
    symptom: str
    topic: str
    page_ref: str
    clues: list[IncidentClue] = Field(default_factory=list)
    root_cause_options: list[str] = Field(default_factory=list)
    correct_option_index: int = 0
    ground_truth_explanation: str = ""


class IncidentEvaluation(BaseModel):
    resolved_correctly: bool
    score: int  # 0..100
    investigation_bonus: int = 0
    feedback: str
    violated_principle: str
    page_ref: str
    post_mortem_question: str


class TrojanStatement(BaseModel):
    id: int
    topic: str
    statement: str
    is_buggy: bool
    bug_explanation: str = ""
    page_ref: str = ""


class TrojanChallenge(BaseModel):
    title: str
    context_summary: str
    statements: list[TrojanStatement] = Field(default_factory=list)


class DevilsAdvocateQuestion(BaseModel):
    persona: str
    question: str
    conflicting_sections: list[str] = Field(default_factory=list)
    hidden_trap: str
    sample_strong_answer: str = ""


class DefenseEvaluation(BaseModel):
    defense_score: int  # 0..100
    verdict: str  # Thuyết phục | Cần chặt chẽ hơn | Mâu thuẫn tài liệu
    strengths: list[str] = Field(default_factory=list)
    gaps_or_contradictions: list[str] = Field(default_factory=list)
    strong_answer_outline: str
    next_hard_follow_up: str
