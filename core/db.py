"""Tầng lưu trữ SQLite cho Hồ sơ năng lực người học (Learner DNA), lỗi tư duy, lịch sử quiz và telemetry LLM."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import config
from core.contracts import (
    CoachTurnResult,
    ConceptMastery,
    DocumentConceptMap,
    LearnerDNA,
    MisconceptionRecord,
    QuizResult,
)

DEFAULT_DOC_ID = "default_curriculum"

DEMO_LEARNER_IDS = {"me", "intern_demo", "senior_demo"}
_VALID_OVERALL_LEVELS = {"Mới bắt đầu", "Trung bình", "Nâng cao"}

_BLOOM_RANK = {"Nhớ": 1, "Hiểu": 2, "Vận dụng": 3, "Phân tích": 4, "Đánh giá": 5, "Sáng tạo": 6}
_BLOOM_CAP = {"Nhớ": 68.0, "Hiểu": 84.0, "Vận dụng": 96.0, "Phân tích": 100.0, "Đánh giá": 100.0, "Sáng tạo": 100.0}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# Mốc khởi động tiến trình: "ngân sách phiên" chỉ tính các lần gọi từ thời điểm này trở đi.
_PROCESS_START = _now_iso()


def _due_iso(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_db_path(db_path: str | Path | None = None) -> Path:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect(db_path: str | Path | None = None) -> Iterator[sqlite3.Connection]:
    path = _resolve_db_path(db_path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Khởi tạo schema SQLite và nạp sẵn 3 hồ sơ người học mẫu nếu bảng còn trống."""
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS learners (
                learner_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role_label TEXT NOT NULL DEFAULT '',
                overall_level TEXT NOT NULL DEFAULT 'Mới bắt đầu',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS concept_mastery (
                learner_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                concept TEXT NOT NULL,
                mastery_score REAL NOT NULL DEFAULT 0,
                bloom_level TEXT NOT NULL DEFAULT 'Nhớ',
                attempts INTEGER NOT NULL DEFAULT 0,
                correct INTEGER NOT NULL DEFAULT 0,
                last_reviewed TEXT NOT NULL DEFAULT '',
                next_review_due TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (learner_id, doc_id, concept)
            );

            CREATE TABLE IF NOT EXISTS misconceptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                description TEXT NOT NULL,
                occurrences INTEGER NOT NULL DEFAULT 1,
                resolved INTEGER NOT NULL DEFAULT 0,
                last_seen TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                weak_topics_json TEXT NOT NULL DEFAULT '[]',
                misconceptions_json TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                purpose TEXT NOT NULL,
                tokens_in INTEGER NOT NULL DEFAULT 0,
                tokens_out INTEGER NOT NULL DEFAULT 0,
                latency_ms INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                is_demo INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS confidence_stats (
                learner_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                solid_correct INTEGER NOT NULL DEFAULT 0,
                lucky_guess INTEGER NOT NULL DEFAULT 0,
                aware_gap INTEGER NOT NULL DEFAULT 0,
                blind_spot INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (learner_id, doc_id)
            );

            CREATE TABLE IF NOT EXISTS learner_documents (
                learner_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                doc_text TEXT NOT NULL,
                sections_json TEXT NOT NULL,
                concept_map_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (learner_id, doc_id)
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                turn_json TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                learner_id TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        existing = conn.execute("SELECT COUNT(*) AS cnt FROM learners").fetchone()["cnt"]
        if existing == 0:
            _seed_default_learners(conn)

        # Dọn sạch các profile cũ từng tạo không qua tài khoản mật khẩu (bảng users).
        legacy_rows = conn.execute(
            """
            SELECT learner_id FROM learners
            WHERE learner_id NOT IN ('me', 'intern_demo', 'senior_demo')
              AND learner_id NOT IN (SELECT learner_id FROM users)
            """
        ).fetchall()
        for r in legacy_rows:
            lid = r["learner_id"]
            for tbl in ("chat_messages", "learner_documents", "confidence_stats", "quiz_attempts", "misconceptions", "concept_mastery", "learners"):
                conn.execute(f"DELETE FROM {tbl} WHERE learner_id = ?", (lid,))


def _seed_default_learners(conn: sqlite3.Connection) -> None:
    now = _now_iso()
    past_due = _due_iso(-2)
    future_due = _due_iso(4)
    today_due = _due_iso(0)

    conn.executemany(
        "INSERT INTO learners (learner_id, name, role_label, overall_level, updated_at) VALUES (?, ?, ?, ?, ?)",
        [
            ("me", "Học viên của tôi (Tự học từ đầu)", "Tài khoản tự do", "Mới bắt đầu", now),
            ("intern_demo", "Minh — Intern mới học", "Trình độ Cơ bản", "Mới bắt đầu", now),
            ("senior_demo", "Lan — Kỹ sư Backend", "Trình độ Nâng cao", "Nâng cao", now),
        ],
    )

    # Dữ liệu mẫu cho Minh (Intern mới học) trên bộ giáo trình mặc định.
    conn.executemany(
        """
        INSERT INTO concept_mastery
            (learner_id, doc_id, concept, mastery_score, bloom_level, attempts, correct, last_reviewed, next_review_due)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("intern_demo", DEFAULT_DOC_ID, "RAG là gì", 62.0, "Hiểu", 3, 2, now, future_due),
            ("intern_demo", DEFAULT_DOC_ID, "Chunking và Overlap", 28.0, "Nhớ", 4, 1, now, past_due),
            ("intern_demo", DEFAULT_DOC_ID, "Embeddings và Cosine", 35.0, "Nhớ", 3, 1, now, today_due),
            ("intern_demo", DEFAULT_DOC_ID, "Cấu trúc lời gọi & Messages", 42.0, "Nhớ", 2, 1, now, today_due),
            ("intern_demo", DEFAULT_DOC_ID, "Structured outputs & Function calling", 18.0, "Nhớ", 1, 0, now, past_due),
            # Dữ liệu mẫu cho Lan (Kỹ sư Backend) trên bộ giáo trình mặc định.
            ("senior_demo", DEFAULT_DOC_ID, "RAG là gì", 90.0, "Vận dụng", 4, 4, now, future_due),
            ("senior_demo", DEFAULT_DOC_ID, "Chunking và Overlap", 86.0, "Phân tích", 5, 4, now, future_due),
            ("senior_demo", DEFAULT_DOC_ID, "Embeddings và Cosine", 84.0, "Vận dụng", 4, 4, now, future_due),
            ("senior_demo", DEFAULT_DOC_ID, "Structured outputs & Function calling", 88.0, "Vận dụng", 4, 4, now, future_due),
            ("senior_demo", DEFAULT_DOC_ID, "Cấu trúc lời gọi & Messages", 78.0, "Hiểu", 3, 3, now, future_due),
            ("senior_demo", DEFAULT_DOC_ID, "Đánh giá chất lượng RAG", 48.0, "Hiểu", 3, 1, past_due, past_due),
        ],
    )

    conn.executemany(
        """
        INSERT INTO misconceptions
            (learner_id, doc_id, topic, description, occurrences, resolved, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "intern_demo",
                DEFAULT_DOC_ID,
                "Chunking và Overlap",
                "Gán nhầm hậu quả mất ngữ cảnh của chunk quá nhỏ sang chunk quá lớn",
                2,
                0,
                now,
            ),
            (
                "intern_demo",
                DEFAULT_DOC_ID,
                "Cấu trúc lời gọi & Messages",
                "Tưởng mô hình OpenAI tự nhớ hội thoại trước đó mà không cần gửi lại lịch sử messages",
                1,
                0,
                now,
            ),
            (
                "senior_demo",
                DEFAULT_DOC_ID,
                "Đánh giá chất lượng RAG",
                "Chỉ kiểm tra tỉ lệ trả lời đúng mà bỏ qua chỉ số trích đúng nguồn trang",
                1,
                0,
                now,
            ),
            (
                "senior_demo",
                DEFAULT_DOC_ID,
                "Chunking và Overlap",
                "Đặt chunk quá lớn không có overlap làm lẫn lộn nhiều ý trong truy xuất",
                1,
                1,
                past_due,
            ),
        ],
    )

    conn.executemany(
        """
        INSERT INTO quiz_attempts
            (learner_id, doc_id, created_at, model, difficulty, score, total, weak_topics_json, misconceptions_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "intern_demo",
                DEFAULT_DOC_ID,
                now,
                config.DEFAULT_MODEL,
                "Trung bình",
                2,
                5,
                json.dumps(["Chunking và Overlap", "Embeddings và Cosine"], ensure_ascii=False),
                json.dumps(
                    ["Gán nhầm hậu quả mất ngữ cảnh của chunk quá nhỏ sang chunk quá lớn"],
                    ensure_ascii=False,
                ),
            ),
            (
                "senior_demo",
                DEFAULT_DOC_ID,
                now,
                config.STRONG_MODEL,
                "Khó",
                4,
                5,
                json.dumps(["Đánh giá chất lượng RAG"], ensure_ascii=False),
                json.dumps(
                    ["Chỉ kiểm tra tỉ lệ trả lời đúng mà bỏ qua chỉ số trích đúng nguồn trang"],
                    ensure_ascii=False,
                ),
            ),
        ],
    )

    conn.executemany(
        """
        INSERT INTO confidence_stats (learner_id, doc_id, solid_correct, lucky_guess, aware_gap, blind_spot)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("me", DEFAULT_DOC_ID, 0, 0, 0, 0),
            ("intern_demo", DEFAULT_DOC_ID, 3, 2, 2, 2),
            ("senior_demo", DEFAULT_DOC_ID, 11, 1, 1, 1),
        ],
    )


def list_learners(db_path: str | Path | None = None) -> list[dict]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT learner_id, name, role_label, overall_level, updated_at FROM learners ORDER BY rowid"
        ).fetchall()
    return [dict(r) for r in rows]


def _slugify_learner_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "learner"


def _generate_learner_id(conn: sqlite3.Connection, clean_name: str) -> str:
    slug = _slugify_learner_name(clean_name)[:16]
    digest = hashlib.sha1(clean_name.lower().encode("utf-8")).hexdigest()[:8]
    base = f"user_{slug}_{digest}"
    candidate = base
    suffix = 1
    while conn.execute("SELECT 1 FROM learners WHERE learner_id = ?", (candidate,)).fetchone() is not None:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def login_or_register_learner(
    name: str,
    overall_level: str = "Mới bắt đầu",
    db_path: str | Path | None = None,
) -> str:
    """Đăng nhập theo tên hiển thị hoặc tạo tài khoản người học mới, trả về learner_id."""
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("Tên học viên không được để trống.")
    level = overall_level if overall_level in _VALID_OVERALL_LEVELS else "Mới bắt đầu"
    init_db(db_path)
    now = _now_iso()

    with _connect(db_path) as conn:
        demo_placeholders = ", ".join("?" for _ in DEMO_LEARNER_IDS)
        existing = conn.execute(
            f"""
            SELECT learner_id FROM learners
            WHERE lower(name) = lower(?) AND learner_id NOT IN ({demo_placeholders})
            ORDER BY rowid
            """,
            (clean_name, *sorted(DEMO_LEARNER_IDS)),
        ).fetchone()
        if existing is not None:
            learner_id = existing["learner_id"]
            conn.execute(
                "UPDATE learners SET updated_at = ? WHERE learner_id = ?",
                (now, learner_id),
            )
            return learner_id

        learner_id = _generate_learner_id(conn, clean_name)
        conn.execute(
            "INSERT INTO learners (learner_id, name, role_label, overall_level, updated_at) VALUES (?, ?, ?, ?, ?)",
            (learner_id, clean_name, "Học viên", level, now),
        )
        return learner_id


def list_user_learners(db_path: str | Path | None = None) -> list[dict]:
    """Danh sách tài khoản người học do người dùng tự tạo (bỏ qua hồ sơ demo)."""
    return [row for row in list_learners(db_path) if row["learner_id"] not in DEMO_LEARNER_IDS]


_USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")
_PASSWORD_MIN_LENGTH = 6
_PBKDF2_ITERATIONS = 200000


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS).hex()


def register_user(
    username: str,
    password: str,
    display_name: str = "",
    db_path: str | Path | None = None,
) -> str:
    """Đăng ký tài khoản username/password, tạo hồ sơ người học và trả về learner_id."""
    clean_username = (username or "").strip().lower()
    if not _USERNAME_RE.match(clean_username):
        raise ValueError("Tên đăng nhập 3-32 ký tự, chỉ gồm chữ cái không dấu, số, _ . -")
    if not password or len(password) < _PASSWORD_MIN_LENGTH:
        raise ValueError("Mật khẩu tối thiểu 6 ký tự.")

    init_db(db_path)
    now = _now_iso()

    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (clean_username,)
        ).fetchone()
        if existing is not None:
            raise ValueError("Tên đăng nhập đã tồn tại.")

        learner_id = "user_" + uuid.uuid4().hex[:12]
        learner_name = display_name.strip() or clean_username
        conn.execute(
            "INSERT INTO learners (learner_id, name, role_label, overall_level, updated_at) VALUES (?, ?, ?, ?, ?)",
            (learner_id, learner_name, "Học viên", "Mới bắt đầu", now),
        )

        salt_bytes = secrets.token_bytes(16)
        password_hash = _hash_password(password, salt_bytes)
        conn.execute(
            """
            INSERT INTO users (username, display_name, learner_id, password_hash, salt)
            VALUES (?, ?, ?, ?, ?)
            """,
            (clean_username, learner_name, learner_id, password_hash, salt_bytes.hex()),
        )
        return learner_id


def authenticate_user(
    username: str,
    password: str,
    db_path: str | Path | None = None,
) -> str | None:
    """Xác thực username/password; trả về learner_id khi đúng, None khi sai."""
    clean_username = (username or "").strip().lower()
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT learner_id, password_hash, salt FROM users WHERE username = ?",
            (clean_username,),
        ).fetchone()

    if row is None:
        # Thực hiện một phép hash giả để thời gian phản hồi không tiết lộ username có tồn tại hay không.
        _hash_password(password or "", b"\x00" * 16)
        return None

    computed = _hash_password(password or "", bytes.fromhex(row["salt"]))
    if hmac.compare_digest(computed, row["password_hash"]):
        return row["learner_id"]
    return None


def get_user_display_name(learner_id: str, db_path: str | Path | None = None) -> str:
    """Trả về tên hiển thị của tài khoản theo learner_id, hoặc chuỗi rỗng nếu không có."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT display_name FROM users WHERE learner_id = ?", (learner_id,)
        ).fetchone()
    return row["display_name"] if row is not None else ""


def _is_due(next_review_due: str) -> bool:
    if not next_review_due:
        return False
    try:
        due = datetime.strptime(next_review_due, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return due <= datetime.now(timezone.utc)
    except ValueError:
        return False


def _compute_overall_level(
    scores: list[float],
    existing_level: str,
    active_misconceptions_count: int = 0,
    total_concepts: int | None = None,
) -> str:
    if not scores:
        return existing_level
    avg = sum(scores) / len(scores)
    min_breadth = min(3, total_concepts) if total_concepts and total_concepts > 0 else 3
    if (
        avg >= 76.0
        and active_misconceptions_count <= 1
        and (len(scores) >= min_breadth or existing_level == "Nâng cao")
    ):
        return "Nâng cao"
    if (len(scores) >= 2 and avg >= 48.0) or (len(scores) == 1 and avg >= 60.0) or (
        existing_level in ("Trung bình", "Nâng cao") and avg >= 48.0
    ):
        return "Trung bình"
    return "Mới bắt đầu"


def _sync_learner_level_conn(
    conn: sqlite3.Connection,
    learner_id: str,
    doc_id: str,
    now: str,
    total_concepts: int | None = None,
) -> str:
    all_rows = conn.execute(
        "SELECT mastery_score FROM concept_mastery WHERE learner_id = ? AND doc_id = ? AND attempts > 0",
        (learner_id, doc_id),
    ).fetchall()
    misc_row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM misconceptions WHERE learner_id = ? AND doc_id = ? AND resolved = 0",
        (learner_id, doc_id),
    ).fetchone()
    learner_row = conn.execute(
        "SELECT overall_level FROM learners WHERE learner_id = ?", (learner_id,)
    ).fetchone()
    existing_lvl = learner_row["overall_level"] if learner_row else "Mới bắt đầu"
    active_cnt = int(misc_row["cnt"]) if misc_row else 0
    new_lvl = _compute_overall_level(
        [float(r["mastery_score"]) for r in all_rows],
        existing_lvl,
        active_misconceptions_count=active_cnt,
        total_concepts=total_concepts,
    )
    conn.execute(
        "UPDATE learners SET overall_level = ?, updated_at = ? WHERE learner_id = ?",
        (new_lvl, now, learner_id),
    )
    return new_lvl


def get_learner_dna(
    learner_id: str,
    doc_id: str = DEFAULT_DOC_ID,
    concepts_order: list[str] | None = None,
    db_path: str | Path | None = None,
) -> LearnerDNA:
    init_db(db_path)
    with _connect(db_path) as conn:
        learner = conn.execute(
            "SELECT learner_id, name, role_label, overall_level FROM learners WHERE learner_id = ?",
            (learner_id,),
        ).fetchone()
        if learner is None:
            conn.execute(
                "INSERT INTO learners (learner_id, name, role_label, overall_level, updated_at) VALUES (?, ?, ?, ?, ?)",
                (learner_id, learner_id, "Học viên", "Mới bắt đầu", _now_iso()),
            )
            learner_dict = {
                "learner_id": learner_id,
                "name": learner_id,
                "role_label": "Học viên",
                "overall_level": "Mới bắt đầu",
            }
        else:
            learner_dict = dict(learner)

        mastery_rows = conn.execute(
            """
            SELECT concept, mastery_score, bloom_level, attempts, correct, last_reviewed, next_review_due
            FROM concept_mastery
            WHERE learner_id = ? AND doc_id = ?
            """,
            (learner_id, doc_id),
        ).fetchall()

        misc_rows = conn.execute(
            """
            SELECT id, topic, description, occurrences, resolved, last_seen
            FROM misconceptions
            WHERE learner_id = ? AND doc_id = ?
            ORDER BY resolved ASC, occurrences DESC, id DESC
            """,
            (learner_id, doc_id),
        ).fetchall()

    active_misc: list[MisconceptionRecord] = []
    resolved_misc: list[MisconceptionRecord] = []
    active_topics: set[str] = set()
    for r in misc_rows:
        rec = MisconceptionRecord(
            id=r["id"],
            topic=r["topic"],
            description=r["description"],
            occurrences=r["occurrences"],
            resolved=bool(r["resolved"]),
            last_seen=r["last_seen"],
        )
        if rec.resolved:
            resolved_misc.append(rec)
        else:
            active_misc.append(rec)
            active_topics.add(r["topic"].strip().lower())

    by_concept = {r["concept"]: dict(r) for r in mastery_rows}
    ordered_names: list[str] = []
    if concepts_order:
        ordered_names.extend(concepts_order)
    for cname in by_concept:
        if cname not in ordered_names:
            ordered_names.append(cname)

    concepts: list[ConceptMastery] = []
    for cname in ordered_names:
        row = by_concept.get(cname)
        has_active_misc = cname.strip().lower() in active_topics
        if row is None:
            concepts.append(
                ConceptMastery(
                    concept=cname,
                    mastery_score=0.0,
                    bloom_level="Nhớ",
                    attempts=0,
                    correct=0,
                    status="misconception" if has_active_misc else "unseen",
                )
            )
            continue

        score = float(row["mastery_score"])
        due = _is_due(row["next_review_due"])
        if has_active_misc or (row["attempts"] > 0 and score < 40.0):
            status = "misconception"
        elif score >= 75.0 and not due:
            status = "mastered"
        elif row["attempts"] > 0:
            status = "learning"
        else:
            status = "unseen"

        concepts.append(
            ConceptMastery(
                concept=cname,
                mastery_score=round(score, 1),
                bloom_level=row["bloom_level"] or "Nhớ",
                attempts=int(row["attempts"]),
                correct=int(row["correct"]),
                last_reviewed=row["last_reviewed"],
                next_review_due=row["next_review_due"],
                due_for_review=due,
                status=status,
            )
        )

    assessed_scores = [c.mastery_score for c in concepts if c.attempts > 0]
    mastery_avg = round(sum(assessed_scores) / len(assessed_scores), 1) if assessed_scores else 0.0
    effective_level = _compute_overall_level(
        assessed_scores,
        learner_dict["overall_level"],
        active_misconceptions_count=len(active_misc),
        total_concepts=len(concepts) if concepts else None,
    )

    return LearnerDNA(
        learner_id=learner_dict["learner_id"],
        name=learner_dict["name"],
        role_label=learner_dict["role_label"],
        overall_level=effective_level,
        mastery_avg=mastery_avg,
        concepts=concepts,
        active_misconceptions=active_misc,
        resolved_misconceptions=resolved_misc,
    )


def update_concept_mastery(
    learner_id: str,
    doc_id: str,
    concept: str,
    is_correct: bool | None = None,
    score_override: float | None = None,
    bloom_level: str = "Hiểu",
    confidence: str = "Bình thường",
    total_concepts: int | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Cập nhật điểm thành thạo theo cấp độ Bloom, mức tự tin và lịch ôn Spaced Repetition."""
    init_db(db_path)
    now = _now_iso()
    concept = concept.strip() or "Kiến thức chung"
    cap = _BLOOM_CAP.get(bloom_level, 84.0)

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT mastery_score, bloom_level, attempts, correct
            FROM concept_mastery
            WHERE learner_id = ? AND doc_id = ? AND concept = ?
            """,
            (learner_id, doc_id, concept),
        ).fetchone()

        prev_score = float(row["mastery_score"]) if row else 30.0
        prev_bloom = row["bloom_level"] if row else "Nhớ"
        attempts = int(row["attempts"]) if row else 0
        correct = int(row["correct"]) if row else 0

        attempts += 1
        effective_correct = bool(is_correct)
        if score_override is not None:
            target = max(0.0, min(100.0, float(score_override)))
            effective_correct = target >= 60.0 and (is_correct is not False)
            if attempts == 1:
                new_score = target
            elif target >= 80.0 and prev_score < 70.0:
                # Bứt phá khi học viên thực sự làm chủ bản chất (>=80) sau các lần hổng trước đó.
                new_score = max(0.35 * prev_score + 0.65 * target, target - 10.0)
            else:
                new_score = 0.35 * prev_score + 0.65 * target
            if (
                bloom_level in ("Phân tích", "Đánh giá", "Sáng tạo")
                and target >= 95.0
            ):
                # Làm chủ bậc tư duy cao với điểm gần tuyệt đối: công nhận thẳng 100%.
                new_score = 100.0
            if target >= 70.0 and (is_correct is not False):
                correct += 1
        else:
            if is_correct:
                correct += 1
                if confidence == "Đoán mò":
                    gain = 6.0  # Đúng do đoán mò chỉ tăng nhẹ, chưa công nhận vững
                else:
                    gain = 18.0 if _BLOOM_RANK.get(bloom_level, 2) >= 2 else 12.0
                new_score = min(cap, prev_score + gain)
            else:
                penalty = 22.0 if confidence == "Chắc chắn" else 16.0
                new_score = max(10.0, prev_score - penalty)

        # Nâng cấp Bloom đạt được nếu trả lời đúng thực chất (không phải đoán mò).
        new_bloom = prev_bloom
        if (score_override is not None and float(score_override) >= 75.0 and effective_correct) or (
            score_override is None and is_correct and confidence != "Đoán mò"
        ):
            if _BLOOM_RANK.get(bloom_level, 1) > _BLOOM_RANK.get(prev_bloom, 1):
                new_bloom = bloom_level

        # Khoảng cách ôn tập lặp lại ngắt quãng (Spaced Repetition).
        if (score_override is not None and not effective_correct) or (confidence == "Đoán mò" and is_correct):
            due_days = 0
        elif new_score >= 82.0:
            due_days = 5
        elif new_score >= 60.0:
            due_days = 2
        else:
            due_days = 0
        next_due = _due_iso(due_days)

        conn.execute(
            """
            INSERT INTO concept_mastery
                (learner_id, doc_id, concept, mastery_score, bloom_level, attempts, correct, last_reviewed, next_review_due)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id, doc_id, concept) DO UPDATE SET
                mastery_score = excluded.mastery_score,
                bloom_level = excluded.bloom_level,
                attempts = excluded.attempts,
                correct = excluded.correct,
                last_reviewed = excluded.last_reviewed,
                next_review_due = excluded.next_review_due
            """,
            (learner_id, doc_id, concept, round(new_score, 1), new_bloom, attempts, correct, now, next_due),
        )

        _sync_learner_level_conn(conn, learner_id, doc_id, now, total_concepts=total_concepts)


def record_misconception(
    learner_id: str,
    doc_id: str,
    topic: str,
    description: str,
    db_path: str | Path | None = None,
) -> None:
    desc = description.strip()
    if not desc or desc.lower() == "chính xác":
        return
    init_db(db_path)
    now = _now_iso()
    norm_new = " ".join(desc.lower().replace("'", "").replace('"', "").rstrip(".").split())
    with _connect(db_path) as conn:
        candidates = conn.execute(
            """
            SELECT id, description, occurrences FROM misconceptions
            WHERE learner_id = ? AND doc_id = ? AND lower(topic) = lower(?) AND resolved = 0
            """,
            (learner_id, doc_id, topic),
        ).fetchall()
        matched_id = None
        matched_occ = 0
        for row in candidates:
            norm_old = " ".join(
                row["description"].lower().replace("'", "").replace('"', "").rstrip(".").split()
            )
            if norm_old == norm_new or (len(norm_new) > 15 and (norm_new in norm_old or norm_old in norm_new)):
                matched_id = row["id"]
                matched_occ = int(row["occurrences"])
                break
        if matched_id is not None:
            conn.execute(
                "UPDATE misconceptions SET occurrences = ?, last_seen = ? WHERE id = ?",
                (matched_occ + 1, now, matched_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO misconceptions (learner_id, doc_id, topic, description, occurrences, resolved, last_seen)
                VALUES (?, ?, ?, ?, 1, 0, ?)
                """,
                (learner_id, doc_id, topic, desc, now),
            )
        _sync_learner_level_conn(conn, learner_id, doc_id, now)


def resolve_misconception(
    learner_id: str,
    doc_id: str,
    topic: str,
    db_path: str | Path | None = None,
) -> int:
    """Đánh dấu đã khắc phục lỗi tư duy của một chủ đề khi học viên vượt qua kiểm tra."""
    init_db(db_path)
    now = _now_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE misconceptions SET resolved = 1, last_seen = ?
            WHERE learner_id = ? AND doc_id = ? AND lower(topic) = lower(?) AND resolved = 0
            """,
            (now, learner_id, doc_id, topic),
        )
        if cur.rowcount > 0:
            _sync_learner_level_conn(conn, learner_id, doc_id, now)
        return cur.rowcount


def save_quiz_attempt(
    learner_id: str,
    doc_id: str,
    model: str,
    difficulty: str,
    result: QuizResult,
    db_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO quiz_attempts
                (learner_id, doc_id, created_at, model, difficulty, score, total, weak_topics_json, misconceptions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id,
                doc_id,
                _now_iso(),
                model,
                difficulty,
                result.score,
                result.total,
                json.dumps(result.weak_topics, ensure_ascii=False),
                json.dumps(result.detected_misconceptions, ensure_ascii=False),
            ),
        )


def recent_quiz_attempts(
    learner_id: str,
    doc_id: str | None = None,
    limit: int = 10,
    db_path: str | Path | None = None,
) -> list[dict]:
    init_db(db_path)
    with _connect(db_path) as conn:
        if doc_id:
            rows = conn.execute(
                """
                SELECT id, created_at, model, difficulty, score, total, weak_topics_json, misconceptions_json
                FROM quiz_attempts WHERE learner_id = ? AND doc_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (learner_id, doc_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, created_at, model, difficulty, score, total, weak_topics_json, misconceptions_json
                FROM quiz_attempts WHERE learner_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (learner_id, limit),
            ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["weak_topics"] = json.loads(item.pop("weak_topics_json", "[]"))
        item["misconceptions"] = json.loads(item.pop("misconceptions_json", "[]"))
        out.append(item)
    return out


def log_llm_call(
    model: str,
    purpose: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: int = 0,
    cost_usd: float | None = None,
    is_demo: bool = False,
    db_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    cost = (
        round(cost_usd, 6)
        if cost_usd is not None
        else config.estimate_cost_usd(model, tokens_in, tokens_out)
    )
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO llm_calls (created_at, model, purpose, tokens_in, tokens_out, latency_ms, cost_usd, is_demo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (_now_iso(), model, purpose, int(tokens_in), int(tokens_out), int(latency_ms), cost, int(is_demo)),
        )


def get_session_cost_usd(db_path: str | Path | None = None, since: str | None = None) -> float:
    """Chi phí LLM thật của PHIÊN hiện tại (từ lúc tiến trình khởi động, hoặc từ `since` nếu truyền vào)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total_cost FROM llm_calls WHERE is_demo = 0 AND created_at >= ?",
            (since or _PROCESS_START,),
        ).fetchone()
    return round(float(row["total_cost"]), 6)


def get_telemetry_summary(db_path: str | Path | None = None, limit: int = 20) -> dict:
    init_db(db_path)
    with _connect(db_path) as conn:
        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS total_calls,
                COALESCE(SUM(tokens_in), 0) AS total_tokens_in,
                COALESCE(SUM(tokens_out), 0) AS total_tokens_out,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                COALESCE(SUM(cost_usd), 0) AS total_cost_usd
            FROM llm_calls
            """
        ).fetchone()
        by_model = conn.execute(
            """
            SELECT model, COUNT(*) AS calls,
                   COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                   COALESCE(SUM(tokens_in + tokens_out), 0) AS total_tokens,
                   COALESCE(SUM(cost_usd), 0) AS cost_usd
            FROM llm_calls
            GROUP BY model
            ORDER BY calls DESC
            """
        ).fetchall()
        recent = conn.execute(
            """
            SELECT created_at, model, purpose, tokens_in, tokens_out, latency_ms, cost_usd, is_demo
            FROM llm_calls ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {
        "total_calls": int(agg["total_calls"]),
        "total_tokens_in": int(agg["total_tokens_in"]),
        "total_tokens_out": int(agg["total_tokens_out"]),
        "avg_latency_ms": round(float(agg["avg_latency_ms"]), 1),
        "total_cost_usd": round(float(agg["total_cost_usd"]), 6),
        "by_model": [dict(r) for r in by_model],
        "recent_calls": [dict(r) for r in recent],
    }


def reset_learner_profile(
    learner_id: str,
    doc_id: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Đưa hồ sơ học viên về trạng thái sạch hoặc khôi phục mẫu mặc định nếu là tài khoản demo."""
    init_db(db_path)
    now = _now_iso()
    with _connect(db_path) as conn:
        if doc_id:
            conn.execute("DELETE FROM concept_mastery WHERE learner_id = ? AND doc_id = ?", (learner_id, doc_id))
            conn.execute("DELETE FROM misconceptions WHERE learner_id = ? AND doc_id = ?", (learner_id, doc_id))
            conn.execute("DELETE FROM quiz_attempts WHERE learner_id = ? AND doc_id = ?", (learner_id, doc_id))
            conn.execute("DELETE FROM confidence_stats WHERE learner_id = ? AND doc_id = ?", (learner_id, doc_id))
        else:
            conn.execute("DELETE FROM concept_mastery WHERE learner_id = ?", (learner_id,))
            conn.execute("DELETE FROM misconceptions WHERE learner_id = ?", (learner_id,))
            conn.execute("DELETE FROM quiz_attempts WHERE learner_id = ?", (learner_id,))
            conn.execute("DELETE FROM confidence_stats WHERE learner_id = ?", (learner_id,))
        conn.execute(
            "UPDATE learners SET overall_level = ?, updated_at = ? WHERE learner_id = ?",
            ("Nâng cao" if learner_id == "senior_demo" else "Mới bắt đầu", now, learner_id),
        )


def restore_demo_profiles(db_path: str | Path | None = None) -> None:
    """Khôi phục dữ liệu mẫu ban đầu cho các hồ sơ demo."""
    init_db(db_path)
    with _connect(db_path) as conn:
        for table in ("concept_mastery", "misconceptions", "quiz_attempts", "confidence_stats", "learners"):
            conn.execute(f"DELETE FROM {table} WHERE learner_id IN ('me', 'intern_demo', 'senior_demo')")
        _seed_default_learners(conn)


def update_confidence_stats(
    learner_id: str,
    doc_id: str,
    solid: int = 0,
    lucky: int = 0,
    aware: int = 0,
    blind: int = 0,
    db_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO confidence_stats (learner_id, doc_id, solid_correct, lucky_guess, aware_gap, blind_spot)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id, doc_id) DO UPDATE SET
                solid_correct = solid_correct + excluded.solid_correct,
                lucky_guess = lucky_guess + excluded.lucky_guess,
                aware_gap = aware_gap + excluded.aware_gap,
                blind_spot = blind_spot + excluded.blind_spot
            """,
            (learner_id, doc_id, int(solid), int(lucky), int(aware), int(blind)),
        )


def get_confidence_stats(
    learner_id: str,
    doc_id: str = DEFAULT_DOC_ID,
    db_path: str | Path | None = None,
) -> dict:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT solid_correct, lucky_guess, aware_gap, blind_spot
            FROM confidence_stats WHERE learner_id = ? AND doc_id = ?
            """,
            (learner_id, doc_id),
        ).fetchone()
    if row is None:
        return {"solid_correct": 0, "lucky_guess": 0, "aware_gap": 0, "blind_spot": 0, "total": 0}
    data = dict(row)
    data["total"] = data["solid_correct"] + data["lucky_guess"] + data["aware_gap"] + data["blind_spot"]
    return data


def save_learner_document(
    learner_id: str,
    doc_id: str,
    title: str,
    doc_text: str,
    sections: list[dict],
    concept_map: DocumentConceptMap,
    db_path: str | Path | None = None,
) -> None:
    """Lưu hoặc cập nhật tài liệu đã tải lên của một hồ sơ người học."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO learner_documents
                (learner_id, doc_id, title, doc_text, sections_json, concept_map_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(learner_id, doc_id) DO UPDATE SET
                title = excluded.title,
                doc_text = excluded.doc_text,
                sections_json = excluded.sections_json,
                concept_map_json = excluded.concept_map_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                learner_id,
                doc_id,
                title,
                doc_text,
                json.dumps(sections, ensure_ascii=False),
                concept_map.model_dump_json(),
            ),
        )


def list_learner_documents(
    learner_id: str,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Danh sách tài liệu của một hồ sơ người học, mới nhất trước."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT doc_id, title, updated_at FROM learner_documents
            WHERE learner_id = ?
            ORDER BY updated_at DESC, rowid DESC
            """,
            (learner_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def load_learner_document(
    learner_id: str,
    doc_id: str,
    db_path: str | Path | None = None,
) -> dict | None:
    """Tải đầy đủ nội dung một tài liệu đã lưu, trả về None nếu không tìm thấy."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT doc_id, title, doc_text, sections_json, concept_map_json
            FROM learner_documents
            WHERE learner_id = ? AND doc_id = ?
            """,
            (learner_id, doc_id),
        ).fetchone()
    if row is None:
        return None
    return {
        "doc_id": row["doc_id"],
        "title": row["title"],
        "doc_text": row["doc_text"],
        "sections": json.loads(row["sections_json"]),
        "concept_map": DocumentConceptMap.model_validate_json(row["concept_map_json"]),
    }


def append_chat_message(
    learner_id: str,
    doc_id: str,
    role: str,
    content: str,
    turn: CoachTurnResult | dict | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Ghi một lượt hội thoại của chatbot; kèm JSON của CoachTurnResult khi có."""
    init_db(db_path)
    turn_json = ""
    if turn is not None:
        if isinstance(turn, CoachTurnResult):
            turn_json = turn.model_dump_json()
        else:
            turn_json = json.dumps(turn, ensure_ascii=False)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (learner_id, doc_id, role, content, turn_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (learner_id, doc_id, role, content, turn_json, _now_iso()),
        )


def load_chat_history(
    learner_id: str,
    doc_id: str,
    limit: int = 40,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Lịch sử hội thoại theo thứ tự thời gian tăng dần (cũ → mới), tối đa `limit` lượt gần nhất."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, turn_json, created_at
            FROM chat_messages
            WHERE learner_id = ? AND doc_id = ?
            ORDER BY id DESC LIMIT ?
            """,
            (learner_id, doc_id, limit),
        ).fetchall()
    out: list[dict] = []
    for r in reversed(rows):
        item = dict(r)
        raw = (item.pop("turn_json", "") or "").strip()
        turn = None
        if raw:
            try:
                turn = CoachTurnResult.model_validate_json(raw)
            except ValueError:
                turn = None
        item["turn"] = turn
        out.append(item)
    return out


def clear_chat_history(
    learner_id: str,
    doc_id: str,
    db_path: str | Path | None = None,
) -> None:
    """Xóa toàn bộ lịch sử hội thoại của một người học trên một tài liệu."""
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE learner_id = ? AND doc_id = ?",
            (learner_id, doc_id),
        )
