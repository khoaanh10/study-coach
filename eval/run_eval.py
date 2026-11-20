"""Bộ đánh giá chất lượng Study Coach trên 20 câu hỏi mẫu (16 trong tài liệu + 4 câu bẫy ngoài tài liệu).

Chạy mặc định (không cần API key): python eval/run_eval.py
Chạy so sánh 2 model thật (cần API key): python eval/run_eval.py --live

Phạm vi đo: chỉ phần TRUY XUẤT (embedding cục bộ, không gọi LLM): nhận biết câu nằm trong tài liệu,
truy xuất đúng file nguồn, từ chối câu ngoài tài liệu. KHÔNG đo độ đúng của câu trả lời do LLM sinh ra.
Bộ 20 câu này đã được dùng để chỉnh các ngưỡng nên điểm 100% là điểm trên bộ đã biết; bộ giữ riêng
`eval/holdout.json` (24 câu viết tách biệt) cho ước lượng tổng quát hóa trung thực hơn.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from core.loader import load_default_samples  # noqa: E402
from core.rag import DualMemoryKnowledgeBase, LocalNGramEmbeddings  # noqa: E402


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _run_naive_baseline(items: list[dict], sections: list[dict]) -> dict:
    """Cấu hình đối chứng: cùng metadata nguồn và cùng embedding, chỉ khác chunk lớn (1400), overlap=0
    và KHÔNG có cổng từ chối ngoài tài liệu (chỉ xét ngưỡng tương đồng >= 0.05).

    Chênh lệch giữa hai cấu hình gồm hai yếu tố tách biệt: (1) cách chia chunk ảnh hưởng thế nào tới việc
    truy xuất đúng nguồn; (2) có/không có cổng từ chối. Mục (2) là bật/tắt một tính năng, không phải kết quả
    đo cải tiến độc lập.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=0)
    raw = [
        Document(page_content=sec["text"], metadata={"source": sec.get("source", ""), "page": sec.get("page", "")})
        for sec in sections
        if (sec.get("text") or "").strip()
    ]
    store = FAISS.from_documents(splitter.split_documents(raw), LocalNGramEmbeddings())

    in_scope_hits = 0
    in_scope_total = 0
    source_hits = 0
    out_refusals = 0
    out_total = 0
    passed_cnt = 0

    for q in items:
        expected = q["expected_answer"]
        src = q["source"]
        pairs = store.similarity_search_with_score(q["question"], k=3)
        top_sim = max(0.0, min(1.0, 1.0 - float(pairs[0][1]) / 2.0)) if pairs else 0.0
        pred_in_scope = top_sim >= 0.05
        retrieved_sources = [str(d.metadata.get("source", "")) for d, _ in pairs]

        if expected is not None:
            in_scope_total += 1
            if pred_in_scope:
                in_scope_hits += 1
            if pred_in_scope and src in retrieved_sources:
                source_hits += 1
                passed_cnt += 1
        else:
            out_total += 1
            if not pred_in_scope:
                out_refusals += 1
                passed_cnt += 1

    return {
        "name": "Đối chứng (Chunk=1400, Overlap=0, cùng metadata, không có cổng ngoài-tài-liệu)",
        "in_scope_recall": f"{in_scope_hits}/{in_scope_total}",
        "source_citation_accuracy": f"{source_hits}/{in_scope_total}",
        "out_of_scope_refusal": f"{out_refusals}/{out_total}",
        "overall_pass_rate": round(100.0 * passed_cnt / len(items), 1),
    }


def run_holdout_eval(kb: DualMemoryKnowledgeBase) -> dict:
    """Bộ giữ riêng: câu ngắn, diễn đạt khác, có câu không dấu và câu ngoài tài liệu gần chủ đề AI."""
    path = ROOT / "eval" / "holdout.json"
    if not path.exists():
        return {"available": False}
    items = json.loads(path.read_text(encoding="utf-8"))
    in_total = in_ok = out_total = out_ok = 0
    failures: list[dict] = []
    for q in items:
        chunks, is_in_scope = kb.retrieve(q["question"], top_k=3)
        if q["in_scope"]:
            in_total += 1
            ok = is_in_scope and (q["source"] in [c.source for c in chunks])
            in_ok += ok
        else:
            out_total += 1
            ok = not is_in_scope
            out_ok += ok
        if not ok:
            failures.append({"id": q["id"], "question": q["question"], "in_scope": q["in_scope"]})
    return {
        "available": True,
        "in_scope_correct": f"{in_ok}/{in_total}",
        "out_of_scope_refused": f"{out_ok}/{out_total}",
        "overall_pass_rate": round(100.0 * (in_ok + out_ok) / len(items), 1),
        "failures": failures,
    }


def run_offline_eval(embeddings=None, result_suffix: str = "") -> dict:
    """Đo truy xuất. embeddings=None dùng embedding cục bộ (offline); truyền embedding OpenAI để đo cấu hình Live."""
    questions_path = ROOT / "eval" / "questions.json"
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    if len(items) != 20:
        raise SystemExit(f"Kỳ vọng 20 câu hỏi trong questions.json, hiện có {len(items)}.")

    samples_dir = ROOT / "samples"
    sources_norm = {
        p.name: _norm(p.read_text(encoding="utf-8"))
        for p in samples_dir.glob("*.txt")
    }

    combined, sections, _ = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=embeddings or LocalNGramEmbeddings(), fallback_text=combined)

    schema_errors: list[str] = []
    evidence_errors: list[str] = []
    in_scope_hits = 0
    in_scope_total = 0
    source_match_hits = 0
    out_scope_refusals = 0
    out_scope_total = 0
    latencies_ms: list[float] = []
    details: list[dict] = []

    required_keys = {"id", "source", "difficulty", "question", "expected_answer", "evidence"}
    for q in items:
        missing = required_keys - set(q.keys())
        if missing:
            schema_errors.append(f"{q.get('id', '?')}: thiếu trường {missing}")
            continue

        expected = q["expected_answer"]
        ev = q["evidence"]
        src = q["source"]

        if expected is not None:
            in_scope_total += 1
            src_text = sources_norm.get(src, "")
            if not ev or _norm(ev) not in src_text:
                evidence_errors.append(f"{q['id']}: evidence không khớp nguyên văn file {src}")
        else:
            out_scope_total += 1

        t0 = time.perf_counter()
        chunks, is_in_scope = kb.retrieve(q["question"], top_k=3)
        dt_ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies_ms.append(dt_ms)

        retrieved_sources = [c.source for c in chunks]
        if expected is not None:
            if is_in_scope:
                in_scope_hits += 1
            if is_in_scope and src in retrieved_sources:
                source_match_hits += 1
            passed = is_in_scope and (src in retrieved_sources)
        else:
            if not is_in_scope:
                out_scope_refusals += 1
            passed = not is_in_scope

        details.append(
            {
                "id": q["id"],
                "type": "trong_tai_lieu" if expected is not None else "ngoai_tai_lieu",
                "difficulty": q["difficulty"],
                "is_in_scope_pred": is_in_scope,
                "top_source": retrieved_sources[0] if retrieved_sources else "",
                "top_score": chunks[0].score if chunks else 0.0,
                "passed": passed,
                "latency_ms": dt_ms,
            }
        )

    baseline = _run_naive_baseline(items, sections)
    holdout = run_holdout_eval(kb)
    overall_pass = round(100.0 * sum(1 for d in details if d["passed"]) / len(items), 1)
    after_row = {
        "name": "Hiện tại (Dual-Memory RAG: Chunk=550, Overlap=90 + Metadata + Cổng ngoài-tài-liệu)",
        "in_scope_recall": f"{in_scope_hits}/{in_scope_total}",
        "source_citation_accuracy": f"{source_match_hits}/{in_scope_total}",
        "out_of_scope_refusal": f"{out_scope_refusals}/{out_scope_total}",
        "overall_pass_rate": overall_pass,
    }

    summary = {
        "total_questions": len(items),
        "schema_errors": schema_errors,
        "evidence_errors": evidence_errors,
        "in_scope_recall": after_row["in_scope_recall"],
        "source_citation_accuracy": after_row["source_citation_accuracy"],
        "out_of_scope_refusal": after_row["out_of_scope_refusal"],
        "overall_retrieval_pass_rate": overall_pass,
        "avg_retrieval_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 2),
        "baseline_comparison": [baseline, after_row],
        "holdout": holdout,
        "notes": [
            "Chỉ đo truy xuất bằng embedding cục bộ; không đo độ đúng câu trả lời của LLM, độ trễ hay chi phí.",
            "Bộ 20 câu đã dùng để chỉnh ngưỡng: điểm trên bộ này không phải ước lượng tổng quát hóa.",
            "Dòng Đối chứng tắt cổng ngoài-tài-liệu có chủ đích nên không phản ánh một cải tiến đã đo độc lập.",
        ],
        "details": details,
    }

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"latest_eval{result_suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"baseline_vs_current{result_suffix}.json").write_text(
        json.dumps(summary["baseline_comparison"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_live_comparison() -> dict:
    """So sánh chất lượng và độ trễ giữa GPT-4o-mini và GPT-4.1-mini trên bộ mẫu."""
    if not config.has_live_api_key():
        raise SystemExit("Thiếu OPENAI_API_KEY để chạy chế độ --live.")
    from core.adaptive import extract_concept_map
    from core.coach_graph import run_coach_turn
    from core import db

    questions_path = ROOT / "eval" / "questions.json"
    items = json.loads(questions_path.read_text(encoding="utf-8"))[:6]
    combined, sections, title = load_default_samples()
    kb = DualMemoryKnowledgeBase(sections, embeddings=LocalNGramEmbeddings(), fallback_text=combined)
    cmap = extract_concept_map(combined, db.DEFAULT_DOC_ID, title, demo_mode=True)
    dna = db.get_learner_dna("intern_demo", db.DEFAULT_DOC_ID, [c.name for c in cmap.concepts])

    results = {}
    for model_name in config.MODEL_CHOICES:
        latencies = []
        costs = []
        refusal_ok = 0
        for q in items:
            turn = run_coach_turn(
                query=q["question"],
                kb=kb,
                learner_dna=dna,
                concept_map=cmap,
                doc_id=db.DEFAULT_DOC_ID,
                model=model_name,
                demo_mode=False,
            )
            latencies.append(turn.latency_ms)
            costs.append(turn.cost_usd)
            if q["expected_answer"] is None and turn.route_taken == "refuse_out_of_doc":
                refusal_ok += 1
        results[model_name] = {
            "samples": len(items),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "total_cost_usd": round(sum(costs), 6),
            "out_of_scope_refused": f"{refusal_ok}/{sum(1 for q in items if q['expected_answer'] is None)}",
        }
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_model_compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Chạy benchmark gọi API thật để so sánh model")
    parser.add_argument(
        "--openai-embeddings",
        action="store_true",
        help="Đo truy xuất với embedding OpenAI thật (cần key; tốn vài chục lượt embed, không đáng kể)",
    )
    args = parser.parse_args()

    if args.openai_embeddings:
        if not config.has_live_api_key():
            raise SystemExit("Thiếu OPENAI_API_KEY để đo với embedding OpenAI.")
        from core.rag import build_embeddings

        summary = run_offline_eval(embeddings=build_embeddings(demo_mode=False), result_suffix="_openai_embeddings")
        print("(Đã đo với embedding OpenAI)")
    else:
        summary = run_offline_eval()
    print("=== KET QUA KIEM TRA BO 20 CAU EVAL ===")
    print(f"- Loi schema: {len(summary['schema_errors'])}")
    print(f"- Loi khop bang chung nguon: {len(summary['evidence_errors'])}")
    print(f"- Nhan dien trong tai lieu: {summary['in_scope_recall']}")
    print(f"- Truy xuat dung file nguon: {summary['source_citation_accuracy']}")
    print(f"- Tu choi dung cau ngoai tai lieu: {summary['out_of_scope_refusal']}")
    print(f"- Ti le dat tong: {summary['overall_retrieval_pass_rate']}%")
    print(f"- Do tre truy xuat trung binh: {summary['avg_retrieval_latency_ms']} ms")
    ho = summary["holdout"]
    if ho.get("available"):
        print("=== BO GIU RIENG (holdout, khong dung de chinh nguong) ===")
        print(f"- Nhan dien dung cau trong tai lieu: {ho['in_scope_correct']}")
        print(f"- Tu choi dung cau ngoai tai lieu: {ho['out_of_scope_refused']}")
        print(f"- Ti le dat: {ho['overall_pass_rate']}%")
        for f in ho["failures"]:
            print("  SAI:", f["id"], f["question"])

    if summary["schema_errors"] or summary["evidence_errors"]:
        for err in summary["schema_errors"] + summary["evidence_errors"]:
            print("LOI:", err)
        return 1

    if summary["overall_retrieval_pass_rate"] < 90.0:
        for d in summary["details"]:
            if not d["passed"]:
                print("CHUA DAT:", d)
        return 1

    if args.live:
        live_res = run_live_comparison()
        print("=== LIVE MODEL COMPARISON ===", live_res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
