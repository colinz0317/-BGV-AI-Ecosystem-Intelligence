import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


GOOGLE_WEAK_ALIASES = {"android", "google play"}
NON_GOOGLE_AI_ACTORS = {
    "openai", "chatgpt", "sora", "gpt", "microsoft", "copilot",
    "nvidia", "anthropic", "claude", "meta", "llama",
}


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def subthemes(record: Dict[str, Any]) -> List[str]:
    raw = record.get("sub_theme", [])
    if isinstance(raw, list):
        return [safe_str(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return [safe_str(x) for x in parsed]
            except Exception:
                pass
        return [text]
    return [safe_str(raw)]


def read_scores(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_google_weak_alias_risk(record: Dict[str, Any]) -> bool:
    if record.get("company") != "Google" or not record.get("is_usable_for_iss"):
        return False
    text = " ".join([
        safe_str(record.get("source_title", "")),
        safe_str(record.get("evidence_span", "")),
        safe_str(record.get("reason_short", "")),
    ]).lower()
    has_weak_google = any(alias in text for alias in GOOGLE_WEAK_ALIASES)
    has_strong_google = any(alias in text for alias in ["google", "alphabet", "gemini", "deepmind", "youtube", "vertex ai"])
    has_other_actor = any(actor in text for actor in NON_GOOGLE_AI_ACTORS)
    return has_weak_google and has_other_actor and not has_strong_google


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-run audit for ISS labeling/scoring outputs.")
    parser.add_argument("--labels", required=True, help="Labeled JSONL file.")
    parser.add_argument("--scores", required=True, help="Top-level ISS score CSV.")
    parser.add_argument("--output", required=True, help="Output JSON audit report.")
    parser.add_argument("--low-event-threshold", type=int, default=10)
    parser.add_argument("--examples-per-section", type=int, default=25)
    args = parser.parse_args()

    labels_path = Path(args.labels)
    scores_path = Path(args.scores)
    output_path = Path(args.output)

    total = 0
    usable = 0
    company_counts = Counter()
    usable_company_counts = Counter()
    owned_by_other_examples = []
    google_alias_examples = []
    startup_cvc_examples = []
    cvc_counts = Counter()
    startup_counts = Counter()

    for record in load_jsonl(labels_path):
        total += 1
        company = safe_str(record.get("company", ""))
        company_counts[company] += 1
        if record.get("is_usable_for_iss"):
            usable += 1
            usable_company_counts[company] += 1

        reason = safe_str(record.get("reason_short", ""))
        if "owned by another company" in reason:
            if len(owned_by_other_examples) < args.examples_per_section:
                owned_by_other_examples.append({
                    "company": company,
                    "chunk_id": record.get("chunk_id"),
                    "source_title": record.get("source_title"),
                    "reason_short": reason,
                    "evidence_span": safe_str(record.get("evidence_span", ""))[:500],
                })

        if is_google_weak_alias_risk(record):
            if len(google_alias_examples) < args.examples_per_section:
                google_alias_examples.append({
                    "chunk_id": record.get("chunk_id"),
                    "source_title": record.get("source_title"),
                    "evidence_span": safe_str(record.get("evidence_span", ""))[:500],
                    "reason_short": reason,
                })

        tags = set(subthemes(record))
        is_cvc = record.get("source_type") == "CVC_from_google_news"
        is_startup = "startup_ecosystem" in tags
        if record.get("is_usable_for_iss") and (is_cvc or is_startup):
            if is_cvc:
                cvc_counts[company] += 1
            if is_startup:
                startup_counts[company] += 1
            if len(startup_cvc_examples) < args.examples_per_section:
                startup_cvc_examples.append({
                    "company": company,
                    "chunk_id": record.get("chunk_id"),
                    "source_type": record.get("source_type"),
                    "sub_theme": record.get("sub_theme"),
                    "source_title": record.get("source_title"),
                    "evidence_span": safe_str(record.get("evidence_span", ""))[:500],
                })

    low_sample_themes = []
    for row in read_scores(scores_path):
        try:
            n_events = int(float(row.get("num_events_after_dedup", 0)))
        except Exception:
            n_events = 0
        if n_events < args.low_event_threshold:
            low_sample_themes.append({
                "company": row.get("company"),
                "theme": row.get("theme"),
                "iis_score_0_100": row.get("iis_score_0_100"),
                "num_events_after_dedup": n_events,
            })

    report = {
        "input_files": {
            "labels": str(labels_path),
            "scores": str(scores_path),
        },
        "summary": {
            "total_records": total,
            "usable_records": usable,
            "rejected_records": total - usable,
            "company_counts": dict(company_counts),
            "usable_company_counts": dict(usable_company_counts),
        },
        "owned_by_other_company": {
            "example_count": len(owned_by_other_examples),
            "examples": owned_by_other_examples,
        },
        "low_sample_themes": {
            "threshold": args.low_event_threshold,
            "count": len(low_sample_themes),
            "themes": low_sample_themes,
        },
        "google_alias_risk": {
            "example_count": len(google_alias_examples),
            "examples": google_alias_examples,
        },
        "startup_ecosystem_and_cvc": {
            "cvc_usable_counts": dict(cvc_counts),
            "startup_ecosystem_usable_counts": dict(startup_counts),
            "examples": startup_cvc_examples,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"owned_by_other_examples={len(owned_by_other_examples)}")
    print(f"low_sample_themes={len(low_sample_themes)}")
    print(f"google_alias_risk_examples={len(google_alias_examples)}")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
