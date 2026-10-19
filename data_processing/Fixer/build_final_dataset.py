"""
Build the final labeling-ready dataset.

Combines three operations into one pass:
  1. Apply Fix 1: convert fiscal-period document_dates ("2025-Q3", "2024-FY")
     to ISO dates per company fiscal calendar. Affects sec_filing and
     earning_call chunks.
  2. Drop ALL existing announcement chunks (source_type == "official_website_announcement").
     This is the merge step - we're replacing them entirely. The 23
     profile-page noise chunks Fix 2 would have caught are dropped here too.
  3. Append the 35,487 new announcement chunks from ann_chunks.csv (the
     output of running v3 scraper -> patched cleaner -> chunker).

Inputs
------
    --base       Original 60K labeling-ready JSONL
                 (all_chunks_for_labeling.jsonl)
    --new        New announcement chunks CSV from re-run pipeline
                 (ann_chunks.csv)
    --output     Final dataset JSONL ready for labeling

Output
------
A single JSONL with:
  - All non-announcement chunks from base (with Fix 1 applied to financial chunks)
  - All announcement chunks from new

Plus a brief audit log printed to stdout.

Usage
-----
    python3 build_final_dataset.py \\
        --base   all_chunks_for_labeling.jsonl \\
        --new    ann_chunks.csv \\
        --output all_chunks_for_labeling_final.jsonl
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# FISCAL CALENDAR (same logic as fix_chunks_dates.py)
# ============================================================

FISCAL_PATTERN = re.compile(r"^(\d{4})-(Q[1-4]|FY)$")
ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


def fiscal_period_to_iso_date(company: str, fiscal_year: int, period: str) -> str | None:
    """Map (company, FYxxxx, period) to the ISO date marking that period's end."""
    p = period.upper()

    if company == "Google":
        # Calendar-year aligned
        return {"Q1": f"{fiscal_year:04d}-03-31", "Q2": f"{fiscal_year:04d}-06-30",
                "Q3": f"{fiscal_year:04d}-09-30", "Q4": f"{fiscal_year:04d}-12-31",
                "FY": f"{fiscal_year:04d}-12-31"}.get(p)

    if company == "Microsoft":
        # FY ends June 30 of fiscal_year
        prev = fiscal_year - 1
        return {"Q1": f"{prev:04d}-09-30", "Q2": f"{prev:04d}-12-31",
                "Q3": f"{fiscal_year:04d}-03-31", "Q4": f"{fiscal_year:04d}-06-30",
                "FY": f"{fiscal_year:04d}-06-30"}.get(p)

    if company == "NVIDIA":
        # FY ends late January of fiscal_year (approximated to month-end)
        prev = fiscal_year - 1
        return {"Q1": f"{prev:04d}-04-30", "Q2": f"{prev:04d}-07-31",
                "Q3": f"{prev:04d}-10-31", "Q4": f"{fiscal_year:04d}-01-31",
                "FY": f"{fiscal_year:04d}-01-31"}.get(p)

    return None


def fix_fiscal_period(chunk: dict) -> bool:
    """Apply Fix 1 to a single chunk in-place. Return True if modified."""
    doc_date = chunk.get("document_date", "") or ""
    if ISO_PATTERN.match(doc_date):
        return False
    m = FISCAL_PATTERN.match(doc_date.strip())
    if not m:
        return False
    iso = fiscal_period_to_iso_date(chunk.get("company", ""), int(m.group(1)), m.group(2))
    if iso is None:
        return False
    chunk["original_document_date"] = doc_date
    chunk["document_date"] = iso
    chunk["document_date_source"] = "fiscal_period_mapped"
    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def build(base_path: Path, new_path: Path, output_path: Path) -> dict:
    stats = {
        "base_total": 0,
        "base_kept_non_announcement": 0,
        "base_dropped_announcement": 0,
        "fiscal_fixed": 0,
        "new_added": 0,
        "by_source_kept": Counter(),
        "by_source_dropped": Counter(),
    }

    with output_path.open("w") as fout:

        # ===== Pass 1: read base, apply Fix 1, drop announcements, write rest =====
        with base_path.open("r") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                stats["base_total"] += 1
                source_type = chunk.get("source_type", "")

                # Drop all announcement chunks - they will be replaced
                if source_type == "official_website_announcement":
                    stats["base_dropped_announcement"] += 1
                    stats["by_source_dropped"][source_type] += 1
                    continue

                # Apply Fix 1 to financial chunks (sec_filing, earning_call)
                if fix_fiscal_period(chunk):
                    stats["fiscal_fixed"] += 1

                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                stats["base_kept_non_announcement"] += 1
                stats["by_source_kept"][source_type] += 1

        # ===== Pass 2: read new announcement CSV, write to same output =====
        with new_path.open("r", newline="") as fcsv:
            reader = csv.DictReader(fcsv)
            for row in reader:
                # CSV row already has the same field names as JSONL chunks.
                # Coerce to dict, write as one JSONL line.
                # Strip empty strings to None where appropriate? Not necessary.
                chunk = dict(row)
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                stats["new_added"] += 1
                stats["by_source_kept"][chunk.get("source_type", "")] += 1

    return stats


def print_summary(stats: dict, output_path: Path) -> None:
    final_total = stats["base_kept_non_announcement"] + stats["new_added"]
    print("=" * 60)
    print("Final dataset build - Summary")
    print("=" * 60)
    print(f"Base input:                       {stats['base_total']:>7,}")
    print(f"  Kept (non-announcement):        {stats['base_kept_non_announcement']:>7,}")
    print(f"  Dropped (old announcements):    {stats['base_dropped_announcement']:>7,}")
    print(f"  Fiscal periods mapped (Fix 1):  {stats['fiscal_fixed']:>7,}")
    print()
    print(f"New announcement chunks added:    {stats['new_added']:>7,}")
    print()
    print(f"FINAL OUTPUT:                     {final_total:>7,} chunks")
    print()
    print("Final source_type breakdown:")
    for st, n in sorted(stats["by_source_kept"].items(), key=lambda x: -x[1]):
        print(f"  {st:35} {n:>7,}")
    print()
    print(f"Output: {output_path}")


def main():
    p = argparse.ArgumentParser(description="Build final labeling-ready dataset")
    p.add_argument("--base", type=Path, required=True,
                   help="Original JSONL (e.g. all_chunks_for_labeling.jsonl)")
    p.add_argument("--new", type=Path, required=True,
                   help="New announcement chunks CSV (e.g. ann_chunks.csv)")
    p.add_argument("--output", type=Path, required=True,
                   help="Final JSONL ready for labeling pipeline")
    args = p.parse_args()

    for path, label in [(args.base, "base"), (args.new, "new")]:
        if not path.exists():
            raise SystemExit(f"ERROR: {label} input not found: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats = build(args.base, args.new, args.output)
    print_summary(stats, args.output)


if __name__ == "__main__":
    main()
