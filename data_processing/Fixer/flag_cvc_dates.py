"""
Flag CVC chunks with unreliable document_date for ISS exclusion.

Background
----------
CVC chunks come from Google News RSS, where the `published` field reflects
Google News indexing time rather than the article's true publication date.
This produces clustering: many unrelated chunks share the same date because
they were indexed by Google News on the same day. Statistical signature:
a single date appears in 20-60+ chunks instead of the expected 2-5/day.

Effect on ISS scoring: these chunks get max recency weight (because their
"document_date" looks recent), polluting the recency dimension. The right
mitigation is to mark them as not-usable for ISS while keeping them
available for GMS (which is recency-independent capability matching).

Detection heuristic
-------------------
A CVC chunk is flagged unreliable if its document_date appears >= THRESHOLD
times across all CVC chunks. Default THRESHOLD=15.

Field design (consistent with existing is_usable_for_iss/is_usable_for_gms)
---------------------------------------------------------------------------
Each CVC chunk gets the following added fields:

  is_usable_for_iss:    bool
    True  = date verified, ISS may use this chunk normally
    False = date unreliable, ISS algorithm must skip this chunk

  is_usable_for_gms:    bool
    Always True for CVC (capability matching is recency-independent)

  iss_exclusion_reason: string (only when is_usable_for_iss=False)
    "unreliable_document_date_clustering"

Non-CVC chunks pass through unchanged.

Downstream ISS algorithm change (one line)
------------------------------------------
At the entry of the ISS pipeline, filter chunks:

    chunks = [c for c in chunks if c.get("is_usable_for_iss") != False]

Usage
-----
    python3 flag_cvc_dates.py \\
        --input  all_chunks_for_labeling_final.jsonl \\
        --output all_chunks_for_labeling_final_v2.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path


EXCLUSION_REASON = "unreliable_document_date_clustering"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Input JSONL (all_chunks_for_labeling_final.jsonl)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL with usability flags added")
    p.add_argument("--threshold", type=int, default=15,
                   help="Flag CVC chunks if their date appears >= N times. Default 15.")
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ===== Pass 1: count CVC date frequency =====
    print(f"Pass 1: scanning CVC date distribution...")
    cvc_date_counts = Counter()
    cvc_total = 0
    with args.input.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("source_type") == "CVC_from_google_news":
                date = chunk.get("document_date", "")[:10]
                if date:
                    cvc_date_counts[date] += 1
                    cvc_total += 1
    print(f"  CVC chunks total: {cvc_total:,}")
    print(f"  Distinct CVC dates: {len(cvc_date_counts):,}")

    suspect_dates = {d for d, n in cvc_date_counts.items() if n >= args.threshold}
    will_exclude = sum(n for d, n in cvc_date_counts.items() if n >= args.threshold)
    print(f"  Dates appearing >= {args.threshold} times: {len(suspect_dates):,}")
    print(f"  CVC chunks that will be excluded from ISS: {will_exclude:,} "
          f"({will_exclude/cvc_total*100:.1f}%)")

    # ===== Pass 2: rewrite JSONL with usability flags =====
    print(f"\nPass 2: writing flagged dataset...")
    stats = {
        "total_input": 0,
        "cvc_total": 0,
        "cvc_excluded_from_iss": 0,
        "cvc_usable_for_iss": 0,
        "non_cvc": 0,
    }

    with args.input.open("r") as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            stats["total_input"] += 1

            if chunk.get("source_type") == "CVC_from_google_news":
                stats["cvc_total"] += 1
                date = chunk.get("document_date", "")[:10]

                if date in suspect_dates:
                    chunk["is_usable_for_iss"] = False
                    chunk["is_usable_for_gms"] = True  # GMS is recency-independent
                    chunk["iss_exclusion_reason"] = EXCLUSION_REASON
                    stats["cvc_excluded_from_iss"] += 1
                else:
                    chunk["is_usable_for_iss"] = True
                    chunk["is_usable_for_gms"] = True
                    stats["cvc_usable_for_iss"] += 1
            else:
                stats["non_cvc"] += 1

            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"CVC date flagging - Summary")
    print(f"{'='*60}")
    print(f"Total chunks processed:    {stats['total_input']:,}")
    print(f"  Non-CVC (untouched):     {stats['non_cvc']:,}")
    print(f"  CVC total:               {stats['cvc_total']:,}")
    print(f"    is_usable_for_iss=False: {stats['cvc_excluded_from_iss']:,}  "
          f"({stats['cvc_excluded_from_iss']/stats['cvc_total']*100:.1f}%)")
    print(f"    is_usable_for_iss=True:  {stats['cvc_usable_for_iss']:,}  "
          f"({stats['cvc_usable_for_iss']/stats['cvc_total']*100:.1f}%)")
    print(f"\n  All CVC chunks have is_usable_for_gms=True "
          f"(GMS is recency-independent).")
    print(f"\nThreshold used:            >= {args.threshold} chunks/day")
    print(f"\nOutput: {args.output}")
    print(f"\nTop 10 dates that triggered ISS exclusion:")
    for d, n in sorted(cvc_date_counts.items(), key=lambda x: -x[1])[:10]:
        if d in suspect_dates:
            print(f"  {d}: {n} chunks")

    print(f"\nNext step: tell ISS owner to add ONE line at ISS pipeline entry:")
    print(f'  chunks = [c for c in chunks if c.get("is_usable_for_iss") != False]')


if __name__ == "__main__":
    main()
