"""
Chunk extracted 10-K sections (Item 1 Business and Item 7 MD&A) into
subsection-aware chunks for downstream capability labeling.

Strategy
--------
1. For each section .txt, detect subsection headers using conservative rules
   that avoid false positives from bullet markers, table cells, sentence
   fragments, and page-header noise.
2. Build a list of (subsection_header, body_text) pairs.
3. Within each subsection, apply a sliding window to chunk into ~2000 char
   pieces with sentence-boundary respect and 200-char overlap.
4. Emit JSONL with the same field naming as your existing chunks
   (chunk_id, source_type, company, document_date, chunk_text, etc.) so that
   uploads to your existing Pinecone namespace work without schema changes.

The output schema does NOT include theme/event_type/strength because those
are ISS-specific labels; 10-K capability data is labeled separately for GMS.
We mark every chunk with is_usable_for_iss=false, is_usable_for_gms=true,
and include subsection_path so the labeling step has the structural context.

Usage
-----
    # Default: read from sec_10k_sections/, use filing_info.json from
    # sec_10k_data/<company>/filing_info.json
    python3 chunk_10k.py

    # Custom paths
    python3 chunk_10k.py --sections-dir my_sections \
                        --filings-dir my_filings \
                        --output-dir my_chunks
"""

import argparse
import json
import re
from pathlib import Path

# -----------------------------------------------------------------------------
# Subsection header detection
# -----------------------------------------------------------------------------

# Lines that look like headers but are actually noise. Match on the stripped
# line. Items here are dropped from any consideration as headers OR body.
NOISE_LINES = {
    # Microsoft drop-cap fragments
    "USINESS", "ANAGEMENT'S DISCUSSION AND ANALYSIS",
    # Bullet markers
    "•", "◦", "○", "*", "-", "—",
}
NOISE_PATTERNS = [
    re.compile(r"^Item\s+\d+[A-Z]?\s*$"),     # "Item 1", "Item 7" page footer
    re.compile(r"^ITEM\s+\d+[A-Z]?\.?\s*$"),  # "ITEM 1." standalone
    re.compile(r"^PART\s+I+\s*$", re.IGNORECASE),
    re.compile(r"^Note\s+\d+\s*$"),           # "Note 15" cross-reference
]

def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False  # blank handled separately
    if s in NOISE_LINES:
        return True
    if any(p.match(s) for p in NOISE_PATTERNS):
        return True
    return False


def is_likely_subsection_header(line: str, lines: list, idx: int) -> bool:
    """
    Conservative subsection header detection. We prefer to miss real headers
    rather than emit false positives, since false positives break chunking.
    """
    s = line.strip()
    if not s:
        return False
    # Length bounds (tight: real headers are 3-70 chars)
    if len(s) < 3 or len(s) > 70:
        return False
    # Must start with an uppercase letter
    if not s[0].isalpha() or not s[0].isupper():
        return False
    # No terminal punctuation (real headers don't end in .,;:?!)
    if s[-1] in ".,;:?!":
        return False
    # No digits anywhere (excludes "As of December 31, 2025", "November 2025",
    # "Up 65%", "71.1", and most financial-table fragments)
    if re.search(r"\d", s):
        return False
    # Must be majority alphabetic
    alpha_count = sum(1 for c in s if c.isalpha())
    if alpha_count < len(s) * 0.5:
        return False
    # Exclude predefined noise patterns
    if is_noise_line(line):
        return False
    # Backward-look: previous substantive line must be a finished paragraph
    # (long enough, ends in terminal punctuation). This filters out table row
    # labels and iXBRL-split sentence fragments, which are surrounded by
    # other short fragments rather than completed paragraphs.
    prev_substantive = None
    for j in range(idx - 1, max(-1, idx - 15), -1):
        nl = lines[j].strip()
        if not nl or is_noise_line(nl):
            continue
        prev_substantive = nl
        break
    if prev_substantive is not None:
        # Allow if previous line is a long, finished paragraph
        if len(prev_substantive) < 100 or prev_substantive[-1] not in ".!?":
            return False
    # Forward-look: within next 5 non-blank lines, must find a body paragraph
    # (>= 150 chars). Disambiguates real headers from short fragments whose
    # surroundings are also short.
    seen = 0
    for j in range(idx + 1, min(idx + 12, len(lines))):
        nl = lines[j].strip()
        if not nl:
            continue
        if is_noise_line(nl):
            continue
        seen += 1
        if len(nl) >= 150:
            return True
        if seen >= 5:
            return False
    return False


def parse_subsections(section_text: str, section_title: str) -> list:
    """
    Walk the section text and split it into a list of:
        [{"subsection": str, "body": str}, ...]
    The first chunk's subsection is the section title itself, until we hit
    the first detected subsection header.
    """
    lines = section_text.split("\n")

    subsections = []
    current_header = section_title  # default until first explicit header
    current_body_lines: list = []

    def flush():
        body = "\n".join(current_body_lines).strip()
        if body:
            subsections.append({"subsection": current_header, "body": body})

    for i, line in enumerate(lines):
        if is_noise_line(line):
            continue
        if is_likely_subsection_header(line, lines, i):
            flush()
            current_header = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)
    flush()
    return subsections


# -----------------------------------------------------------------------------
# Sliding-window chunker
# -----------------------------------------------------------------------------

TARGET_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 800       # below this, merge with next subsection's chunk
MAX_CHUNK_SIZE = 2500      # hard cap; force a split before reaching this
OVERLAP_SIZE = 200         # character overlap between consecutive chunks


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n\n+")

def split_sentences(text: str) -> list:
    """Cheap sentence splitter that respects . ! ? followed by capital letter,
    plus paragraph breaks. Good enough for chunk boundaries."""
    parts = SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def sliding_window(text: str, target: int = TARGET_CHUNK_SIZE,
                   max_size: int = MAX_CHUNK_SIZE,
                   overlap: int = OVERLAP_SIZE) -> list:
    """Return a list of chunk strings. Tries to break on sentence boundaries."""
    text = text.strip()
    if len(text) <= max_size:
        return [text]

    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent) + 1  # account for the joining space
        if current_len + sent_len > target and current:
            chunks.append(" ".join(current))
            # Build overlap from the tail of the current chunk
            tail = []
            tail_len = 0
            for s in reversed(current):
                if tail_len + len(s) + 1 > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current = tail
            current_len = tail_len
        current.append(sent)
        current_len += sent_len

        # Force split if a single sentence pushes us past max
        if current_len > max_size:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current))
    return chunks


# -----------------------------------------------------------------------------
# Metadata + JSONL output
# -----------------------------------------------------------------------------

# Map our internal slug to display values
SECTION_DISPLAY = {
    "item_1_business": "Item 1 - Business",
    "item_7_mdna": "Item 7 - MD&A",
}


def build_chunks_for_section(section_key: str, section_text: str,
                              filing_info: dict, company_slug: str) -> list:
    section_title = SECTION_DISPLAY.get(section_key, section_key)
    subsections = parse_subsections(section_text, section_title)

    chunks = []
    chunk_seq = 0
    for sub_idx, sub in enumerate(subsections):
        sub_chunks = sliding_window(sub["body"])
        for c_idx, chunk_text in enumerate(sub_chunks):
            # Skip chunks that are too short and not the only one in subsection
            if len(chunk_text) < MIN_CHUNK_SIZE and len(sub_chunks) > 1:
                continue
            chunk_id = f"{company_slug}_10k_{section_key}_{sub_idx:03d}_{c_idx:02d}"
            subsection_path = f"{section_title} > {sub['subsection']}" \
                              if sub['subsection'] != section_title else section_title

            chunks.append({
                "chunk_id": chunk_id,
                "source_type": "annual_report_10k",
                "source_name": "SEC EDGAR",
                "source_title": filing_info.get("primary_document", ""),
                "source_url": filing_info.get("source_url", ""),
                "document_date": filing_info.get("report_date", ""),
                "filing_date": filing_info.get("filing_date", ""),
                "company": filing_info.get("company", company_slug.title()),
                "section": section_title,
                "subsection": sub["subsection"],
                "subsection_path": subsection_path,
                "chunk_seq": chunk_seq,
                "chunk_text": chunk_text,
                # ISS/GMS routing flags - 10-K capability data is for GMS only
                "is_usable_for_iss": False,
                "is_usable_for_gms": True,
                "chunk_kind": "capability_evidence",
            })
            chunk_seq += 1
    return chunks


def process_company(sections_dir: Path, filings_dir: Path,
                    output_dir: Path, company_slug: str) -> dict:
    print(f"\n>>> {company_slug}")
    company_sections_dir = sections_dir / company_slug
    if not company_sections_dir.is_dir():
        print(f"    No sections dir found: {company_sections_dir}, skipping")
        return None

    filing_info_path = filings_dir / company_slug / "filing_info.json"
    if not filing_info_path.exists():
        print(f"    !!! No filing_info.json at {filing_info_path}, skipping")
        return None
    filing_info = json.loads(filing_info_path.read_text())

    all_chunks = []
    section_summary = []
    for section_key in ("item_1_business", "item_7_mdna"):
        section_path = company_sections_dir / f"{section_key}.txt"
        if not section_path.exists():
            print(f"    [{section_key}] file missing, skipping")
            continue
        text = section_path.read_text()
        chunks = build_chunks_for_section(section_key, text, filing_info, company_slug)
        all_chunks.extend(chunks)
        # Stats
        n_subsections = len(set(c["subsection"] for c in chunks))
        avg_len = int(sum(len(c["chunk_text"]) for c in chunks) / max(len(chunks), 1))
        section_summary.append({
            "section": section_key,
            "n_chunks": len(chunks),
            "n_subsections": n_subsections,
            "avg_chunk_chars": avg_len,
        })
        print(f"    [{section_key}] {len(chunks)} chunks across {n_subsections} subsections "
              f"(avg {avg_len} chars)")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{company_slug}_chunks.jsonl"
    with out_path.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"    -> {out_path} ({len(all_chunks)} total chunks)")

    return {
        "company": company_slug,
        "n_chunks": len(all_chunks),
        "sections": section_summary,
        "output_file": str(out_path.relative_to(output_dir.parent)),
    }


def main():
    p = argparse.ArgumentParser(description="Chunk 10-K sections into capability-labeling-ready chunks")
    p.add_argument("--sections-dir", type=Path, default=Path("sec_10k_sections"),
                   help="Directory with extracted section .txt files (default: sec_10k_sections)")
    p.add_argument("--filings-dir", type=Path, default=Path("sec_10k_data"),
                   help="Directory with per-company filing_info.json (default: sec_10k_data)")
    p.add_argument("--output-dir", type=Path, default=Path("sec_10k_chunks"),
                   help="Output directory for JSONL chunks (default: sec_10k_chunks)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for company_dir in sorted(args.sections_dir.iterdir()):
        if not company_dir.is_dir():
            continue
        info = process_company(args.sections_dir, args.filings_dir, args.output_dir, company_dir.name)
        if info:
            manifest.append(info)

    manifest_path = args.output_dir / "chunking_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    main()
