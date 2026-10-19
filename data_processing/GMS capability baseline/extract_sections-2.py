"""
Extract Item 1 (Business) and Item 7 (MD&A) from a 10-K HTML filing.
Designed to handle Workiva-generated iXBRL files from Microsoft, NVIDIA, and Alphabet.

Strategy
--------
Three companies, three different header conventions:
  - Alphabet:  ALL CAPS, "ITEM 1.\nBUSINESS"      (multi-line)
  - Microsoft: ALL CAPS, "ITEM 1. B\nUSINESS..."  (drop cap splits "B" from "USINESS")
  - NVIDIA:    Title Case, "Item 1. Business"    (single line)

To handle all three, we:
  1. Build a whitespace-stripped version of the text. This collapses every
     possible split or line break, so "BUSINESS" written as "B\nUSINESS" or
     "ITEM 1.\nBUSINESS" becomes the same compact form.
  2. Search on the compact text with simple patterns.
  3. Map the compact-text match position back to the original-text position
     using an index map built during step 1.
  4. To distinguish a real section header from a Table of Contents entry,
     check the original-text region following the match: TOC entries are
     followed by short lines and page numbers; real headers are followed
     by substantial paragraph content.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# -----------------------------------------------------------------------------
# Section definitions
# -----------------------------------------------------------------------------
# For each section we define start and end markers as (compact_pattern, ...).
# Compact patterns are matched against the whitespace-stripped text.
# Try the ALL CAPS pattern first (Alphabet, Microsoft); fall back to Title Case (NVIDIA).
# Patterns are case-sensitive on purpose - that's how we tell ALL CAPS apart.

SECTIONS = {
    "item_1_business": {
        "title": "Item 1. Business",
        "start_patterns": [
            re.compile(r"ITEM1\.BUSINESS"),   # ALL CAPS form
            re.compile(r"Item1\.Business"),   # Title Case form
        ],
        "end_patterns": [
            re.compile(r"ITEM1A\.RISKFACTORS"),
            re.compile(r"Item1A\.RiskFactors"),
        ],
    },
    "item_7_mdna": {
        "title": "Item 7. Management's Discussion and Analysis",
        "start_patterns": [
            # ALL CAPS variants (with curly or straight apostrophe, or no apostrophe)
            re.compile(r"ITEM7\.MANAGEMENT.{0,3}SDISCUSSION"),
            # Title Case variants
            re.compile(r"Item7\.Management.{0,3}sDiscussion"),
        ],
        "end_patterns": [
            re.compile(r"ITEM7A\.QUANTITATIVE"),
            re.compile(r"Item7A\.Quantitative"),
        ],
    },
}


# Page noise patterns - lines that should be removed during cleanup.
NOISE_LINE_PATTERNS = [
    re.compile(r"^Table of Contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d{1,3}\.\s*$"),
    re.compile(r"^\s*\d{1,3}\s*$"),
    re.compile(r"^\s*$"),
]
COMPANY_HEADER_LINES = [
    re.compile(r"^Alphabet Inc\.?\s*$"),
    re.compile(r"^Microsoft Corporation\s*$"),
    re.compile(r"^NVIDIA Corporation\s*$"),
    re.compile(r"^\s*PART\s+I+\s*$"),  # "PART I", "PART II" page header noise
]


# -----------------------------------------------------------------------------
# Compact-text utilities
# -----------------------------------------------------------------------------

def build_compact_text(text: str):
    """
    Return (compact_text, pos_map) where compact_text contains no whitespace
    characters and pos_map[i] is the index in the original text corresponding
    to compact_text[i].
    """
    out_chars = []
    pos_map = []
    for i, c in enumerate(text):
        # Treat NBSP (U+00A0) as whitespace explicitly, since str.isspace doesn't on all Pythons
        if c.isspace() or c == "\xa0":
            continue
        out_chars.append(c)
        pos_map.append(i)
    return "".join(out_chars), pos_map


def is_likely_toc_match(original_text: str, original_pos: int, lookahead_chars: int = 5000) -> bool:
    """
    Heuristic: a TOC has many "Item N." or "Item NA." labels packed close
    together. Real section content has at most one or two cross-references
    in 5000 chars. Counting standalone page-number lines does not work
    because financial tables also produce isolated 1-3 digit lines.
    """
    snippet = original_text[original_pos:original_pos + lookahead_chars]
    item_label_lines = len(re.findall(r"(?im)^\s*Item\s*\d+[A-Z]?\.\s*$", snippet))
    return item_label_lines >= 4


def find_first_real_match(text: str, compact: str, pos_map: list, patterns, after_pos: int = 0):
    """
    Scan `compact` for each pattern in order. For every match, map back to
    original position. Skip matches that look like TOC entries. Return the
    first surviving match's original position, or None.
    """
    candidates = []
    for pat in patterns:
        for m in pat.finditer(compact):
            orig_pos = pos_map[m.start()]
            if orig_pos < after_pos:
                continue
            candidates.append((orig_pos, pat))
    candidates.sort()  # earliest original position first

    for orig_pos, pat in candidates:
        if is_likely_toc_match(text, orig_pos):
            continue
        return orig_pos
    return None


# -----------------------------------------------------------------------------
# Section extraction
# -----------------------------------------------------------------------------

def extract_full_text(html_path: Path) -> str:
    raw = html_path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    return soup.get_text(separator="\n", strip=False)


def extract_section(text: str, compact: str, pos_map: list, section_def: dict):
    start = find_first_real_match(text, compact, pos_map, section_def["start_patterns"])
    if start is None:
        return None
    # End must come after start; also must pass TOC check (which it will, since
    # by the time we hit the next section we are well past TOC).
    end = find_first_real_match(text, compact, pos_map, section_def["end_patterns"], after_pos=start + 200)
    if end is None:
        return None
    return text[start:end]


def clean_section_text(raw_section: str) -> str:
    lines = raw_section.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if any(p.match(s) for p in NOISE_LINE_PATTERNS):
            continue
        if any(p.match(s) for p in COMPANY_HEADER_LINES):
            continue
        cleaned.append(s)
    out = []
    prev_blank = False
    for line in cleaned:
        if line == "":
            if not prev_blank:
                out.append(line)
            prev_blank = True
        else:
            out.append(line)
            prev_blank = False
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def extract_one_filing(html_path: Path, output_dir: Path, company_slug: str) -> dict:
    print(f"\n>>> {company_slug}: {html_path.name}")
    text = extract_full_text(html_path)
    print(f"    Full text length: {len(text):,} chars")

    compact, pos_map = build_compact_text(text)
    print(f"    Compact text length: {len(compact):,} chars (whitespace removed)")

    company_dir = output_dir / company_slug
    company_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for key, section_def in SECTIONS.items():
        raw = extract_section(text, compact, pos_map, section_def)
        if raw is None:
            print(f"    [{key}] NOT FOUND")
            summaries.append({"section_key": key, "title": section_def["title"], "found": False})
            continue
        cleaned = clean_section_text(raw)
        out_path = company_dir / f"{key}.txt"
        out_path.write_text(cleaned)
        summaries.append({
            "section_key": key,
            "title": section_def["title"],
            "found": True,
            "raw_chars": len(raw),
            "cleaned_chars": len(cleaned),
            "cleaned_lines": cleaned.count("\n") + 1,
            "output_file": str(out_path.relative_to(output_dir.parent)),
        })
        print(f"    [{key}] {len(cleaned):,} chars ({cleaned.count(chr(10))+1:,} lines) -> {out_path.name}")

    return {
        "company_slug": company_slug,
        "source_html": html_path.name,
        "full_text_chars": len(text),
        "sections": summaries,
    }


def main():
    p = argparse.ArgumentParser(description="Extract sections from 10-K filings")
    p.add_argument("--input-dir", type=Path, default=Path("sec_10k_data"))
    p.add_argument("--output-dir", type=Path, default=Path("sec_10k_sections"))
    p.add_argument("--single", type=Path, default=None,
                   help="Process a single HTML file (use with --slug)")
    p.add_argument("--slug", type=str, default=None,
                   help="Company slug for --single mode")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    if args.single:
        if not args.slug:
            p.error("--single requires --slug")
        manifest.append(extract_one_filing(args.single, args.output_dir, args.slug))
    else:
        for company_dir in sorted(args.input_dir.iterdir()):
            if not company_dir.is_dir():
                continue
            html_files = [f for f in company_dir.glob("10k.*") if f.suffix.lower() in (".htm", ".html")]
            if not html_files:
                print(f"!!! No 10-K HTML found in {company_dir}, skipping")
                continue
            manifest.append(extract_one_filing(html_files[0], args.output_dir, company_dir.name))

    manifest_path = args.output_dir / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    main()
