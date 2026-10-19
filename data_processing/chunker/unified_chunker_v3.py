"""
BGV Unified Sliding Window Chunker v3
=======================================
One script, three data types, separate outputs.

Input:
  - tc_cleaned.csv              → TechCrunch articles
  - ann_cleaned.csv             → Company announcements (Google/MSFT/NVIDIA)
  - sec_texts/ folder           → SEC filings (.txt files)

Output (separate):
  - tc_chunks.csv + tc_chunks.json
  - ann_chunks.csv + ann_chunks.json
  - sec_text_chunks.csv + sec_text_chunks.json

Chunking strategies:
  - TechCrunch (avg 2959 chars): sentence-level sliding window for long articles,
    direct chunk for short ones
  - Announcements (avg 3883 chars): same strategy
  - SEC 8-K (short): single chunk or light split
  - SEC 10-K (very long): section-based + sliding window hybrid
"""

import os
import re
import json
import glob
import pandas as pd
from datetime import datetime


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    abbrevs = ['Inc', 'Corp', 'Ltd', 'Co', 'Mr', 'Mrs', 'Dr', 'Jr', 'Sr',
               'U.S', 'U.K', 'E.U', 'D.C', 'A.I', 'No', 'vs', 'approx',
               'dept', 'Gov', 'Rep', 'Sen', 'Gen', 'Vol', 'Sec', 'Art']
    for ab in abbrevs:
        text = text.replace(f'{ab}. ', f'{ab}≈ ')

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    sentences = [s.replace('≈', '.') for s in sentences]
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    return sentences


def sliding_window_sentences(sentences, window_size=8, overlap=2):
    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    chunks = []
    step = max(1, window_size - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i : i + window_size]
        chunks.append(" ".join(window))
        if i + window_size >= len(sentences):
            break
    return chunks


def chunk_text(text, min_for_split=1500):
    """Decide chunking strategy based on text length."""
    if len(text) < min_for_split:
        return [text], "single"

    sentences = split_sentences(text)
    if len(sentences) <= 10:
        return [" ".join(sentences)], "single"

    chunks = sliding_window_sentences(sentences, window_size=8, overlap=2)
    return chunks, "sliding_window"


# ============================================================
# 1. TECHCRUNCH CHUNKER
# ============================================================

def process_techcrunch(input_csv):
    print("=" * 60)
    print("  CHUNKING: TechCrunch")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} articles\n")

    all_chunks = []

    for _, row in df.iterrows():
        text = str(row.get('clean_content', '')).strip()
        if len(text) < 100:
            continue

        title = str(row.get('title', '')).strip()
        companies_raw = str(row.get('companies', ''))
        companies = parse_companies(companies_raw)
        date = str(row.get('date', ''))[:10]
        url = str(row.get('url', ''))

        text_chunks, method = chunk_text(text)

        for comp in companies:
            for idx, chunk_text_str in enumerate(text_chunks):
                prefix = comp[:4].lower()
                chunk_id = f"{prefix}_tc_{row.get('id', '')}_{idx+1:03d}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source_type": "techCrunch",
                    "source_name": "TechCrunch",
                    "source_title": title,
                    "source_url": url,
                    "document_date": date,
                    "company": comp,
                    "chunk_text": chunk_text_str,
                })

    print(f"  Generated {len(all_chunks)} chunks")
    print(f"  By company:")
    df_out = pd.DataFrame(all_chunks)
    for comp in ["Google", "Microsoft", "NVIDIA", "OpenAI"]:
        count = len(df_out[df_out['company'] == comp])
        if count > 0:
            print(f"    {comp}: {count}")

    return all_chunks


# ============================================================
# 2. ANNOUNCEMENTS CHUNKER
# ============================================================

def process_announcements(input_csv):
    print("\n" + "=" * 60)
    print("  CHUNKING: Announcements")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} articles\n")

    all_chunks = []

    for row_idx, row in df.iterrows():
        text = str(row.get('clean_content', '')).strip()
        if len(text) < 100:
            continue

        company = str(row.get('company', ''))
        title = str(row.get('title', '')).strip()
        date = str(row.get('date', ''))[:10]
        url = str(row.get('url', ''))
        source_name = str(row.get('source_name', f'{company} Blog'))

        text_chunks, method = chunk_text(text)

        prefix = company[:4].lower()
        for idx, chunk_text_str in enumerate(text_chunks):
            chunk_id = f"{prefix}_ann_{row_idx+1:05d}_{idx+1:03d}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "source_type": "official_website_announcement",
                "source_name": source_name,
                "source_title": title,
                "source_url": url,
                "document_date": date,
                "company": company,
                "chunk_text": chunk_text_str,
            })

    print(f"  Generated {len(all_chunks)} chunks")
    df_out = pd.DataFrame(all_chunks)
    for comp in ["Google", "Microsoft", "NVIDIA"]:
        count = len(df_out[df_out['company'] == comp])
        if count > 0:
            print(f"    {comp}: {count}")

    return all_chunks


# ============================================================
# 3. SEC TEXT CHUNKER
# ============================================================

def parse_sec_header(content):
    meta = {"company": "", "filing_type": "", "filing_date": "",
            "items": "", "matched_themes": "", "url": ""}

    text_idx = content.find("Text:")
    if text_idx < 0:
        return meta, content

    header = content[:text_idx]
    body = content[text_idx + 5:].strip()

    patterns = {
        "company": r'Company:\s*(.+?)(?:\s{2,}|Filing)',
        "filing_type": r'Filing Type:\s*(\S+)',
        "filing_date": r'Filing Date:\s*(\S+)',
        "matched_themes": r'Matched Themes:\s*(.+?)(?:\s{2,}|URL)',
        "url": r'URL:\s*(\S+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, header)
        if match:
            meta[key] = match.group(1).strip()

    return meta, body


def clean_sec_boilerplate(text):
    boilerplate = [
        r'UNITED STATES SECURITIES AND EXCHANGE COMMISSION.*?FORM \d+-\w+',
        r'CURRENT REPORT.*?Securities Exchange Act of 1934',
        r'Check the appropriate box below.*?Exchange Act\.',
        r'Securities registered pursuant to Section 12\(b\).*?(?=Item)',
        r'Indicate by check mark.*?Exchange Act\.',
        r'Emerging growth company.*?Exchange Act\.',
        r'SIGNATURE\s+Pursuant to the requirements.*$',
        r'Exhibit No\.\s+Description.*$',
        r'Cover Page Interactive Data File.*',
        r'\(State or other jurisdiction.*?\)',
        r'\(Commission File Number\)',
        r'\(IRS Employer Identification No\.\)',
        r'\(Address of principal executive.*?\)',
        r'\(Registrant.?s telephone number.*?\)',
        r'Not Applicable.*?since last report\)',
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_sec_texts(sec_folder):
    print("\n" + "=" * 60)
    print("  CHUNKING: SEC Text Files")
    print("=" * 60)

    txt_files = sorted(glob.glob(os.path.join(sec_folder, "*.txt")))
    if not txt_files:
        print(f"  No .txt files found in {sec_folder}")
        return []

    print(f"  Found {len(txt_files)} files\n")

    all_chunks = []

    for fpath in txt_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            meta, body = parse_sec_header(content)
            cleaned = clean_sec_boilerplate(body)

            if len(cleaned) < 50:
                continue

            # Standardize company
            company = meta["company"]
            if "Alphabet" in company or "Google" in company:
                company = "Google"
            elif "Microsoft" in company:
                company = "Microsoft"
            elif "NVIDIA" in company or "Nvidia" in company:
                company = "NVIDIA"
            elif "OpenAI" in company:
                company = "OpenAI"

            filing_type = meta["filing_type"]
            prefix = company[:4].lower()

            # Chunk
            text_chunks, method = chunk_text(cleaned)

            for idx, chunk_text_str in enumerate(text_chunks):
                chunk_id = f"{prefix}_sec_{filing_type.lower().replace('-','')}_{meta['filing_date']}_{idx+1:03d}"

                # Build source_title from filing type and date
                source_title = f"{meta['company']} {filing_type} Filing - {meta['filing_date']}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source_type": "sec_filing",
                    "source_name": f"SEC {filing_type}",
                    "source_title": source_title,
                    "source_url": meta["url"],
                    "document_date": meta["filing_date"],
                    "company": company,
                    "chunk_text": chunk_text_str,
                })

            n = len(text_chunks)
            print(f"  ✅ {filing_type} {meta['filing_date']} → {n} chunks ({method})")

        except Exception as e:
            print(f"  ❌ {fname}: {e}")

    print(f"\n  SEC Text Total: {len(all_chunks)} chunks")
    return all_chunks


# ============================================================
# HELPERS
# ============================================================

COMPANY_MAP = {
    "Google": "Google", "Alphabet": "Google",
    "Microsoft": "Microsoft",
    "NVIDIA": "NVIDIA", "Nvidia": "NVIDIA",
    "OpenAI": "OpenAI",
}

def parse_companies(companies_str):
    if not isinstance(companies_str, str):
        return []
    raw = [c.strip() for c in companies_str.split(';') if c.strip()]
    result = []
    for c in raw:
        std = COMPANY_MAP.get(c, c)
        if std in ["Google", "Microsoft", "NVIDIA", "OpenAI"] and std not in result:
            result.append(std)
    return result if result else ["Unknown"]


def save_output(chunks, name, output_dir):
    if not chunks:
        return
    df = pd.DataFrame(chunks)
    csv_path = os.path.join(output_dir, f"{name}_chunks.csv")
    json_path = os.path.join(output_dir, f"{name}_chunks.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  → {csv_path} ({len(chunks)} chunks)")
    print(f"  → {json_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    start = datetime.now()
    print("\n" + "=" * 60)
    print("  BGV Unified Chunker v3")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1. TechCrunch ---
    tc_path = os.path.join(current_dir, "tc_cleaned.csv")
    tc_chunks = []
    if os.path.exists(tc_path):
        tc_chunks = process_techcrunch(tc_path)
    else:
        print(f"  ⚠ TechCrunch file not found: {tc_path}")

    # --- 2. Announcements ---
    ann_path = os.path.join(current_dir, "ann_cleaned.csv")
    ann_chunks = []
    if os.path.exists(ann_path):
        ann_chunks = process_announcements(ann_path)
    else:
        print(f"  ⚠ Announcements file not found: {ann_path}")

    # --- 3. SEC Text ---
    sec_folder = os.path.join(current_dir, "sec_texts")
    sec_chunks = []
    if os.path.exists(sec_folder):
        sec_chunks = process_sec_texts(sec_folder)
    else:
        print(f"  ⚠ SEC folder not found: {sec_folder}")

    # --- Save separate outputs ---
    print(f"\n{'='*60}")
    print("  SAVING OUTPUTS")
    print(f"{'='*60}")
    save_output(tc_chunks, "tc", current_dir)
    save_output(ann_chunks, "ann", current_dir)
    save_output(sec_chunks, "sec_text", current_dir)

    # --- Summary ---
    total = len(tc_chunks) + len(ann_chunks) + len(sec_chunks)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"  TechCrunch:     {len(tc_chunks)} chunks")
    print(f"  Announcements:  {len(ann_chunks)} chunks")
    print(f"  SEC Text:       {len(sec_chunks)} chunks")
    print(f"  Total:          {total} chunks\n")


if __name__ == "__main__":
    main()
