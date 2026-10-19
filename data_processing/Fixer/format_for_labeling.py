"""
BGV Format Converter for Labeling Pipeline
=============================================
Converts existing chunk files to the standardized labeling input format.

Already in correct format (from unified_chunker_v3):
  - tc_chunks.csv
  - ann_chunks.csv
  - sec_text_chunks.csv

Need conversion (from earlier chunking scripts):
  - cvc_chunks_output.csv       → source_type: "CVC_from_google_news"
  - sec_chunks_output.csv       → source_type: "sec_filing"
  - speaker_chunks_output.csv   → source_type: "earning_call"

Output:
  - cvc_chunks_labeled_format.csv/json
  - sec_structured_chunks_labeled_format.csv/json
  - ec_chunks_labeled_format.csv/json
  - (optional) all_chunks_for_labeling.jsonl — everything merged

Target schema:
  chunk_id, source_type, source_name, source_title,
  source_url, document_date, company, chunk_text
"""

import os
import re
import json
import pandas as pd


# ============================================================
# COMPANY STANDARDIZATION
# ============================================================

COMPANY_MAP = {
    "Google": "Google", "Alphabet": "Google", "Alphabet Inc.": "Google",
    "Microsoft": "Microsoft", "Microsoft Corporation": "Microsoft",
    "NVIDIA": "NVIDIA", "NVIDIA Corporation": "NVIDIA", "Nvidia": "NVIDIA",
    "OpenAI": "OpenAI",
}


def std_company(name):
    return COMPANY_MAP.get(name, name)


# ============================================================
# 1. CONVERT CVC CHUNKS
# ============================================================

def convert_cvc(input_csv, output_name):
    print(f"  Converting CVC: {input_csv}")

    df = pd.read_csv(input_csv)

    # Strip metadata header from text_chunk if present
    def strip_header(text):
        text = str(text)
        # Remove "[Company: ... | Date: ... ]" header
        text = re.sub(r'^\[.*?\]\s*\n*', '', text, flags=re.DOTALL)
        return text.strip()

    records = []
    for _, row in df.iterrows():
        company = std_company(str(row.get('company', '')))
        if company not in ["Google", "Microsoft", "NVIDIA", "OpenAI"]:
            continue

        chunk_text = strip_header(row.get('text_chunk', ''))
        if len(chunk_text) < 20:
            continue

        # Build source_title from the title field or first line of text
        title = str(row.get('title', ''))
        if not title or title == 'nan':
            title = chunk_text.split('.')[0][:150]

        records.append({
            "chunk_id": row.get('chunk_id', ''),
            "source_type": "CVC_from_google_news",
            "source_name": "Google News",
            "source_title": title,
            "source_url": str(row.get('source_url', '')),
            "document_date": str(row.get('event_date', ''))[:10],
            "company": company,
            "chunk_text": chunk_text,
        })

    print(f"    → {len(records)} chunks")
    return records


# ============================================================
# 2. CONVERT SEC STRUCTURED CHUNKS
# ============================================================

def convert_sec_structured(input_csv, output_name):
    print(f"  Converting SEC Structured: {input_csv}")

    df = pd.read_csv(input_csv)

    records = []
    for _, row in df.iterrows():
        company = std_company(str(row.get('company', '')))
        if company not in ["Google", "Microsoft", "NVIDIA", "OpenAI"]:
            continue

        chunk_text = str(row.get('text_chunk', ''))
        if len(chunk_text) < 20:
            continue

        # Build source title from metric and form type
        metric = str(row.get('metric_name', ''))
        form = str(row.get('form_type', ''))
        event_date = str(row.get('event_date', ''))
        source_title = f"{company} {form} - {metric}" if metric and metric != 'nan' else f"{company} {form} Filing"

        records.append({
            "chunk_id": row.get('chunk_id', ''),
            "source_type": "sec_filing",
            "source_name": f"SEC {form}" if form and form != 'nan' else "SEC Filing",
            "source_title": source_title,
            "source_url": "",
            "document_date": event_date[:10] if event_date else "",
            "company": company,
            "chunk_text": chunk_text,
        })

    print(f"    → {len(records)} chunks")
    return records


# ============================================================
# 3. CONVERT EARNINGS CALL CHUNKS
# ============================================================

def convert_earnings_call(input_csv, output_name):
    print(f"  Converting Earnings Call: {input_csv}")

    df = pd.read_csv(input_csv)

    # Strip metadata header from text_chunk
    def strip_header(text):
        text = str(text)
        text = re.sub(r'^\[Company:.*?\]\s*\n*', '', text, flags=re.DOTALL)
        return text.strip()

    records = []
    for _, row in df.iterrows():
        company = std_company(str(row.get('company', '')))
        if company not in ["Google", "Microsoft", "NVIDIA", "OpenAI"]:
            continue

        chunk_text = strip_header(row.get('text_chunk', ''))
        if len(chunk_text) < 20:
            continue

        # Build source title
        quarter = str(row.get('quarter', ''))
        fy = str(row.get('fiscal_year', ''))
        chunk_type = str(row.get('chunk_type', ''))
        speaker = str(row.get('speaker_name', ''))
        source_title = f"{company} {quarter} FY{fy} Earnings Call"
        if speaker and speaker != 'nan':
            source_title += f" - {speaker}"
        if chunk_type == 'qa_pair':
            source_title += " (Q&A)"

        records.append({
            "chunk_id": row.get('chunk_id', ''),
            "source_type": "earning_call",
            "source_name": f"{company} Earnings Call",
            "source_title": source_title,
            "source_url": "",
            "document_date": str(row.get('event_date', ''))[:10],
            "company": company,
            "chunk_text": chunk_text,
        })

    print(f"    → {len(records)} chunks")
    return records


# ============================================================
# SAVE HELPERS
# ============================================================

def save_converted(records, name, output_dir):
    if not records:
        return
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, f"{name}.csv")
    json_path = os.path.join(output_dir, f"{name}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {csv_path}")


def save_merged_jsonl(all_records, output_dir):
    """Save all chunks as a single JSONL file for the labeling pipeline."""
    jsonl_path = os.path.join(output_dir, "all_chunks_for_labeling.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\n  Merged JSONL: {jsonl_path} ({len(all_records)} chunks)")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  BGV Format Converter for Labeling Pipeline")
    print("=" * 60 + "\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    all_records = []

    # --- Convert old-format chunks ---
    print("  [1] Converting old-format chunk files...\n")

    # CVC
    cvc_path = os.path.join(current_dir, "cvc_chunks_output.csv")
    if os.path.exists(cvc_path):
        cvc = convert_cvc(cvc_path, "cvc")
        save_converted(cvc, "cvc_chunks_labeled_format", current_dir)
        all_records.extend(cvc)

    # SEC Structured
    sec_path = os.path.join(current_dir, "sec_chunks_output.csv")
    if os.path.exists(sec_path):
        sec = convert_sec_structured(sec_path, "sec_structured")
        save_converted(sec, "sec_structured_chunks_labeled_format", current_dir)
        all_records.extend(sec)

    # Earnings Call
    ec_path = os.path.join(current_dir, "speaker_chunks_output.csv")
    if os.path.exists(ec_path):
        ec = convert_earnings_call(ec_path, "earnings_call")
        save_converted(ec, "ec_chunks_labeled_format", current_dir)
        all_records.extend(ec)

    # --- Load already-correct-format chunks ---
    print("\n  [2] Loading already-formatted chunk files...\n")

    for name, filename in [("tc", "tc_chunks.csv"), ("ann", "ann_chunks.csv"), ("sec_text", "sec_text_chunks.csv")]:
        path = os.path.join(current_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            records = df.to_dict('records')
            all_records.extend(records)
            print(f"    Loaded {filename}: {len(records)} chunks")
        else:
            print(f"    ⚠ Not found: {filename}")

    # --- Save merged JSONL ---
    print(f"\n  [3] Saving merged output...")
    save_merged_jsonl(all_records, current_dir)

    # --- Summary ---
    df_all = pd.DataFrame(all_records)
    print(f"\n{'='*60}")
    print(f"  FORMAT CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total chunks: {len(df_all)}")

    print(f"\n  By source_type:")
    for st, count in df_all['source_type'].value_counts().items():
        print(f"    {st:35}: {count}")

    print(f"\n  By company:")
    for comp in ["Google", "Microsoft", "NVIDIA", "OpenAI"]:
        count = len(df_all[df_all['company'] == comp])
        if count > 0:
            print(f"    {comp:12}: {count}")

    print(f"\n  Schema check:")
    required = ['chunk_id', 'source_type', 'source_name', 'source_title',
                'source_url', 'document_date', 'company', 'chunk_text']
    for col in required:
        present = col in df_all.columns
        filled = df_all[col].notna().sum() if present else 0
        print(f"    {col:20}: {'✅' if present else '❌'} ({filled}/{len(df_all)} filled)")

    print()


if __name__ == "__main__":
    main()
