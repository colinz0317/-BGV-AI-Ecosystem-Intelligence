"""
BGV Speaker-Role Based Chunking v2
====================================
Parses raw earnings call transcripts and produces Q&A-paired chunks
with full metadata for downstream ISS scoring and Pinecone storage.

Supports three transcript formats:
  - Google/Alphabet: "Name, Role: Text"
  - Microsoft: "NAME IN CAPS:" on its own line
  - NVIDIA: "Name\\nTitle, Company\\n" with separator lines and Q/A markers

Core feature: Q&A PAIR BINDING
  Analyst question + all executive responses → single chunk
  This preserves conversational context as required by the BGV architecture.

Usage:
    python speaker_chunking_v2.py

Input:  transcripts/ folder with .txt files
Output: speaker_chunks_output.csv, speaker_chunks.json
"""

import os
import re
import json
import glob
import pandas as pd
from datetime import datetime


# ============================================================
# FORMAT DETECTION
# ============================================================

def detect_format(text):
    """Auto-detect which company's transcript format this is."""
    # NVIDIA: has "CORPORATE PARTICIPANTS" header and dotted separator lines
    if "CORPORATE PARTICIPANTS" in text and "....." in text:
        return "nvidia"
    # Microsoft: speakers in ALL CAPS followed by colon
    if re.search(r'^[A-Z]{2,}\s+[A-Z]{2,}.*:', text, re.MULTILINE):
        # Check for typical Microsoft patterns
        if "BRETT IVERSEN" in text or "SATYA NADELLA" in text or "AMY HOOD" in text:
            return "microsoft"
    # Google: "Name, Role:" pattern
    if re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+,\s+\w+.*:', text, re.MULTILINE):
        return "google"
    # Fallback: check more broadly
    if re.search(r'^[A-Z ]{5,}:', text, re.MULTILINE):
        return "microsoft"
    return "google"  # default


def extract_metadata_from_filename(filename):
    """Extract company, date, quarter from filename."""
    meta = {"company": "Unknown", "date": "", "quarter": "", "fiscal_year": ""}
    fname = os.path.basename(filename)

    # Google: "2024_Q1_Earnings_Call_Transcript.txt"
    m = re.match(r'(\d{4})_Q(\d)_Earnings', fname)
    if m:
        meta["company"] = "Google"
        meta["fiscal_year"] = m.group(1)
        meta["quarter"] = f"Q{m.group(2)}"
        meta["date"] = f"{m.group(1)}-Q{m.group(2)}"
        return meta

    # Microsoft: "2024-10-30_Microsoft_Fiscal_Year_2025_First_Quarter..."
    m = re.match(r'(\d{4}-\d{2}-\d{2})_Microsoft_Fiscal_Year_(\d{4})_(\w+)_Quarter', fname)
    if m:
        meta["company"] = "Microsoft"
        meta["date"] = m.group(1)
        meta["fiscal_year"] = m.group(2)
        quarter_map = {"First": "Q1", "Second": "Q2", "Third": "Q3", "Fourth": "Q4"}
        meta["quarter"] = quarter_map.get(m.group(3), m.group(3))
        return meta

    # NVIDIA: "NVIDIA_2026_Q1_transcript.txt"
    m = re.match(r'NVIDIA_(\d{4})_Q(\d)', fname)
    if m:
        meta["company"] = "NVIDIA"
        meta["fiscal_year"] = m.group(1)
        meta["quarter"] = f"Q{m.group(2)}"
        meta["date"] = f"{m.group(1)}-Q{m.group(2)}"
        return meta

    # Try to infer company from content if filename doesn't match
    if "microsoft" in fname.lower():
        meta["company"] = "Microsoft"
    elif "nvidia" in fname.lower() or "nvda" in fname.lower():
        meta["company"] = "NVIDIA"
    elif "google" in fname.lower() or "alphabet" in fname.lower():
        meta["company"] = "Google"

    return meta


# ============================================================
# ROLE CLASSIFICATION
# ============================================================

EXECUTIVE_TITLES = [
    "ceo", "cfo", "coo", "cto", "president", "chief executive",
    "chief financial", "chief operating", "chief technology",
    "vice president", "director", "executive", "founder",
    "general counsel", "secretary", "accounting officer",
    "investor relations",
]

ANALYST_TITLES = [
    "analyst", "morgan stanley", "jpmorgan", "jp morgan", "goldman sachs",
    "jefferies", "ubs", "barclays", "bank of america", "bofa", "baml",
    "wells fargo", "bernstein", "evercore", "baird", "citi", "citigroup",
    "cantor", "melius", "deutsche bank", "rbc", "piper", "piper sandler",
    "oppenheimer", "wolfe", "td cowen", "cowen", "wedbush", "raymond james",
    "loop capital", "keybanc", "stifel", "needham", "rosenblatt",
    "mizuho", "bmo", "hsbc", "macquarie", "susquehanna", "truist",
    "william blair", "daiwa", "nomura", "credit suisse", "societe generale",
]


def classify_role(name, role_text):
    """Classify speaker as 'executive', 'analyst', or 'operator'."""
    combined = f"{name} {role_text}".lower()

    if "operator" in combined:
        return "operator"

    for title in ANALYST_TITLES:
        if title in combined:
            return "analyst"

    for title in EXECUTIVE_TITLES:
        if title in combined:
            return "executive"

    # If role contains a company name from analyst firms, it's an analyst
    if re.search(r'\b(morgan|goldman|jpmorgan|jefferies|ubs|barclays|bofa|baml|wells|bernstein|evercore|baird|citi|cantor|melius|deutsche|rbc|piper|cowen|mizuho|bmo|truist|macquarie)\b', combined, re.I):
        return "analyst"

    return "executive"  # default to executive if unclear


# ============================================================
# PARSER: GOOGLE FORMAT
# ============================================================

def parse_google(text, meta):
    """
    Google format:
      Executive: "Sundar Pichai, CEO Alphabet and Google: text..."
      Analyst:   "Brian Nowak (Morgan Stanley): text..."
      Operator:  "Operator: text..."
    """
    segments = []

    # Three specific patterns for Google transcripts
    # 1. Operator: text
    # 2. Name Name, Title/Role: text  (executive)
    # 3. Name Name (Company): text    (analyst)
    pattern = re.compile(
        r'^('
        r'Operator'                                          # Operator
        r'|[A-Z][a-z]+(?:\s+[A-Z][a-z\.]+){0,3}\s*\([^)]+\)'  # Name (Company)
        r'|[A-Z][a-z]+(?:\s+[A-Z][a-z\.]+){0,3},\s*[A-Z][A-Za-z\s\&\;\,\.]+?(?=\s*:)' # Name, Title
        r')\s*:\s*',
        re.MULTILINE
    )

    splits = list(pattern.finditer(text))

    for i, match in enumerate(splits):
        speaker_raw = match.group(1).strip()
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        speech = text[start:end].strip()

        # Clean page numbers and artifacts
        speech = re.sub(r'^\d+\s*$', '', speech, flags=re.MULTILINE)
        speech = re.sub(r'\n{3,}', '\n\n', speech)
        speech = speech.strip()

        if not speech or len(speech) < 10:
            continue

        # Parse name and role
        if speaker_raw == "Operator":
            name, role = "Operator", "Operator"
        elif "(" in speaker_raw:
            name = speaker_raw.split("(")[0].strip()
            role = speaker_raw.split("(")[1].rstrip(")").strip()
        elif "," in speaker_raw:
            parts = speaker_raw.split(",", 1)
            name = parts[0].strip()
            role = parts[1].strip()
        else:
            name = speaker_raw
            role = ""

        role_type = classify_role(name, role)

        segments.append({
            "name": name,
            "role": role,
            "role_type": role_type,
            "text": speech,
        })

    return segments


# ============================================================
# PARSER: MICROSOFT FORMAT
# ============================================================

def parse_microsoft(text, meta):
    """
    Microsoft format has TWO sub-formats:
      Block (prepared remarks): "NAME:\\n" then text on next lines
      Inline (Q&A): "NAME, Company: text on same line"
    Both need to be captured.
    """
    segments = []

    # Unified pattern: matches both block and inline formats
    # ALL CAPS name (2+ words), optional ", Company", then colon
    pattern = re.compile(
        r'^([A-Z][A-Z\s\.\-\']+?(?:,\s*[A-Za-z\s\&\.]+?)?)\s*:\s*(.*)',
        re.MULTILINE
    )

    splits = list(pattern.finditer(text))

    # Filter out false matches (lines inside paragraphs that happen to match)
    filtered = []
    for match in splits:
        speaker = match.group(1).strip()
        # Must have at least 2 uppercase words or be "Operator"
        words = speaker.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w != ','):
            filtered.append(match)
        elif speaker.upper() == "OPERATOR":
            filtered.append(match)
    splits = filtered

    for i, match in enumerate(splits):
        speaker_raw = match.group(1).strip()
        inline_text = match.group(2).strip()  # Text on the same line as speaker

        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        following_text = text[start:end].strip()

        # Combine inline text + following text
        if inline_text:
            speech = inline_text + "\n" + following_text
        else:
            speech = following_text
        speech = speech.strip()

        # Clean artifacts
        speech = re.sub(r'\n{3,}', '\n\n', speech).strip()

        if not speech or len(speech) < 10:
            continue

        # Parse name and role
        if "," in speaker_raw:
            parts = speaker_raw.split(",", 1)
            name = parts[0].strip().title()
            role = parts[1].strip()
        else:
            name = speaker_raw.strip().title()
            role = ""

        if "OPERATOR" in speaker_raw.upper():
            name, role = "Operator", "Operator"

        role_type = classify_role(name, role)

        segments.append({
            "name": name,
            "role": role,
            "role_type": role_type,
            "text": speech,
        })

    return segments


# ============================================================
# PARSER: NVIDIA FORMAT
# ============================================================

def parse_nvidia(text, meta):
    """
    NVIDIA format (FactSet CallStreet):
    Speaker Name
    Title, Company [Q or A marker]
    Text...
    .....separator.....
    """
    segments = []

    # Remove header junk (copyright, page numbers, repeated headers)
    text = re.sub(r'1-877-FACTSET.*?www\.callstreet\.com', '', text)
    text = re.sub(r'Total Pages:.*?\n', '', text)
    text = re.sub(r'Copyright.*?FactSet CallStreet.*?\n', '', text)
    text = re.sub(r'NVIDIA Corp\.\s*\(NVDA\)\nQ\d\s+\d{4}\s+Earnings Call\nCorrected Transcript\n\d{2}-\w+-\d{4}', '', text)

    # Split by separator lines
    blocks = re.split(r'\s*\.{20,}\s*', text)

    for block in blocks:
        block = block.strip()
        if not block or len(block) < 20:
            continue

        lines = block.split('\n')
        lines = [l.strip() for l in lines if l.strip()]

        if not lines:
            continue

        # Check if starts with "Operator:"
        if lines[0].startswith("Operator:"):
            speech = " ".join(lines)
            name = "Operator"
            role = "Operator"
            role_type = "operator"
        elif len(lines) >= 2:
            # First line: Name, Second line: Title + possible Q/A marker
            name = lines[0].strip()
            role_line = lines[1].strip()

            # Remove Q/A marker at end
            qa_marker = ""
            if role_line.endswith(" Q"):
                qa_marker = "Q"
                role_line = role_line[:-2].strip()
            elif role_line.endswith(" A"):
                qa_marker = "A"
                role_line = role_line[:-2].strip()

            role = role_line
            speech = "\n".join(lines[2:]).strip()

            if not speech:
                continue

            role_type = classify_role(name, role)

            # Override with Q/A marker if available
            if qa_marker == "Q":
                role_type = "analyst"
            elif qa_marker == "A":
                role_type = "executive"
        else:
            continue

        # Clean artifacts
        speech = re.sub(r'\n{3,}', '\n\n', speech).strip()

        if speech and len(speech) >= 10:
            segments.append({
                "name": name,
                "role": role,
                "role_type": role_type,
                "text": speech,
            })

    return segments


# ============================================================
# Q&A PAIR BINDING
# ============================================================

def bind_qa_pairs(segments, meta):
    """
    Core feature: Bind analyst questions with executive answers into single chunks.
    
    Strategy:
    1. Split segments into "prepared remarks" and "Q&A" sections
    2. In Q&A: group analyst question + all executive responses until next analyst/operator
    3. Each group becomes one chunk
    
    Output chunk types:
    - "prepared_remarks": Executive's prepared speech (chunked by speaker)
    - "qa_pair": Analyst question + executive answer(s) bound together
    """
    chunks = []

    # Find Q&A boundary
    # Strategy: find the LAST operator line that introduces Q&A, not the first
    # (the first operator often just describes the call format)
    qa_start_idx = None
    
    # Look for explicit Q&A transition markers
    qa_transition_patterns = [
        "take your questions",
        "move over to q&a",
        "let's go to q&a",
        "we'll now move",
        "first question comes from",
        "first question is from",
        "your first question",
        "begin the question",
        "open it up for questions",
        "open the line for questions",
        "operator instructions",
    ]
    
    # Search from the MIDDLE of the transcript forward (Q&A never starts in first 20%)
    min_start = len(segments) // 5
    
    for i in range(min_start, len(segments)):
        seg = segments[i]
        if seg["role_type"] == "operator":
            text_lower = seg["text"].lower()
            if any(kw in text_lower for kw in qa_transition_patterns):
                qa_start_idx = i
                break

    # If no explicit marker found, find first analyst after the halfway point
    if qa_start_idx is None:
        midpoint = len(segments) // 3
        for i in range(midpoint, len(segments)):
            if segments[i]["role_type"] == "analyst":
                qa_start_idx = i
                break

    if qa_start_idx is None:
        qa_start_idx = len(segments)  # No Q&A found

    # --- PART 1: Prepared Remarks ---
    for seg in segments[:qa_start_idx]:
        if seg["role_type"] == "operator":
            continue  # Skip operator lines in prepared remarks
        if len(seg["text"]) < 50:
            continue

        chunks.append({
            "chunk_type": "prepared_remarks",
            "speaker_name": seg["name"],
            "speaker_role": seg["role"],
            "speaker_type": seg["role_type"],
            "question_text": "",
            "answer_text": seg["text"],
            "full_text": seg["text"],
        })

    # --- PART 2: Q&A Pairs ---
    qa_segments = segments[qa_start_idx:]
    i = 0

    while i < len(qa_segments):
        seg = qa_segments[i]

        # Skip operator lines
        if seg["role_type"] == "operator":
            i += 1
            continue

        # Found an analyst question
        if seg["role_type"] == "analyst":
            question_name = seg["name"]
            question_role = seg["role"]
            question_text = seg["text"]

            # Collect all executive responses until next analyst or operator-intro
            answers = []
            j = i + 1
            while j < len(qa_segments):
                next_seg = qa_segments[j]
                if next_seg["role_type"] == "analyst":
                    break
                if next_seg["role_type"] == "operator":
                    # Check if operator is introducing next question
                    if any(kw in next_seg["text"].lower() for kw in
                           ["next question", "last question", "final question",
                            "concludes", "no further questions"]):
                        break
                    j += 1
                    continue
                if next_seg["role_type"] == "executive":
                    answers.append(next_seg)
                j += 1

            # Build the paired chunk
            answer_parts = []
            for ans in answers:
                answer_parts.append(f"[{ans['name']}, {ans['role']}]: {ans['text']}")
            answer_text = "\n\n".join(answer_parts)

            # Full text = question + answers bound together
            full_text = (
                f"[QUESTION by {question_name}, {question_role}]:\n{question_text}"
                f"\n\n[ANSWER]:\n{answer_text}"
            )

            chunks.append({
                "chunk_type": "qa_pair",
                "speaker_name": question_name,
                "speaker_role": question_role,
                "speaker_type": "analyst",
                "question_text": question_text,
                "answer_text": answer_text,
                "full_text": full_text,
            })

            i = j  # Jump to next analyst
        else:
            # Executive speaking in Q&A but not as answer (e.g., follow-up)
            i += 1

    return chunks


# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def process_transcript(file_path):
    """Process a single transcript file end-to-end."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Clean hidden Unicode direction markers (found in some Google Q2 files)
    text = text.replace('\u202D', '').replace('\u202C', '')
    text = text.replace('\u200E', '').replace('\u200F', '')  # LTR/RTL marks
    text = text.replace('\u2069', '').replace('\u2066', '')  # isolate markers

    meta = extract_metadata_from_filename(file_path)
    fmt = detect_format(text)

    # Parse into speaker segments
    if fmt == "nvidia":
        segments = parse_nvidia(text, meta)
    elif fmt == "microsoft":
        segments = parse_microsoft(text, meta)
    else:
        segments = parse_google(text, meta)

    # Bind Q&A pairs
    chunks = bind_qa_pairs(segments, meta)

    # Enrich with metadata
    enriched = []
    for idx, chunk in enumerate(chunks):
        # Build the enriched text chunk with metadata header
        header = (
            f"[Company: {meta['company']} | Date: {meta['date']} | "
            f"Quarter: {meta['quarter']} FY{meta['fiscal_year']} | "
            f"Type: {chunk['chunk_type']} | "
            f"Speaker: {chunk['speaker_name']}]"
        )
        enriched_text = f"{header}\n\n{chunk['full_text']}"

        enriched.append({
            "chunk_id": f"ec_{meta['company'].lower()}_{meta['quarter'].lower()}_fy{meta['fiscal_year']}_{idx+1:03d}",
            "text_chunk": enriched_text,
            "company": meta["company"],
            "event_date": meta["date"],
            "quarter": meta["quarter"],
            "fiscal_year": meta["fiscal_year"],
            "chunk_type": chunk["chunk_type"],
            "speaker_name": chunk["speaker_name"],
            "speaker_role": chunk["speaker_role"],
            "speaker_type": chunk["speaker_type"],
            "event_type": "Strategic Commitment" if chunk["chunk_type"] == "prepared_remarks" else "Commercial Traction",
            "category": "Earnings Call",
            "signal_strength": 4,  # Earnings calls are high-reliability
            "source_type": "Earnings Call Transcript",
            "source_file": os.path.basename(file_path),
        })

    return enriched


def main():
    print("\n" + "=" * 60)
    print("  BGV Speaker-Role Chunking v2")
    print("  With Q&A Pair Binding")
    print("=" * 60 + "\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Look for transcript files in transcripts/ subfolder or current dir
    transcript_dir = os.path.join(current_dir, "transcripts")
    if not os.path.exists(transcript_dir):
        transcript_dir = current_dir

    txt_files = glob.glob(os.path.join(transcript_dir, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in {transcript_dir}")
        print("Please create a 'transcripts/' folder and put your transcript files inside.")
        return

    print(f"Found {len(txt_files)} transcript files in {transcript_dir}\n")

    all_chunks = []

    for fpath in sorted(txt_files):
        fname = os.path.basename(fpath)
        try:
            chunks = process_transcript(fpath)
            all_chunks.extend(chunks)

            # Stats for this file
            prepared = sum(1 for c in chunks if c["chunk_type"] == "prepared_remarks")
            qa = sum(1 for c in chunks if c["chunk_type"] == "qa_pair")
            print(f"  ✅ {fname}")
            print(f"     {chunks[0]['company'] if chunks else '?'} {chunks[0]['quarter'] if chunks else '?'} → {prepared} prepared + {qa} Q&A pairs = {len(chunks)} chunks")

        except Exception as e:
            print(f"  ❌ {fname}: {e}")

    if not all_chunks:
        print("\nNo chunks generated. Check file formats.")
        return

    # Save CSV
    df = pd.DataFrame(all_chunks)
    csv_path = os.path.join(current_dir, "speaker_chunks_output.csv")
    df.to_csv(csv_path, index=False)

    # Save JSON
    json_path = os.path.join(current_dir, "speaker_chunks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("  CHUNKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Prepared remarks: {sum(1 for c in all_chunks if c['chunk_type'] == 'prepared_remarks')}")
    print(f"  Q&A pairs: {sum(1 for c in all_chunks if c['chunk_type'] == 'qa_pair')}")
    print(f"\n  By company:")
    for comp in df["company"].unique():
        count = len(df[df["company"] == comp])
        print(f"    {comp}: {count}")
    print(f"\n  Avg chunk length: {df['text_chunk'].str.len().mean():.0f} chars")
    print(f"\n  CSV: {csv_path}")
    print(f"  JSON: {json_path}\n")


if __name__ == "__main__":
    main()
