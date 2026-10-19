"""
BGV TechCrunch Data Cleaner
=============================
Cleans the full-content TechCrunch data for downstream chunking.

Cleaning steps:
  1. Remove article-end boilerplate (newsletter signups, Disrupt promos, etc.)
  2. Clean HTML entities and artifacts
  3. Standardize company names
  4. Deduplicate articles
  5. Filter out non-relevant content (event promos, job listings, etc.)
  6. Flag content quality (full / fallback / short)
  7. Add standardized metadata columns for chunking pipeline

Input:  techcrunch_full_content.csv
Output: tc_cleaned.csv
"""

import os
import re
import pandas as pd
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "techcrunch_full_content.csv"
OUTPUT_CSV = "tc_cleaned.csv"

VALID_COMPANIES = ["Google", "Microsoft", "NVIDIA", "OpenAI"]

COMPANY_STANDARD = {
    "Google": "Google",
    "Alphabet": "Google",
    "Microsoft": "Microsoft",
    "NVIDIA": "NVIDIA",
    "Nvidia": "NVIDIA",
    "OpenAI": "OpenAI",
}

MIN_CONTENT_LENGTH = 150  # Minimum chars to keep an article


# ============================================================
# TEXT CLEANING
# ============================================================

# Patterns to strip from the END of articles (newsletter promos, etc.)
TAIL_NOISE_PATTERNS = [
    # Newsletter signups
    r'Sign up for\s+.{0,200}(?:newsletter|TechCrunch|inbox|email).*$',
    r'Subscribe to\s+.{0,200}(?:newsletter|TechCrunch|podcast).*$',
    r'Get the latest\s+.{0,100}(?:newsletter|inbox).*$',
    r'Want more\s+.{0,100}(?:newsletter|TechCrunch).*$',
    r'Stay up to date\s+.{0,100}(?:newsletter|inbox).*$',

    # TechCrunch event promos
    r'(?:Join us at|Grab your|Get your|Don.t miss)\s+.{0,200}(?:TechCrunch Disrupt|TC Disrupt|Startup Battlefield).*$',
    r'TechCrunch Disrupt\s+\d{4}\s+(?:is|takes|will).*$',
    r'(?:Last \d+ hours?|Early bird|Limited time).{0,200}(?:pass|ticket|discount|Disrupt).*$',

    # Read more / related
    r'Read more on TechCrunch.*$',
    r'Related(?:\s+articles?|\s+stories?)?:?\s*$',
    r'(?:More|Also)\s+(?:from|on)\s+TechCrunch.*$',

    # Author bios at the end
    r'(?:By\s+)?[A-Z][a-z]+ [A-Z][a-z]+ (?:is|covers|writes|reports)\s+.{0,200}(?:TechCrunch|tech|startups).*$',
]

# Patterns to strip from ANYWHERE in the text
INLINE_NOISE_PATTERNS = [
    r'\[(?:Image|Photo|Video|Chart|Graph)\s*:?\s*[^\]]*\]',  # Image/video captions
    r'(?:Image|Photo)\s+(?:Credits?|via|courtesy)\s*:?\s*[^\n]{0,100}',  # Photo credits
    r'Correction:?\s+.{0,300}',  # Correction notices (keep short ones)
    r'Editor.s note:?\s+.{0,200}',  # Editor's notes
]

# Full-article noise: articles that are entirely promotional
PROMOTIONAL_TITLE_PATTERNS = [
    r'grab your.{0,50}pass',
    r'last \d+ hours?.{0,50}(ticket|pass|discount)',
    r'early.bird.{0,50}(ticket|pass)',
    r'save \d+%.{0,50}(ticket|pass|disrupt)',
    r'TechCrunch Disrupt \d{4}.{0,50}(agenda|speaker|schedule)',
    r'^Introducing TechCrunch',
    r'^Apply to .{0,50}Startup Battlefield',
    r'^Meet the .{0,50}Startup Battlefield',
]


def clean_article_text(text):
    """Clean a single article's text content."""
    if not isinstance(text, str) or len(text) < 10:
        return ""

    # Step 1: Clean HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&#8217;", "'")
    text = text.replace("&#8220;", '"')
    text = text.replace("&#8221;", '"')
    text = text.replace("&#8211;", "–")
    text = text.replace("&#8212;", "—")
    text = re.sub(r'&#\d+;', '', text)  # Remove remaining numeric entities
    text = re.sub(r'&\w+;', '', text)   # Remove remaining named entities

    # Step 2: Remove any stray HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Step 3: Remove inline noise
    for pattern in INLINE_NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Step 4: Remove tail noise (work backwards from end of article)
    for pattern in TAIL_NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Step 5: Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


def is_promotional(title, text):
    """Check if an article is purely promotional (TC event ads, etc.)."""
    title_lower = str(title).lower()

    for pattern in PROMOTIONAL_TITLE_PATTERNS:
        if re.search(pattern, title_lower):
            return True

    # Short articles about TC events
    if len(str(text)) < 500:
        text_lower = str(text).lower()
        if "techcrunch disrupt" in text_lower and "ticket" in text_lower:
            return True
        if "startup battlefield" in text_lower and "apply" in text_lower:
            return True

    return False


# ============================================================
# COMPANY PARSING
# ============================================================

def parse_companies(companies_str):
    """Parse and standardize company names from the companies column."""
    if not isinstance(companies_str, str):
        return []

    raw = [c.strip() for c in companies_str.split(';') if c.strip()]
    standardized = []

    for c in raw:
        std = COMPANY_STANDARD.get(c, c)
        if std in VALID_COMPANIES and std not in standardized:
            standardized.append(std)

    return standardized


# ============================================================
# MAIN PIPELINE
# ============================================================

def clean_techcrunch(input_path, output_path):
    print("\n" + "=" * 60)
    print("  BGV TechCrunch Data Cleaner")
    print("=" * 60)

    # --- Load ---
    print(f"\n  Loading: {input_path}")
    df = pd.read_csv(input_path)
    initial = len(df)
    print(f"  Raw articles: {initial}")

    # --- Step 1: Clean text ---
    print(f"\n  [Step 1] Cleaning article text...")
    df['clean_content'] = df['full_content'].apply(clean_article_text)
    df['clean_length'] = df['clean_content'].str.len()

    avg_before = df['full_content'].fillna('').str.len().mean()
    avg_after = df['clean_length'].mean()
    print(f"    Avg content: {avg_before:.0f} → {avg_after:.0f} chars (removed {avg_before-avg_after:.0f} avg noise)")

    # --- Step 2: Remove promotional articles ---
    print(f"  [Step 2] Filtering promotional content...")
    promo_mask = df.apply(lambda r: is_promotional(r['title'], r['clean_content']), axis=1)
    promo_count = promo_mask.sum()
    df = df[~promo_mask]
    print(f"    Removed {promo_count} promotional articles → {len(df)} remaining")

    # --- Step 3: Remove too-short articles ---
    print(f"  [Step 3] Removing short articles (<{MIN_CONTENT_LENGTH} chars)...")
    short_mask = df['clean_length'] < MIN_CONTENT_LENGTH
    short_count = short_mask.sum()
    df = df[~short_mask]
    print(f"    Removed {short_count} short articles → {len(df)} remaining")

    # --- Step 4: Deduplicate ---
    print(f"  [Step 4] Deduplicating...")
    before = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    df = df.drop_duplicates(subset=['title'], keep='first')
    print(f"    {before} → {len(df)} (removed {before - len(df)} dupes)")

    # --- Step 5: Standardize companies ---
    print(f"  [Step 5] Standardizing company names...")
    df['companies_list'] = df['companies'].apply(parse_companies)
    no_company = (df['companies_list'].str.len() == 0).sum()
    print(f"    Articles with no target company: {no_company}")

    # --- Step 6: Parse dates ---
    print(f"  [Step 6] Parsing dates...")
    df['date'] = pd.to_datetime(df['published_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    invalid_dates = df['date'].isna().sum()
    print(f"    Invalid dates: {invalid_dates}")

    # --- Step 7: Content quality flag ---
    print(f"  [Step 7] Flagging content quality...")
    df['content_quality'] = 'full'
    df.loc[df['fetch_status'] == 'fallback_to_preview', 'content_quality'] = 'preview_only'
    df.loc[df['clean_length'] < 500, 'content_quality'] = 'short'

    # --- Step 8: Build output ---
    print(f"  [Step 8] Building output...")
    output_cols = [
        'id',
        'date',
        'title',
        'url',
        'companies',            # Original semicolon-separated
        'companies_list',       # Standardized list
        'matched_core_keywords',
        'clean_content',        # Cleaned full text
        'clean_length',
        'content_quality',
        'excerpt',              # Keep original excerpt for reference
    ]

    df_out = df[output_cols].copy()
    df_out = df_out.sort_values('date', ascending=False).reset_index(drop=True)

    # --- Save ---
    df_out.to_csv(output_path, index=False)

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"  Input:  {initial} articles")
    print(f"  Output: {len(df_out)} articles")
    print(f"  Removed: {initial - len(df_out)} (promo + short + dupes)")

    print(f"\n  By company:")
    all_comps = df_out['companies_list'].explode()
    for comp in VALID_COMPANIES:
        count = (all_comps == comp).sum()
        print(f"    {comp}: {count}")

    print(f"\n  Content quality:")
    print(f"    {df_out['content_quality'].value_counts().to_string()}")

    print(f"\n  Content length (after cleaning):")
    print(f"    Mean:   {df_out['clean_length'].mean():.0f} chars")
    print(f"    Median: {df_out['clean_length'].median():.0f} chars")
    print(f"    Min:    {df_out['clean_length'].min()}")
    print(f"    Max:    {df_out['clean_length'].max()}")

    print(f"\n  Date range: {df_out['date'].min()} → {df_out['date'].max()}")
    print(f"\n  Output: {output_path}\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, INPUT_CSV)
    output_path = os.path.join(current_dir, OUTPUT_CSV)

    if os.path.exists(input_path):
        clean_techcrunch(input_path, output_path)
    else:
        print(f"Error: {INPUT_CSV} not found")
        print(f"Place techcrunch_full_content.csv next to this script.")
