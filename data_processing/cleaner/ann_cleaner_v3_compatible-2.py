"""
BGV Announcement Data Cleaner
===============================
Merges and cleans announcement data from three sources:
  - Google:    announcements_scrape_progress.csv (10K+ from sitemap)
  - Microsoft: announcements_msft_nvda_v2.csv
  - NVIDIA:    announcements_msft_nvda_v2.csv

Cleaning steps:
  1. Merge all three companies
  2. Filter Google: remove non-English, non-article pages
  3. Clean HTML entities and boilerplate
  4. Remove duplicates
  5. Remove very short / empty content
  6. Standardize metadata

Input:  announcements_scrape_progress.csv + announcements_msft_nvda_v2.csv
Output: ann_cleaned.csv
"""

import os
import re
import pandas as pd
from datetime import datetime


INPUT_GOOGLE = "announcements_google_v3.csv"  # CHANGED: now reads v3 output (was announcements_scrape_progress.csv)
INPUT_MSFT_NVDA = "announcements_msft_nvda_v2.csv"
OUTPUT_CSV = "ann_cleaned.csv"

MIN_CONTENT_LENGTH = 150


# ============================================================
# TEXT CLEANING
# ============================================================

TAIL_NOISE = [
    r'Share\s+this\s+(?:article|page|post).*$',
    r'(?:Tags|Categories)\s*:.*$',
    r'(?:Related|Recommended)\s+(?:posts?|articles?|stories?).*$',
    r'About\s+(?:Google|Microsoft|NVIDIA|the\s+author).*$',
    r'Posted\s+(?:in|by).*$',
    r'Follow\s+us\s+(?:on|at).*$',
    r'Subscribe\s+to\s+(?:our|the).*$',
    r'Sign\s+up\s+for.*$',
    r'Forward-Looking Statements.*$',
    r'Safe Harbor.*$',
    r'©\s*\d{4}.*$',
    r'All rights reserved.*$',
]

def clean_text(text):
    if not isinstance(text, str) or len(text) < 10:
        return ""

    # HTML entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&\w+;', '', text)

    # Stray HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Tail noise
    for pattern in TAIL_NOISE:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ============================================================
# GOOGLE FILTER
# ============================================================

# PATCHED: Set to False to drop non-English /intl/ pages (legacy behavior).
# Default True keeps localized announcements (e.g. blog.google/intl/ja-jp/),
# which are real Google announcements for non-English markets, not translations.
KEEP_NON_ENGLISH_INTL = False


def filter_google(df, keep_non_english_intl=KEEP_NON_ENGLISH_INTL):
    """Filter Google blog articles: optionally keep non-English, remove junk."""
    initial = len(df)

    # PATCHED: drop non-English /intl/ pages only if explicitly disabled
    if not keep_non_english_intl:
        intl_non_en = df['url'].str.contains('/intl/', na=False) & ~df['url'].str.contains('/intl/en-', na=False)
        df = df[~intl_non_en]
        print(f"    Removed {intl_non_en.sum()} non-English international pages")
    else:
        intl_non_en_n = (df['url'].str.contains('/intl/', na=False) & ~df['url'].str.contains('/intl/en-', na=False)).sum()
        print(f"    Kept {intl_non_en_n} non-English international pages (KEEP_NON_ENGLISH_INTL=True)")

    # Remove product/platform help pages (not announcements)
    junk_patterns = [
        r'/products/admob/',
        r'/products/adwords/',
        r'/products/marketingplatform/',
        r'/help/',
        r'/support/',
        r'/developers/',
    ]
    for pattern in junk_patterns:
        mask = df['url'].str.contains(pattern, na=False)
        if mask.sum() > 0:
            df = df[~mask]

    print(f"    Google: {initial} → {len(df)}")
    return df


# ============================================================
# MAIN
# ============================================================

def clean_announcements():
    print("\n" + "=" * 60)
    print("  BGV Announcement Data Cleaner")
    print("=" * 60)

    # --- Load & Merge ---
    print(f"\n  [Step 1] Loading and merging data...")
    dfs = []

    if os.path.exists(INPUT_GOOGLE):
        df_g = pd.read_csv(INPUT_GOOGLE)
        df_g = df_g[df_g['company'] == 'Google']
        print(f"    Google: {len(df_g)} records")
        dfs.append(df_g)
    else:
        print(f"    ⚠ Google file not found: {INPUT_GOOGLE}")

    if os.path.exists(INPUT_MSFT_NVDA):
        df_mn = pd.read_csv(INPUT_MSFT_NVDA)
        for comp in ['Microsoft', 'NVIDIA']:
            sub = df_mn[df_mn['company'] == comp]
            print(f"    {comp}: {len(sub)} records")
        dfs.append(df_mn)
    else:
        print(f"    ⚠ MSFT/NVDA file not found: {INPUT_MSFT_NVDA}")

    if not dfs:
        print("  No data loaded. Exiting.")
        return

    # Standardize columns across files
    # PATCHED: also preserve date_source and discovery_source if present (v3 only)
    all_dfs = []
    for d in dfs:
        std = pd.DataFrame({
            'url': d['url'],
            'company': d['company'],
            'title': d.get('title', ''),
            'date': d.get('date', ''),
            'date_source': d.get('date_source', '') if 'date_source' in d.columns else '',
            'discovery_source': d.get('discovery_source', '') if 'discovery_source' in d.columns else '',
            'full_content': d.get('full_content', ''),
            'content_length': d.get('content_length', 0),
            'fetch_status': d.get('fetch_status', ''),
        })
        all_dfs.append(std)

    df = pd.concat(all_dfs, ignore_index=True)
    initial = len(df)
    print(f"\n    Merged total: {initial}")

    # --- Step 2: Filter Google ---
    print(f"\n  [Step 2] Filtering Google non-English/junk...")
    google_mask = df['company'] == 'Google'
    google_df = filter_google(df[google_mask])
    other_df = df[~google_mask]
    df = pd.concat([google_df, other_df], ignore_index=True)

    # --- Step 3: Remove failed fetches ---
    print(f"\n  [Step 3] Removing failed fetches...")
    ok_statuses = ['success', 'partial']
    failed = ~df['fetch_status'].isin(ok_statuses)
    print(f"    Removing {failed.sum()} non-success records")
    df = df[~failed]

    # --- Step 4: Clean text ---
    print(f"\n  [Step 4] Cleaning text...")
    df['clean_content'] = df['full_content'].apply(clean_text)
    df['clean_length'] = df['clean_content'].str.len()

    # --- Step 5: Remove short ---
    print(f"\n  [Step 5] Removing short articles (<{MIN_CONTENT_LENGTH} chars)...")
    short = df['clean_length'] < MIN_CONTENT_LENGTH
    print(f"    Removing {short.sum()} short articles")
    df = df[~short]

    # --- Step 6: Deduplicate ---
    # PATCHED: removed title-based dedup. Same title can occur on multiple
    # genuine articles (e.g. 12 distinct 'Made by Google 2025' articles).
    # URL-based dedup is sufficient.
    print(f"\n  [Step 6] Deduplicating by URL...")
    before = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    print(f"    {before} → {len(df)}")

    # --- Step 7: Standardize company names ---
    print(f"\n  [Step 7] Standardizing...")
    company_map = {
        'Google': 'Google', 'Alphabet': 'Google',
        'Microsoft': 'Microsoft',
        'NVIDIA': 'NVIDIA', 'Nvidia': 'NVIDIA',
    }
    df['company'] = df['company'].map(company_map).fillna(df['company'])
    valid = df['company'].isin(['Google', 'Microsoft', 'NVIDIA'])
    df = df[valid]

    # Clean date
    df['date'] = df['date'].fillna('').str[:10]

    # Generate source_name
    source_name_map = {
        'Google': 'Google Blog',
        'Microsoft': 'Microsoft Blog',
        'NVIDIA': 'NVIDIA Newsroom',
    }
    df['source_name'] = df['company'].map(source_name_map)

    # --- Step 8: Output ---
    # PATCHED: include date_source so downstream knows which dates came from
    # html_meta vs sitemap_lastmod fallback
    output_cols = [
        'url', 'company', 'title', 'date', 'date_source', 'source_name',
        'clean_content', 'clean_length',
    ]
    df_out = df[output_cols].sort_values(['company', 'date'], ascending=[True, False]).reset_index(drop=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"  CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"  Input:  {initial}")
    print(f"  Output: {len(df_out)}")
    print(f"  Removed: {initial - len(df_out)}")

    print(f"\n  By company:")
    for comp in ['Google', 'Microsoft', 'NVIDIA']:
        sub = df_out[df_out['company'] == comp]
        print(f"    {comp:12}: {len(sub)} articles (avg {sub['clean_length'].mean():.0f} chars)")

    print(f"\n  Content length:")
    print(f"    Mean:   {df_out['clean_length'].mean():.0f} chars")
    print(f"    Median: {df_out['clean_length'].median():.0f} chars")

    print(f"\n  Output: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    clean_announcements()
