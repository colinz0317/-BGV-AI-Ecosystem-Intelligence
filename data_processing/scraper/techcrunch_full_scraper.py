"""
TechCrunch Full Article Scraper
=================================
Uses your existing techcrunch_news_all_companies.csv (5418 article URLs)
to fetch the FULL article text instead of the truncated 500-char preview.

Features:
  - Auto-resume: saves progress every 50 articles, can restart without re-fetching
  - Rate limiting: 1-2 sec between requests to avoid blocking
  - Multiple extraction methods: og:description → article body → paragraphs
  - Progress bar in terminal

Usage:
    pip install requests beautifulsoup4 pandas lxml tqdm
    python techcrunch_full_scraper.py

Input:  techcrunch_news_all_companies.csv (your existing file)
Output: techcrunch_full_content.csv (with complete article text)
"""

import os
import re
import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Tip: pip install tqdm for a progress bar\n")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "techcrunch_news_all_companies.csv"
OUTPUT_CSV = "techcrunch_full_content.csv"
PROGRESS_CSV = "techcrunch_scrape_progress.csv"  # Auto-resume file

# Rate limiting (be polite to TechCrunch servers)
MIN_DELAY = 1.0   # seconds between requests
MAX_DELAY = 2.0

# Save progress every N articles
SAVE_INTERVAL = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# ARTICLE TEXT EXTRACTION
# ============================================================

def extract_article_text(html):
    """
    Extract full article text from a TechCrunch page.
    Uses multiple strategies in order of reliability.
    """
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: Find the article body container
    # TechCrunch uses specific CSS classes for article content
    article_selectors = [
        {"class_": "article-content"},
        {"class_": "entry-content"},
        {"class_": re.compile(r"post-content|article__content|article-body")},
        {"name": "article"},
    ]

    for selector in article_selectors:
        container = soup.find("div", **selector) if "name" not in selector else soup.find(**selector)
        if container:
            # Remove unwanted elements
            for unwanted in container.find_all(["script", "style", "nav", "aside",
                                                  "figure", "figcaption", "iframe"]):
                unwanted.decompose()

            # Remove ad blocks and related articles
            for ad in container.find_all(class_=re.compile(r"ad-|promo|related|newsletter|signup")):
                ad.decompose()

            paragraphs = container.find_all("p")
            if paragraphs:
                text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                if len(text) > 200:
                    return clean_text(text)

    # Strategy 2: Get all <p> tags within the main content area
    main = soup.find("main") or soup.find("body")
    if main:
        paragraphs = main.find_all("p")
        # Filter out short navigation/footer text
        good_paras = [p.get_text(strip=True) for p in paragraphs
                      if len(p.get_text(strip=True)) > 30]
        if good_paras:
            text = "\n\n".join(good_paras)
            if len(text) > 200:
                return clean_text(text)

    # Strategy 3: meta description (better than nothing)
    meta = soup.find("meta", attrs={"property": "og:description"})
    if not meta:
        meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        return clean_text(meta["content"])

    return ""


def clean_text(text):
    """Clean extracted text."""
    # Remove HTML entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================
# SCRAPER WITH AUTO-RESUME
# ============================================================

def load_progress(progress_path):
    """Load previously scraped results for resume capability."""
    if os.path.exists(progress_path):
        df = pd.read_csv(progress_path)
        print(f"  Resuming from previous progress: {len(df)} articles already fetched")
        return set(df['id'].astype(str)), df.to_dict('records')
    return set(), []


def save_progress(records, progress_path):
    """Save current progress to disk."""
    if records:
        pd.DataFrame(records).to_csv(progress_path, index=False)


def scrape_full_articles(input_csv, output_csv, progress_csv):
    """Main scraping loop."""
    print("\n" + "=" * 60)
    print("  TechCrunch Full Article Scraper")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load source data
    print(f"\n  Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"  Total articles: {total}")

    # Load progress
    done_ids, results = load_progress(progress_csv)
    remaining = df[~df['id'].astype(str).isin(done_ids)]
    print(f"  Already fetched: {len(done_ids)}")
    print(f"  Remaining: {len(remaining)}")

    if len(remaining) == 0:
        print("\n  All articles already fetched! Generating output...")
        _generate_output(df, results, output_csv)
        return

    # Set up progress tracking
    start_time = datetime.now()
    success_count = 0
    fail_count = 0
    new_results = []

    iterator = tqdm(remaining.iterrows(), total=len(remaining), desc="  Fetching") if HAS_TQDM else remaining.iterrows()

    for idx, row in iterator:
        article_id = str(row['id'])
        url = row['url']

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)

            if resp.status_code == 200:
                full_text = extract_article_text(resp.text)

                new_results.append({
                    "id": article_id,
                    "full_content": full_text,
                    "content_length_new": len(full_text),
                    "fetch_status": "success" if len(full_text) > 200 else "partial",
                })
                success_count += 1
            else:
                new_results.append({
                    "id": article_id,
                    "full_content": "",
                    "content_length_new": 0,
                    "fetch_status": f"http_{resp.status_code}",
                })
                fail_count += 1

        except requests.exceptions.Timeout:
            new_results.append({
                "id": article_id,
                "full_content": "",
                "content_length_new": 0,
                "fetch_status": "timeout",
            })
            fail_count += 1

        except Exception as e:
            new_results.append({
                "id": article_id,
                "full_content": "",
                "content_length_new": 0,
                "fetch_status": f"error: {str(e)[:50]}",
            })
            fail_count += 1

        # Save progress periodically
        if len(new_results) % SAVE_INTERVAL == 0 and new_results:
            all_results = results + new_results
            save_progress(all_results, progress_csv)
            if not HAS_TQDM:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = len(new_results) / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - len(new_results)) / rate / 60 if rate > 0 else 0
                print(f"    Progress: {len(new_results)}/{len(remaining)} "
                      f"({success_count} ok, {fail_count} fail) "
                      f"ETA: {eta:.0f} min")

        # Rate limiting
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    # Final save
    all_results = results + new_results
    save_progress(all_results, progress_csv)

    # Generate output
    _generate_output(df, all_results, output_csv)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"  SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Output:  {output_csv}")


def _generate_output(df_original, results, output_csv):
    """Merge fetched content with original data and save."""
    df_results = pd.DataFrame(results)
    df_results['id'] = df_results['id'].astype(str)
    df_original['id'] = df_original['id'].astype(str)

    # Merge: keep all original columns + add full_content
    df_merged = df_original.merge(
        df_results[['id', 'full_content', 'content_length_new', 'fetch_status']],
        on='id',
        how='left'
    )

    # Where fetch failed, fall back to original content_preview
    df_merged['full_content'] = df_merged['full_content'].fillna('')
    mask_empty = df_merged['full_content'].str.len() < 100
    df_merged.loc[mask_empty, 'full_content'] = df_merged.loc[mask_empty, 'content_preview']
    df_merged.loc[mask_empty, 'fetch_status'] = 'fallback_to_preview'

    df_merged.to_csv(output_csv, index=False)

    # Stats
    print(f"\n  Output Statistics:")
    print(f"    Total articles:  {len(df_merged)}")
    print(f"    Full text fetched: {(df_merged['fetch_status'] == 'success').sum()}")
    print(f"    Partial:         {(df_merged['fetch_status'] == 'partial').sum()}")
    print(f"    Fallback to 500: {(df_merged['fetch_status'] == 'fallback_to_preview').sum()}")
    print(f"    Failed:          {(df_merged['fetch_status'].str.startswith('http_')).sum() + (df_merged['fetch_status'].str.startswith('error')).sum()}")

    avg_old = df_merged['content_preview'].str.len().mean()
    avg_new = df_merged['full_content'].str.len().mean()
    print(f"\n    Avg content length (before): {avg_old:.0f} chars")
    print(f"    Avg content length (after):  {avg_new:.0f} chars")
    print(f"    Improvement: {avg_new/avg_old:.1f}x more text")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(current_dir, INPUT_CSV)
    output_path = os.path.join(current_dir, OUTPUT_CSV)
    progress_path = os.path.join(current_dir, PROGRESS_CSV)

    if not os.path.exists(input_path):
        print(f"Error: {INPUT_CSV} not found in {current_dir}")
        print("Place your techcrunch_news_all_companies.csv next to this script.")
        exit(1)

    print(f"\n  ⏱️  Estimated time: ~2-3 hours for 5418 articles")
    print(f"  💡 You can stop anytime (Ctrl+C) and resume later — progress is auto-saved\n")

    scrape_full_articles(input_path, output_path, progress_path)
