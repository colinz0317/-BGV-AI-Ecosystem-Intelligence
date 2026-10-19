"""
BGV CVC Data Scraper v3
========================
Based on debug results:
  ✅ Google News RSS  → main source (100 entries per query)
  ✅ TechCrunch RSS   → supplement (20 entries, main feed only)
  ❌ TechCrunch API   → removed (404)
  ❌ TechCrunch Funding RSS → removed (404)

Usage:
    pip install feedparser requests beautifulsoup4 pandas lxml
    python cvc_scraper_v3.py
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
from datetime import datetime
from urllib.parse import quote_plus

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COMPANIES = ["NVIDIA", "Microsoft", "Google", "OpenAI"]

COMPANY_ALIASES = {
    "NVIDIA":    ["NVIDIA", "Nvidia", "nvidia", "NVDA"],
    "Microsoft": ["Microsoft", "microsoft", "MSFT", "M12 venture", "M12 Ventures"],
    "Google":    ["Google", "google", "Alphabet", "alphabet", "GOOGL", "Google Ventures", "GV ", "CapitalG"],
    "OpenAI":    ["OpenAI", "openai", "Open AI"],
}

CATEGORY_MAP = {
    "AI Infrastructure":   ["data center", "gpu", "chip", "compute", "infrastructure", "hardware", "semiconductor", "hpc", "accelerator", "server"],
    "Cloud Computing":     ["cloud", "azure", "gcp", "aws", "saas", "iaas", "paas", "serverless"],
    "AI/ML Platform":      ["machine learning", "deep learning", "llm", "foundation model", "generative ai", "ai platform", "mlops", "ai model", "large language"],
    "Cybersecurity":       ["security", "cybersecurity", "threat", "zero trust", "encryption"],
    "Enterprise Software": ["enterprise", "erp", "crm", "workflow", "automation", "productivity", "copilot", "office"],
    "Healthcare AI":       ["health", "biotech", "pharma", "medical", "clinical", "drug"],
    "Autonomous/Robotics": ["autonomous", "self-driving", "robotics", "drone", "robot"],
    "Developer Tools":     ["developer", "devops", "api", "sdk", "code", "github", "open source"],
    "Data & Analytics":    ["data", "analytics", "database", "warehouse", "lakehouse"],
    "Vertical AI":         ["fintech", "legaltech", "edtech", "agritech"],
}

EVENT_TYPE_MAP = {
    "Strategic Investment": ["invests", "investment", "invested", "backing", "backs", "venture", "fund", "stake"],
    "Acquisition":         ["acquires", "acquisition", "acquired", "buys", "bought", "takeover", "merger", "merge"],
    "Strategic Partnership":["partnership", "partners", "collaborate", "collaboration", "alliance", "deal", "agreement"],
    "Funding Round":       ["series a", "series b", "series c", "series d", "seed", "funding round", "raises", "raised", "round"],
    "Product/Platform Launch": ["launches", "launched", "unveils", "announces", "introduces", "release", "rolls out"],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36"
}


# ============================================================
# HELPERS
# ============================================================

def classify_category(text):
    text_lower = text.lower()
    scores = {}
    for cat, kws in CATEGORY_MAP.items():
        score = sum(1 for kw in kws if kw in text_lower)
        if score > 0:
            scores[cat] = score
    return max(scores, key=scores.get) if scores else "General Technology"


def classify_event_type(text):
    text_lower = text.lower()
    scores = {}
    for et, kws in EVENT_TYPE_MAP.items():
        score = sum(1 for kw in kws if kw in text_lower)
        if score > 0:
            scores[et] = score
    return max(scores, key=scores.get) if scores else "Other"


def extract_amount(text):
    patterns = [
        r'\$(\d+(?:\.\d+)?)\s*(billion|B)\b',
        r'\$(\d+(?:\.\d+)?)\s*(million|M)\b',
        r'\$(\d+(?:\.\d+)?)\s*(thousand|K)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1))
            unit = match.group(2).lower()
            if unit in ['billion', 'b']:
                return f"${amount}B"
            elif unit in ['million', 'm']:
                return f"${amount}M"
    return "Undisclosed"


def match_company(text):
    matched = []
    for company, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            if alias in text:
                matched.append(company)
                break
    return list(set(matched))


def parse_date(date_str):
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return datetime.now().strftime("%Y-%m-%d")


def clean_html(raw_html):
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r'\s+', ' ', text).strip()


# ============================================================
# SCRAPER 1 (MAIN): GOOGLE NEWS RSS
# ============================================================

def scrape_google_news():
    """
    Main data source. Each query returns ~100 articles.
    Strategy: many targeted queries per company to maximize coverage.
    """
    print("=" * 60)
    print("SCRAPER 1 [MAIN]: Google News RSS")
    print("=" * 60)

    results = []
    seen_urls = set()

    # Build comprehensive query list
    # More queries = more coverage. Google News handles volume well.
    query_templates = [
        # CVC / investment queries
        '"{company}" invests startup',
        '"{company}" investment AI startup',
        '"{company}" venture capital',
        '"{company}" strategic investment',
        '"{company}" funding round',
        '"{company}" leads funding',
        '"{company}" backs startup',
        '"{company}" corporate venture',
        # Acquisition queries
        '"{company}" acquires',
        '"{company}" acquisition AI',
        '"{company}" acquisition startup',
        # Partnership queries
        '"{company}" partnership AI',
        '"{company}" strategic partnership startup',
        '"{company}" collaboration AI',
    ]

    # Add CVC arm-specific queries
    cvc_arm_queries = {
        "Microsoft": ['"M12" investment', '"M12 Ventures" startup'],
        "Google":    ['"Google Ventures" investment', '"GV" leads funding', '"CapitalG" investment'],
        "NVIDIA":    ['"NVentures" investment', '"NVIDIA" GPU startup'],
        "OpenAI":    ['"OpenAI" startup fund', '"OpenAI" investment'],
    }

    total_queries = 0
    
    for company in TARGET_COMPANIES:
        print(f"\n  📡 {company}")
        company_count = 0

        # Standard queries
        for template in query_templates:
            query = template.format(company=company)
            count = _fetch_google_news_query(query, company, results, seen_urls)
            company_count += count
            total_queries += 1
            time.sleep(1.0)  # polite rate limiting

        # CVC arm queries
        for extra_query in cvc_arm_queries.get(company, []):
            count = _fetch_google_news_query(extra_query, company, results, seen_urls)
            company_count += count
            total_queries += 1
            time.sleep(1.0)

        print(f"     {company} subtotal: {company_count} new records")

    print(f"\n  Total queries executed: {total_queries}")
    print(f"  Google News total: {len(results)} records\n")
    return results


def _fetch_google_news_query(query, primary_company, results, seen_urls):
    """Fetch a single Google News RSS query and append results."""
    encoded = quote_plus(query)
    feed_url = (
        f"https://news.google.com/rss/search?"
        f"q={encoded}&hl=en-US&gl=US&ceid=US:en"
    )

    new_count = 0
    try:
        resp = requests.get(feed_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"     ❌ [{resp.status_code}] {query}")
            return 0

        feed = feedparser.parse(resp.text)
        entries = feed.entries if feed.entries else []

        for entry in entries:
            url = entry.get("link", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = clean_html(entry.get("title", ""))
            summary = clean_html(entry.get("summary", ""))
            published = entry.get("published", "")
            full_text = f"{title} {summary}"

            # Verify at least one target company is mentioned
            companies = match_company(full_text)
            if not companies:
                companies = [primary_company]

            for comp in companies:
                results.append({
                    "date": parse_date(published),
                    "company": comp,
                    "title": title,
                    "description": summary[:500] if summary else title,
                    "event_type": classify_event_type(full_text),
                    "category": classify_category(full_text),
                    "amount": extract_amount(full_text),
                    "target_company": "Unknown",
                    "source_url": url,
                    "source_type": "Google News"
                })
                new_count += 1

    except Exception as e:
        print(f"     ⚠ Error on '{query}': {e}")

    return new_count


# ============================================================
# SCRAPER 2 (SUPPLEMENT): TECHCRUNCH RSS
# ============================================================

def scrape_techcrunch_rss():
    """
    Supplement. Only the main feed works (20 entries).
    Relaxed filtering: any mention of target company gets captured.
    """
    print("=" * 60)
    print("SCRAPER 2 [SUPPLEMENT]: TechCrunch RSS")
    print("=" * 60)

    results = []
    seen_urls = set()

    feeds = [
        ("https://techcrunch.com/feed/", "Main Feed"),
        ("https://techcrunch.com/category/venture/feed/", "Venture"),
        ("https://techcrunch.com/category/artificial-intelligence/feed/", "AI"),
        ("https://techcrunch.com/category/startups/feed/", "Startups"),
    ]

    for feed_url, feed_name in feeds:
        print(f"  Fetching {feed_name}...")
        try:
            feed = feedparser.parse(feed_url)
            entries = feed.entries if feed.entries else []
            matched = 0

            for entry in entries:
                url = entry.get("link", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title = entry.get("title", "")
                summary = clean_html(entry.get("summary", ""))
                published = entry.get("published", "")
                full_text = f"{title} {summary}"

                companies = match_company(full_text)
                if not companies:
                    continue

                matched += 1
                for comp in companies:
                    results.append({
                        "date": parse_date(published),
                        "company": comp,
                        "title": title,
                        "description": summary[:500] if summary else title,
                        "event_type": classify_event_type(full_text),
                        "category": classify_category(full_text),
                        "amount": extract_amount(full_text),
                        "target_company": "Unknown",
                        "source_url": url,
                        "source_type": "TechCrunch"
                    })

            print(f"    {feed_name}: {len(entries)} entries → {matched} matched")
            time.sleep(1)

        except Exception as e:
            print(f"    Error on {feed_name}: {e}")

    print(f"\n  TechCrunch RSS total: {len(results)} records\n")
    return results


# ============================================================
# ENRICHMENT: Fetch full article text for better descriptions
# ============================================================

def enrich_articles(df, max_fetch=50):
    """
    Fetch full article pages to:
    1. Get better descriptions (meta description / first paragraphs)
    2. Extract investment amounts from full text
    3. Try to identify target companies
    """
    print("=" * 60)
    print("ENRICHMENT: Fetching article details")
    print("=" * 60)

    # Prioritize articles with short descriptions
    df = df.copy()
    short_mask = df['description'].str.len() < 100
    to_fetch_idx = df[short_mask].head(max_fetch).index.tolist()

    # Also fetch some with "Unknown" target company
    unknown_mask = (df['target_company'] == "Unknown") & (~df.index.isin(to_fetch_idx))
    to_fetch_idx += df[unknown_mask].head(max(0, max_fetch - len(to_fetch_idx))).index.tolist()
    to_fetch_idx = list(set(to_fetch_idx))[:max_fetch]

    print(f"  Fetching {len(to_fetch_idx)} article pages for enrichment...\n")

    fetched = 0
    for idx in to_fetch_idx:
        url = df.at[idx, 'source_url']

        # Skip Google News redirect URLs (they redirect, hard to scrape)
        if 'news.google.com' in url:
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10, allow_redirects=True)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)

            # 1. Better description
            meta = soup.find("meta", attrs={"property": "og:description"})
            if not meta:
                meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content") and len(meta["content"]) > 50:
                df.at[idx, 'description'] = meta["content"][:500]
            else:
                paragraphs = soup.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paragraphs[:5])
                if len(text) > 50:
                    df.at[idx, 'description'] = text[:500]

            # 2. Extract amount from full text
            if df.at[idx, 'amount'] == "Undisclosed":
                amount = extract_amount(page_text)
                if amount != "Undisclosed":
                    df.at[idx, 'amount'] = amount

            fetched += 1
            time.sleep(0.8)

        except Exception:
            pass

    print(f"  Successfully enriched {fetched} articles.\n")
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 60)
    print("  BGV CVC DATA SCRAPER v3")
    print(f"  Target: {', '.join(TARGET_COMPANIES)}")
    print(f"  Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cvc_raw_data")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # --- Main source: Google News ---
    gn = scrape_google_news()
    all_results.extend(gn)

    # --- Supplement: TechCrunch RSS ---
    tc = scrape_techcrunch_rss()
    all_results.extend(tc)

    if not all_results:
        print("\n⚠️  NO DATA COLLECTED. Check internet connection.")
        return

    # --- Combine ---
    df = pd.DataFrame(all_results)
    print(f"Raw records collected: {len(df)}")

    # --- Dedup ---
    df = df.drop_duplicates(subset=['source_url'], keep='first')
    df['_key'] = df['company'] + '|' + df['title'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['_key'], keep='first')
    df = df.drop(columns=['_key'])
    print(f"After dedup: {len(df)}")

    # --- Enrich ---
    df = enrich_articles(df, max_fetch=40)

    # --- Save ---
    combined_path = os.path.join(output_dir, "cvc_raw_combined.csv")
    df.to_csv(combined_path, index=False)

    # Per-source saves
    for src in df['source_type'].unique():
        src_df = df[df['source_type'] == src]
        safe = src.lower().replace(" ", "_")
        src_df.to_csv(os.path.join(output_dir, f"cvc_raw_{safe}.csv"), index=False)

    # --- Summary ---
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'=' * 60}")
    print("  SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total records:  {len(df)}")
    print(f"  Time elapsed:   {elapsed:.0f} seconds")
    print(f"\n  By Company:")
    for comp in TARGET_COMPANIES:
        count = len(df[df['company'] == comp])
        print(f"    {comp}: {count}")
    print(f"\n  By Source:")
    for src, count in df['source_type'].value_counts().items():
        print(f"    {src}: {count}")
    print(f"\n  By Event Type:")
    for et, count in df['event_type'].value_counts().items():
        print(f"    {et}: {count}")
    print(f"\n  Date Range: {df['date'].min()} → {df['date'].max()}")
    
    # Amount stats
    has_amount = len(df[df['amount'] != 'Undisclosed'])
    print(f"  Records with $ amount: {has_amount}")
    
    print(f"\n  Output: {combined_path}")
    print(f"\n  Next step: python cvc_cleaner.py\n")


if __name__ == "__main__":
    main()
