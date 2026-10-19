"""
BGV Announcement Scraper v3 — Google-only (Fixed)
==================================================
Scrapes announcements from blog.google with reliable date extraction.

Why this exists
---------------
v1 used <lastmod> from sitemap.xml as document_date, but lastmod is the
URL's last-modified time, not the article's publication date. When Google
regenerated their sitemap (e.g. Jan 2026 and Mar 2026), thousands of old
URLs got their lastmod refreshed, causing 11k+ Google chunks to be wrongly
dated to those two periods.

v2 fixed this for Microsoft and NVIDIA by extracting date from HTML meta
tags (article:published_time, etc.) at fetch time. v3 applies the same
fix to Google.

What's different from v1/v2
---------------------------
1. fetch_article extracts date from HTML meta tags FIRST, then falls back
   to discovery-stage date (RSS published / sitemap lastmod) only if the
   page itself has no embedded date metadata.
2. Adds a `date_source` column tracking provenance of every date so you can
   audit data quality. Values: html_meta | rss_published | sitemap_lastmod
   | google_news_published | none
3. Single-company (Google only). Microsoft and NVIDIA continue to use v2.

Output schema (compatible with v1/v2 + one new column)
------------------------------------------------------
    url, company, title, date, date_source, full_content,
    content_length, fetch_status, discovery_source

Usage
-----
    pip install requests beautifulsoup4 lxml feedparser pandas tqdm
    python announcement_scraper_google_v3.py

Input:  none (configures Google blog directly)
Output: announcements_google_v3.csv
"""

import os
import re
import time
import random
import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import quote_plus

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_CSV = "announcements_google_v3.csv"
PROGRESS_CSV = "announcements_google_v3_progress.csv"

DATE_START = "2023-01-01"
DATE_END = "2026-12-31"

MIN_DELAY = 1.0
MAX_DELAY = 2.5
SAVE_INTERVAL = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ============================================================
# GOOGLE BLOG CONFIG (copied from v1, validated)
# ============================================================

CONFIG = {
    "name": "Google Blog",
    "domain": "blog.google",
    "rss_feeds": [
        "https://blog.google/rss/",
        "https://blog.google/technology/ai/rss/",
        "https://blog.google/products/google-cloud/rss/",
        "https://blog.google/outreach-initiatives/google-org/rss/",
    ],
    "sitemap_urls": [
        "https://blog.google/sitemap.xml",
        "https://blog.google/sitemap-posts.xml",
    ],
    "google_news_queries": [
        'site:blog.google AI',
        'site:blog.google cloud',
        'site:blog.google partnership',
        'site:blog.google investment',
        'site:blog.google announcement',
        'site:blog.google launch',
    ],
    "article_selectors": [
        {"class_": re.compile(r"article-body|article__body|article-content|uni-article-body")},
        {"class_": re.compile(r"post-content|entry-content")},
    ],
}


# ============================================================
# URL DISCOVERY (with date provenance tracking)
# ============================================================

def discover_urls_rss():
    """RSS discovery. Returns dict: url -> {title, date, date_source, discovery_source}"""
    urls = {}
    for feed_url in CONFIG["rss_feeds"]:
        try:
            print(f"    RSS: {feed_url}")
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                url = entry.get("link", "")
                if not url:
                    continue
                pub_date = entry.get("published", "")
                parsed_date = _parse_date(pub_date)
                urls[url] = {
                    "title": entry.get("title", ""),
                    "date": parsed_date,
                    # RSS-published is reasonably reliable but not authoritative
                    "date_source": "rss_published" if parsed_date else "none",
                    "discovery_source": "rss",
                }
            print(f"      -> {len(feed.entries)} entries")
            time.sleep(0.5)
        except Exception as e:
            print(f"      Error: {e}")
    return urls


def discover_urls_sitemap():
    """Sitemap discovery. lastmod is captured but flagged as unreliable -
    fetch_article will overwrite this with HTML meta date if available."""
    urls = {}
    for sitemap_url in CONFIG["sitemap_urls"]:
        try:
            print(f"    Sitemap: {sitemap_url}")
            resp = requests.get(sitemap_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"      -> HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "lxml-xml")
            sub_sitemaps = soup.find_all("sitemap")

            if sub_sitemaps:
                # Sitemap index - recurse into post sitemaps only
                print(f"      -> sitemap index with {len(sub_sitemaps)} sub-sitemaps")
                for sm in sub_sitemaps:
                    loc = sm.find("loc")
                    if loc:
                        sub_url = loc.text.strip()
                        if any(kw in sub_url.lower() for kw in ["post", "blog", "news", "press"]):
                            urls.update(_parse_sitemap_xml(sub_url))
                            time.sleep(0.5)
            else:
                urls.update(_parse_sitemap_content(resp.text))

            print(f"      -> {len(urls)} URLs so far")
            time.sleep(0.5)
        except Exception as e:
            print(f"      Error: {e}")
    return urls


def _parse_sitemap_xml(sitemap_url):
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return {}
        return _parse_sitemap_content(resp.text)
    except Exception:
        return {}


def _parse_sitemap_content(xml_text):
    urls = {}
    soup = BeautifulSoup(xml_text, "lxml-xml")
    for url_tag in soup.find_all("url"):
        loc = url_tag.find("loc")
        lastmod = url_tag.find("lastmod")
        if loc:
            url = loc.text.strip()
            lastmod_date = lastmod.text.strip()[:10] if lastmod else ""

            # Filter by date range. lastmod can be wrong, but if it's WAY out
            # of range we skip the URL entirely - this is just a coarse filter.
            if lastmod_date and (lastmod_date < DATE_START or lastmod_date > DATE_END):
                continue

            urls[url] = {
                "title": "",
                # Provisional date from sitemap; will be overridden by HTML
                # meta if the article page has a real publish date.
                "date": lastmod_date,
                "date_source": "sitemap_lastmod" if lastmod_date else "none",
                "discovery_source": "sitemap",
            }
    return urls


def discover_urls_google_news():
    """Google News RSS search."""
    urls = {}
    for query in CONFIG["google_news_queries"]:
        try:
            encoded = quote_plus(query)
            feed_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(feed_url)

            for entry in feed.entries:
                url = entry.get("link", "")
                title = entry.get("title", "")
                pub_date = entry.get("published", "")

                if CONFIG["domain"] not in url:
                    if CONFIG["domain"].split('.')[0] not in title.lower():
                        continue

                parsed_date = _parse_date(pub_date)
                urls[url] = {
                    "title": title,
                    "date": parsed_date,
                    "date_source": "google_news_published" if parsed_date else "none",
                    "discovery_source": "google_news",
                }
            time.sleep(1.0)
        except Exception:
            pass

    print(f"    Google News: {len(urls)} URLs from {len(CONFIG['google_news_queries'])} queries")
    return urls


# ============================================================
# ARTICLE FETCH (with HTML meta date extraction - the actual fix)
# ============================================================

def fetch_article(url):
    """
    Fetch article and extract title, date, and full text.
    Date extraction priority:
      1. HTML meta: article:published_time / datePublished / date
      2. <time datetime="..."> tag
      3. (caller falls back to discovery-stage date if both above empty)
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            return {
                "title": "", "date": "", "date_source": "",
                "full_content": "", "fetch_status": f"http_{resp.status_code}",
            }

        soup = BeautifulSoup(resp.text, "lxml")

        # --- Title ---
        title = ""
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"]
        elif soup.find("title"):
            title = soup.find("title").get_text(strip=True)

        # --- Date (the actual fix) ---
        date = ""
        date_source = ""
        for meta_name in ["article:published_time", "datePublished", "date"]:
            meta = (soup.find("meta", attrs={"property": meta_name}) or
                    soup.find("meta", attrs={"name": meta_name}))
            if meta and meta.get("content"):
                date = meta["content"][:10]
                date_source = "html_meta"
                break
        if not date:
            time_tag = soup.find("time", attrs={"datetime": True})
            if time_tag:
                date = time_tag["datetime"][:10]
                date_source = "html_meta"

        # --- Article body ---
        full_text, status = _extract_article_text(soup, CONFIG["article_selectors"])

        return {
            "title": title,
            "date": date,
            "date_source": date_source,
            "full_content": full_text,
            "fetch_status": status,
        }

    except requests.exceptions.Timeout:
        return {
            "title": "", "date": "", "date_source": "",
            "full_content": "", "fetch_status": "timeout",
        }
    except Exception as e:
        return {
            "title": "", "date": "", "date_source": "",
            "full_content": "", "fetch_status": f"error: {str(e)[:50]}",
        }


def _extract_article_text(soup, selectors):
    """Extract full article body. Returns (text, status)."""
    # Try selectors first
    for selector in selectors:
        container = soup.find("div", **selector)
        if container:
            text = _extract_text(container)
            if len(text) > 200:
                return text, "success"

    # Fallback: <article> tag
    article = soup.find("article")
    if article:
        text = _extract_text(article)
        if len(text) > 200:
            return text, "success"

    # Fallback: paragraphs in main/body
    main = soup.find("main") or soup.find("body")
    if main:
        paras = [p.get_text(strip=True) for p in main.find_all("p")
                 if len(p.get_text(strip=True)) > 30]
        text = _clean("\n\n".join(paras))
        if len(text) > 200:
            return text, "success"

    # Meta description as last resort
    meta_desc = soup.find("meta", attrs={"property": "og:description"})
    if meta_desc and meta_desc.get("content"):
        return _clean(meta_desc["content"]), "partial"

    return "", "title_only"


def _extract_text(container):
    """Strip noise from container and join paragraphs."""
    for tag in container.find_all(["script", "style", "nav", "aside",
                                     "figure", "figcaption", "iframe", "video"]):
        tag.decompose()
    for ad in container.find_all(class_=re.compile(r"ad-|promo|related|newsletter|signup|share|social")):
        ad.decompose()
    paras = container.find_all(["p", "h2", "h3", "li"])
    texts = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 15]
    return _clean("\n\n".join(texts))


def _clean(text):
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _parse_date(date_str):
    """Parse common RSS/HTTP date formats to YYYY-MM-DD."""
    if not date_str:
        return ""
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


# ============================================================
# PROGRESS / RESUME
# ============================================================

def load_progress():
    if os.path.exists(PROGRESS_CSV):
        df = pd.read_csv(PROGRESS_CSV)
        print(f"  Resuming: {len(df)} URLs already processed")
        return set(df['url']), df.to_dict('records')
    return set(), []


def save_progress(records):
    if records:
        pd.DataFrame(records).to_csv(PROGRESS_CSV, index=False)


# ============================================================
# MAIN
# ============================================================

def main():
    start = datetime.now()
    print("\n" + "=" * 60)
    print("  BGV Announcement Scraper v3 (Google only)")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  Domain: {CONFIG['domain']}")
    print(f"  Date range: {DATE_START} to {DATE_END}")

    done_urls, all_results = load_progress()

    # ===== URL Discovery =====
    print(f"\n[Discovery]")
    print(f"  [1] RSS feeds...")
    rss_urls = discover_urls_rss()
    print(f"  [2] Sitemaps...")
    sitemap_urls = discover_urls_sitemap()
    print(f"  [3] Google News...")
    gn_urls = discover_urls_google_news()

    # Merge with priority: rss > google_news > sitemap
    # (RSS published date and Google News date are more reliable than sitemap lastmod)
    all_urls = {}
    all_urls.update(sitemap_urls)
    all_urls.update(gn_urls)
    all_urls.update(rss_urls)  # last update wins

    # Filter out already-processed
    pending = {u: m for u, m in all_urls.items() if u not in done_urls}
    print(f"\n  Total discovered: {len(all_urls)}")
    print(f"  Already processed: {len(done_urls)}")
    print(f"  To fetch: {len(pending)}")

    if not pending:
        print("\n  All URLs already processed.")
        _write_output(all_results)
        return

    # ===== Fetch articles =====
    print(f"\n[Fetching {len(pending)} articles]")
    items = list(pending.items())
    iterator = tqdm(items, desc="  Fetching") if HAS_TQDM else items

    fetched = 0
    success = 0
    date_source_stats = {"html_meta": 0, "rss_published": 0,
                         "sitemap_lastmod": 0, "google_news_published": 0,
                         "none": 0}

    for url, meta in iterator:
        article = fetch_article(url)

        # Date logic: HTML meta wins; if no HTML meta, fall back to discovery date
        final_date = article["date"]
        final_date_source = article["date_source"]
        if not final_date:
            final_date = meta.get("date", "")
            final_date_source = meta.get("date_source", "none") if final_date else "none"

        date_source_stats[final_date_source] = date_source_stats.get(final_date_source, 0) + 1

        all_results.append({
            "url": url,
            "company": "Google",
            "title": article["title"] or meta.get("title", ""),
            "date": final_date,
            "date_source": final_date_source,
            "full_content": article["full_content"],
            "content_length": len(article["full_content"]),
            "fetch_status": article["fetch_status"],
            "discovery_source": meta.get("discovery_source", ""),
        })

        done_urls.add(url)
        fetched += 1
        if article["fetch_status"] == "success":
            success += 1

        if fetched % SAVE_INTERVAL == 0:
            save_progress(all_results)
            if not HAS_TQDM:
                elapsed = (datetime.now() - start).total_seconds()
                rate = fetched / elapsed if elapsed > 0 else 0
                eta = (len(pending) - fetched) / rate / 60 if rate > 0 else 0
                print(f"    Progress: {fetched}/{len(pending)} "
                      f"({success} ok) ETA: {eta:.0f}min")

        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    # Final save
    save_progress(all_results)
    _write_output(all_results)

    # Summary
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  COMPLETE - {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"  Total URLs fetched: {fetched}")
    print(f"  Successful (full text): {success}")
    print()
    print(f"  Date provenance breakdown:")
    for src, n in sorted(date_source_stats.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f"    {src:30s} {n}")
    print()
    print(f"  Output: {OUTPUT_CSV}")


def _write_output(records):
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
