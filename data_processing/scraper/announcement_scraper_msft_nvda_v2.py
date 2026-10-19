"""
BGV Announcement Scraper v2 - Microsoft + NVIDIA (Fixed)
=========================================================
Fixes from v1:
  - Resolves Google News redirect URLs to real article URLs
  - Adds sitemap crawling (MSFT/NVDA blogs are small, won't explode)
  - Uses RSS feeds properly

Usage:
    python announcement_scraper_msft_nvda_v2.py
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
# CONFIG
# ============================================================

OUTPUT_CSV = "announcements_msft_nvda_v2.csv"
PROGRESS_CSV = "announcements_msft_nvda_v2_progress.csv"
MIN_DELAY = 1.0
MAX_DELAY = 2.0
SAVE_INTERVAL = 30

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
}


# ============================================================
# GOOGLE NEWS URL RESOLVER
# ============================================================

def resolve_google_news_url(gn_url):
    """
    Resolve a Google News redirect URL to the actual article URL.
    Google News URLs redirect via HTTP 302/303 to the real article.
    """
    try:
        # Don't follow redirects automatically - capture the redirect location
        resp = requests.get(gn_url, headers=HEADERS, timeout=15, allow_redirects=True)
        final_url = resp.url

        # If we ended up at news.google.com still, try parsing the page
        if "news.google.com" in final_url:
            soup = BeautifulSoup(resp.text, "lxml")
            # Look for the actual article link in the page
            link = soup.find("a", attrs={"data-n-au": True})
            if link and link.get("href"):
                return link["href"]

            # Try meta refresh
            meta = soup.find("meta", attrs={"http-equiv": "refresh"})
            if meta and meta.get("content"):
                match = re.search(r'url=(.+)', meta["content"], re.I)
                if match:
                    return match.group(1).strip()

            # Try finding any external link
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "google.com" not in href:
                    return href

            return None
        else:
            return final_url

    except Exception:
        return None


# ============================================================
# SITEMAP PARSER (simple, filtered)
# ============================================================

def parse_sitemap_filtered(sitemap_url, domain_filter, date_start="2023-01-01"):
    """Parse sitemap and filter URLs by domain and date."""
    urls = {}
    try:
        resp = requests.get(sitemap_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return urls

        soup = BeautifulSoup(resp.text, "lxml-xml")

        # Check if sitemap index
        sub_sitemaps = soup.find_all("sitemap")
        if sub_sitemaps:
            print(f"      Sitemap index: {len(sub_sitemaps)} sub-sitemaps")
            for sm in sub_sitemaps:
                loc = sm.find("loc")
                if loc:
                    sub_url = loc.text.strip()
                    # Only process post/blog sitemaps
                    if any(kw in sub_url.lower() for kw in ["post", "blog", "news", "press", "article"]):
                        sub_urls = _parse_sitemap_urls(sub_url, domain_filter, date_start)
                        urls.update(sub_urls)
                        print(f"        {sub_url[:60]}... → {len(sub_urls)} URLs")
                        time.sleep(0.5)
        else:
            urls = _parse_sitemap_urls(sitemap_url, domain_filter, date_start, soup=soup)

    except Exception as e:
        print(f"      Sitemap error: {e}")

    return urls


def _parse_sitemap_urls(url, domain_filter, date_start, soup=None):
    """Parse URL entries from a sitemap."""
    urls = {}
    try:
        if soup is None:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                return urls
            soup = BeautifulSoup(resp.text, "lxml-xml")

        for url_tag in soup.find_all("url"):
            loc = url_tag.find("loc")
            lastmod = url_tag.find("lastmod")
            if not loc:
                continue

            page_url = loc.text.strip()

            # Filter by domain
            if domain_filter not in page_url:
                continue

            # Filter by date
            date = lastmod.text.strip()[:10] if lastmod else ""
            if date and date < date_start:
                continue

            urls[page_url] = {"title": "", "date": date}

    except Exception:
        pass

    return urls


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def fetch_article(url, selectors):
    """Fetch full article text from URL."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            return "", f"http_{resp.status_code}"

        soup = BeautifulSoup(resp.text, "lxml")

        # Extract title
        title = ""
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"]
        elif soup.find("title"):
            title = soup.find("title").get_text(strip=True)

        # Extract date
        date = ""
        for meta_name in ["article:published_time", "datePublished", "date"]:
            meta = soup.find("meta", attrs={"property": meta_name}) or soup.find("meta", attrs={"name": meta_name})
            if meta and meta.get("content"):
                date = meta["content"][:10]
                break
        # Also try time tag
        if not date:
            time_tag = soup.find("time", attrs={"datetime": True})
            if time_tag:
                date = time_tag["datetime"][:10]

        # Try selectors for article body
        for selector in selectors:
            container = soup.find("div", **selector)
            if container:
                text = _extract_text(container)
                if len(text) > 200:
                    return _make_result(title, date, text), "success"

        # Fallback: article tag
        article = soup.find("article")
        if article:
            text = _extract_text(article)
            if len(text) > 200:
                return _make_result(title, date, text), "success"

        # Fallback: all paragraphs
        main = soup.find("main") or soup.find("body")
        if main:
            paras = [p.get_text(strip=True) for p in main.find_all("p") if len(p.get_text(strip=True)) > 30]
            text = "\n\n".join(paras)
            if len(text) > 200:
                return _make_result(title, date, _clean(text)), "success"

        # Meta description
        meta_desc = soup.find("meta", attrs={"property": "og:description"})
        if meta_desc and meta_desc.get("content") and "google" not in meta_desc["content"].lower():
            return _make_result(title, date, meta_desc["content"]), "partial"

        return _make_result(title, date, ""), "title_only"

    except requests.exceptions.Timeout:
        return "", "timeout"
    except Exception as e:
        return "", f"error: {str(e)[:50]}"


def _make_result(title, date, text):
    """Pack title, date, text into a structured string for later parsing."""
    return f"TITLE:::{title}|||DATE:::{date}|||TEXT:::{text}"


def _parse_result(result_str):
    """Unpack structured result string."""
    parts = result_str.split("|||")
    title = parts[0].replace("TITLE:::", "") if len(parts) > 0 else ""
    date = parts[1].replace("DATE:::", "") if len(parts) > 1 else ""
    text = parts[2].replace("TEXT:::", "") if len(parts) > 2 else ""
    return title, date, text


def _extract_text(container):
    for tag in container.find_all(["script", "style", "nav", "aside", "figure",
                                    "figcaption", "iframe", "video"]):
        tag.decompose()
    for ad in container.find_all(class_=re.compile(r"ad-|promo|related|newsletter|signup|share|social")):
        ad.decompose()
    paras = container.find_all(["p", "h2", "h3", "li"])
    texts = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 15]
    return _clean("\n\n".join(texts))


def _clean(text):
    text = text.replace("\xa0", " ").replace("&nbsp;", " ")
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ============================================================
# COMPANY CONFIGS
# ============================================================

COMPANIES = {
    "Microsoft": {
        "rss_feeds": [
            "https://blogs.microsoft.com/feed/",
            "https://blogs.microsoft.com/blog/feed/",
            "https://blogs.microsoft.com/ai/feed/",
            "https://blogs.microsoft.com/on-the-issues/feed/",
        ],
        "sitemaps": [
            "https://blogs.microsoft.com/sitemap.xml",
            "https://blogs.microsoft.com/sitemap_index.xml",
        ],
        "domain_filter": "blogs.microsoft.com",
        "google_queries": [
            'site:blogs.microsoft.com AI',
            'site:blogs.microsoft.com Azure',
            'site:blogs.microsoft.com partnership',
            'site:blogs.microsoft.com Copilot',
            'site:blogs.microsoft.com OpenAI',
            'site:blogs.microsoft.com investment',
            'site:blogs.microsoft.com startup',
            'site:blogs.microsoft.com announcement',
        ],
        "selectors": [
            {"class_": re.compile(r"entry-content|post-content|article-content")},
        ],
    },
    "NVIDIA": {
        "rss_feeds": [
            "https://nvidianews.nvidia.com/rss",
            "https://blogs.nvidia.com/feed/",
        ],
        "sitemaps": [
            "https://nvidianews.nvidia.com/sitemap.xml",
            "https://blogs.nvidia.com/sitemap.xml",
        ],
        "domain_filter": "nvidia.com",
        "google_queries": [
            'site:nvidianews.nvidia.com 2025',
            'site:nvidianews.nvidia.com 2024',
            'site:nvidianews.nvidia.com 2023',
            'site:blogs.nvidia.com AI',
            'site:blogs.nvidia.com partnership',
            'site:blogs.nvidia.com data center',
            'site:blogs.nvidia.com enterprise',
        ],
        "selectors": [
            {"class_": re.compile(r"press-release-body|article-body|content-area")},
            {"class_": re.compile(r"entry-content|post-content")},
        ],
    },
}


# ============================================================
# MAIN
# ============================================================

def load_progress():
    if os.path.exists(PROGRESS_CSV):
        df = pd.read_csv(PROGRESS_CSV)
        print(f"  Resuming: {len(df)} already done")
        return set(df['url']), df.to_dict('records')
    return set(), []


def save_progress(records):
    if records:
        pd.DataFrame(records).to_csv(PROGRESS_CSV, index=False)


def main():
    start = datetime.now()
    print("\n" + "=" * 60)
    print("  BGV Announcement Scraper v2 (Microsoft + NVIDIA)")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    done_urls, all_results = load_progress()

    # Also load v1 results to extract and resolve Google News URLs
    v1_path = "announcements_msft_nvda.csv"
    gn_urls_to_resolve = {}
    if os.path.exists(v1_path):
        v1 = pd.read_csv(v1_path)
        gn = v1[v1['url'].str.contains('news.google.com', na=False)]
        for _, r in gn.iterrows():
            if r['url'] not in done_urls:
                gn_urls_to_resolve[r['url']] = {
                    "title": str(r.get('title', '')),
                    "company": r['company'],
                }
        print(f"\n  Found {len(gn_urls_to_resolve)} Google News URLs from v1 to resolve")

    for company, config in COMPANIES.items():
        print(f"\n{'='*60}")
        print(f"  {company}")
        print(f"{'='*60}")

        all_urls = {}  # url → {"title", "date"}

        # --- Source 1: RSS ---
        print(f"\n  [1] RSS Feeds...")
        for feed_url in config["rss_feeds"]:
            try:
                feed = feedparser.parse(feed_url)
                count = 0
                for entry in feed.entries:
                    url = entry.get("link", "")
                    if url and url not in done_urls:
                        all_urls[url] = {
                            "title": entry.get("title", ""),
                            "date": entry.get("published", "")[:10] if entry.get("published") else "",
                        }
                        count += 1
                print(f"    {feed_url[:50]}... → {count} new")
                time.sleep(0.5)
            except Exception as e:
                print(f"    Error: {e}")

        # --- Source 2: Sitemap ---
        print(f"\n  [2] Sitemaps...")
        for sm_url in config["sitemaps"]:
            print(f"    {sm_url}")
            sm_urls = parse_sitemap_filtered(sm_url, config["domain_filter"])
            new = {u: m for u, m in sm_urls.items() if u not in done_urls}
            all_urls.update(new)
            print(f"      → {len(new)} new URLs")
            time.sleep(0.5)

        # --- Source 3: Resolve Google News URLs from v1 ---
        company_gn = {u: m for u, m in gn_urls_to_resolve.items() if m["company"] == company}
        if company_gn:
            print(f"\n  [3] Resolving {len(company_gn)} Google News URLs...")
            resolved = 0
            for gn_url, meta in list(company_gn.items())[:200]:  # Cap at 200 to save time
                real_url = resolve_google_news_url(gn_url)
                if real_url and config["domain_filter"] in real_url and real_url not in done_urls:
                    all_urls[real_url] = {"title": meta["title"], "date": ""}
                    resolved += 1
                done_urls.add(gn_url)
                time.sleep(0.5)
            print(f"    Resolved: {resolved} real URLs")

        print(f"\n  Total URLs to fetch: {len(all_urls)}")

        # --- Fetch articles ---
        if not all_urls:
            print(f"  ✅ No new URLs for {company}")
            continue

        print(f"\n  Fetching articles...")
        items = list(all_urls.items())
        iterator = tqdm(items, desc=f"  {company}") if HAS_TQDM else items
        fetched = 0
        success = 0

        for url, meta in iterator:
            result, status = fetch_article(url, config["selectors"])

            if result and "|||" in result:
                title, date, text = _parse_result(result)
            else:
                title = meta.get("title", "")
                date = meta.get("date", "")
                text = result

            if not title:
                title = meta.get("title", "")
            if not date:
                date = meta.get("date", "")

            all_results.append({
                "url": url,
                "company": company,
                "title": title,
                "date": date,
                "full_content": text,
                "content_length": len(text),
                "fetch_status": status,
            })
            done_urls.add(url)
            fetched += 1
            if status == "success":
                success += 1

            if fetched % SAVE_INTERVAL == 0:
                save_progress(all_results)
                if not HAS_TQDM:
                    print(f"    Progress: {fetched}/{len(items)} ({success} success)")

            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

        save_progress(all_results)
        print(f"  ✅ {company}: {fetched} fetched, {success} success")

    # --- Final output ---
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*60}")
    print(f"  COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"  Total: {len(df)}")
    for comp in ["Microsoft", "NVIDIA"]:
        sub = df[df['company'] == comp]
        ok = (sub['fetch_status'] == 'success').sum()
        print(f"    {comp:12}: {len(sub)} ({ok} with full text)")
    print(f"\n  Output: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
