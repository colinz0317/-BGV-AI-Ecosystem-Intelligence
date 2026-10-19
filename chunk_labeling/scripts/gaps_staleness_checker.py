"""
Gap Staleness Checker
=====================
Takes a filtered gap file (output of gap_critic.py) and checks whether each gap
is still valid by searching for recent news about the company and capability.

For each gap:
1. Constructs a search query from company + gap_title + capability tags
2. Calls Tavily to get recent news (within FRESHNESS_MONTHS)
3. Feeds search results to GPT to judge: STALE / STILL_VALID / UNCLEAR
4. Appends freshness_status + freshness_reason + search_urls to each record

Output:
- gaps_freshness_checked.jsonl  : all gaps with freshness fields added
- gaps_stale.jsonl              : only STALE gaps (for review/discard)
- gaps_still_valid.jsonl        : only STILL_VALID gaps (clean output)
- gaps_freshness_unclear.jsonl  : UNCLEAR gaps (for manual review)
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paths import (
    BASE_DIR,
    FILE_GAPS_CRITIC_FILTERED,
    FILE_GAPS_FRESHNESS_CHECKED,
    FILE_GAPS_STALE,
    FILE_GAPS_VALID,
    FILE_GAPS_UNCLEAR
)

from openai import OpenAI
from tavily import TavilyClient

# Auto-load .env
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())


# =========================
# Configuration
# =========================
INPUT_FILE = str(FILE_GAPS_CRITIC_FILTERED)

OUTPUT_ALL = str(FILE_GAPS_FRESHNESS_CHECKED)
OUTPUT_STALE = str(FILE_GAPS_STALE)
OUTPUT_VALID = str(FILE_GAPS_VALID)
OUTPUT_UNCLEAR = str(FILE_GAPS_UNCLEAR)

MODEL_NAME = "gpt-4o-mini"
FRESHNESS_MONTHS = 12
OLD_EVIDENCE_MONTHS = 6    # Evidence older than this triggers GA-verification secondary search
                           # Set low (6mo) because evidence_dates from Actor are unreliable;
                           # dates are now read directly from evidence_quotes[].document_date
MAX_SEARCH_RESULTS = 5
SLEEP_BETWEEN_GAPS = 1.5
MAX_RETRIES = 3

# Source types that represent "research / vision" statements (as opposed to product announcements)
# These age poorly and require active GA verification when the evidence is old.
RESEARCH_VISION_SOURCE_TYPES = {
    "official_website_announcement",
    "research_blog",
    "blog_post",
    "keynote_transcript",
}

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# =========================
# Prompts
# =========================
STALENESS_SYSTEM_PROMPT = f"""You are an analyst for a venture capital intelligence platform.

Your task: given a capability gap description and recent news about the company,
determine whether the gap is still valid TODAY or has been addressed.

Today's date: {datetime.now().strftime("%B %Y")}.
Only consider news from the last {FRESHNESS_MONTHS} months as relevant.

========================
TIMESTAMP WARNING
========================
Many news aggregator sites re-publish old articles with today's date. You MUST NOT
trust article timestamps. Instead, look for explicit time anchors INSIDE the article
body itself — specific product launch dates, version numbers, event names, quarter
references (e.g. "Q1 2026"), or named releases. If no such internal anchor exists,
treat the article as undated and weight it accordingly.

========================
STALE CRITERIA (strict)
========================
A gap is STALE ONLY IF the news contains clear evidence of ONE of the following:
1. General Availability (GA) launch: the company has shipped a production-ready
   product or feature that directly addresses this gap (not beta, not preview,
   not "coming soon", not "we are exploring").
2. Acquisition: the company has completed (closed, not announced) an acquisition
   of a company whose core product directly fills this gap.

The following do NOT make a gap STALE:
- Research papers, blog posts, or vision statements about future plans
- "We are investing in X" or "We plan to build X" announcements
- Beta, limited preview, or early-access releases
- Partnerships where the capability is still provided by the external partner
  (that signals the gap still exists internally)
- PR language like "we are committed to", "we are exploring", "long-term roadmap"

========================
OLD EVIDENCE RULE (critical)
========================
Some gaps are flagged as OLD EVIDENCE (original source is 18+ months old and is a
research or vision announcement). For these gaps:
- The original statement "we are researching X" or "we plan to build X" is stale
  BY DESIGN — large companies routinely ship capabilities 12-24 months after announcing them.
- You MUST treat the GA Verification Search results as the primary signal.
- If the GA Verification Search finds evidence of a production launch, mark STALE.
- If the GA Verification Search finds only more research/beta/partnership news,
  mark STILL_VALID (the gap persists) but note that the original evidence is old.
- Do NOT simply confirm STILL_VALID because the original research intent remains
  unfulfilled — actively look for whether the capability was shipped.

========================
VERDICT DEFINITIONS
========================
- STALE: Evidence of a GA launch or closed acquisition that directly closes the gap.
- STILL_VALID: No qualifying evidence that the gap has been closed. Includes cases
  where the company is only in early stages, announced plans, or relies on partners.
- UNCLEAR: News is ambiguous or covers a related but not identical capability.
  Cannot determine with confidence even after applying the strict criteria above.

Return ONLY a JSON object with exactly two fields:
{{
  "freshness_status": "STALE" | "STILL_VALID" | "UNCLEAR",
  "freshness_reason": "One sentence citing a specific fact from article content (not just the headline or date)."
}}

No markdown. No explanation outside the JSON object.
"""

STALENESS_USER_TEMPLATE = """Company: {company}

Gap title: {gap_title}

Gap description:
{gap_text}

Capability tags: {capability}

Original evidence age: {evidence_age_note}

Recent news (last {months} months):
{news_text}
{ga_verification_section}
Based on the news above, is this gap STALE (already addressed), STILL_VALID, or UNCLEAR?
"""


# =========================
# Helpers
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def parse_quarter_date(date_str: str) -> Optional[datetime]:
    """
    Parse a document_date string like '2022-Q1' or '2024-Q3' into an approximate datetime.
    Returns None if parsing fails.
    """
    import re
    m = re.match(r"(\d{4})-Q([1-4])", date_str.strip())
    if m:
        year = int(m.group(1))
        quarter = int(m.group(2))
        month = (quarter - 1) * 3 + 1  # Q1→Jan, Q2→Apr, Q3→Jul, Q4→Oct
        return datetime(year, month, 1)
    # Try plain YYYY-MM or YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(date_str.strip()[:10], fmt)
        except ValueError:
            pass
    return None


def is_old_evidence(gap: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Returns (is_old, age_note).
    Evidence is "old" if the oldest evidence date is > OLD_EVIDENCE_MONTHS months ago
    AND the source_type is a research/vision type.

    Date source priority:
      1. evidence_quotes[].document_date  (most reliable — set directly from chunk metadata)
      2. evidence_dates[]                 (legacy fallback — may be unreliable if Actor didn't fill it)
    """
    source_types = set(gap.get("source_types", []))

    if not source_types.intersection(RESEARCH_VISION_SOURCE_TYPES):
        return False, "Source is not a research/vision type."

    # --- Primary: pull document_date from each evidence_quote ---
    date_strings: List[str] = []
    for eq in gap.get("evidence_quotes", []):
        d = eq.get("document_date")
        if d:
            date_strings.append(d)

    # --- Fallback: legacy evidence_dates field ---
    if not date_strings:
        date_strings = gap.get("evidence_dates", [])

    if not date_strings:
        return False, "Evidence date unknown."

    oldest = None
    for ds in date_strings:
        parsed = parse_quarter_date(ds)
        if parsed and (oldest is None or parsed < oldest):
            oldest = parsed

    if oldest is None:
        return False, "Evidence date could not be parsed."

    age_months = (datetime.now() - oldest).days / 30.44
    age_note = f"Original evidence is from ~{oldest.strftime('%B %Y')} (~{int(age_months)} months ago)."

    if age_months > OLD_EVIDENCE_MONTHS:
        return True, age_note
    return False, age_note


def build_search_query(gap: Dict[str, Any]) -> str:
    """
    Build a focused Tavily search query from gap fields.
    Short and specific — Tavily performs best with 3-8 word queries.
    """
    company = gap.get("company", "")
    title = gap.get("gap_title", "")
    caps = gap.get("capability", [])
    cap_hint = caps[0].replace("_", " ") if caps else ""
    query = f"{company} {title} {cap_hint}"
    return query[:120].strip()


def build_ga_verification_query(gap: Dict[str, Any]) -> str:
    """
    Build a targeted product-launch verification query for old research/vision evidence.

    Extracts meaningful domain/product keywords from the gap title by stripping
    generic filler words (AI, Solutions, for, and, Platform, etc.) that add no
    search signal. This produces queries like:
      "NVIDIA Isaac Sim Robotics Humanoid Quadruped GA 2025 2026"
      "Google Pathways Multisensory Complex Task GA 2025 2026"
      "Google indigenous language model Siouan GA 2025 2026"

    Avoids capability tags (ai_agents, ai_strategy_analytics) which are internal
    vocabulary and return no useful Tavily results.
    """
    # Generic words that carry no search signal for product GA queries
    STOPWORDS = {
        "ai", "ai-powered", "ai-driven", "ai-enhanced", "ai-based",
        "for", "and", "the", "of", "in", "on", "to", "a", "an",
        "with", "using", "via", "through", "by", "from",
        "integration", "development", "solutions", "solution",
        "capabilities", "capability", "enhancement", "enhancements",
        "optimization", "platform", "tools", "tool", "systems", "system",
        "framework", "infrastructure", "deployment", "access", "support",
        "driven", "powered", "based", "advanced", "new", "enterprise",
        "scalable", "real-time", "automated", "autonomous",
        "model", "models", "large",
    }

    company = gap.get("company", "")
    title = gap.get("gap_title", "")

    # Tokenise and filter
    raw_words = title.replace("-", " ").split()
    key_words = [w for w in raw_words if w.lower() not in STOPWORDS][:6]
    title_hint = " ".join(key_words)

    query = f"{company} {title_hint} GA generally available 2025 2026"
    return query[:120].strip()


def search_recent_news(query: str) -> List[Dict[str, Any]]:
    """
    Call Tavily search and return a list of result dicts.
    """
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=MAX_SEARCH_RESULTS,
            include_answer=False,
        )
        results = response.get("results", [])
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:1200],
                "published_date": r.get("published_date", ""),
            })
        return formatted
    except Exception as e:
        print(f"    Tavily search failed: {e}")
        return []


def format_news_for_prompt(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No recent news found."
    lines = []
    for i, r in enumerate(results, 1):
        date_str = r.get("published_date", "unknown date")
        lines.append(
            f"[{i}] {r['title']} ({date_str})\n"
            f"URL: {r['url']}\n"
            f"{r['content']}\n"
        )
    return "\n---\n".join(lines)


def judge_staleness(
    gap: Dict[str, Any],
    news_results: List[Dict[str, Any]],
    ga_results: Optional[List[Dict[str, Any]]] = None,
    evidence_age_note: str = "",
) -> Dict[str, Any]:
    """
    Ask GPT to judge whether the gap is STALE / STILL_VALID / UNCLEAR.
    If ga_results is provided, include a second GA-verification block in the prompt.
    """
    news_text = format_news_for_prompt(news_results)

    ga_verification_section = ""
    if ga_results:
        ga_text = format_news_for_prompt(ga_results)
        ga_verification_section = (
            "\n========================\n"
            "GA VERIFICATION SEARCH (targeted product-launch query)\n"
            "========================\n"
            "[OLD EVIDENCE FLAG] The original gap evidence is 18+ months old and was a "
            "research or vision announcement. Apply the OLD EVIDENCE RULE from the system "
            "prompt. Treat this section as the primary signal for whether the gap was closed.\n\n"
            + ga_text
            + "\n"
        )

    user_prompt = STALENESS_USER_TEMPLATE.format(
        company=gap.get("company", ""),
        gap_title=gap.get("gap_title", ""),
        gap_text=gap.get("gap_text", ""),
        capability=", ".join(gap.get("capability", [])),
        months=FRESHNESS_MONTHS,
        news_text=news_text,
        evidence_age_note=evidence_age_note or "Evidence date not available.",
        ga_verification_section=ga_verification_section,
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": STALENESS_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                import re
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?\s*```$", "", raw).strip()

            result = json.loads(raw)

            status = result.get("freshness_status", "UNCLEAR")
            if status not in ("STALE", "STILL_VALID", "UNCLEAR"):
                status = "UNCLEAR"

            all_urls = [r["url"] for r in news_results]
            if ga_results:
                all_urls += [r["url"] for r in ga_results]

            return {
                "freshness_status": status,
                "freshness_reason": result.get("freshness_reason", ""),
                "search_urls": all_urls,
            }

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            print(f"    Attempt {attempt} JSON parse failed: {last_error}")
            time.sleep(2.0 * attempt)

        except Exception as e:
            last_error = str(e)
            print(f"    Attempt {attempt} failed: {last_error}")
            time.sleep(2.0 * attempt)

    return {
        "freshness_status": "UNCLEAR",
        "freshness_reason": f"Judgment failed after {MAX_RETRIES} attempts: {last_error}",
        "search_urls": [r["url"] for r in news_results],
    }


# =========================
# Main
# =========================
def main():
    gaps = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(gaps)} gaps from {INPUT_FILE}")
    print(f"Freshness window: {FRESHNESS_MONTHS} months  |  Old-evidence threshold: {OLD_EVIDENCE_MONTHS} months")
    print(f"Model: {MODEL_NAME}\n")

    # Resume: skip gaps already written to any output file
    done_titles = set()
    for fp in (OUTPUT_ALL, OUTPUT_VALID, OUTPUT_STALE, OUTPUT_UNCLEAR):
        if Path(fp).exists():
            for r in load_jsonl(fp):
                t = r.get("gap_title", "")
                if t:
                    done_titles.add(t)
    if done_titles:
        print(f"Resuming: {len(done_titles)} gaps already processed, skipping.\n")

    # Count existing records for summary
    existing_valid = len(load_jsonl(OUTPUT_VALID)) if Path(OUTPUT_VALID).exists() else 0
    existing_stale = len(load_jsonl(OUTPUT_STALE)) if Path(OUTPUT_STALE).exists() else 0
    existing_unclear = len(load_jsonl(OUTPUT_UNCLEAR)) if Path(OUTPUT_UNCLEAR).exists() else 0

    # Open all output files in append mode
    f_all = open(OUTPUT_ALL, "a", encoding="utf-8")
    f_valid = open(OUTPUT_VALID, "a", encoding="utf-8")
    f_stale = open(OUTPUT_STALE, "a", encoding="utf-8")
    f_unclear = open(OUTPUT_UNCLEAR, "a", encoding="utf-8")

    valid_count = existing_valid
    stale_count = existing_stale
    unclear_count = existing_unclear

    for i, gap in enumerate(gaps, 1):
        company = gap.get("company", "?")
        title = gap.get("gap_title", "?")

        # Resume: skip already-processed gaps
        if title in done_titles:
            print(f"[{i}/{len(gaps)}] SKIP (already done): {company} | {title}")
            continue

        print(f"[{i}/{len(gaps)}] {company} | {title}")

        # Detect old research/vision evidence
        old_evidence, age_note = is_old_evidence(gap)
        if old_evidence:
            print(f"  ⚠️  OLD EVIDENCE: {age_note}")

        # Primary search
        query = build_search_query(gap)
        print(f"  Search query: {query}")
        news_results = search_recent_news(query)
        print(f"  Found {len(news_results)} search results")

        # Secondary GA-verification search (only for old research/vision evidence)
        ga_results = None
        ga_query = None
        if old_evidence:
            ga_query = build_ga_verification_query(gap)
            print(f"  GA verification query: {ga_query}")
            time.sleep(0.5)
            ga_results = search_recent_news(ga_query)
            print(f"  Found {len(ga_results)} GA verification results")

        judgment = judge_staleness(
            gap,
            news_results,
            ga_results=ga_results,
            evidence_age_note=age_note,
        )
        judgment["search_query_used"] = query
        if ga_query:
            judgment["ga_verification_query_used"] = ga_query
        judgment["old_evidence_flag"] = old_evidence

        updated_gap = {**gap, **judgment}

        status = judgment["freshness_status"]
        reason = judgment["freshness_reason"]
        urls = judgment["search_urls"]
        print(f"  Status: {status}")
        print(f"  Reason: {reason[:120]}")
        if urls:
            print(f"  URLs: {urls[0]}" + (f" (+{len(urls)-1} more)" if len(urls) > 1 else ""))

        # Write immediately
        f_all.write(json.dumps(updated_gap, ensure_ascii=False) + "\n")
        f_all.flush()

        if status == "STALE":
            f_stale.write(json.dumps(updated_gap, ensure_ascii=False) + "\n")
            f_stale.flush()
            stale_count += 1
        elif status == "STILL_VALID":
            f_valid.write(json.dumps(updated_gap, ensure_ascii=False) + "\n")
            f_valid.flush()
            valid_count += 1
        else:
            f_unclear.write(json.dumps(updated_gap, ensure_ascii=False) + "\n")
            f_unclear.flush()
            unclear_count += 1

        time.sleep(SLEEP_BETWEEN_GAPS)

    f_all.close()
    f_valid.close()
    f_stale.close()
    f_unclear.close()

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  STILL_VALID : {valid_count}")
    print(f"  STALE       : {stale_count}")
    print(f"  UNCLEAR     : {unclear_count}")
    print(f"\nOutputs:")
    print(f"  All checked  → {OUTPUT_ALL}")
    print(f"  Still valid  → {OUTPUT_VALID}")
    print(f"  Stale        → {OUTPUT_STALE}")
    print(f"  Unclear      → {OUTPUT_UNCLEAR}")

    from collections import Counter
    print("\nBy company:")
    all_checked = load_jsonl(OUTPUT_ALL)
    for company in sorted({g["company"] for g in gaps}):
        co_gaps = [g for g in all_checked if g["company"] == company]
        statuses = Counter(g["freshness_status"] for g in co_gaps)
        print(f"  {company}: {dict(statuses)}")


if __name__ == "__main__":
    main()