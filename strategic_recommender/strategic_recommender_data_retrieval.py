#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import csv
from typing import List, Dict, Any, Optional


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv_file(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def smart_load(path: str) -> Any:
    if path.endswith(".json"):
        return load_json_file(path)
    if path.endswith(".jsonl"):
        return load_jsonl_file(path)
    if path.endswith(".csv"):
        return load_csv_file(path)
    raise ValueError(f"Unsupported file type: {path}")


# ============================================================
# TODO: ADAPT THESE THREE TO YOUR ACTUAL DATA STRUCTURE
# ============================================================

def load_chunks(path: str) -> List[Dict[str, Any]]:
    """
    Expected rough schema:
    {
        "chunk_id": "123",
        "company": "Microsoft",
        "theme": "Horizontal AI",
        "sub_theme": "copilot",
        "text": "chunk text here...",
        "source": "earnings_call",
        "date": "2026-04-01"
    }
    """
    data = smart_load(path)

    if isinstance(data, dict):
        if "chunks" in data:
            return data["chunks"]
        return list(data.values())

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported chunk file structure")


def load_portfolio(path: str) -> List[Dict[str, Any]]:
    """
    Expected rough schema:
    {
        "portfolio_id": "p1",
        "company_name": "Startup X",
        "description": "...",
        "capabilities": ["rag_tooling", "audit_logging"],
        "industry": ["healthcare"],
        "geography": ["US", "IN"]
    }
    """
    data = smart_load(path)

    if isinstance(data, dict):
        if "portfolio" in data:
            return data["portfolio"]
        return list(data.values())

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported portfolio file structure")


def load_gms_results(path: str) -> List[Dict[str, Any]]:
    """
    Loads GMS results. Each row represents one gap-company match result.
    Expected fields:
    gap_id, gap_title, gap_text, target_company, candidate_company,
    semantic_score, capability_score, industry_score, geography_score,
    final_score_raw, display_score, reason, tech_coverage, industry_fit, geography_fit
    """
    return load_jsonl_file(path)


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).lower().split())


def keyword_overlap_score(query_terms: List[str], text: str) -> int:
    text_norm = normalize_text(text)
    score = 0
    for term in query_terms:
        term_norm = normalize_text(term)
        if term_norm and term_norm in text_norm:
            score += 1
    return score


def build_query_terms(portfolio: Dict[str, Any], gap: Dict[str, Any]) -> List[str]:
    terms = []

    if gap.get("gap_text"):
        terms.append(str(gap["gap_text"]))

    for item in gap.get("capability", []) or []:
        terms.append(str(item))

    for item in gap.get("industry", []) or []:
        terms.append(str(item))

    for item in gap.get("geography", []) or []:
        terms.append(str(item))

    if portfolio.get("description"):
        terms.append(str(portfolio["description"]))

    for item in portfolio.get("capability", portfolio.get("capabilities", [])) or []:
        terms.append(str(item))

    for item in portfolio.get("industry", []) or []:
        terms.append(str(item))

    for item in portfolio.get("geography", portfolio.get("geography_all", [])) or []:
        terms.append(str(item))

    return terms


def local_retrieve_chunks(
    chunks: List[Dict[str, Any]],
    portfolio: Dict[str, Any],
    gap: Dict[str, Any],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Prototype top-k retrieval:
    score chunks by keyword overlap against gap + portfolio context.
    Returns top_k chunks with retrieval_score attached.
    """
    query_terms = build_query_terms(portfolio, gap)

    scored = []
    total_scanned = 0

    for chunk in chunks:
        total_scanned += 1

        chunk_text = str(chunk.get("chunk_text", ""))
        company = str(chunk.get("company", "")).strip()

        score = keyword_overlap_score(query_terms, chunk_text)

        gap_company = str(gap.get("company", "")).strip()
        if gap_company and company and gap_company.lower() == company.lower():
            score += 2

        if score <= 0:
            continue

        chunk_with_score = dict(chunk)
        chunk_with_score["retrieval_score"] = score

        scored.append(chunk_with_score)

    scored.sort(key=lambda x: x.get("retrieval_score", 0), reverse=True)

    print(f"Top-k retrieval scanned {total_scanned} chunks")
    print(f"Top-k retrieval found {len(scored)} matching chunks")
    print(f"Returning top {min(top_k, len(scored))} chunks")

    return scored[:top_k]


def vectordb_retrieve_chunks(
    portfolio: Dict[str, Any],
    gap: Dict[str, Any],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    TODO:
    Replace with Pinecone / vector DB query later.
    """
    raise NotImplementedError("Vector DB retrieval not implemented yet")


def retrieve_chunks(
    chunks: List[Dict[str, Any]],
    portfolio: Dict[str, Any],
    gap: Dict[str, Any],
    mode: str = "local",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    if mode == "local":
        return local_retrieve_chunks(chunks, portfolio, gap, top_k=top_k)
    elif mode == "vectordb":
        return vectordb_retrieve_chunks(portfolio, gap, top_k=top_k)
    else:
        raise ValueError(f"Unsupported retrieval mode: {mode}")


def find_first_portfolio(portfolios: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not portfolios:
        raise ValueError("No portfolio data found")
    return portfolios[0]


def find_first_gms_result(gms_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not gms_results:
        raise ValueError("No GMS results found")
    return gms_results[0]

def select_top_gms_results(
    gms_results: List[Dict[str, Any]],
    candidate_company: Optional[str] = None,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Select top-N GMS gap matches.
    If candidate_company is provided, filter to that portfolio company first.
    Sort by display_score, then final_score_raw.
    """
    rows = gms_results

    if candidate_company:
        rows = [
            r for r in rows
            if str(r.get("candidate_company", "")).lower().strip()
            == candidate_company.lower().strip()
        ]

    def score_key(row):
        display_score = row.get("display_score")
        final_score_raw = row.get("final_score_raw")

        try:
            display_score = float(display_score)
        except (TypeError, ValueError):
            display_score = -1

        try:
            final_score_raw = float(final_score_raw)
        except (TypeError, ValueError):
            final_score_raw = -1

        return (display_score, final_score_raw)

    rows = sorted(rows, key=score_key, reverse=True)
    return rows[:top_n]


# Compatibility aliases for main.py
load_gaps = load_gms_results
find_first_gap = find_first_gms_result

