from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import json
import os
import re
import sys

# Load backend/.env (OPENAI_API_KEY, etc.) before anything else reads env vars.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Option A: allow backend to reuse strategic_recommender pipeline without moving files.
_SR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../strategic_recommender")
if _SR_DIR not in sys.path:
    sys.path.insert(0, _SR_DIR)

from strategic_recommender_data_retrieval import (
    load_chunks,
    load_portfolio,
    load_gaps,
    retrieve_chunks,
)
from strategic_recommender_prompt_builder import build_recommendation_prompt
from strategic_recommender_main import convert_gms_to_gap, find_matching_portfolio, call_openai_chat

# Option B: allow backend to reuse chunk_labeling scripts
_CHUNK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../chunk_labeling/scripts")
if _CHUNK_DIR not in sys.path:
    sys.path.insert(0, _CHUNK_DIR)
from iss_scoring import compute_iis_for_scope, IISConfig
from datetime import datetime
from pydantic import BaseModel

app = FastAPI(title="BGV Platform API")

# Allow CORS for React frontend (Vite usually runs on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. In production, restrict to frontend URL.
    allow_credentials=False,  # Wildcard origin + credentials is rejected by browsers per CORS spec.
    allow_methods=["*"],
    allow_headers=["*"],
)

# File Paths (relative to backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_CSV = os.path.join(BASE_DIR, "../chunk_labeling/data/iss/iis_theme_matrix.csv")
AUDIT_JSONL = os.path.join(BASE_DIR, "../chunk_labeling/data/iss/iis_audit.jsonl")
PORTFOLIO_JSON = os.path.join(BASE_DIR, "../chunk_labeling/data/input/bgv_portfolio_tagged.json")

_ACRONYMS = {"ai", "gpu", "cpu", "llm", "api", "sdk", "ml", "nlp", "rag", "iot", "saas", "paas", "iaas", "gtm", "b2b", "b2c", "ui", "ux", "hpc", "aws", "gcp", "cv", "rl"}

def _strip_markdown(s: str) -> str:
    """Strip common markdown markers so the memo renders cleanly in a plain-text block."""
    if not s:
        return s
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)           # # / ## / ### headings
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)            # bold **x**
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", s)  # italic *x* (not ** or ***)
    s = re.sub(r"(?m)^\s*[-*+]\s+", "- ", s)          # normalize bullets
    return s


def _smart_title(s: str) -> str:
    if not s:
        return s
    words = s.split()
    out = []
    for w in words:
        if w.lower() in _ACRONYMS:
            out.append(w.upper())
        else:
            out.append(w[:1].upper() + w[1:].lower() if w else w)
    return " ".join(out)


@app.get("/api/companies")
def get_companies():
    """Returns a list of companies and a list of themes."""
    df = pd.read_csv(MATRIX_CSV)
    companies = df["company"].tolist()
    themes = [col for col in df.columns if col != "company"]
    return {"companies": companies, "themes": themes}

@app.get("/api/matrix")
def get_matrix():
    """Returns the 2D array of scores matching the companies and themes."""
    df = pd.read_csv(MATRIX_CSV)
    themes = [col for col in df.columns if col != "company"]
    matrix = df[themes].values.tolist()
    return {"scores": matrix}

class MatrixRecalculateRequest(BaseModel):
    dataSources: dict

_CACHED_EVENTS = None
def get_cached_events():
    global _CACHED_EVENTS
    if _CACHED_EVENTS is not None:
        return _CACHED_EVENTS
    csv_path = os.path.join(BASE_DIR, "../chunk_labeling/data/iss/iis_normalized_events.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path)
    events = df.to_dict('records')
    for e in events:
        if "event_date" in e and pd.notna(e["event_date"]):
            e["event_date_obj"] = datetime.strptime(str(e["event_date"]), "%Y-%m-%d").date()
    _CACHED_EVENTS = events
    return events

FRONTEND_SOURCE_MAP = {
    "TechCrunch News": "techCrunch",                    # matches CSV: 'techCrunch'
    "Company Official Site": "official_website_announcement",
    "Earnings Calls": "earning_call",                   # matches CSV: 'earning_call' (no trailing s)
    "CVC News": "CVC_from_google_news",                 # matches CSV: 'CVC_from_google_news'
}

@app.post("/api/matrix/recalculate")
def recalculate_matrix(req: MatrixRecalculateRequest):
    data_sources = req.dataSources
    cached = get_cached_events()

    # Work on per-request copies so we never mutate the in-memory cache.
    # Also cast time_weight explicitly to float in case the CSV was read as str.
    inverted_map = {v: k for k, v in FRONTEND_SOURCE_MAP.items()}
    events = []
    for e in cached:
        ec = dict(e)  # shallow copy — preserves original time_weight intact
        source_type = ec.get("source_type", "")
        frontend_key = inverted_map.get(source_type)
        source_reliability = float(data_sources.get(frontend_key, 1.0)) if frontend_key else 1.0
        time_weight = float(ec.get("time_weight", 1.0))
        ec["event_weight"] = time_weight * source_reliability
        events.append(ec)

    grouped = {}
    for e in events:
        if e.get("event_weight", 0) > 0:
            company = e.get("company")
            theme = e.get("theme")
            if company and theme:
                grouped.setdefault((company, theme), []).append(e)
            
    df = pd.read_csv(MATRIX_CSV)
    companies = df["company"].tolist()
    themes = [col for col in df.columns if col != "company"]
    
    config = IISConfig()
    
    matrix = []
    for company in companies:
        row = []
        for theme in themes:
            group_events = grouped.get((company, theme), [])
            if group_events:
                result = compute_iis_for_scope(group_events, config)
                row.append(result["iis_score_0_100"])
            else:
                row.append(0.0)
        matrix.append(row)
        
    return {"scores": matrix}

@app.get("/api/evidence")
def get_evidence(
    company: str = None, 
    theme: str = None, 
    companyIndex: int = None, 
    themeIndex: int = None
):
    """
    Returns evidence for a given company and theme.
    Supports either string names (company, theme) or integer indices (companyIndex, themeIndex).
    """
    # Resolve indices to names if indices are provided
    if companyIndex is not None or themeIndex is not None:
        df = pd.read_csv(MATRIX_CSV)
        companies = df["company"].tolist()
        themes = [col for col in df.columns if col != "company"]
        
        if companyIndex is not None and 0 <= companyIndex < len(companies):
            company = companies[companyIndex]
        if themeIndex is not None and 0 <= themeIndex < len(themes):
            theme = themes[themeIndex]

    if not company or not theme:
        return {"error": "Missing valid company or theme parameter"}

    evidence_data = []
    
    # Read the JSONL audit file to extract top events
    if os.path.exists(AUDIT_JSONL):
        with open(AUDIT_JSONL, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                # Check if this record matches the requested company and theme
                if record.get("company") == company and record.get("theme") == theme:
                    # Collect all top events
                    event_groups = [
                        "top_A_events", "top_B_events", "top_C_events", 
                        "top_D_events", "top_E_events", "top_F_events"
                    ]
                    for group in event_groups:
                        events = record.get(group) or []
                        for event in events:
                            evidence_data.append({
                                "type": event.get("event_type", "Event").replace("_", " ").title(),
                                "text": event.get("evidence_span", ""),
                                "source": f"{event.get('source_type', 'Source')}, {event.get('event_date', '')}",
                                "strength": event.get("strength_1_5", 1)
                            })
                    break  # Found the matching record in main audit
                    
    subThemes = []
    SUBTHEME_JSONL = os.path.join(BASE_DIR, "../chunk_labeling/data/iss/iis_subtheme_audit.jsonl")
    if os.path.exists(SUBTHEME_JSONL):
        with open(SUBTHEME_JSONL, "r") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                if record.get("company") == company and record.get("theme") == theme:
                    s_events = []
                    event_groups = [
                        "top_A_events", "top_B_events", "top_C_events", 
                        "top_D_events", "top_E_events", "top_F_events"
                    ]
                    for group in event_groups:
                        events = record.get(group) or []
                        for event in events:
                            s_events.append({
                                "type": event.get("event_type", "Event").replace("_", " ").title(),
                                "text": event.get("evidence_span", ""),
                                "source": f"{event.get('source_type', 'Source')}, {event.get('event_date', '')}",
                                "strength": event.get("strength_1_5", 1)
                            })
                    if s_events:
                        raw_sub = record.get("sub_theme") or "Other"
                        if "|" in raw_sub:
                            raw_sub = raw_sub.split("|")[-1].strip()
                        subThemes.append({
                            "name": _smart_title(raw_sub),
                            "iss": round(float(record.get("iis_score_0_100") or 0)),
                            "events": s_events
                        })
    
    # Sort subThemes by ISS descending
    subThemes = sorted(subThemes, key=lambda x: x["iss"], reverse=True)
    
    return {"evidence": evidence_data, "subThemes": subThemes}

@app.get("/api/portfolio-companies")
def get_portfolio_companies():
    """Returns the list of portfolio company names."""
    if not os.path.exists(PORTFOLIO_JSON):
        return {"companies": []}
        
    with open(PORTFOLIO_JSON, "r") as f:
        data = json.load(f)
    
    companies = [item.get("company_name", "") for item in data if "company_name" in item]
    return {"companies": companies}

@app.get("/api/portfolio/{company}")
def get_portfolio_detail(company: str):
    """Returns details for a specific portfolio company."""
    if not os.path.exists(PORTFOLIO_JSON):
        return {"description": ""}
        
    with open(PORTFOLIO_JSON, "r") as f:
        data = json.load(f)
        
    for item in data:
        if item.get("company_name", "").lower() == company.lower():
            return {"description": item.get("description", "")}
            
    return {"description": ""}

@app.get("/api/settings")
def get_settings():
    """Returns the current settings configuration."""
    return {
        "issWeights": {
            "Strategic Commitment": 25,
            "Resourcing & Talent": 20,
            "Product Shipping": 20,
            "Commercial Traction": 15,
            "Partnerships & GTM": 10,
            "Strategic Consistency": 10
        },
        "dataSources": {
            "TechCrunch News": True,
            "Company Official Site": True,
            "Earnings Calls": True,
            "SEC Filings": False
        }
    }

@app.put("/api/settings")
def update_settings(settings: dict):
    """Updates the settings configuration (mocked)."""
    return {"status": "success", "settings": settings}

GAPS_JSONL = os.path.join(BASE_DIR, "../chunk_labeling/data/gaps_pipeline/gaps_still_valid.jsonl")
REFRESH_MANIFEST = os.path.join(BASE_DIR, "../chunk_labeling/config/refresh_manifest.json")

@app.get("/api/match")
def match_portfolio(company: str, top_k: int = 10):
    """Returns matches for a given portfolio company."""
    GMS_RESULTS_FILE = os.path.join(BASE_DIR, "../chunk_labeling/data/gms/gms_results.jsonl")
    results = []
    if not os.path.exists(GMS_RESULTS_FILE):
        return {"matches": []}
        
    skipped = 0
    with open(GMS_RESULTS_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"[/api/match] skipping malformed line {line_no}: {e}")
                continue
            cand = record.get("candidate_company", "")
            if company.lower() == cand.lower():
                cap_score = record.get("capability_score")
                ind_score = record.get("industry_score")
                raw_gms = float(record.get("final_score_raw") or record.get("final_score") or 0)
                relative_gap_rank_score = float(record.get("display_score") or 0)

                results.append({
                    "gap": record.get("gap_title", ""),
                    "company_need": record.get("target_company", ""),
                    "gms": round(raw_gms, 1),
                    "relative_gap_rank_score": round(relative_gap_rank_score),
                    "capability_match": round(float(cap_score) / 100, 2) if cap_score is not None else 0,
                    "industry_match": round(float(ind_score) / 100, 2) if ind_score is not None else None,
                    "gap_text": record.get("gap_text", ""),
                    "evidence": record.get("reason", ""),
                    "source": "Strategic Gaps Library",
                    # evidence fields — joined below
                    "evidence_quotes": [],
                    "evidence_dates": [],
                    "source_types": [],
                    "gap_strength_reason": "",
                })

    results = sorted(results, key=lambda x: x["gms"], reverse=True)
    results = results[:top_k]

    # Join evidence from gaps_still_valid using (gap_title, target_company) as key
    if results and os.path.exists(GAPS_JSONL):
        gaps_index = {}
        with open(GAPS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    g = json.loads(line)
                except Exception:
                    continue
                key = (g.get("gap_title", "").strip(), g.get("company", "").strip())
                gaps_index[key] = g

        for r in results:
            key = (r["gap"].strip(), r["company_need"].strip())
            gap_rec = gaps_index.get(key)
            if gap_rec:
                r["evidence_quotes"]     = gap_rec.get("evidence_quotes", []) or []
                r["evidence_dates"]      = gap_rec.get("evidence_dates", []) or []
                r["source_types"]        = gap_rec.get("source_types", []) or []
                r["gap_strength_reason"] = gap_rec.get("gap_strength_reason", "")

    response = {"matches": results}
    if skipped:
        response["warnings"] = [f"Skipped {skipped} malformed line(s) in gms_results.jsonl"]
    return response



@app.get("/api/gaps")
def get_gaps():
    """Returns the extracted gaps library from chunk_labeling/gaps_extracted.jsonl."""
    if not os.path.exists(GAPS_JSONL):
        return {"gaps": []}

    gaps = []
    with open(GAPS_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            gaps.append({
                "company": r.get("company", ""),
                "gap_title": r.get("gap_title", ""),
                "gap_text": r.get("gap_text", ""),
                "geography": r.get("geography", []) or [],
                "gap_strength": r.get("gap_strength", 0),
                "gap_strength_reason": r.get("gap_strength_reason", ""),
                "capability": r.get("capability", []) or [],
                "industry": r.get("industry", []) or [],
                "evidence_quotes": r.get("evidence_quotes", []) or [],
                "evidence_dates": r.get("evidence_dates", []) or [],
                "source_types": r.get("source_types", []) or [],
            })
    return {"gaps": gaps}


@app.get("/api/refresh-status")
def get_refresh_status():
    """
    Dynamically computes last-refresh timestamps from the mtime of each pipeline
    output file. No manual JSON updates needed — runs automatically reflect here.
    """
    from datetime import datetime, timezone

    def file_date(rel_path: str):
        """Return ISO date string (YYYY-MM-DD) for file mtime, or None if missing."""
        full = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full):
            return None
        ts = os.path.getmtime(full)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    # Map each pipeline stage → its canonical output file
    stage_files = {
        "ISS Scoring":     "../chunk_labeling/data/iss/iis_theme_matrix.csv",
        "Gap Extraction":  "../chunk_labeling/data/gaps_pipeline/gaps_extracted_with_10k.jsonl",
        "Critic Review":   "../chunk_labeling/data/gaps_pipeline/gaps_critic_filtered.jsonl",
        "Staleness Check": "../chunk_labeling/data/gaps_pipeline/gaps_still_valid.jsonl",
        "GMS Matching":    "../chunk_labeling/data/gms/gms_results.jsonl",
    }

    sources = {}
    all_dates = []
    for stage, rel in stage_files.items():
        d = file_date(rel)
        sources[stage] = {"last_updated": d, "status": "ok" if d else "missing"}
        if d:
            all_dates.append(d)

    global_last = max(all_dates) if all_dates else None
    return {"sources": sources, "global_last_updated": global_last}


_RECO_CHUNKS_FILE = os.path.join(BASE_DIR, "../chunk_labeling/data/input/all_chunks_labeled_v2.jsonl")
_RECO_PORTFOLIO_FILE = os.path.join(BASE_DIR, "../chunk_labeling/data/input/bgv_portfolio_tagged.json")
_RECO_GMS_FILE = os.path.join(BASE_DIR, "../chunk_labeling/data/gms/gms_results.jsonl")

# mtime-keyed in-memory cache so Generate Report doesn't re-parse 24MB JSONL each click.
_FILE_CACHE = {}

def _cached_load(path, loader):
    mtime = os.path.getmtime(path)
    entry = _FILE_CACHE.get(path)
    if entry and entry[0] == mtime:
        return entry[1]
    data = loader(path)
    _FILE_CACHE[path] = (mtime, data)
    return data


@app.get("/api/recommendation")
def get_recommendation(company: str, top_n_gaps: int = 5, top_k_chunks: int = 5):
    """Generates Strategic Investment Memo sections for the portfolio company's top GMS gaps."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on the server.")

    for p in (_RECO_CHUNKS_FILE, _RECO_PORTFOLIO_FILE, _RECO_GMS_FILE):
        if not os.path.exists(p):
            raise HTTPException(status_code=500, detail=f"Missing data file: {p}")

    top_n_gaps = max(1, min(int(top_n_gaps), 20))
    top_k_chunks = max(1, min(int(top_k_chunks), 20))

    gms_results = _cached_load(_RECO_GMS_FILE, load_gaps)
    candidate_rows = [
        r for r in gms_results
        if str(r.get("candidate_company", "")).lower().strip() == company.lower().strip()
    ]

    if not candidate_rows:
        raise HTTPException(status_code=404, detail=f"No GMS results found for portfolio company: {company}")

    def score_key(row):
        try:
            return float(row.get("final_score_raw", 0) or 0)
        except (TypeError, ValueError):
            return 0

    top_rows = sorted(candidate_rows, key=score_key, reverse=True)[:top_n_gaps]

    chunks = _cached_load(_RECO_CHUNKS_FILE, load_chunks)
    portfolios = _cached_load(_RECO_PORTFOLIO_FILE, load_portfolio)

    recommendations = []

    for rank, row in enumerate(top_rows, start=1):
        gap = convert_gms_to_gap(row)

        candidate_name = (gap.get("candidate_company") or "").strip().lower()
        portfolio = next(
            (p for p in portfolios if (p.get("company_name") or "").strip().lower() == candidate_name),
            None,
        )

        if portfolio is None:
            continue

        retrieved = retrieve_chunks(
            chunks=chunks,
            portfolio=portfolio,
            gap=gap,
            mode="local",
            top_k=top_k_chunks
        )

        prompt = build_recommendation_prompt(portfolio, gap, retrieved)

        try:
            memo = call_openai_chat(prompt)
            memo = _strip_markdown(memo)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed for gap rank {rank}: {e}")

        recommendations.append({
            "rank": rank,
            "memo": memo,
            "gap_title": gap.get("gap_title", ""),
            "target_company": gap.get("target_company", ""),
            "display_score": row.get("display_score", 0),
            "final_score_raw": row.get("final_score_raw", 0),
            "retrieved_chunk_count": len(retrieved),
        })

    if not recommendations:
        raise HTTPException(status_code=404, detail=f"No valid recommendation records generated for: {company}")

    return {
        "portfolio_company": company,
        "top_n_gaps": len(recommendations),
        "top_k_chunks": top_k_chunks,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
