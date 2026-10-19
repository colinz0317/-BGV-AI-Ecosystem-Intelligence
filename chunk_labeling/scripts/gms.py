import json
import math
import os
import re
import time
from typing import Dict, List, Any, Tuple, Optional

from paths import (
    BASE_DIR,
    FILE_GAPS_VALID,
    FILE_PORTFOLIO_JSONL,
    FILE_CAPABILITY_LOOKUP,
    FILE_INDUSTRY_LOOKUP,
    FILE_GMS_RESULTS_JSON,
    FILE_GMS_RESULTS_JSONL
)

from openai import OpenAI

# Auto-load .env
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())


# =========================================================
# 1. CONFIG
# =========================================================

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 128

FULL_MATCH_THRESHOLD = 0.95
PARTIAL_MATCH_THRESHOLD = 0.01

CAPABILITY_NONE_FLOOR = 0.30
INDUSTRY_NONE_FLOOR = 0.00

# =========================================================
# DISPLAY SCORE CONFIG
# =========================================================
# "minmax" = linear 0-100 within each gap
# "power"  = min-max first, then apply gamma to widen separation
DISPLAY_SCORE_MODE = "power"
DISPLAY_SCORE_GAMMA = 1.2
FLAT_SCORE_FALLBACK = 50.0

INDUSTRY_VOCAB = {
    "IND_HEALTHCARE", "IND_MANUFACTURING", "IND_ENERGY_OIL_GAS",
    "IND_AUTOMOTIVE", "IND_INTERNET_PLATFORM",
    "IND_FINANCIAL_SERVICES", "IND_INSURANCE", "IND_CONSTRUCTION",
    "IND_LOGISTICS", "IND_RETAIL", "IND_FOOD_SERVICE",
    "IND_MEDIA_DIGITAL", "IND_UTILITIES", "IND_WASTE_MANAGEMENT",
    "IND_MEDICAL_DEVICES", "IND_INDUSTRIAL_INSPECTION",
    "IND_TELECOMMUNICATIONS", "IND_PUBLIC_SECTOR",
    "IND_SUSTAINABILITY", "IND_ECOMMERCE",
    "IND_SECURITY", "IND_CLOUD_INFRA", "IND_AI_INFRA",
    "IND_SALES_MARKETING", "IND_DEV_TOOLS", "IND_GENERAL",
}

RAW_TO_STANDARD_INDUSTRY = {
    "healthcare": "IND_HEALTHCARE",
    "manufacturing": "IND_MANUFACTURING",
    "energy": "IND_ENERGY_OIL_GAS",
    "oil_gas": "IND_ENERGY_OIL_GAS",
    "oil&gas": "IND_ENERGY_OIL_GAS",
    "automotive": "IND_AUTOMOTIVE",
    "internet_platform": "IND_INTERNET_PLATFORM",
    "financial_services": "IND_FINANCIAL_SERVICES",
    "insurance": "IND_INSURANCE",
    "construction": "IND_CONSTRUCTION",
    "logistics": "IND_LOGISTICS",
    "retail": "IND_RETAIL",
    "food_service": "IND_FOOD_SERVICE",
    "food": "IND_FOOD_SERVICE",
    "media": "IND_MEDIA_DIGITAL",
    "media_digital": "IND_MEDIA_DIGITAL",
    "entertainment": "IND_MEDIA_DIGITAL",
    "utilities": "IND_UTILITIES",
    "waste_management": "IND_WASTE_MANAGEMENT",
    "medical_devices": "IND_MEDICAL_DEVICES",
    "industrial_inspection": "IND_INDUSTRIAL_INSPECTION",
    "telecommunications": "IND_TELECOMMUNICATIONS",
    "telecom": "IND_TELECOMMUNICATIONS",
    "government": "IND_PUBLIC_SECTOR",
    "public_sector": "IND_PUBLIC_SECTOR",
    "sustainability": "IND_SUSTAINABILITY",
    "ecommerce": "IND_ECOMMERCE",
    "security": "IND_SECURITY",
    "cybersecurity": "IND_SECURITY",
    "cloud_infra": "IND_CLOUD_INFRA",
    "enterprise_software": "IND_CLOUD_INFRA",
    "infrastructure": "IND_CLOUD_INFRA",
    "ai_infra": "IND_AI_INFRA",
    "sales_marketing": "IND_SALES_MARKETING",
    "dev_tools": "IND_DEV_TOOLS",
    "developer_tools": "IND_DEV_TOOLS",
    "general": "IND_GENERAL",
    "GENERAL": "IND_GENERAL",

    "gaming": "IND_MEDIA_DIGITAL",
    "agriculture": "IND_GENERAL",
    "technology": "IND_GENERAL",
    "consumer_electronics": "IND_GENERAL",
    "smart_home": "IND_GENERAL",
    "emerging_markets": "IND_GENERAL",
    "legal": "IND_GENERAL",
    "professional_services": "IND_GENERAL",
    "wellness": "IND_GENERAL",
    "robotics": "IND_GENERAL",
    "semiconductors": "IND_GENERAL",
    "transportation": "IND_LOGISTICS",
    "cultural_preservation": "IND_GENERAL",
    "education": "IND_GENERAL",
    "advertising": "IND_SALES_MARKETING",
    "marketing_automation": "IND_SALES_MARKETING",
    "emergency_services": "IND_GENERAL",
}

GEO_ALIASES = {
    "global": "GLOBAL",
    "GLOBAL": "GLOBAL",

    "us": "UNITED_STATES",
    "usa": "UNITED_STATES",
    "u.s.": "UNITED_STATES",
    "u.s": "UNITED_STATES",
    "united states": "UNITED_STATES",
    "United States": "UNITED_STATES",

    "uk": "UNITED_KINGDOM",
    "u.k.": "UNITED_KINGDOM",
    "united kingdom": "UNITED_KINGDOM",
    "United Kingdom": "UNITED_KINGDOM",

    "europe": "EUROPE",
    "Europe": "EUROPE",

    "north_america": "NORTH_AMERICA",
    "North America": "NORTH_AMERICA",
    "latin america": "LATIN_AMERICA",
    "Latin America": "LATIN_AMERICA",
    "asia": "ASIA",
    "Asia": "ASIA",
    "asia_pacific": "ASIA_PACIFIC",
    "africa": "AFRICA",
    "Africa": "AFRICA",
    "west_africa": "WEST_AFRICA",
    "middle_east": "MIDDLE_EAST",
    "emerging_markets": "EMERGING_MARKETS",

    "india": "INDIA",
    "India": "INDIA",
    "france": "FRANCE",
    "France": "FRANCE",
    "thailand": "THAILAND",
    "Thailand": "THAILAND",
    "kenya": "KENYA",
    "Kenya": "KENYA",
    "japan": "JAPAN",
    "Japan": "JAPAN",
    "singapore": "SINGAPORE",
    "Singapore": "SINGAPORE",
    "ghana": "GHANA",
    "Ghana": "GHANA",
    "israel": "ISRAEL",
    "Israel": "ISRAEL",
    "spain": "SPAIN",
    "Spain": "SPAIN",
    "poland": "POLAND",
    "Poland": "POLAND",
    "ukraine": "UKRAINE",
    "Ukraine": "UKRAINE",
    "mexico": "MEXICO",
    "Mexico": "MEXICO",
    "china": "CHINA",
    "China": "CHINA",

    "accra": "ACCRA",
    "tema": "TEMA",
    "kumasi": "KUMASI",
}

GEO_PARENTS = {
    "UNITED_STATES": ["NORTH_AMERICA", "GLOBAL"],
    "MEXICO": ["LATIN_AMERICA", "NORTH_AMERICA", "GLOBAL"],
    "FRANCE": ["EUROPE", "GLOBAL"],
    "SPAIN": ["EUROPE", "GLOBAL"],
    "UNITED_KINGDOM": ["EUROPE", "GLOBAL"],
    "POLAND": ["EUROPE", "GLOBAL"],
    "UKRAINE": ["EUROPE", "GLOBAL"],
    "CHINA": ["ASIA", "ASIA_PACIFIC", "GLOBAL"],

    "INDIA": ["ASIA", "ASIA_PACIFIC", "GLOBAL"],
    "JAPAN": ["ASIA", "ASIA_PACIFIC", "GLOBAL"],
    "THAILAND": ["ASIA", "ASIA_PACIFIC", "GLOBAL"],
    "SINGAPORE": ["ASIA", "ASIA_PACIFIC", "GLOBAL"],

    "KENYA": ["AFRICA", "GLOBAL"],
    "GHANA": ["WEST_AFRICA", "AFRICA", "GLOBAL"],
    "ISRAEL": ["MIDDLE_EAST", "GLOBAL"],

    "ACCRA": ["GHANA", "WEST_AFRICA", "AFRICA", "GLOBAL"],
    "TEMA": ["GHANA", "WEST_AFRICA", "AFRICA", "GLOBAL"],
    "KUMASI": ["GHANA", "WEST_AFRICA", "AFRICA", "GLOBAL"],
}

GEO_SENSITIVITY_BY_INDUSTRY = {
    "IND_HEALTHCARE": "high",
    "IND_FINANCIAL_SERVICES": "high",
    "IND_INSURANCE": "high",
    "IND_PUBLIC_SECTOR": "high",
    "IND_UTILITIES": "high",
    "IND_TELECOMMUNICATIONS": "high",
    "IND_ENERGY_OIL_GAS": "high",
    "IND_MEDICAL_DEVICES": "high",
    "IND_WASTE_MANAGEMENT": "high",
    "IND_CONSTRUCTION": "high",
    "IND_LOGISTICS": "high",

    "IND_RETAIL": "medium",
    "IND_FOOD_SERVICE": "medium",
    "IND_AUTOMOTIVE": "medium",
    "IND_MANUFACTURING": "medium",
    "IND_INDUSTRIAL_INSPECTION": "medium",
    "IND_SUSTAINABILITY": "medium",
    "IND_ECOMMERCE": "medium",
    "IND_MEDIA_DIGITAL": "medium",
    "IND_INTERNET_PLATFORM": "medium",

    "IND_AI_INFRA": "low",
    "IND_CLOUD_INFRA": "low",
    "IND_DEV_TOOLS": "low",
    "IND_SECURITY": "low",
    "IND_SALES_MARKETING": "low",
    "IND_GENERAL": "low",
}

GEO_SENSITIVITY_WEIGHT = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.25,
}

GEO_MATCH_MULTIPLIERS = {
    "exact": 1.10,
    "partial": 0.85,
    "neutral": 1.00,
    "none": 0.00,
}


# =========================================================
# 2. IO HELPERS
# =========================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================================================
# 2.5 DISPLAY SCORE HELPERS
# =========================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_display_score(
    raw_score: float,
    min_score: float,
    max_score: float,
    mode: str = "power",
    gamma: float = 1.5,
    flat_score_fallback: float = 50.0,
) -> float:
    if max_score == min_score:
        return round(flat_score_fallback, 2)

    normalized = (raw_score - min_score) / (max_score - min_score)
    normalized = clamp(normalized, 0.0, 1.0)

    if mode == "minmax":
        display = normalized * 100.0
    elif mode == "power":
        display = (normalized ** gamma) * 100.0
    else:
        raise ValueError(f"Unsupported DISPLAY_SCORE_MODE: {mode}")

    return round(display, 2)


def add_display_scores(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    gap_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        gap_id = str(r.get("gap_id", ""))
        gap_to_records.setdefault(gap_id, []).append(r)

    updated_records: List[Dict[str, Any]] = []

    for gap_id, gap_records in gap_to_records.items():
        raw_scores = [float(r.get("final_score", 0.0)) for r in gap_records]
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        for r in gap_records:
            raw_final_score = float(r.get("final_score", 0.0))

            r["final_score_raw"] = raw_final_score
            r["display_score"] = compute_display_score(
                raw_score=raw_final_score,
                min_score=min_score,
                max_score=max_score,
                mode=DISPLAY_SCORE_MODE,
                gamma=DISPLAY_SCORE_GAMMA,
                flat_score_fallback=FLAT_SCORE_FALLBACK,
            )
            r["display_score_meta"] = {
                "mode": DISPLAY_SCORE_MODE,
                "gamma": DISPLAY_SCORE_GAMMA if DISPLAY_SCORE_MODE == "power" else None,
                "gap_min_raw_score": round(min_score, 4),
                "gap_max_raw_score": round(max_score, 4),
            }

            updated_records.append(r)

    updated_records.sort(
        key=lambda x: (x.get("gap_id", ""), -float(x.get("final_score_raw", 0.0)))
    )
    return updated_records


# =========================================================
# 3. NORMALIZATION HELPERS
# =========================================================

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def normalize_industry_token(raw: str) -> str:
    raw = normalize_text(raw)
    if raw in INDUSTRY_VOCAB:
        return raw

    key = raw.lower()
    mapped = RAW_TO_STANDARD_INDUSTRY.get(key)
    if mapped:
        return mapped

    return "IND_GENERAL"


def map_gap_industries(raw_industries: List[str]) -> List[str]:
    normalized = [normalize_industry_token(x) for x in (raw_industries or [])]
    normalized = [x for x in normalized if x in INDUSTRY_VOCAB]
    return sorted(set(normalized)) if normalized else ["IND_GENERAL"]


def normalize_geo_token(raw: str) -> str:
    raw = normalize_text(raw)
    if raw in GEO_ALIASES:
        return GEO_ALIASES[raw]

    lower = raw.lower()
    if lower in GEO_ALIASES:
        return GEO_ALIASES[lower]

    return raw.upper().replace(" ", "_")


def normalize_geo_list(geo_list: List[str]) -> List[str]:
    if not geo_list:
        return []
    normalized = [normalize_geo_token(g) for g in geo_list if str(g).strip()]
    return sorted(set(normalized))


def expand_geo(token: str) -> List[str]:
    expanded = [token]
    for parent in GEO_PARENTS.get(token, []):
        if parent not in expanded:
            expanded.append(parent)
    return expanded


def normalize_capability_token(cap: str) -> str:
    return normalize_text(cap).lower().replace(" ", "_")


# =========================================================
# 4. EMBEDDING HELPERS
# =========================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_to_score(cos_sim: float) -> float:
    scaled = (cos_sim + 1.0) / 2.0
    scaled = max(0.0, min(1.0, scaled))
    sharpened = scaled ** 1.35
    return round(sharpened * 100.0, 2)


def _weak_capability_sentence_from_gap(gap_record: Dict[str, Any]) -> str:
    caps = [normalize_text(x).replace("_", " ") for x in gap_record.get("capability", []) if normalize_text(x)]
    if not caps:
        return ""
    cap_hint = ", ".join(sorted(set(caps))[:3])
    return f"This gap is related to missing capabilities such as {cap_hint}."


def _weak_industry_sentence_from_gap(gap_record: Dict[str, Any]) -> str:
    inds = [
        normalize_industry_token(x).replace("IND_", "").replace("_", " ").lower()
        for x in gap_record.get("industry", [])
        if normalize_text(x)
    ]
    if not inds:
        return ""
    ind_hint = ", ".join(sorted(set(inds))[:3])
    return f"It is especially relevant in contexts such as {ind_hint}."


def _weak_capability_sentence_from_company(company_record: Dict[str, Any]) -> str:
    caps = [normalize_text(x).replace("_", " ") for x in company_record.get("capability", []) if normalize_text(x)]
    if not caps:
        return ""
    cap_hint = ", ".join(sorted(set(caps))[:4])
    return f"The company appears to provide solutions related to {cap_hint}."


def _weak_industry_sentence_from_company(company_record: Dict[str, Any]) -> str:
    inds = [
        normalize_text(x).replace("IND_", "").replace("_", " ").lower()
        for x in company_record.get("industry", [])
        if normalize_text(x)
    ]
    if not inds:
        return ""
    ind_hint = ", ".join(sorted(set(inds))[:3])
    return f"It is positioned around markets such as {ind_hint}."


def build_gap_embedding_text(gap_record: Dict[str, Any]) -> str:
    target_company = normalize_text(gap_record.get("company", ""))
    theme = normalize_text(gap_record.get("theme", ""))
    gap_text = normalize_text(gap_record.get("gap_text", ""))

    capability_sentence = _weak_capability_sentence_from_gap(gap_record)
    industry_sentence = _weak_industry_sentence_from_gap(gap_record)

    parts = [
        "Capability gap profile.",
        f"Target company: {target_company}",
        f"Primary theme: {theme}",
        f"Gap description: {gap_text}",
    ]
    if capability_sentence:
        parts.append(capability_sentence)
    if industry_sentence:
        parts.append(industry_sentence)

    parts.append("This text describes an unmet need or missing solution area.")
    return "\n".join(parts).strip()


def build_company_embedding_text(company_record: Dict[str, Any]) -> str:
    company_name = normalize_text(company_record.get("company_name", ""))
    description = normalize_text(company_record.get("description", ""))

    capability_sentence = _weak_capability_sentence_from_company(company_record)
    industry_sentence = _weak_industry_sentence_from_company(company_record)

    parts = [
        "Portfolio company profile.",
        f"Company name: {company_name}",
        f"Business description: {description}",
    ]
    if capability_sentence:
        parts.append(capability_sentence)
    if industry_sentence:
        parts.append(industry_sentence)

    parts.append("This text describes the company's solution areas, products, and business focus.")
    return "\n".join(parts).strip()


def get_embeddings_in_batches(
    client: OpenAI,
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
    sleep_seconds: float = 0.0,
) -> List[List[float]]:
    all_embeddings: List[List[float]] = []

    for start_idx in range(0, len(texts), batch_size):
        batch = texts[start_idx:start_idx + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return all_embeddings


def prepare_embedding_maps(
    client: OpenAI,
    gap_records: List[Dict[str, Any]],
    company_records: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    gap_ids = [f"g{i:03d}" for i in range(1, len(gap_records) + 1)]
    gap_texts = [build_gap_embedding_text(r) for r in gap_records]

    company_names = [r["company_name"] for r in company_records]
    company_texts = [build_company_embedding_text(r) for r in company_records]

    print("Generating gap embeddings...")
    gap_vectors = get_embeddings_in_batches(client, gap_texts, model=EMBEDDING_MODEL)

    print("Generating company embeddings...")
    company_vectors = get_embeddings_in_batches(client, company_texts, model=EMBEDDING_MODEL)

    gap_embedding_map = {gid: vec for gid, vec in zip(gap_ids, gap_vectors)}
    company_embedding_map = {name: vec for name, vec in zip(company_names, company_vectors)}

    return gap_embedding_map, company_embedding_map


def compute_semantic_score_from_embeddings(
    gap_embedding: List[float],
    company_embedding: List[float],
) -> float:
    cos_sim = cosine_similarity(gap_embedding, company_embedding)
    return cosine_to_score(cos_sim)


# =========================================================
# 5. LOOKUP-BASED STRUCTURED FIT
# =========================================================

def get_lookup_score(
    lookup: Dict[str, Dict[str, float]],
    source_key: str,
    target_key: str,
    default: float = 0.0,
) -> float:
    if source_key in lookup and target_key in lookup[source_key]:
        return float(lookup[source_key][target_key])
    if target_key in lookup and source_key in lookup[target_key]:
        return float(lookup[target_key][source_key])
    return default


def summarize_match_type(avg_score: float) -> str:
    if avg_score >= FULL_MATCH_THRESHOLD:
        return "full"
    if avg_score >= PARTIAL_MATCH_THRESHOLD:
        return "partial"
    return "none"


def compute_capability_fit(
    gap_capabilities: List[str],
    company_capabilities: List[str],
    capability_lookup: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    gap_caps = sorted(set(normalize_capability_token(x) for x in (gap_capabilities or []) if normalize_text(x)))
    company_caps = sorted(set(normalize_capability_token(x) for x in (company_capabilities or []) if normalize_text(x)))

    if not gap_caps:
        return {
            "capability_score": 50.0,
            "tech_coverage": {
                "match_type": "partial",
                "multiplier": 0.5,
                "matched_capabilities": [],
                "missing_capabilities": [],
                "best_capability_matches": [],
            }
        }

    best_capability_matches = []
    matched_capabilities = []
    missing_capabilities = []
    best_scores = []

    for gap_cap in gap_caps:
        best_company_cap = None
        best_score = 0.0

        for company_cap in company_caps:
            score = get_lookup_score(capability_lookup, gap_cap, company_cap, default=0.0)
            if score > best_score:
                best_score = score
                best_company_cap = company_cap

        best_scores.append(best_score)

        if best_score >= PARTIAL_MATCH_THRESHOLD:
            matched_capabilities.append(gap_cap)
        else:
            missing_capabilities.append(gap_cap)

        best_capability_matches.append({
            "gap_capability": gap_cap,
            "best_company_capability": best_company_cap,
            "score": round(best_score, 4),
        })

    avg_score = sum(best_scores) / max(len(best_scores), 1)
    match_type = summarize_match_type(avg_score)

    if match_type == "none":
        multiplier = CAPABILITY_NONE_FLOOR
    else:
        multiplier = round(avg_score, 4)

    capability_score = round(avg_score * 100.0, 2)

    return {
        "capability_score": capability_score,
        "tech_coverage": {
            "match_type": match_type,
            "multiplier": multiplier,
            "matched_capabilities": matched_capabilities,
            "missing_capabilities": missing_capabilities,
            "best_capability_matches": best_capability_matches,
        }
    }


def compute_industry_fit(
    gap_industries_raw: List[str],
    company_industries: List[str],
    industry_lookup: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    gap_industries = sorted(set(map_gap_industries(gap_industries_raw)))
    company_industries_norm = sorted(
        set(normalize_industry_token(x) for x in (company_industries or []) if normalize_text(x))
    )

    if gap_industries == ["IND_GENERAL"]:
        return {
            "industry_score": None,
            "industry_fit": {
                "match_type": "neutral",
                "multiplier": 1.0,
                "matched_industries": [],
                "gap_industries_normalized": gap_industries,
                "company_industries": company_industries_norm,
                "best_industry_matches": [],
            }
        }

    if not gap_industries:
        gap_industries = ["IND_GENERAL"]

    if not company_industries_norm:
        return {
            "industry_score": 0.0,
            "industry_fit": {
                "match_type": "none",
                "multiplier": INDUSTRY_NONE_FLOOR,
                "matched_industries": [],
                "gap_industries_normalized": gap_industries,
                "company_industries": [],
                "best_industry_matches": [],
            }
        }

    best_industry_matches = []
    best_scores = []

    for gap_ind in gap_industries:
        best_company_ind = None
        best_score = 0.0

        for company_ind in company_industries_norm:
            score = get_lookup_score(industry_lookup, gap_ind, company_ind, default=0.0)
            if score > best_score:
                best_score = score
                best_company_ind = company_ind

        best_scores.append(best_score)
        best_industry_matches.append({
            "gap_industry": gap_ind,
            "best_company_industry": best_company_ind,
            "score": round(best_score, 4),
        })

    avg_score = sum(best_scores) / max(len(best_scores), 1)
    match_type = summarize_match_type(avg_score)

    if match_type == "none":
        multiplier = INDUSTRY_NONE_FLOOR
        matched_industries = []
    else:
        multiplier = round(avg_score, 4)
        matched_industries = [
            item["best_company_industry"]
            for item in best_industry_matches
            if item["score"] >= PARTIAL_MATCH_THRESHOLD and item["best_company_industry"] is not None
        ]

    industry_score = round(avg_score * 100.0, 2)

    return {
        "industry_score": industry_score,
        "industry_fit": {
            "match_type": match_type,
            "multiplier": multiplier,
            "matched_industries": sorted(set(matched_industries)),
            "gap_industries_normalized": gap_industries,
            "company_industries": company_industries_norm,
            "best_industry_matches": best_industry_matches,
        }
    }


# =========================================================
# 6. GEOGRAPHY FIT
# =========================================================

def compute_pair_geo_score(gap_geo: str, company_geo: str) -> Tuple[float, str]:
    if gap_geo == company_geo:
        return GEO_MATCH_MULTIPLIERS["exact"], "exact"

    gap_expanded = set(expand_geo(gap_geo))
    company_expanded = set(expand_geo(company_geo))

    if gap_geo in company_expanded or company_geo in gap_expanded:
        return GEO_MATCH_MULTIPLIERS["partial"], "partial"

    if gap_expanded & company_expanded:
        return GEO_MATCH_MULTIPLIERS["partial"], "partial"

    return GEO_MATCH_MULTIPLIERS["none"], "none"


def compute_geography_fit(gap_geos_raw: List[str], company_geos_raw: List[str]) -> Dict[str, Any]:
    gap_geos = normalize_geo_list(gap_geos_raw)
    company_geos = normalize_geo_list(company_geos_raw)

    if not gap_geos:
        return {
            "geography_score": None,
            "geography_fit": {
                "match_type": "neutral",
                "multiplier": 1.0,
                "matched_gap_geography": [],
                "matched_company_geography": [],
                "gap_geography_normalized": [],
                "company_geography_normalized": company_geos,
            }
        }

    if set(gap_geos) == {"GLOBAL"}:
        return {
            "geography_score": None,
            "geography_fit": {
                "match_type": "neutral",
                "multiplier": 1.0,
                "matched_gap_geography": ["GLOBAL"],
                "matched_company_geography": [],
                "gap_geography_normalized": gap_geos,
                "company_geography_normalized": company_geos,
            }
        }

    if not company_geos:
        return {
            "geography_score": 0.0,
            "geography_fit": {
                "match_type": "none",
                "multiplier": 0.0,
                "matched_gap_geography": [],
                "matched_company_geography": [],
                "gap_geography_normalized": gap_geos,
                "company_geography_normalized": [],
            }
        }

    best_score = 0.0
    best_gap_geo = None
    best_company_geo = None
    best_pair_type = "none"

    for g in gap_geos:
        if g == "GLOBAL":
            continue
        for c in company_geos:
            score, pair_type = compute_pair_geo_score(g, c)
            if score > best_score:
                best_score = score
                best_gap_geo = g
                best_company_geo = c
                best_pair_type = pair_type

    if best_pair_type == "exact":
        fit_type = "full"
    elif best_score > 0:
        fit_type = "partial"
    else:
        fit_type = "none"

    geography_score = round(min(best_score, 1.0) * 100.0, 2)

    return {
        "geography_score": geography_score,
        "geography_fit": {
            "match_type": fit_type,
            "multiplier": round(best_score, 2),
            "matched_gap_geography": [best_gap_geo] if best_gap_geo else [],
            "matched_company_geography": [best_company_geo] if best_company_geo else [],
            "gap_geography_normalized": gap_geos,
            "company_geography_normalized": company_geos,
        }
    }


def get_geo_sensitivity(mapped_gap_industries: List[str]) -> str:
    order = {"low": 1, "medium": 2, "high": 3}
    levels = [GEO_SENSITIVITY_BY_INDUSTRY.get(ind, "medium") for ind in mapped_gap_industries]
    if not levels:
        return "medium"
    return max(levels, key=lambda x: order[x])


def compute_geo_adjustment(mapped_gap_industries: List[str], geography_fit: Dict[str, Any]) -> Tuple[float, str, float]:
    match_type = geography_fit["match_type"]
    raw_geo_multiplier = geography_fit["multiplier"]

    if match_type == "neutral":
        return 1.0, "neutral", 0.0

    sensitivity = get_geo_sensitivity(mapped_gap_industries)
    w = GEO_SENSITIVITY_WEIGHT[sensitivity]
    adjustment = (1 - w) + w * raw_geo_multiplier
    adjustment = round(adjustment, 4)
    return adjustment, sensitivity, w


# =========================================================
# 7. REASON
# =========================================================

def build_reason(
    semantic_score: float,
    tech_coverage: Dict[str, Any],
    industry_fit: Dict[str, Any],
    geography_fit: Dict[str, Any],
    geo_adjustment: float,
) -> str:
    tech = tech_coverage["tech_coverage"]
    ind = industry_fit["industry_fit"]
    geo = geography_fit["geography_fit"]

    parts = [f"The candidate company shows a semantic relevance score of {semantic_score:.1f} to the target gap."]

    if tech["match_type"] == "full":
        parts.append("It fully covers the required capabilities listed in the gap.")
    elif tech["match_type"] == "partial":
        parts.append("It partially covers the required capabilities based on the capability match lookup.")
    else:
        parts.append("It does not cover the required capabilities well, which materially reduces the score.")

    if ind["match_type"] == "neutral":
        parts.append("The gap does not impose a specific industry constraint, so industry is treated as neutral and does not increase or reduce the score.")
    elif ind["match_type"] == "full":
        parts.append("Its industry alignment is strong and directly relevant to the gap.")
    elif ind["match_type"] == "partial":
        parts.append("Its industry alignment is partial based on the industry match lookup, so the structured fit is reduced.")
    else:
        parts.append("Its industry alignment is not sufficient, which heavily limits the match quality.")

    if geo["match_type"] == "neutral":
        parts.append("The gap does not impose a specific geographic constraint, so geography is treated as neutral and does not increase or reduce the score.")
    elif geo["match_type"] == "full":
        parts.append("Geographically, the company operates in the same normalized geography as the gap, which slightly boosts the result.")
    elif geo["match_type"] == "partial":
        parts.append(f"Geographically, the company has partial overlap with the gap after normalization and hierarchy expansion. The geography adjustment applied is {geo_adjustment:.2f}.")
    else:
        parts.append(f"Geographically, the company does not show deployable overlap with the gap after normalization, and the geography adjustment is {geo_adjustment:.2f}.")

    return " ".join(parts)


# =========================================================
# 8. MAIN SCORING
# =========================================================

def score_one_pair(
    gap_record: Dict[str, Any],
    company_record: Dict[str, Any],
    gap_id: str,
    gap_embedding_map: Dict[str, List[float]],
    company_embedding_map: Dict[str, List[float]],
    capability_lookup: Dict[str, Dict[str, float]],
    industry_lookup: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    gap_embedding = gap_embedding_map[gap_id]
    company_embedding = company_embedding_map[company_record["company_name"]]

    semantic_score = compute_semantic_score_from_embeddings(gap_embedding, company_embedding)

    tech_result = compute_capability_fit(
        gap_capabilities=gap_record.get("capability", []),
        company_capabilities=company_record.get("capability", []),
        capability_lookup=capability_lookup,
    )

    industry_result = compute_industry_fit(
        gap_industries_raw=gap_record.get("industry", []),
        company_industries=company_record.get("industry", []),
        industry_lookup=industry_lookup,
    )

    geography_result = compute_geography_fit(
        gap_geos_raw=gap_record.get("geography", []),
        company_geos_raw=company_record.get("geography_all", []),
    )

    mapped_gap_industries = map_gap_industries(gap_record.get("industry", []))
    geo_adjustment, geo_sensitivity, geo_weight = compute_geo_adjustment(
        mapped_gap_industries=mapped_gap_industries,
        geography_fit=geography_result["geography_fit"],
    )

    final_score = (
        semantic_score
        * tech_result["tech_coverage"]["multiplier"]
        * industry_result["industry_fit"]["multiplier"]
        * geo_adjustment
    )
    final_score = round(final_score, 2)

    geography_result["geography_fit"]["sensitivity"] = geo_sensitivity
    geography_result["geography_fit"]["sensitivity_weight"] = geo_weight
    geography_result["geography_fit"]["adjustment"] = geo_adjustment

    reason = build_reason(
        semantic_score=semantic_score,
        tech_coverage=tech_result,
        industry_fit=industry_result,
        geography_fit=geography_result,
        geo_adjustment=geo_adjustment,
    )

    return {
        "gap_id": gap_id,
        "gap_title": gap_record.get("gap_title", ""),
        "gap_text": gap_record.get("gap_text", ""),
        "target_company": gap_record.get("company", ""),
        "candidate_company": company_record.get("company_name", ""),
        "semantic_score": semantic_score,
        "capability_score": tech_result["capability_score"],
        "industry_score": industry_result["industry_score"],
        "geography_score": geography_result["geography_score"],
        "tech_coverage": tech_result["tech_coverage"],
        "industry_fit": industry_result["industry_fit"],
        "geography_fit": geography_result["geography_fit"],
        "final_score": final_score,
        "reason": reason,
    }


def run_gms_pipeline(
    gap_jsonl_path: str,
    company_jsonl_path: str,
    capability_lookup_path: str,
    industry_lookup_path: str,
    output_json_path: str,
    output_jsonl_path: str,
    api_key: Optional[str] = None,
    top_k_per_gap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    gap_records = load_jsonl(gap_jsonl_path)
    company_records = load_jsonl(company_jsonl_path)

    capability_lookup = load_json(capability_lookup_path)
    industry_lookup = load_json(industry_lookup_path)

    gap_embedding_map, company_embedding_map = prepare_embedding_maps(
        client=client,
        gap_records=gap_records,
        company_records=company_records,
    )

    all_results = []

    for gap_idx, gap_record in enumerate(gap_records, start=1):
        gap_id = f"g{gap_idx:03d}"
        pair_results = []

        for company_record in company_records:
            result = score_one_pair(
                gap_record=gap_record,
                company_record=company_record,
                gap_id=gap_id,
                gap_embedding_map=gap_embedding_map,
                company_embedding_map=company_embedding_map,
                capability_lookup=capability_lookup,
                industry_lookup=industry_lookup,
            )
            pair_results.append(result)

        pair_results = sorted(pair_results, key=lambda x: x["final_score"], reverse=True)

        if top_k_per_gap is not None:
            pair_results = pair_results[:top_k_per_gap]

        all_results.extend(pair_results)

    all_results = add_display_scores(all_results)

    save_json(output_json_path, all_results)
    save_jsonl(output_jsonl_path, all_results)
    return all_results


# =========================================================
# 9. EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":
    GAP_JSONL_PATH = FILE_GAPS_VALID
    COMPANY_JSONL_PATH = FILE_PORTFOLIO_JSONL

    CAPABILITY_LOOKUP_PATH = FILE_CAPABILITY_LOOKUP
    INDUSTRY_LOOKUP_PATH = FILE_INDUSTRY_LOOKUP

    OUTPUT_JSON_PATH = FILE_GMS_RESULTS_JSON
    OUTPUT_JSONL_PATH = FILE_GMS_RESULTS_JSONL

    results = run_gms_pipeline(
        gap_jsonl_path=str(GAP_JSONL_PATH),
        company_jsonl_path=str(COMPANY_JSONL_PATH),
        capability_lookup_path=str(CAPABILITY_LOOKUP_PATH),
        industry_lookup_path=str(INDUSTRY_LOOKUP_PATH),
        output_json_path=str(OUTPUT_JSON_PATH),
        output_jsonl_path=str(OUTPUT_JSONL_PATH),
        top_k_per_gap=10,
    )

    print(f"Done. Generated {len(results)} GMS records.")