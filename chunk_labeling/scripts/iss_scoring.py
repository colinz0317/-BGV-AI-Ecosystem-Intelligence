import ast
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Set

import pandas as pd

from paths import (
    FILE_ALL_CHUNKS,
    FILE_ISS_SCORES,
    FILE_ISS_THEME_MATRIX,
    FILE_ISS_NORMALIZED_EVENTS,
    FILE_ISS_AUDIT,
    FILE_ISS_SUBTHEME_SCORES,
    FILE_ISS_SUBTHEME_AUDIT,
    FILE_ISS_REJECTED
)


"""
Deterministic Investment Intent Score (IIS / ISS) pipeline.

Updated for dual-theme chunk labels:
- Each chunk can contribute to:
  1) development_scope: Vertical AI / Horizontal AI
  2) agent_architecture: single-agent AI / multi-agent AI
- A chunk may contribute to both dimensions at the same time.
- Final top-level output always includes 16 rows:
  4 companies x 4 themes
- Sub-theme drill-down is also generated:
  - For development themes: use raw sub_theme
  - For agent themes: use a combined label such as
      "Vertical AI | healthcare"
      "Horizontal AI | enterprise_automation"
"""


# =========================================================
# Global constants
# =========================================================

VALID_SOURCE_TYPES = {
    "sec_filing",
    "earning_call",
    "official_website_announcement",
    "techCrunch",
    "CVC_from_google_news",
}

DEFAULT_SOURCE_RELIABILITY_MAP = {
    "sec_filing": 1.0,
    "earning_call": 1.0,
    "official_website_announcement": 0.75,
    "techCrunch": 0.8,
    "CVC_from_google_news": 0.55,
}

VALID_DEVELOPMENT_SCOPES = {
    "Vertical AI",
    "Horizontal AI",
    "unknown",
}

VALID_AGENT_ARCHITECTURES = {
    "single-agent AI",
    "multi-agent AI",
    "unknown",
}

VALID_THEMES = {
    "Vertical AI",
    "Horizontal AI",
    "single-agent AI",
    "multi-agent AI",
}

VALID_EVENT_TYPES = {
    "Strategic Commitment",
    "Resourcing",
    "Product Shipping & Iteration",
    "Commercial Traction",
    "Partnerships & GTM Motions",
    "COMMITMENT",
    "RESOURCING",
    "PRODUCT",
    "DEAL_TRACTION",
    "PARTNERSHIP",
}

DEFAULT_COMPANIES = ["Google", "Microsoft", "NVIDIA", "OpenAI"]


# =========================================================
# Configuration
# =========================================================

@dataclass
class IISConfig:
    """
    Central configuration for the Investment Intent Score (IIS/ISS) pipeline.
    """

    # Input / output
    input_file: str = str(FILE_ALL_CHUNKS)

    # Top-level outputs
    output_score_file: str = str(FILE_ISS_SCORES)
    output_theme_matrix_file: str = str(FILE_ISS_THEME_MATRIX)

    # Event / audit outputs
    output_event_file: str = str(FILE_ISS_NORMALIZED_EVENTS)
    output_audit_file: str = str(FILE_ISS_AUDIT)

    # Sub-theme outputs
    output_subtheme_score_file: str = str(FILE_ISS_SUBTHEME_SCORES)
    output_subtheme_audit_file: str = str(FILE_ISS_SUBTHEME_AUDIT)

    # Rejected log
    output_rejected_file: str = str(FILE_ISS_REJECTED)

    # Fixed company universe for guaranteed 16 rows
    companies: List[str] = field(default_factory=lambda: list(DEFAULT_COMPANIES))

    # Scoring date ("now")
    score_date: date = field(default_factory=lambda: date.today())

    # Event-level time decay per month
    lambda_month: float = 0.12

    # Bucket-level decay for consistency score
    bucket_decay_beta: float = 0.85

    # Lookback settings
    consistency_lookback_months: int = 12
    consistency_num_buckets: int = 4
    consistency_bucket_event_threshold: int = 3
    product_window_days: int = 365

    # Product shipping score parameters
    product_cap_n_star: int = 8
    product_mix_alpha: float = 0.6

    # Resourcing scoring mode: "mean" or "saturation"
    resourcing_mode: str = "mean"
    resourcing_saturation_tau: float = 2.5

    # Source reliability behavior
    use_source_reliability: bool = True

    # Canonical source types allowed in current pipeline
    valid_source_types: Set[str] = field(default_factory=lambda: set(VALID_SOURCE_TYPES))

    # Optional user override
    source_reliability_map: Optional[Dict[str, float]] = None

    # Final weights for A-F
    weights: Dict[str, float] = field(default_factory=lambda: {
        "A_commitment": 20.0,
        "B_resourcing": 22.0,
        "C_product": 18.0,
        "D_traction": 22.0,
        "E_partnership": 13.0,
        "F_consistency": 5.0,
    })

    # Agent architecture themes emphasize shipped/deployed agent capability more
    # than broad strategic commitment.
    agent_weights: Dict[str, float] = field(default_factory=lambda: {
        "A_commitment": 5.0,
        "B_resourcing": 15.0,
        "C_product": 35.0,
        "D_traction": 25.0,
        "E_partnership": 15.0,
        "F_consistency": 5.0,
    })

    # Event label mapping
    event_type_map: Dict[str, str] = field(default_factory=lambda: {
        "Strategic Commitment": "COMMITMENT",
        "Resourcing": "RESOURCING",
        "Product Shipping & Iteration": "PRODUCT",
        "Commercial Traction": "DEAL_TRACTION",
        "Partnerships & GTM Motions": "PARTNERSHIP",
        "COMMITMENT": "COMMITMENT",
        "RESOURCING": "RESOURCING",
        "PRODUCT": "PRODUCT",
        "DEAL_TRACTION": "DEAL_TRACTION",
        "PARTNERSHIP": "PARTNERSHIP",
    })


# =========================================================
# Utility functions
# =========================================================

def safe_str(x: Any) -> str:
    """Safely convert a value to string."""
    if pd.isna(x) if isinstance(x, float) else False:
        return ""
    return "" if x is None else str(x)


def load_records(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL or CSV into a list of dictionaries."""
    if file_path.endswith(".jsonl"):
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path).to_dict(orient="records")

    raise ValueError("Unsupported file format. Please use .jsonl or .csv")


def save_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    """Save a list of dicts to JSONL."""
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_date(date_str: str) -> Optional[date]:
    """Parse common date formats into a Python date."""
    if not date_str:
        return None

    date_str = date_str.strip()
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%Y-%m",
        "%Y/%m",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if fmt in ("%Y-%m", "%Y/%m"):
                return date(dt.year, dt.month, 1)
            return dt.date()
        except ValueError:
            continue
    return None


def months_between(d1: date, d2: date) -> int:
    """Integer month difference between two dates."""
    return max(0, (d1.year - d2.year) * 12 + (d1.month - d2.month))


def days_between(d1: date, d2: date) -> int:
    """Difference in days between two dates."""
    return (d1 - d2).days


def strength_to_unit_interval(strength_1_5: float) -> float:
    """Map 1..5 to 0..1."""
    try:
        s = float(strength_1_5)
    except Exception:
        s = 1.0
    s = min(max(s, 1.0), 5.0)
    return (s - 1.0) / 4.0


def weighted_mean(values_and_weights: List[Tuple[float, float]]) -> float:
    """Weighted mean with safe handling."""
    numerator = 0.0
    denominator = 0.0
    for value, weight in values_and_weights:
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def re_normalize_text(text: str) -> str:
    """Normalize text for de-duplication."""
    text = safe_str(text).lower().strip()
    text = " ".join(text.split())
    return text


# =========================================================
# Source reliability resolution
# =========================================================

def resolve_source_reliability_map(config: IISConfig) -> Dict[str, float]:
    """Resolve the final source reliability map."""
    final_map = dict(DEFAULT_SOURCE_RELIABILITY_MAP)

    if config.source_reliability_map is None:
        return final_map

    invalid_keys = set(config.source_reliability_map.keys()) - set(config.valid_source_types)
    if invalid_keys:
        raise ValueError(
            f"Invalid source_type(s) in source_reliability_map: {sorted(invalid_keys)}. "
            f"Allowed values are: {sorted(config.valid_source_types)}"
        )

    for k, v in config.source_reliability_map.items():
        try:
            v = float(v)
        except Exception:
            raise ValueError(f"Source reliability for '{k}' must be numeric, got: {v}")

        if not 0 <= v <= 1:
            raise ValueError(
                f"Source reliability for '{k}' must be between 0 and 1, got: {v}"
            )

        final_map[k] = v

    return final_map


# =========================================================
# Validation helpers
# =========================================================

def validate_config(config: IISConfig) -> None:
    """Validate important config assumptions."""
    if config.resourcing_mode not in {"mean", "saturation"}:
        raise ValueError("resourcing_mode must be either 'mean' or 'saturation'.")

    if config.consistency_num_buckets <= 0:
        raise ValueError("consistency_num_buckets must be > 0.")

    if config.consistency_lookback_months <= 0:
        raise ValueError("consistency_lookback_months must be > 0.")

    if config.consistency_bucket_event_threshold <= 0:
        raise ValueError("consistency_bucket_event_threshold must be > 0.")

    if config.product_window_days <= 0:
        raise ValueError("product_window_days must be > 0.")

    if config.product_cap_n_star <= 0:
        raise ValueError("product_cap_n_star must be > 0.")

    if not 0 <= config.product_mix_alpha <= 1:
        raise ValueError("product_mix_alpha must be between 0 and 1.")

    missing_weight_keys = {
        "A_commitment",
        "B_resourcing",
        "C_product",
        "D_traction",
        "E_partnership",
        "F_consistency",
    } - set(config.weights.keys())

    if missing_weight_keys:
        raise ValueError(f"Missing required weight keys: {missing_weight_keys}")

    missing_agent_weight_keys = {
        "A_commitment",
        "B_resourcing",
        "C_product",
        "D_traction",
        "E_partnership",
        "F_consistency",
    } - set(config.agent_weights.keys())

    if missing_agent_weight_keys:
        raise ValueError(f"Missing required agent weight keys: {missing_agent_weight_keys}")

    resolved_map = resolve_source_reliability_map(config)
    missing_source_types = config.valid_source_types - set(resolved_map.keys())
    if missing_source_types:
        raise ValueError(
            f"Resolved source_reliability_map is missing source types: {missing_source_types}"
        )


# =========================================================
# Event normalization and expansion
# =========================================================

def parse_subthemes(raw_subtheme: Any) -> List[str]:
    """
    Normalize sub_theme into a clean list of labels.

    Supports all common inputs from labeling output:
    - ["ai_infra", "model_ops"]
    - "['ai_infra', 'model_ops']"
    - "ai_infra"
    - "ai_infra, model_ops"

    This prevents audit/scoring output like "['ai_infra']" and enables
    sub-theme drill-down expansion while preserving top-level score behavior.
    """
    if raw_subtheme is None:
        return ["unknown"]

    values: List[Any]

    if isinstance(raw_subtheme, list):
        values = raw_subtheme
    elif isinstance(raw_subtheme, tuple) or isinstance(raw_subtheme, set):
        values = list(raw_subtheme)
    else:
        text = safe_str(raw_subtheme).strip()
        if not text:
            return ["unknown"]

        # Handle strings like "['ai_infra', 'model_ops']" produced by CSV/JSONL serialization.
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    values = list(parsed)
                else:
                    values = [parsed]
            except Exception:
                values = [text]
        elif "," in text:
            values = [part.strip() for part in text.split(",")]
        else:
            values = [text]

    cleaned: List[str] = []
    seen: Set[str] = set()
    for v in values:
        tag = safe_str(v).strip().strip('"').strip("'")
        if not tag:
            continue
        if tag not in seen:
            cleaned.append(tag)
            seen.add(tag)

    if not cleaned:
        return ["unknown"]

    # If model returned ["unknown", "ai_infra"], keep the informative tag(s).
    non_unknown = [t for t in cleaned if t != "unknown"]
    return non_unknown or ["unknown"]


def normalize_development_subthemes(raw_subtheme: Any, development_scope: str) -> List[str]:
    """Normalize subthemes for development_scope themes."""
    if development_scope in {"Vertical AI", "Horizontal AI"}:
        return parse_subthemes(raw_subtheme)
    return ["unknown"]


def derive_agent_subtheme_group(development_scope: str, sub_theme: str) -> str:
    """
    For agent themes, build a combined drill-down label such as:
    - Vertical AI | healthcare
    - Horizontal AI | enterprise_automation
    - unknown
    """
    development_scope = safe_str(development_scope).strip()
    sub_theme = safe_str(sub_theme).strip()

    if development_scope in {"Vertical AI", "Horizontal AI"}:
        if sub_theme and sub_theme != "unknown":
            return f"{development_scope} | {sub_theme}"
        return f"{development_scope} | unknown"

    return "unknown"


def count_unique_source_events(events: List[Dict[str, Any]]) -> int:
    """Count original scoring events before sub_theme expansion."""
    return len({e.get("source_event_id", e.get("fingerprint", "")) for e in events})


def normalize_record_base(
    record: Dict[str, Any],
    config: IISConfig,
    resolved_source_reliability_map: Dict[str, float],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Normalize shared fields first.
    One raw chunk record may later expand into 0, 1, or 2 scoring events.
    """
    company = safe_str(record.get("company", "")).strip()
    development_scope = safe_str(record.get("development_scope", "")).strip()
    agent_architecture = safe_str(record.get("agent_architecture", "")).strip()
    raw_sub_themes = parse_subthemes(record.get("sub_theme", ["unknown"]))
    raw_event_type = safe_str(record.get("event_type", "")).strip()
    source_type = safe_str(record.get("source_type", "")).strip()

    if not company:
        return None, "missing_company"

    if raw_event_type in ("", "unknown"):
        return None, "missing_or_unknown_event_type"

    if raw_event_type not in config.event_type_map:
        return None, "unmapped_event_type"

    if source_type not in config.valid_source_types:
        return None, "invalid_source_type"

    if development_scope not in VALID_DEVELOPMENT_SCOPES:
        development_scope = "unknown"

    if agent_architecture not in VALID_AGENT_ARCHITECTURES:
        agent_architecture = "unknown"

    # At least one of the two theme dimensions must be usable
    if development_scope == "unknown" and agent_architecture == "unknown":
        return None, "both_theme_dimensions_unknown"

    event_type = config.event_type_map.get(raw_event_type)
    if event_type is None:
        return None, "unmapped_event_type"

    event_date_str = safe_str(record.get("event_date", "")).strip()
    document_date_str = safe_str(record.get("document_date", "")).strip()
    event_dt = parse_date(event_date_str) or parse_date(document_date_str)
    if event_dt is None:
        return None, "invalid_or_missing_date"

    evidence_span = safe_str(record.get("evidence_span", "")).strip()
    chunk_id = safe_str(record.get("chunk_id", "")).strip()
    doc_id = safe_str(record.get("source_title", "")).strip() or safe_str(record.get("source_name", "")).strip()

    strength_1_5 = record.get("strength_1_5", 1)
    s_unit = strength_to_unit_interval(strength_1_5)

    age_months = months_between(config.score_date, event_dt)
    time_weight = math.exp(-config.lambda_month * age_months)

    source_reliability = 1.0
    if config.use_source_reliability:
        source_reliability = resolved_source_reliability_map[source_type]

    event_weight = time_weight * source_reliability

    normalized = {
        "company": company,
        "development_scope": development_scope,
        "agent_architecture": agent_architecture,
        "raw_sub_themes": raw_sub_themes,
        "raw_sub_theme": raw_sub_themes[0] if raw_sub_themes else "unknown",
        "event_type": event_type,
        "event_date": event_dt.isoformat(),
        "event_date_obj": event_dt,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_type": source_type,
        "strength_1_5": float(strength_1_5) if str(strength_1_5).strip() != "" else 1.0,
        "strength_norm": s_unit,
        "time_weight": time_weight,
        "source_reliability": source_reliability,
        "event_weight": event_weight,
        "evidence_span": evidence_span,
    }
    return normalized, None


def expand_base_record_to_scoring_events(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand one base normalized chunk record into scoring events.

    Important update for list-valued sub_theme:
    - If sub_theme = ["advertising", "search_discovery"], the development view
      is expanded into two sub-theme events.
    - Each expanded event receives subtheme_fraction = 1 / N and its event_weight
      is multiplied by that fraction, so total top-level contribution is conserved.
    - source_event_id is shared across expanded sub-theme events, so product cadence
      can count the original event once instead of double-counting multi-label chunks.
    """
    company = base["company"]
    development_scope = base["development_scope"]
    agent_architecture = base["agent_architecture"]
    raw_subthemes = parse_subthemes(base.get("raw_sub_themes", base.get("raw_sub_theme", "unknown")))
    event_type = base["event_type"]
    event_date = base["event_date"]
    event_date_obj = base["event_date_obj"]
    evidence_span = base["evidence_span"]

    month_bucket = event_date_obj.strftime("%Y-%m")
    evidence_key = re_normalize_text(evidence_span)[:120]

    out = []

    # View 1: development_scope
    if development_scope in {"Vertical AI", "Horizontal AI"}:
        subthemes = normalize_development_subthemes(raw_subthemes, development_scope)
        fraction = 1.0 / max(1, len(subthemes))
        source_event_id = f"{company}|{development_scope}|{event_type}|{month_bucket}|{evidence_key}"

        for sub_theme in subthemes:
            fingerprint = f"{company}|{development_scope}|{sub_theme}|{event_type}|{month_bucket}|{evidence_key}"

            dev_event = dict(base)
            dev_event["theme"] = development_scope
            dev_event["sub_theme"] = sub_theme
            dev_event["theme_dimension"] = "development_scope"
            dev_event["subtheme_fraction"] = fraction
            dev_event["source_event_id"] = source_event_id
            dev_event["base_event_weight"] = base["event_weight"]
            dev_event["event_weight"] = base["event_weight"] * fraction
            dev_event["fingerprint"] = fingerprint
            out.append(dev_event)

    # View 2: agent_architecture
    if agent_architecture in {"single-agent AI", "multi-agent AI"}:
        # Agent drill-down also expands across the same development sub_theme list.
        # This avoids labels like "Horizontal AI | ['model_ops']".
        if development_scope in {"Vertical AI", "Horizontal AI"}:
            subthemes = normalize_development_subthemes(raw_subthemes, development_scope)
        else:
            subthemes = ["unknown"]

        fraction = 1.0 / max(1, len(subthemes))
        source_event_id = f"{company}|{agent_architecture}|{event_type}|{month_bucket}|{evidence_key}"

        for sub_theme in subthemes:
            sub_theme_group = derive_agent_subtheme_group(development_scope, sub_theme)
            fingerprint = f"{company}|{agent_architecture}|{sub_theme_group}|{event_type}|{month_bucket}|{evidence_key}"

            agent_event = dict(base)
            agent_event["theme"] = agent_architecture
            agent_event["sub_theme"] = sub_theme_group
            agent_event["theme_dimension"] = "agent_architecture"
            agent_event["subtheme_fraction"] = fraction
            agent_event["source_event_id"] = source_event_id
            agent_event["base_event_weight"] = base["event_weight"]
            agent_event["event_weight"] = base["event_weight"] * fraction
            agent_event["fingerprint"] = fingerprint
            out.append(agent_event)

    return out

def normalize_and_expand_record(
    record: Dict[str, Any],
    config: IISConfig,
    resolved_source_reliability_map: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Normalize one raw chunk record, then expand it into scoring events.
    """
    base, reason = normalize_record_base(record, config, resolved_source_reliability_map)
    if base is None:
        return [], reason

    expanded = expand_base_record_to_scoring_events(base)
    if not expanded:
        return [], "no_scoring_views_generated"

    return expanded, None


def deduplicate_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    De-duplicate repeated reporting of the same underlying event.
    De-dup is performed within each expanded scoring view.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        groups.setdefault(e["fingerprint"], []).append(e)

    deduped = []
    for fp, group in groups.items():
        best = max(group, key=lambda x: x["event_weight"] * x["strength_norm"])
        merged_chunk_ids = sorted({g["chunk_id"] for g in group if g["chunk_id"]})
        merged_doc_ids = sorted({g["doc_id"] for g in group if g["doc_id"]})

        e = dict(best)
        e["merged_chunk_ids"] = merged_chunk_ids
        e["merged_doc_ids"] = merged_doc_ids
        e["duplicate_count"] = len(group)
        deduped.append(e)

    return deduped


# =========================================================
# Sub-score computations A-F
# =========================================================

def score_A_commitment(events: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    subset = [e for e in events if e["event_type"] == "COMMITMENT"]
    values_and_weights = [(e["strength_norm"], e["event_weight"]) for e in subset]
    score = weighted_mean(values_and_weights)
    ranked = sorted(subset, key=lambda x: x["strength_norm"] * x["event_weight"], reverse=True)
    return score, ranked


def score_B_resourcing(events: List[Dict[str, Any]], config: IISConfig) -> Tuple[float, List[Dict[str, Any]]]:
    subset = [e for e in events if e["event_type"] == "RESOURCING"]
    ranked = sorted(subset, key=lambda x: x["strength_norm"] * x["event_weight"], reverse=True)

    if not subset:
        return 0.0, ranked

    if config.resourcing_mode == "saturation":
        contributions = [e["event_weight"] * e["strength_norm"] for e in subset]
        total = sum(contributions)
        score = 1.0 - math.exp(-total / config.resourcing_saturation_tau)
        return score, ranked

    values_and_weights = [(e["strength_norm"], e["event_weight"]) for e in subset]
    score = weighted_mean(values_and_weights)
    return score, ranked


def score_C_product(events: List[Dict[str, Any]], config: IISConfig) -> Tuple[float, List[Dict[str, Any]]]:
    subset = [e for e in events if e["event_type"] == "PRODUCT"]
    ranked = sorted(subset, key=lambda x: x["strength_norm"] * x["event_weight"], reverse=True)

    if not subset:
        return 0.0, ranked

    recent_subset = [
        e for e in subset
        if days_between(config.score_date, e["event_date_obj"]) <= config.product_window_days
    ]

    if not recent_subset:
        return 0.0, ranked

    n_distinct = count_unique_source_events(recent_subset)
    cadence_score = min(n_distinct / config.product_cap_n_star, 1.0)

    importance_score = weighted_mean([
        (e["strength_norm"], e["event_weight"]) for e in recent_subset
    ])

    alpha = config.product_mix_alpha
    score = alpha * cadence_score + (1 - alpha) * importance_score
    return score, ranked


def score_D_traction(events: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    subset = [e for e in events if e["event_type"] == "DEAL_TRACTION"]
    values_and_weights = [(e["strength_norm"], e["event_weight"]) for e in subset]
    score = weighted_mean(values_and_weights)
    ranked = sorted(subset, key=lambda x: x["strength_norm"] * x["event_weight"], reverse=True)
    return score, ranked


def score_E_partnership(events: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    subset = [e for e in events if e["event_type"] == "PARTNERSHIP"]
    values_and_weights = [(e["strength_norm"], e["event_weight"]) for e in subset]
    score = weighted_mean(values_and_weights)
    ranked = sorted(subset, key=lambda x: x["strength_norm"] * x["event_weight"], reverse=True)
    return score, ranked


def score_F_consistency(events: List[Dict[str, Any]], config: IISConfig) -> Tuple[float, List[Dict[str, Any]]]:
    if not events:
        return 0.0, []

    lookback_months = config.consistency_lookback_months
    k = config.consistency_num_buckets
    beta = config.bucket_decay_beta
    threshold = config.consistency_bucket_event_threshold

    bucket_size = max(1, lookback_months // k)

    bucket_scores = []
    bucket_debug = []

    for bucket_idx in range(k):
        start_month = bucket_idx * bucket_size
        end_month = (bucket_idx + 1) * bucket_size

        in_bucket = []
        for e in events:
            age_m = months_between(config.score_date, e["event_date_obj"])
            if start_month <= age_m < end_month:
                in_bucket.append(e)

        bucket_event_count = count_unique_source_events(in_bucket)
        bucket_score = min(bucket_event_count / threshold, 1.0)
        bucket_scores.append(bucket_score)
        bucket_debug.extend(in_bucket)

    numerator = sum((beta ** i) * bucket_scores[i] for i in range(k))
    denominator = sum(beta ** i for i in range(k))
    score = numerator / denominator if denominator > 0 else 0.0

    ranked = sorted(bucket_debug, key=lambda x: x["event_date_obj"], reverse=True)
    return score, ranked


# =========================================================
# Final IIS scoring
# =========================================================

def serialize_top_events(events: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Keep a compact audit trail of top contributing events."""
    out = []
    for e in events[:top_n]:
        out.append({
            "doc_id": e.get("doc_id", ""),
            "chunk_id": e.get("chunk_id", ""),
            "event_date": e.get("event_date", ""),
            "event_type": e.get("event_type", ""),
            "strength_1_5": e.get("strength_1_5", ""),
            "strength_norm": round(e.get("strength_norm", 0.0), 4),
            "event_weight": round(e.get("event_weight", 0.0), 4),
            "source_type": e.get("source_type", ""),
            "theme_dimension": e.get("theme_dimension", ""),
            "development_scope": e.get("development_scope", ""),
            "agent_architecture": e.get("agent_architecture", ""),
            "sub_theme": e.get("sub_theme", ""),
            "evidence_span": e.get("evidence_span", ""),
            "duplicate_count": e.get("duplicate_count", 1),
        })
    return out


def compute_iis_for_scope(events: List[Dict[str, Any]], config: IISConfig) -> Dict[str, Any]:
    """
    Reused for:
    - (company, theme)
    - (company, theme, sub_theme)
    """
    A, A_ranked = score_A_commitment(events)
    B, B_ranked = score_B_resourcing(events, config)
    C, C_ranked = score_C_product(events, config)
    D, D_ranked = score_D_traction(events)
    E, E_ranked = score_E_partnership(events)
    F, F_ranked = score_F_consistency(events, config)

    if events and events[0].get("theme") in {"single-agent AI", "multi-agent AI"}:
        w = config.agent_weights
    else:
        w = config.weights
    total = (
        w["A_commitment"] * A +
        w["B_resourcing"] * B +
        w["C_product"] * C +
        w["D_traction"] * D +
        w["E_partnership"] * E +
        w["F_consistency"] * F
    )

    return {
        "A_commitment": A,
        "B_resourcing": B,
        "C_product": C,
        "D_traction": D,
        "E_partnership": E,
        "F_consistency": F,
        "iis_score_0_100": total,
        "top_A_events": serialize_top_events(A_ranked),
        "top_B_events": serialize_top_events(B_ranked),
        "top_C_events": serialize_top_events(C_ranked),
        "top_D_events": serialize_top_events(D_ranked),
        "top_E_events": serialize_top_events(E_ranked),
        "top_F_events": serialize_top_events(F_ranked),
    }


# =========================================================
# Pipeline runner
# =========================================================

def run_iis_pipeline(config: IISConfig) -> None:
    """
    End-to-end pipeline:
    1. load chunk labels
    2. normalize + expand each chunk into up to two scoring events
    3. de-duplicate
    4. compute top-level scores for all 16 company-theme pairs
    5. compute sub-theme drill-down scores
    6. save outputs
    """
    validate_config(config)
    resolved_source_reliability_map = resolve_source_reliability_map(config)

    print("Using source reliability map:")
    for k in sorted(resolved_source_reliability_map.keys()):
        print(f"  {k}: {resolved_source_reliability_map[k]}")

    raw_records = load_records(config.input_file)

    normalized_events = []
    rejected_records = []

    for r in raw_records:
        expanded_events, reason = normalize_and_expand_record(
            r, config, resolved_source_reliability_map
        )
        if expanded_events:
            normalized_events.extend(expanded_events)
        else:
            rejected_records.append({
                "chunk_id": safe_str(r.get("chunk_id", "")),
                "company": safe_str(r.get("company", "")),
                "development_scope": safe_str(r.get("development_scope", "")),
                "agent_architecture": safe_str(r.get("agent_architecture", "")),
                "sub_theme": safe_str(r.get("sub_theme", "")),
                "event_type": safe_str(r.get("event_type", "")),
                "source_type": safe_str(r.get("source_type", "")),
                "is_usable_for_iss": safe_str(r.get("is_usable_for_iss", "")),
                "reason_short": safe_str(r.get("reason_short", "")),
                "evidence_span": safe_str(r.get("evidence_span", "")),
                "source_title": safe_str(r.get("source_title", "")),
                "rejection_reason": reason,
            })

    if not normalized_events:
        raise ValueError(
            "No usable normalized events found. Check your input schema, development_scope, agent_architecture, event labels, and source_type values."
        )

    deduped_events = deduplicate_events(normalized_events)

    # Save normalized event table
    event_df = pd.DataFrame([
        {
            k: (v.isoformat() if isinstance(v, date) else v)
            for k, v in e.items()
            if k != "event_date_obj"
        }
        for e in deduped_events
    ])
    event_df.to_csv(config.output_event_file, index=False)

    # =====================================================
    # Top-level grouping: (company, theme)
    # Always generate all 16 rows
    # =====================================================
    grouped_top: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for e in deduped_events:
        grouped_top.setdefault((e["company"], e["theme"]), []).append(e)

    top_score_rows = []
    top_audit_rows = []

    for company in config.companies:
        for theme in ["Vertical AI", "Horizontal AI", "single-agent AI", "multi-agent AI"]:
            events = grouped_top.get((company, theme), [])

            if events:
                result = compute_iis_for_scope(events, config)
                num_distinct_subthemes = len(set(e["sub_theme"] for e in events))
            else:
                result = {
                    "A_commitment": 0.0,
                    "B_resourcing": 0.0,
                    "C_product": 0.0,
                    "D_traction": 0.0,
                    "E_partnership": 0.0,
                    "F_consistency": 0.0,
                    "iis_score_0_100": 0.0,
                    "top_A_events": [],
                    "top_B_events": [],
                    "top_C_events": [],
                    "top_D_events": [],
                    "top_E_events": [],
                    "top_F_events": [],
                }
                num_distinct_subthemes = 0

            top_score_rows.append({
                "company": company,
                "theme": theme,
                "iis_score_0_100": round(result["iis_score_0_100"], 4),
                "A_commitment": round(result["A_commitment"], 4),
                "B_resourcing": round(result["B_resourcing"], 4),
                "C_product": round(result["C_product"], 4),
                "D_traction": round(result["D_traction"], 4),
                "E_partnership": round(result["E_partnership"], 4),
                "F_consistency": round(result["F_consistency"], 4),
                "num_events_after_dedup": count_unique_source_events(events),
                "num_expanded_subtheme_events": len(events),
                "num_distinct_subthemes": num_distinct_subthemes,
            })

            top_audit_rows.append({
                "level": "theme",
                "company": company,
                "theme": theme,
                "sub_theme": None,
                "num_events_after_dedup": count_unique_source_events(events),
                "num_expanded_subtheme_events": len(events),
                **result,
            })

    top_score_df = pd.DataFrame(top_score_rows).sort_values(["company", "theme"]).reset_index(drop=True)
    top_score_df.to_csv(config.output_score_file, index=False)

    # Theme matrix
    theme_matrix_df = top_score_df.pivot(
        index="company",
        columns="theme",
        values="iis_score_0_100"
    ).reset_index()

    for theme_name in ["Vertical AI", "Horizontal AI", "single-agent AI", "multi-agent AI"]:
        if theme_name not in theme_matrix_df.columns:
            theme_matrix_df[theme_name] = 0.0

    theme_matrix_df = theme_matrix_df[
        ["company", "Vertical AI", "Horizontal AI", "single-agent AI", "multi-agent AI"]
    ].sort_values("company").reset_index(drop=True)

    theme_matrix_df.to_csv(config.output_theme_matrix_file, index=False)

    # =====================================================
    # Sub-theme drill-down grouping: (company, theme, sub_theme)
    # =====================================================
    grouped_sub: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for e in deduped_events:
        grouped_sub.setdefault((e["company"], e["theme"], e["sub_theme"]), []).append(e)

    sub_score_rows = []
    sub_audit_rows = []

    for (company, theme, sub_theme), events in grouped_sub.items():
        result = compute_iis_for_scope(events, config)

        sub_score_rows.append({
            "company": company,
            "theme": theme,
            "sub_theme": sub_theme,
            "iis_score_0_100": round(result["iis_score_0_100"], 4),
            "A_commitment": round(result["A_commitment"], 4),
            "B_resourcing": round(result["B_resourcing"], 4),
            "C_product": round(result["C_product"], 4),
            "D_traction": round(result["D_traction"], 4),
            "E_partnership": round(result["E_partnership"], 4),
            "F_consistency": round(result["F_consistency"], 4),
            "num_events_after_dedup": count_unique_source_events(events),
            "num_expanded_subtheme_events": len(events),
        })

        sub_audit_rows.append({
            "level": "sub_theme",
            "company": company,
            "theme": theme,
            "sub_theme": sub_theme,
            "num_events_after_dedup": count_unique_source_events(events),
            "num_expanded_subtheme_events": len(events),
            **result,
        })

    sub_score_df = pd.DataFrame(sub_score_rows).sort_values(
        ["company", "theme", "sub_theme"]
    ).reset_index(drop=True)
    sub_score_df.to_csv(config.output_subtheme_score_file, index=False)

    save_jsonl(top_audit_rows, config.output_audit_file)
    save_jsonl(sub_audit_rows, config.output_subtheme_audit_file)

    if rejected_records:
        rejected_df = pd.DataFrame(rejected_records)
        rejected_df.to_csv(config.output_rejected_file, index=False)
        print(f"Saved rejected record log to: {config.output_rejected_file}")
        print("Rejected counts by reason:")
        print(rejected_df["rejection_reason"].value_counts().to_string())

    print(f"Loaded raw records: {len(raw_records)}")
    print(f"Usable expanded events before dedup: {len(normalized_events)}")
    print(f"Events after dedup: {len(deduped_events)}")
    print(f"Top-level company-theme rows: {len(top_score_rows)}")
    print(f"Sub-theme rows: {len(sub_score_rows)}")
    print(f"Saved top-level score table to: {config.output_score_file}")
    print(f"Saved theme matrix to: {config.output_theme_matrix_file}")
    print(f"Saved sub-theme score table to: {config.output_subtheme_score_file}")
    print(f"Saved normalized event table to: {config.output_event_file}")
    print(f"Saved top-level audit file to: {config.output_audit_file}")
    print(f"Saved sub-theme audit file to: {config.output_subtheme_audit_file}")


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    config = IISConfig(
        input_file=str(FILE_ALL_CHUNKS),
        output_score_file=str(FILE_ISS_SCORES),
        output_theme_matrix_file=str(FILE_ISS_THEME_MATRIX),
        output_event_file=str(FILE_ISS_NORMALIZED_EVENTS),
        output_audit_file=str(FILE_ISS_AUDIT),
        output_subtheme_score_file=str(FILE_ISS_SUBTHEME_SCORES),
        output_subtheme_audit_file=str(FILE_ISS_SUBTHEME_AUDIT),
        output_rejected_file=str(FILE_ISS_REJECTED),

        score_date=date(2026, 5, 16),

        lambda_month=0.12,
        bucket_decay_beta=0.85,
        consistency_lookback_months=12,
        consistency_num_buckets=4,
        consistency_bucket_event_threshold=3,
        product_window_days=365,
        product_cap_n_star=8,
        product_mix_alpha=0.6,
        resourcing_mode="mean",
        resourcing_saturation_tau=2.5,
        use_source_reliability=True,
        source_reliability_map=None,
    )

    run_iis_pipeline(config)
