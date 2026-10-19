import os
from pathlib import Path

# BASE_DIR is the root of chunk_labeling/
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Directories ---
DATA_INPUT_DIR = BASE_DIR / "data" / "input"
DATA_GAPS_DIR = BASE_DIR / "data" / "gaps_pipeline"
DATA_GMS_DIR = BASE_DIR / "data" / "gms"
DATA_ISS_DIR = BASE_DIR / "data" / "iss"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

# --- Input Data Files ---
FILE_ALL_CHUNKS = DATA_INPUT_DIR / "all_chunks_labeled_v2.jsonl"
FILE_PORTFOLIO_JSON = DATA_INPUT_DIR / "bgv_portfolio_tagged.json"
FILE_PORTFOLIO_JSONL = DATA_INPUT_DIR / "bgv_portfolio_tagged.jsonl"
FILE_MSFT_CHUNKS = DATA_INPUT_DIR / "microsoft_chunks.jsonl"
FILE_GOOG_CHUNKS = DATA_INPUT_DIR / "alphabet_chunks.jsonl"
FILE_NVDA_CHUNKS = DATA_INPUT_DIR / "nvidia_chunks.jsonl"

# --- Gaps Pipeline Outputs ---
FILE_GAPS_EXTRACTED = DATA_GAPS_DIR / "gaps_extracted_with_10k.jsonl"
FILE_GAPS_CRITIC_FILTERED = DATA_GAPS_DIR / "gaps_critic_filtered.jsonl"
FILE_GAPS_CRITIC_REJECTED = DATA_GAPS_DIR / "gaps_critic_rejected.jsonl"
FILE_GAPS_ECOSYSTEM_QUARANTINE = DATA_GAPS_DIR / "gaps_ecosystem_quarantine.jsonl"

FILE_GAPS_FRESHNESS_CHECKED = DATA_GAPS_DIR / "gaps_freshness_checked.jsonl"
FILE_GAPS_STALE = DATA_GAPS_DIR / "gaps_stale.jsonl"
FILE_GAPS_VALID = DATA_GAPS_DIR / "gaps_still_valid.jsonl"
FILE_GAPS_UNCLEAR = DATA_GAPS_DIR / "gaps_freshness_unclear.jsonl"

# --- GMS Outputs ---
FILE_GMS_RESULTS_JSON = DATA_GMS_DIR / "gms_results.json"
FILE_GMS_RESULTS_JSONL = DATA_GMS_DIR / "gms_results.jsonl"
FILE_CAPABILITY_LOOKUP = DATA_GMS_DIR / "capability_match_lookup.json"
FILE_CAPABILITY_SCORES_JSON = DATA_GMS_DIR / "capability_match_scores.json"
FILE_CAPABILITY_SCORES_JSONL = DATA_GMS_DIR / "capability_match_scores.jsonl"
FILE_INDUSTRY_LOOKUP = DATA_GMS_DIR / "industry_match_lookup.json"
FILE_INDUSTRY_SCORES_JSON = DATA_GMS_DIR / "industry_match_scores.json"
FILE_INDUSTRY_SCORES_JSONL = DATA_GMS_DIR / "industry_match_scores.jsonl"

# --- ISS Outputs ---
FILE_ISS_THEME_MATRIX = DATA_ISS_DIR / "iis_theme_matrix.csv"
FILE_ISS_SCORES = DATA_ISS_DIR / "iis_scores.csv"
FILE_ISS_AUDIT = DATA_ISS_DIR / "iis_audit.jsonl"
FILE_ISS_SUBTHEME_AUDIT = DATA_ISS_DIR / "iis_subtheme_audit.jsonl"
FILE_ISS_SUBTHEME_SCORES = DATA_ISS_DIR / "iis_subtheme_scores.csv"
FILE_ISS_NORMALIZED_EVENTS = DATA_ISS_DIR / "iis_normalized_events.csv"
FILE_ISS_REJECTED = DATA_ISS_DIR / "iis_rejected_records.csv"

# --- Config/Logs ---
FILE_REFRESH_MANIFEST = CONFIG_DIR / "refresh_manifest.json"
FILE_CHUNKING_MANIFEST = CONFIG_DIR / "chunking_manifest.json"
FILE_GAPS_EXTRACTION_ERRORS = LOGS_DIR / "gaps_extraction_errors_with_10k.jsonl"
