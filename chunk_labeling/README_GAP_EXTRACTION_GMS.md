# Gap Extraction & GMS Matcher Pipeline

## Overview
This pipeline represents the core strategic intelligence engine. It extracts "Capability Gaps" (areas where hyperscalers like Google or NVIDIA lack a capability) from unstructured text chunks, validates them through multiple rigorous automated review layers (Critic & Freshness), and then calculates a **Gap Match Score (GMS)** to map these hyperscaler gaps to the capabilities of BGV portfolio companies.

The pipeline ensures that only high-confidence, fresh, and product-relevant strategic gaps are presented to the frontend **Strategic Gap Library** and **Portfolio Matcher**.

---

## 1. Pipeline Execution Flow & Scripts

The pipeline consists of 4 sequential steps:

### Step 1: Baseline Gap Extraction (`gms_with_10k.py`)
- **Purpose**: Extracts initial capability gaps from labeled chunk data.
- **Logic**: It uses LLMs to read the source chunks and identify gaps. Crucially, it incorporates **10-K financial reports as a baseline counter-evidence**. If a chunk suggests a company is building a capability, but their 10-K says they already have it, the gap is suppressed to reduce false positives.
- **Input**: `data/input/all_chunks_labeled_v2.jsonl` (The raw chunks annotated with basic tags)
- **Output**: `data/gaps_pipeline/gaps_extracted_with_10k.jsonl`

### Step 2: Critic Review & Filtering (`gms_with_10k_review_critic.py`)
- **Purpose**: A secondary "Critic" LLM pass to aggressively filter out hallucinations or weak evidence.
- **Logic**: Evaluates whether the extracted gap is a genuine "Capability Deficit" (the company explicitly states they lack it) or an "Ecosystem Opportunity" (the company is providing a platform for 3rd parties to build it). 
- **Outputs**:
  - `data/gaps_pipeline/gaps_critic_filtered.jsonl` (High-confidence, accepted gaps passing to the next round)
  - `data/gaps_pipeline/gaps_ecosystem_quarantine.jsonl` (Ecosystem gaps with "low confidence" flagged for manual review)
  - `data/gaps_pipeline/gaps_critic_rejected.jsonl` (Rejected gaps with reasons)

### Step 3: Freshness & Staleness Checker (`gaps_staleness_checker.py`)
- **Purpose**: Validates whether the capability gap is still relevant today or if the company has already launched a product (GA) to fill it.
- **Logic**: If the original chunk evidence is older than 6 months, the script automatically queries the web via the **Tavily API**. It constructs a highly specific, product-focused query based on the gap title (filtering out generic stop-words) and searches for evidence of a "GA (Generally Available)" launch in 2025/2026. Gaps solved by a recent GA launch are marked as STALE.
- **Outputs**:
  - `data/gaps_pipeline/gaps_still_valid.jsonl` (**The final, clean gap library fed directly to the frontend `/api/gaps`**)
  - `data/gaps_pipeline/gaps_freshness_checked.jsonl` (Full log with search URLs and reasons)
  - `data/gaps_pipeline/gaps_stale.jsonl` (Gaps that are no longer valid)

### Step 4: Gap Match Score Calculation (`gms.py`)
- **Purpose**: Maps the validated hyperscaler gaps against BGV's portfolio companies.
- **Logic**: Uses OpenAI `text-embedding-3-small` to semantically match the gap's capability against the portfolio company's capabilities (utilizing `capability_match_lookup.json`). Calculates a deterministic match score.
- **Inputs**: 
  - `data/gaps_pipeline/gaps_still_valid.jsonl` (The validated gaps)
  - `data/input/bgv_portfolio_tagged.json` (The portfolio company profiles)
- **Output**: `data/gms/gms_results.jsonl` (**Fed to the frontend Portfolio Matcher `/api/match` and Recommender API**)

---

## 2. Key Offline / Configuration Files
- **`data/gms/capability_match_lookup.json`**: A static taxonomy mapping specific capabilities to broader capability categories, ensuring the GMS algorithm uses a controlled vocabulary when matching gaps to startups.
- **`data/input/bgv_portfolio_tagged.json`**: The database of BGV portfolio companies, tagged with their respective industries and capabilities.

## 3. Frontend Connection
- **Strategic Gap Library**: Reads directly from `data/gaps_pipeline/gaps_still_valid.jsonl` (served via `backend/app.py` -> `/api/gaps`).
- **Portfolio Matcher**: Reads directly from `data/gms/gms_results.jsonl` (served via `/api/match`).
- **Geography Logic**: Region tags are generated strictly using a standardized list (`"United States"`, `"Europe"`, etc.) in Step 1 to perfectly map to the frontend UI filters.
