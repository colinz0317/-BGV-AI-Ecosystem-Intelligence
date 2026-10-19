# Investment Intent Score (ISS / IIS) Pipeline

## Overview
The ISS (or IIS) pipeline is a deterministic scoring engine that measures the strategic momentum of major hyperscalers (e.g., Google, Microsoft, NVIDIA, OpenAI) across specific AI themes. 

Unlike the Gap Extraction pipeline (which looks for what companies *lack*), the ISS pipeline quantifies what companies are actively *investing in and building*. The resulting data drives the **Opportunity Matrix** heatmap on the frontend.

---

## 1. Pipeline Execution Flow & Scripts

### Step 1: Chunk Labeling (`iss_label.py`)
- **Purpose**: Processes raw text chunks and annotates them with specific event signals.
- **Logic**: It uses an LLM to assign **dual-theme labels** to text chunks. A single chunk can contribute to two dimensions simultaneously:
  1. `development_scope` (Vertical AI vs. Horizontal AI)
  2. `agent_architecture` (Single-agent AI vs. Multi-agent AI)
- **Input**: Raw text chunks from `data/input/all_chunks_labeled_v2.jsonl` (pre-labeled)
- **Output**: `data/input/all_chunks_labeled_v2.jsonl` (ISS labels added in-place / merged)

### Step 2: Deterministic Scoring Engine (`iss_scoring.py`)
- **Purpose**: Translates the labeled text chunks into quantifiable, time-decayed numerical scores.
- **Logic**: 
  - **Normalization**: Standardizes dates, event types, and sources. Evaluates the reliability of the source (e.g., SEC Filing = 1.0, TechCrunch = 0.8).
  - **Time Decay**: Applies a mathematical time-decay factor ($\lambda = 0.12$ per month) so recent events weigh heavier than old news.
  - **Deduplication**: Groups identical events based on fingerprints to prevent double-counting.
  - **Sub-score Computation**: Calculates scores across 6 distinct dimensions:
    - **A**: Strategic Commitment
    - **B**: Resourcing
    - **C**: Product Shipping & Iteration
    - **D**: Commercial Traction
    - **E**: Partnerships & GTM Motions
    - **F**: Consistency (Long-term sustained effort)
  - **Aggregation**: Computes a final `iis_score_0_100` for a guaranteed 16-row matrix (4 companies $\times$ 4 themes).
- **Input**: `chunks_labeled_output_*.jsonl` (Requires `event_type`, `strength_1_5`, `source_type`, etc.)

---

## 2. Key Output Files (Offline Data)

The `iss_scoring.py` script generates several deterministic CSV and JSONL files that are served to the frontend.

### Primary Outputs (Matrix Data)
- **`data/iss/iis_theme_matrix.csv`**: The aggregated 2D matrix data. **This is fed directly to the frontend Opportunity Matrix (`/api/matrix` and `/api/companies`)**.
- **`data/iss/iis_scores.csv`**: The detailed breakdown of the 16 primary rows (4 companies $\times$ 4 themes), including scores for dimensions A through F.

### Drill-down & Audit Logs (Evidence Data)
- **`data/iss/iis_normalized_events.csv`**: The flattened list of all validated, deduplicated scoring events.
- **`data/iss/iis_audit.jsonl`**: The JSON representation of the top events contributing to the scores. **This is fed to the frontend cell drill-down UI (`/api/evidence`)**.
- **`data/iss/iis_subtheme_scores.csv` & `data/iss/iis_subtheme_audit.jsonl`**: Granular scores broken down into specific sub-themes (e.g., "Vertical AI | healthcare").
- **`data/iss/iis_rejected_records.csv`**: A log of chunks that were discarded due to missing dates, unknown event types, or invalid sources.

---

## 3. Frontend Connection
- **Opportunity Matrix Heatmap**: The backend `app.py` reads `iis_theme_matrix.csv` to render the exact cell colors and values.
- **Evidence Drill-Down**: When a user clicks a cell in the matrix, the frontend fetches from `/api/evidence`, which searches the audit logs to display the underlying chunks (e.g., the specific SEC filing or earnings call quote) that contributed to that cell's high or low score.
