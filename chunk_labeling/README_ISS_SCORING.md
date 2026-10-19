# ISS Scoring Documentation

## Overview

This document explains the logic implemented in `iss_scoring.py` for the deterministic Investment Intent Score (IIS / ISS) pipeline.

This version supports **dual-theme chunk labels**:

- `development_scope`
  - `Vertical AI`
  - `Horizontal AI`
  - `unknown`

- `agent_architecture`
  - `single-agent AI`
  - `multi-agent AI`
  - `unknown`

A single chunk can contribute to **both** dimensions at the same time.

Example:

- `development_scope = "Vertical AI"`
- `agent_architecture = "single-agent AI"`

In this case, the same chunk is used:

1. when computing the company’s **Vertical AI** score
2. when computing the company’s **single-agent AI** score

This makes the scoring pipeline compatible with the new labeling logic where a chunk may simultaneously describe:

- a **market / solution scope**
- an **agent architecture pattern**

---

## Core Design Choice

The scoring logic keeps the existing deterministic ISS formula unchanged.

The main update is **not** in the A–F scoring formula itself.  
The main update is in how chunk-level labeled records are converted into scoring events.

### Old behavior
Each chunk had only one `theme`, so each chunk contributed to only one scoring group.

### New behavior
Each chunk may generate up to **two scoring events**:

1. one scoring event for `development_scope`
2. one scoring event for `agent_architecture`

This means one chunk can contribute to multiple top-level theme scores.

---

## Input Schema

The pipeline expects chunk-level labeling output that includes at least the following fields:

- `chunk_id`
- `company`
- `development_scope`
- `agent_architecture`
- `sub_theme`
- `event_type`
- `strength_1_5`
- `document_date` and/or `event_date`
- `source_type`
- `evidence_span`

### Example
```json
{
  "chunk_id": "msft_agent_017",
  "company": "Microsoft",
  "development_scope": "Horizontal AI",
  "agent_architecture": "single-agent AI",
  "sub_theme": "enterprise_automation",
  "event_type": "Product Shipping & Iteration",
  "strength_1_5": 4,
  "document_date": "2026-03-25",
  "source_type": "official_website_announcement",
  "evidence_span": "..."
}
```

---

## High-Level Pipeline Flow

The scoring pipeline runs in the following sequence:

1. **Load labeled chunk records**
2. **Normalize each record**
3. **Expand each record into scoring events**
4. **De-duplicate repeated events**
5. **Compute top-level ISS for each `(company, theme)`**
6. **Compute sub-theme drill-down ISS for each `(company, theme, sub_theme)`**
7. **Save output tables and audit files**

---

## Step 1: Record Normalization

Each input record is normalized into a shared internal schema.

The normalization step validates:

- company is present
- event type is valid
- source type is valid
- at least one of:
  - `development_scope`
  - `agent_architecture`
  is usable

If both `development_scope` and `agent_architecture` are `unknown`, the record is rejected.

---

## Step 2: Event Expansion

This is the most important change in the new scoring logic.

### Rule
Each normalized chunk record may expand into up to two scoring events.

### Expansion view A: development scope
If:

- `development_scope = Vertical AI` or `Horizontal AI`

then create one scoring event with:

- `theme = development_scope`
- `sub_theme = raw sub_theme`

### Expansion view B: agent architecture
If:

- `agent_architecture = single-agent AI` or `multi-agent AI`

then create one scoring event with:

- `theme = agent_architecture`
- `sub_theme = combined context label`

This makes it possible for one chunk to contribute to both:

- a scope-based theme
- an architecture-based theme

### Example
If a chunk is labeled as:

- `development_scope = Horizontal AI`
- `agent_architecture = single-agent AI`
- `sub_theme = enterprise_automation`

then it generates:

#### Event 1
- `theme = Horizontal AI`
- `sub_theme = enterprise_automation`

#### Event 2
- `theme = single-agent AI`
- `sub_theme = Horizontal AI | enterprise_automation`

---

## Sub-theme Logic

### For development themes
If the top-level theme is:

- `Vertical AI`
- `Horizontal AI`

then `sub_theme` is used directly.

Examples:

- `healthcare`
- `financial_services`
- `enterprise_automation`
- `copilot`

### For agent themes
If the top-level theme is:

- `single-agent AI`
- `multi-agent AI`

then the drill-down `sub_theme` is constructed as a combined label:

- `Vertical AI | healthcare`
- `Vertical AI | financial_services`
- `Horizontal AI | enterprise_automation`
- `Horizontal AI | other`
- `unknown`

This preserves the context of where the agent-related signal came from.

### Why this is necessary
If a chunk is only grouped under:

- `single-agent AI`

without scope context, the drill-down view loses important information about whether that agent signal came from:

- a vertical industry workflow
- a horizontal enterprise use case

The combined label solves this.

---

## Step 3: Event De-duplication

After expansion, events are de-duplicated.

### De-duplication key
Events are grouped using a fingerprint built from:

- company
- theme
- sub_theme
- event_type
- month bucket
- normalized evidence snippet

### Why per-view de-duplication matters
Because one chunk may produce two scoring events, de-duplication must happen **within each expanded scoring view**, not before expansion.

This means:

- the Vertical AI version of the chunk is de-duped within Vertical AI
- the single-agent AI version of the chunk is de-duped within single-agent AI

---

## Step 4: Top-Level Theme Scoring

Top-level scores are computed for each:

- `(company, theme)`

Themes are fixed to:

- `Vertical AI`
- `Horizontal AI`
- `single-agent AI`
- `multi-agent AI`

Companies are currently fixed to:

- `Google`
- `Microsoft`
- `NVIDIA`
- `OpenAI`

So the top-level output always contains:

- **4 companies × 4 themes = 16 rows**

Even if no events exist for a pair, the row is still generated with score `0`.

### Why this is useful
This guarantees a stable output shape for:

- frontend matrices
- dashboards
- evaluation scripts

---

## Step 5: Sub-theme Drill-down Scoring

Sub-theme scores are computed for each:

- `(company, theme, sub_theme)`

This creates drill-down views under every top-level theme.

### Examples
For development themes:

- `Microsoft, Vertical AI, healthcare`
- `Google, Horizontal AI, sales_productivity`

For agent themes:

- `Microsoft, single-agent AI, Horizontal AI | enterprise_automation`
- `NVIDIA, multi-agent AI, Horizontal AI | other`

---

## Scoring Formula

The scoring formula itself is unchanged.

The pipeline computes six sub-scores:

- **A. Commitment**
- **B. Resourcing**
- **C. Product**
- **D. Traction**
- **E. Partnership**
- **F. Consistency**

Final score:

```text
ISS = 20*A + 22*B + 18*C + 22*D + 13*E + 5*F
```

This produces a final score on a 0–100 scale.

---

## A–F Component Definitions

### A. Commitment
Measures explicit strategic commitment signals.

Uses events mapped to:

- `COMMITMENT`

### B. Resourcing
Measures tangible resource allocation, such as:

- hiring
- capex
- infrastructure expansion
- dedicated teams

Uses events mapped to:

- `RESOURCING`

### C. Product
Measures product shipping / iteration momentum.

Uses events mapped to:

- `PRODUCT`

Includes:

- a cadence component
- an importance component

### D. Traction
Measures adoption and commercial pull.

Uses events mapped to:

- `DEAL_TRACTION`

### E. Partnership
Measures GTM and ecosystem expansion signals.

Uses events mapped to:

- `PARTNERSHIP`

### F. Consistency
Measures repeated activity over time.

Uses bucketed historical presence across the lookback window.

---

## Time Decay

Each event gets a time-decay weight:

```text
time_weight = exp(-lambda_month * age_in_months)
```

This means newer evidence contributes more than older evidence.

Default:

- `lambda_month = 0.12`

---

## Source Reliability

Each event can also be weighted by source reliability.

Default source weights:

- `sec_filing = 1.0`
- `earning_call = 1.0`
- `official_website_announcement = 0.75`
- `techCrunch = 0.8`
- `CVC_from_google_news = 0.55`

Final event weight:

```text
event_weight = time_weight * source_reliability
```

---

## Output Files

### 1. `iis_scores.csv`
Top-level theme scores.

One row per:

- `(company, theme)`

This should always contain **16 rows**.

Key fields:

- `company`
- `theme`
- `iis_score_0_100`
- `A_commitment`
- `B_resourcing`
- `C_product`
- `D_traction`
- `E_partnership`
- `F_consistency`
- `num_events_after_dedup`
- `num_distinct_subthemes`

---

### 2. `iis_theme_matrix.csv`
Matrix format for frontend use.

One row per company, one column per theme:

- `Vertical AI`
- `Horizontal AI`
- `single-agent AI`
- `multi-agent AI`

---

### 3. `iis_subtheme_scores.csv`
Drill-down sub-theme scores.

One row per:

- `(company, theme, sub_theme)`

This is the main file for evidence panel drill-down.

---

### 4. `iis_normalized_events.csv`
Normalized and expanded event table after preprocessing.

Useful for debugging and inspection.

---

### 5. `iis_audit.jsonl`
Top-level audit output for `(company, theme)`.

Includes:

- final score
- A–F component scores
- top contributing events by component

---

### 6. `iis_subtheme_audit.jsonl`
Sub-theme audit output for `(company, theme, sub_theme)`.

Useful for evidence panel rendering.

---

### 7. `iis_rejected_records.csv`
Records rejected during preprocessing.

Common reasons:

- missing company
- unknown event type
- invalid source type
- both theme dimensions unknown
- invalid or missing date

---

## Important Behavior Notes

### A chunk can contribute to two top-level themes
Example:

- `Vertical AI`
- `single-agent AI`

The same chunk will be counted in both.

This is intentional.

### A chunk cannot contribute to two development scopes
A chunk cannot be both:

- `Vertical AI`
- `Horizontal AI`

### A chunk cannot contribute to two agent architectures
A chunk cannot be both:

- `single-agent AI`
- `multi-agent AI`

### Agent-theme drill-down keeps scope context
Agent theme sub-groups are not plain `sub_theme` values.

They are combined labels such as:

- `Vertical AI | healthcare`
- `Horizontal AI | enterprise_automation`

This preserves interpretability.

---

## Recommended Frontend Usage

### Top-level matrix
Use:

- `iis_theme_matrix.csv`

to render the company × theme score matrix.

### Drill-down
When a user clicks a `(company, theme)` cell:

- read rows from `iis_subtheme_scores.csv`
- filter by `company` and `theme`
- show:
  - sub-theme score
  - A–F component values
  - supporting evidence from `iis_subtheme_audit.jsonl`

---

## Example Interpretation

If a chunk is labeled:

- `development_scope = Horizontal AI`
- `agent_architecture = single-agent AI`
- `sub_theme = enterprise_automation`

then it contributes to:

### Top-level theme scores
- `Horizontal AI`
- `single-agent AI`

### Drill-down scores
- `Horizontal AI -> enterprise_automation`
- `single-agent AI -> Horizontal AI | enterprise_automation`

This is the intended behavior.

---

## Run Example

```python
if __name__ == "__main__":
    from paths import (
        FILE_ALL_CHUNKS, FILE_ISS_SCORES, FILE_ISS_THEME_MATRIX,
        FILE_ISS_NORMALIZED_EVENTS, FILE_ISS_AUDIT, FILE_ISS_SUBTHEME_SCORES,
        FILE_ISS_SUBTHEME_AUDIT, FILE_ISS_REJECTED
    )

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
        product_window_days=180,
        product_cap_n_star=4,
        product_mix_alpha=0.6,
        resourcing_mode="mean",
        resourcing_saturation_tau=2.5,
        use_source_reliability=True,
        source_reliability_map=None,
    )

    run_iis_pipeline(config)
```

---

## Summary

This scoring pipeline now supports the new labeling setup where a single chunk may carry:

- one **development scope**
- one **agent architecture**

The scoring system handles this by expanding each chunk into multiple scoring views while keeping the deterministic ISS formula unchanged.

This allows:

- stable top-level theme scores
- accurate dual-theme contribution
- interpretable sub-theme drill-down for both development and agent themes
