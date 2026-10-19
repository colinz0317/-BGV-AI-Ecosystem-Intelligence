# AI Ecosystem Intelligence Platform — BGV × SCU Practicum

**Collaborative academic project | Market intelligence · AI analytics · Product development**

A research prototype developed by a Santa Clara University team in collaboration with Berkeley Global Ventures (BGV). The platform was designed to support analysis of AI-sector strategy by combining public-information collection, language-model-assisted classification, investment strategy scoring (ISS), capability-gap analysis, portfolio matching (GMS), and an interactive dashboard.

**Contributors:** Colin Zhang, Lanboer Zhao, Rachel He, Xipeng Ma  
**Faculty advisor:** Professor Manoochehr Ghiassi

> **Portfolio/source-code edition:** This repository contains selected project implementation code, not the original deployed application. The original website is offline. Client-related datasets, generated scoring and matching results, internal handoff documentation, and historical archives have been intentionally omitted. The complete workflow **has not been verified to run** from this public edition. No investment recommendations or results are represented here as current.

## Business problem

AI companies announce investments, partnerships, product initiatives, and capability developments across numerous sources. Reviewing those developments manually makes it difficult to compare strategic priorities or systematically identify potential capability gaps. This project explored a repeatable research workflow to structure public signals and support venture-oriented market intelligence.

## What the team built

1. **Data ingestion and processing:** Python scrapers, cleaners, and text chunking for announcements, news, and other research inputs.
2. **Investment Strategy Scoring (ISS):** LLM-assisted event classification and a scoring pipeline to compare strategic activity across defined themes.
3. **Gap analysis:** Extraction, filtering, and freshness checks for possible strategic capability gaps.
4. **Gap Matching System (GMS):** A matching workflow intended to compare gap descriptions with venture-company capabilities.
5. **Recommendation layer:** Code for AI-assisted analysis and memo generation.
6. **Product interface:** A React/Vite frontend and FastAPI backend supporting exploration of scores, gaps, and matches.

## Architecture

```text
Public information sources
         |
  Python collection + cleaning + chunking
         |
  LLM-assisted classification
         |
  ISS scoring + gap extraction / validation
         |
  GMS matching + recommendation generation
         |
  FastAPI backend  <-->  React/Vite dashboard
```

## Repository map

| Directory | Contents |
| --- | --- |
| `data_processing/` | Data collection, cleaning, chunking, and dataset preparation scripts |
| `chunk_labeling/scripts/` | Labeling, ISS scoring, gap validation, and matching logic |
| `chunk_labeling/config/` | Non-sensitive configuration manifest |
| `strategic_recommender/` | Recommendation data retrieval and prompt construction |
| `backend/` | FastAPI application source |
| `frontend/` | React/Vite user-interface source |

## Technical stack

Python, pandas-based data processing, LLM APIs, FastAPI, React, Vite, JavaScript, and CSV/JSON/JSONL data workflows. Some components reference external services and require credentials or data files that are **not included**.

## Reproducibility and limitations

This is a **source-code portfolio** rather than a turnkey demo. The original batch-produced artifacts and venture-portfolio inputs are not distributed. Several backend endpoints and scripts expect those files and will not function without replacement inputs. The frontend may render portions of its interface independently, but data-driven features depend on the backend. No claim is made that the full system currently runs or that any historical output is investment advice.

For local exploration of the UI source:

```bash
cd frontend
npm install
npm run dev
```

The commands start the Vite development environment; they do **not** restore the original backend, data, or production website.

## Disclosure and attribution

This project was developed collaboratively as part of a university–industry practicum. It is presented to demonstrate technical approach, analytical workflow, and product design—not as an official BGV product or endorsement. BGV-specific datasets and internal operational material have been excluded from this edition. Any further public release should be reviewed against the project's applicable client/course confidentiality terms.
