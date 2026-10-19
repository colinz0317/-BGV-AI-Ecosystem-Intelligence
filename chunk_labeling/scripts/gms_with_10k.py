"""
Gap Extraction Pipeline for GMS (Gap Match Score) — taxonomy-constrained version
with 10-K baseline counter-evidence support
===============================================================================
This script reads ISS-labeled chunks (output of iss_label.py) and extracts
entity-level "capability gaps" per company. These gaps are the input to
the GMS matching algorithm.

Key upgrades:
- capability tags are constrained to the same controlled vocabulary used
  in the portfolio company tagging pipeline, so gap capabilities and company
  capabilities share the same taxonomy
- adds gap_title for UI-friendly short display titles
- integrates 10-K baseline chunks as counter-evidence context to reduce
  false gap extraction (gaps for capabilities the company already has)

Pipeline position:
  Raw chunks → [ISS labeling] → labeled chunks → [THIS SCRIPT] → gaps → [GMS matching]
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Set, Optional, Tuple
from pathlib import Path

from paths import (
    BASE_DIR,
    FILE_ALL_CHUNKS,
    FILE_GAPS_EXTRACTED,
    FILE_GAPS_EXTRACTION_ERRORS,
    FILE_MSFT_CHUNKS,
    FILE_GOOG_CHUNKS,
    FILE_NVDA_CHUNKS
)

import numpy as np
import pandas as pd
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


# =========================
# Configuration
# =========================
INPUT_FILE = str(FILE_ALL_CHUNKS)
OUTPUT_FILE = str(FILE_GAPS_EXTRACTED)
ERROR_FILE = str(FILE_GAPS_EXTRACTION_ERRORS)

# 10-K baseline chunk files (unfiltered, used as counter-evidence context only)
TENK_FILES = {
    "Microsoft": str(FILE_MSFT_CHUNKS),
    "Google": str(FILE_GOOG_CHUNKS),
    "NVIDIA": str(FILE_NVDA_CHUNKS),
}

MODEL_NAME = "gpt-4.1-mini"
SLEEP_BETWEEN_CALLS = 1.0
MAX_RETRIES = 3

MIN_STRENGTH = 3
REQUIRE_USABLE = True

MAX_CHUNKS_PER_BATCH = 40
MAX_BATCHES_PER_COMPANY = None  # None = no limit, process all chunks

# Cap the number of previously-extracted gap descriptions injected into each
# batch prompt. Very long prev_gaps lists blow the context window on large runs.
PREV_GAPS_CAP = 50

# Cap on 10-K baseline chunks per batch to avoid blowing context window.
# 10-K files can be large; this keeps prompt size manageable.
MAX_TENK_CHUNKS_PER_BATCH = 30

GAP_DEDUP_SIMILARITY_THRESHOLD = 0.85
EMBEDDING_MODEL = "text-embedding-3-small"

COMPANIES = ["Microsoft", "Google", "NVIDIA", "OpenAI"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================
# Shared Controlled Vocabularies
# =========================
CAPABILITY_VOCAB = {
    "ai_crm", "ai_observability", "ai_agents", "llm_fine_tuning",
    "llm_routing", "ml_in_database", "rag_tooling", "model_governance",
    "ai_strategy_analytics", "finops_ai", "genai_platform",
    "predictive_analytics", "computer_vision", "ai_accessibility_testing",
    "identity_risk_intelligence", "passwordless_auth", "client_side_security",
    "fraud_detection", "anti_money_laundering", "audit_logging",
    "customer_data_platform", "cloud_cost_optimization", "multi_cloud_monitoring",
    "application_deployment", "paas", "mobile_app_deployment",
    "tech_intelligence", "regulatory_automation", "construction_management",
    "drone_delivery", "drone_inspection", "satellite_analytics",
    "recycling_automation", "food_robotics", "logistics_optimization",
    "route_optimization", "carbon_accounting", "customer_success",
    "revenue_intelligence", "buyer_experience_platform",
    "marketing_automation", "digital_transformation_services",
    "ecommerce_platform",
}

INDUSTRY_HINTS = [
    "GENERAL",
    "healthcare", "manufacturing", "energy", "automotive",
    "internet_platform", "financial_services", "insurance",
    "construction", "logistics", "retail", "food_service",
    "media", "utilities", "waste_management", "medical_devices",
    "industrial_inspection", "telecommunications", "government",
    "sustainability", "ecommerce", "security", "cloud_infra",
    "ai_infra", "sales_marketing", "dev_tools", "education",
    "legal", "agriculture", "technology", "consumer_electronics",
    "smart_home", "professional_services", "wellness",
    "robotics", "semiconductors", "transportation",
    "cultural_preservation", "emerging_markets",
]

CAPABILITY_VOCAB_TEXT = ", ".join(sorted(CAPABILITY_VOCAB))
INDUSTRY_HINTS_TEXT = ", ".join(INDUSTRY_HINTS)


# =========================
# System Prompt
# =========================
SYSTEM_PROMPT = f"""You are an expert analyst for a venture capital intelligence platform.

Your task: given a batch of labeled text chunks from ONE company, identify and extract
distinct capability gaps — areas where this company has admitted needs, is actively
seeking solutions, lacks internal capability, or is investing to fill a deficiency.

A gap is NOT simply "the company is doing X." A gap is: "the company needs X but does
not yet have it internally, or is seeking external partners/acquisitions/solutions for it."

========================
GAP SIGNALS
========================
Signals that indicate a gap:
- Executive statements about unmet needs or areas requiring improvement
- Partnership announcements that fill a missing capability (the need is the gap)
- Acquisition of companies to gain capability the acquirer lacked
- Product limitations acknowledged in earnings calls or press
- Customer complaints or feature requests mentioned in public sources
- Areas where the company is investing heavily but results are nascent
- RFPs, developer program expansions, or ecosystem calls targeting specific capabilities

The following do NOT constitute gaps:
- ALREADY BUILDING OR LAUNCHING (CRITICAL): If the source states the company is actively rolling out, launching, releasing, or widely testing a new feature/product (e.g., "Soon Google Maps will...", "Beta is fully available", "Google announced a new feature"), THIS IS NOT A GAP. The company has ALREADY solved it internally. Do not mistake "launching a solution to the market" for "lacking a capability".
- The company is described as already having built or deployed a solution for this need.
- A competitor has a capability, but the source does not indicate the company lacks it.
- General industry trends or challenges not explicitly attributed to this specific company.
- VAGUE SIGNALS: Do not extract gaps purely based on vague social media reactions or public sentiment (e.g., "users think the price is high").

Signals that DO constitute valid gaps:
- INTERNAL VS EXTERNAL SOLUTION: If the company is partnering with an external startup or third-party to solve a problem because they lack the internal tech (e.g. "partnered with InstaDeep for pest detection"), THAT IS A VALID GAP. 
- Acknowledged internal research centers established explicitly because current capabilities are inadequate to meet new threats/needs.

========================
CROSS-CHUNK SYNTHESIS RULES
========================
Cross-chunk synthesis is encouraged for VALIDATING gaps — use all provided chunks
together to check whether the company already has a capability before declaring a gap.
If any chunk in the batch shows the company already possesses the capability in question,
you MUST STILL extract the gap, but document the counter-evidence in counter_evidence_chunks
with a rebuttal explaining why the gap still stands. Do NOT discard the gap.

However, when synthesizing across chunks, the SUBJECT of each factual claim must remain
the same as in the source chunk. Do not transfer attributes, states, or descriptions
from one named entity to another named entity.

Specifically: if chunk A describes Company X as being "in closed beta," you may NOT
use this to describe Company Y as being "in closed beta," even if both chunks appear
in the same batch.

Your gap assessment must be grounded PRIMARILY in the provided chunks, not in your
general knowledge of the company. If the provided chunks do not mention that the
company lacks a capability, do not assume it lacks one based on your prior knowledge
alone.

========================
CAPABILITY BASELINE CONTEXT RULES
========================
Some batches include a "CAPABILITY BASELINE CONTEXT" section drawn from the company's
10-K / product overview filings. These chunks describe capabilities the company
ALREADY HAS as of their filing date.

When this section is present:
- Use it ONLY for counter-evidence checking, not as a source of new gaps.
- GRANULARITY ALIGNMENT (CRITICAL): A baseline chunk can ONLY be used as counter-evidence if it explicitly addresses the EXACT SPECIFIC capability described in the gap.
- DO NOT OVER-EXTRAPOLATE: Generic, high-level statements in the 10-K (e.g., "We invest heavily in AI", "We offer advanced healthcare solutions") DO NOT invalidate a highly specific technical gap (e.g., "We lack AI out-of-distribution detection for medical imaging").
- The 10-K must explicitly prove the company already has a working solution for the *exact specific problem*. If it only proves they operate in that general domain, the gap STILL STANDS.
- 10-K language patterns that indicate the company ALREADY HAS a capability:
    "We offer X", "We provide X", "Our platform enables X", "We have built X",
    "We launched X", "We introduced X", "We developed X", "We operate X",
    "We have invested in X", "X is available", "X is now generally available",
    "customers can use X", "users can access X"
  When you see these patterns in baseline chunks for a capability similar to a gap
  you are about to extract, treat it as counter-evidence.
- If a gap you would otherwise extract is directly contradicted by a baseline chunk,
  you MUST retain it with a reduced gap_strength and a documented rebuttal in
  counter_evidence_chunks explaining why the gap still stands despite the baseline
  evidence (e.g., the baseline capability is partial, outdated, limited in scope,
  or addresses a different use case than the gap). Do NOT discard the gap.
- Baseline chunk_ids are valid and must be referenced by their exact chunk_id when
  cited in counter_evidence_chunks.
- Do NOT reference baseline chunk_ids in evidence_quotes — they may only appear in
  counter_evidence_chunks.

========================
CONTROLLED CAPABILITY VOCABULARY
========================
You MUST choose capability tags ONLY from this controlled vocabulary.
Do NOT invent new capability tags.
Do NOT paraphrase capability tags.
If the exact concept is not present, choose the closest existing tag(s).

Allowed capability tags:
{CAPABILITY_VOCAB_TEXT}

Examples of how to map concepts into the allowed vocabulary:
- domain-specific fine-tuning -> llm_fine_tuning
- multi-agent orchestration / coordination -> ai_agents
- hallucination mitigation / model validation -> ai_observability or model_governance
- enterprise deployment / scalable GenAI rollout -> genai_platform or application_deployment
- AI monitoring / explainability / governance -> ai_observability, model_governance
- fraud / AML / transaction monitoring -> fraud_detection, anti_money_laundering, audit_logging
- accessibility / remediation -> ai_accessibility_testing
- route planning / logistics optimization -> route_optimization, logistics_optimization
- carbon / climate tracking -> carbon_accounting

IMPORTANT:
- capability must contain 1-3 tags
- every capability tag must be from the allowed list above
- if none of the allowed tags are reasonably close, do not force a bad gap; prefer to
  skip that gap

========================
INDUSTRY TAG GUIDANCE
========================
For industry, prefer values from the following list when possible:
{INDUSTRY_HINTS_TEXT}

Use "GENERAL" if the gap is industry-agnostic.

========================
OUTPUT FORMAT
========================
Return a JSON array of gap objects. Each gap must include:

1) gap_title: a short display title for the gap, suitable for a UI card title.
   - Keep it concise: around 5-12 words
   - Use title-style phrasing, not a full paragraph
   - Make it sound like a product / market need headline
   - Prefer concise noun-phrase style
   - Avoid starting with the company name unless necessary
   - Example: "Compliance & audit logging for regulated AI deployments"

2) gap_text: 1-2 sentence natural language description of the gap. Be specific.
   - Every factual claim in gap_text MUST be directly traceable to a specific
     evidence quote you provide in evidence_quotes.
   - Do NOT introduce any information in gap_text that is absent from your
     evidence_quotes. If a claim cannot be supported by a verbatim quote,
     remove the claim.
   - The subject of each claim in gap_text must match the subject as it appears
     in the source chunk. Do not reassign statements made about one company or
     entity to a different company or entity.

3) company: the company name (copied from input)

4) evidence_quotes: array of 1-3 direct quotes from the input chunks that support
   this gap. Each quote must be copied verbatim from the chunk text. Include the
   chunk_id for traceability. The chunk_id MUST be one of the chunk_ids provided
   in the input. Do NOT use baseline chunk_ids as evidence_quotes — baseline chunks
   may only appear in counter_evidence_chunks.

5) capability: array of 1-3 capability tags selected ONLY from the controlled
   vocabulary above.

6) industry: array of 0-3 industry tags where this gap is most relevant.
   Use "GENERAL" if the gap is industry-agnostic.

7) gap_strength: integer 1-5, how clearly and strongly the evidence supports
   this gap existing. Only include gaps with gap_strength >= 3.
   Score definitions:
   - 5: The company itself explicitly acknowledges the need, deficiency, or gap
        in direct statements (e.g., executive quote, official blog post admission)
   - 4: Strong indirect evidence — e.g., a partnership or acquisition announcement
        where the stated purpose is to fill a named missing capability
   - 3: Moderate indirect evidence — e.g., heavy investment in a nascent area,
        or a product limitation noted by a credible third-party source
   - 2: Weak signal — do NOT include in output
   - 1: Speculation with minimal grounding — do NOT include in output

8) gap_strength_reason: a single sentence explaining why you assigned this
   gap_strength score, referencing the specific evidence that drove the rating.

9) source_types: array of source types from the supporting chunks.

10) geography: array of 0-3 standard geography tags indicating where this gap is
    geographically concentrated, based ONLY on explicit evidence in the chunks.
    You MUST ONLY use tags from this exact list:
    ["United States", "Europe", "India", "Israel", "Japan", "Other Regions", "Global"]
    Use "Global" if the gap has no specific geographic focus.

11) gap_evidence_chunks: array of chunk_ids whose content SUPPORTS the existence
    of this gap (i.e., shows the company needs or lacks this capability).
    Do NOT include baseline chunk_ids here.

12) counter_evidence_chunks: array of objects, one per chunk that contains
    evidence the company ALREADY HAS a related capability. Return an empty array
    if none exist in the batch.
    Each object must include:
    - chunk_id: the chunk containing the counter-evidence (may be a baseline chunk_id)
    - counter_evidence_text: the specific phrase or sentence from that chunk
      that suggests the company has this capability
    - rebuttal: a single sentence explaining why this counter-evidence was
      insufficient to discard the gap (e.g., the existing capability is partial,
      deprecated, limited in scope, or addresses a different use case)

13) third_party_subject_flag: boolean. Set to true if the primary subject of ANY
    of the supporting chunks is a competitor, partner, or third party rather than
    the target company itself. When true, gap_text must include only claims that
    are explicitly and directly about the target company, not about the third party.

14) partnership_signal_type: classify the nature of the gap signal.
    Choose exactly ONE of the following three values:

    - "not_done_yet": The target company itself is the grammatical subject and is
      admitting a product-level limitation or expressing a need they want to solve.
      Language patterns: "we haven't eliminated...", "our product has this issue...",
      "our customers are asking for...", "we are partnering with X to help OUR customers
      solve Y", "we are launching a research center because current capabilities are
      inadequate", "we are building X" (but haven't shipped it).

    - "not_building": The partner/third party is the grammatical subject. The target
      company is providing infrastructure or a platform, and the partner is the one
      building the application or solution on top.
      Language patterns: "[Partner] uses [Company]'s technology to build...",
      "[Partner] joined [Company]'s program to develop...", "[Partner] is building X
      on top of [Company]'s platform", "[Company]'s ecosystem partner [Partner] offers..."

    - "n/a": No external partnership or collaboration is involved. The gap is derived
      entirely from the target company's own product acknowledgments, research
      announcements, or capability limitations with no third-party involvement.

    NOTE: If the source chunk uses research-level language ("active area of research",
    "we are exploring", "promising field") rather than product-level language, lean
    toward "not_done_yet" only if the company is using first-person framing. If the
    framing is purely academic/research without first-person product commitment, prefer
    "n/a" and note it in gap_strength_reason.

15) gap_type: classify the strategic nature of the gap for BGV portfolio matching.
    Choose exactly ONE of the following three values:

    - "capability_deficit": The target company urgently needs this capability filled.
      The company is the one who benefits from an external solution, and there is
      pressure (customer demand, product limitation, executive admission) to address it.
      Acquisition, OEM, or deep integration is a plausible outcome.
      Typical indicators: partnership_signal_type is "not_done_yet", OR the company
      uses first-person product-limitation language, OR customers are explicitly
      requesting this capability.

    - "ecosystem_opportunity": The target company is deliberately not building this
      capability, leaving space for third parties in its ecosystem. The company
      benefits from MORE players building on its platform, not from acquiring them.
      Typical indicators: partnership_signal_type is "not_building", the company's
      role in the announcement is as infrastructure/platform provider, and the
      partner/third party is the one delivering end-user value.

    - "ambiguous": Evidence is insufficient to distinguish between the two types.
      Use this when the gap signal could reasonably fit either category, or when
      the source language does not provide enough context to make a confident call.
      Do NOT default to ambiguous — only use it when genuinely uncertain.

========================
RULES
========================
- Extract 3-8 gaps per batch. Prioritize quality over quantity.
- Each gap must be DISTINCT. Do not create overlapping gaps.
- Every gap must have at least 1 evidence quote from the input chunks.
- The chunk_id in each evidence quote MUST be from the gap signal chunks, not baseline.
- Only include gaps with gap_strength >= 3. If fewer than 3 gaps meet this threshold,
  return fewer. Do not pad with weak evidence.

- Do NOT fabricate gaps.
- Do NOT assert that the company lacks a capability unless the chunks explicitly
  support this. Do not rely on your general knowledge to infer missing capabilities.
- Before finalizing each gap, explicitly check: do any of the provided chunks
  (including baseline chunks) show the company already has a comparable capability
  or solution? If yes, either discard the gap or document the counter-evidence in
  counter_evidence_chunks with a rebuttal explaining why the gap still stands.
- If the source chunk is primarily about a competitor or third party, the gap must
  still be attributable to the target company based on explicit statements in the
  chunk. Set third_party_subject_flag to true in such cases.

- Return ONLY the JSON array. No markdown, no explanation, no code fences.


========================
FEW-SHOT EXAMPLE
========================
Input chunks (abbreviated):
- chunk_id: msft_earn_042, text: "...our Azure customers in financial services have been asking for deeper compliance tooling for AI model governance..."
- chunk_id: msft_ann_018, text: "...Microsoft announced a partnership with Datavisor to provide SOC 2 compliant audit trails for AI deployments..."

CAPABILITY BASELINE CONTEXT:
- chunk_id: msft_10k_007, text: "...Microsoft offers Azure Policy and Azure Blueprints for governance and compliance automation across cloud resources..."

Output:
[
  {{
    "gap_title": "Compliance & audit logging for regulated AI deployments",
    "gap_text": "Azure customers in financial services have explicitly requested deeper compliance tooling for AI model governance, and Microsoft has partnered with Datavisor to provide SOC 2 compliant audit trails for AI deployments, indicating a current gap in native compliance and audit infrastructure specifically for AI workloads.",
    "company": "Microsoft",
    "evidence_quotes": [
      {{
        "chunk_id": "msft_earn_042",
        "quote": "our Azure customers in financial services have been asking for deeper compliance tooling for AI model governance"
      }},
      {{
        "chunk_id": "msft_ann_018",
        "quote": "Microsoft announced a partnership with Datavisor to provide SOC 2 compliant audit trails for AI deployments"
      }}
    ],
    "capability": ["audit_logging", "model_governance"],
    "industry": ["financial_services"],
    "gap_strength": 4,
    "gap_strength_reason": "A third-party partnership was announced specifically to fill a named missing capability, and the need is corroborated by direct customer requests cited in an earnings transcript.",
    "source_types": ["earnings_transcript", "company_announcement"],
    "geography": ["GLOBAL"],
    "gap_evidence_chunks": ["msft_earn_042", "msft_ann_018"],
    "counter_evidence_chunks": [
      {{
        "chunk_id": "msft_10k_007",
        "counter_evidence_text": "Microsoft offers Azure Policy and Azure Blueprints for governance and compliance automation across cloud resources",
        "rebuttal": "Azure Policy and Blueprints cover general cloud governance, not AI-specific model audit trails or SOC 2 compliance for AI deployments, so the gap in AI-specific compliance tooling still stands."
      }}
    ],
    "third_party_subject_flag": false,
    "partnership_signal_type": "not_done_yet",
    "gap_type": "capability_deficit"
  }}
]
"""


# =========================
# User Prompt Template
# =========================
USER_PROMPT_TEMPLATE = """Company: {company}

Below are {chunk_count} labeled chunks from this company (batch {batch_num} of {total_batches}).
Each chunk has been pre-filtered for quality (strength >= 3, usable for ISS).

Identify the distinct capability gaps this company has, based on the evidence below.

{prev_gaps_section}

========================
CHUNKS
========================
{chunks_text}"""

PREV_GAPS_TEMPLATE = """========================
PREVIOUSLY IDENTIFIED GAPS (from earlier batches)
========================
Do NOT repeat any of these gaps. Only extract NEW, distinct gaps not covered below.

{prev_gaps_list}
========================
"""


# =========================
# Helpers
# =========================
def load_records(file_path: str) -> List[Dict[str, Any]]:
    if file_path.endswith(".jsonl"):
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    raise ValueError("Unsupported file format. Use .jsonl or .csv")


def save_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: Dict[str, Any], file_path: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_str(value: Any) -> str:
    if pd.isna(value) if isinstance(value, float) else False:
        return ""
    return "" if value is None else str(value)


def normalize_capability_tag(tag: str) -> str:
    return safe_str(tag).strip().lower().replace(" ", "_")


def normalize_industry_tag(tag: str) -> str:
    return safe_str(tag).strip().lower().replace(" ", "_")


def build_gap_title_fallback(gap_text: str, company: str = "") -> str:
    text = safe_str(gap_text).strip()
    if not text:
        return "Untitled Gap"
    if company:
        pattern = re.compile(rf"^{re.escape(company)}\s+", re.IGNORECASE)
        text = pattern.sub("", text)
    text = re.split(r"[.;:]", text)[0].strip()
    text = re.sub(r"[,\-–—\s]+$", "", text).strip()
    words = text.split()
    if len(words) > 10:
        text = " ".join(words[:10]).strip()
    if len(text) > 80:
        text = text[:80].rstrip()
    return text if text else "Untitled Gap"


def extract_json(text: str) -> Any:
    text = text.strip()
    fence_pattern = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)
    fence_match = fence_pattern.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    if text.endswith("```"):
        text = re.sub(r"\n?\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    result = _find_balanced_json(text, "[", "]")
    if result is not None:
        return result
    result = _find_balanced_json(text, "{", "}")
    if result is not None:
        return result
    raise ValueError(f"No valid JSON found in model output. First 200 chars: {text[:200]}")


def _find_balanced_json(text: str, open_char: str, close_char: str) -> Any:
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def filter_chunks(records: List[Dict[str, Any]], company: str) -> List[Dict[str, Any]]:
    filtered = []
    for r in records:
        if safe_str(r.get("company", "")).strip().lower() != company.lower():
            continue
        if REQUIRE_USABLE and not r.get("is_usable_for_iss", False):
            continue
        strength = r.get("strength_1_5", 0)
        try:
            strength = int(strength)
        except (ValueError, TypeError):
            strength = 0
        if strength < MIN_STRENGTH:
            continue
        filtered.append(r)
    return filtered


def load_tenk_chunks(company: str) -> List[Dict[str, Any]]:
    """
    Load 10-K baseline chunks for a company. These are used only for
    counter-evidence checking, not as gap signal sources.
    Returns empty list if no file is configured or file does not exist.
    """
    file_path = TENK_FILES.get(company)
    if not file_path:
        print(f"  No 10-K file configured for {company}, skipping baseline context.")
        return []
    if not Path(file_path).exists():
        print(f"  10-K file '{file_path}' not found for {company}, skipping baseline context.")
        return []
    records = load_records(file_path)
    # Mark each chunk so downstream logic knows its role
    for r in records:
        r["_is_baseline"] = True
        r["company"] = company
    print(f"  Loaded {len(records)} 10-K baseline chunks for {company}")
    return records


def format_chunks_for_prompt(
    chunks: List[Dict[str, Any]],
    baseline_chunks: List[Dict[str, Any]] = None,
) -> str:
    """
    Format gap signal chunks and (optionally) baseline chunks into the prompt string.
    Baseline chunks are appended in a clearly labelled separate section so the LLM
    knows not to extract gaps from them, only to use them for counter-evidence.
    """
    lines = []
    for c in chunks:
        chunk_id = safe_str(c.get("chunk_id", ""))
        source_type = safe_str(c.get("source_type", ""))
        theme = safe_str(c.get("theme", ""))
        event_type = safe_str(c.get("event_type", ""))
        strength = c.get("strength_1_5", "")
        date = safe_str(c.get("document_date", ""))
        text = safe_str(c.get("evidence_span", "")) or safe_str(c.get("chunk_text", ""))
        lines.append(
            f"[chunk_id: {chunk_id}] [source: {source_type}] [date: {date}] "
            f"[theme: {theme}] [event_type: {event_type}] [strength: {strength}]\n"
            f"{text}\n"
        )
    result = "\n---\n".join(lines)

    if baseline_chunks:
        # Cap baseline chunks to avoid context window issues
        capped = baseline_chunks[:MAX_TENK_CHUNKS_PER_BATCH]
        if len(capped) < len(baseline_chunks):
            print(
                f"  NOTE: 10-K baseline capped at {MAX_TENK_CHUNKS_PER_BATCH} chunks "
                f"(of {len(baseline_chunks)} total) to fit context window."
            )
        baseline_lines = []
        for c in capped:
            chunk_id = safe_str(c.get("chunk_id", ""))
            text = safe_str(c.get("chunk_text", "")) or safe_str(c.get("evidence_span", ""))
            baseline_lines.append(f"[chunk_id: {chunk_id}]\n{text}\n")
        baseline_block = "\n---\n".join(baseline_lines)
        result += (
            "\n\n========================\n"
            "CAPABILITY BASELINE CONTEXT (10-K / product overview)\n"
            "========================\n"
            "The following chunks describe capabilities this company ALREADY HAS. "
            "Use them ONLY for counter-evidence checking. Do NOT extract gaps from them. "
            "Do NOT use their chunk_ids in evidence_quotes. "
            "If a gap you would otherwise extract is directly contradicted by content here, "
            "discard that gap or document it in counter_evidence_chunks with a rebuttal.\n\n"
            + baseline_block
        )
    return result


def get_chunk_ids(chunks: List[Dict[str, Any]]) -> Set[str]:
    return {safe_str(c.get("chunk_id", "")) for c in chunks if safe_str(c.get("chunk_id", ""))}


def validate_gaps(
    gaps: List[Dict[str, Any]],
    company: str,
    valid_chunk_ids: Set[str],
    baseline_chunk_ids: Set[str] = None,
) -> List[Dict[str, Any]]:
    """
    Validate and normalize extracted gaps.
    - capability tags filtered to CAPABILITY_VOCAB only
    - gaps with no valid capability tags are dropped
    - evidence_quotes referencing baseline chunk_ids are dropped with a warning
      (baseline chunks must only appear in counter_evidence_chunks)
    - gap_title auto-generated if missing
    """
    if baseline_chunk_ids is None:
        baseline_chunk_ids = set()

    validated = []

    for gap in gaps:
        if not isinstance(gap, dict):
            continue

        # Validate partnership_signal_type
        _VALID_SIGNAL_TYPES = {"not_done_yet", "not_building", "n/a"}
        raw_signal = safe_str(gap.get("partnership_signal_type", "n/a")).strip().lower()
        partnership_signal_type = raw_signal if raw_signal in _VALID_SIGNAL_TYPES else "n/a"

        # Validate gap_type
        _VALID_GAP_TYPES = {"capability_deficit", "ecosystem_opportunity", "ambiguous"}
        raw_gap_type = safe_str(gap.get("gap_type", "ambiguous")).strip().lower()
        gap_type = raw_gap_type if raw_gap_type in _VALID_GAP_TYPES else "ambiguous"

        g = {
            "gap_title": safe_str(gap.get("gap_title", "")),
            "gap_text": safe_str(gap.get("gap_text", "")),
            "company": company,
            "evidence_quotes": [],
            "capability": [],
            "industry": [],
            "gap_strength": 1,
            "gap_strength_reason": safe_str(gap.get("gap_strength_reason", "")),
            "source_types": [],
            "geography": [],
            "gap_evidence_chunks": (
                gap.get("gap_evidence_chunks", [])
                if isinstance(gap.get("gap_evidence_chunks"), list)
                else []
            ),
            "counter_evidence_chunks": (
                gap.get("counter_evidence_chunks", [])
                if isinstance(gap.get("counter_evidence_chunks"), list)
                else []
            ),
            "third_party_subject_flag": bool(gap.get("third_party_subject_flag", False)),
            "partnership_signal_type": partnership_signal_type,
            "gap_type": gap_type,
            # Track whether any counter-evidence came from 10-K baseline
            "_baseline_counter_evidence": False,
        }

        if not g["gap_text"].strip():
            continue

        if not g["gap_title"].strip():
            g["gap_title"] = build_gap_title_fallback(g["gap_text"], company)

        # Validate evidence_quotes — reject any that reference baseline chunk_ids
        quotes = gap.get("evidence_quotes", [])
        if isinstance(quotes, list):
            for q in quotes:
                if isinstance(q, dict) and q.get("quote"):
                    cid = safe_str(q.get("chunk_id", ""))
                    if cid and cid in baseline_chunk_ids:
                        print(
                            f"    WARNING: evidence_quote references baseline chunk_id '{cid}'. "
                            f"Baseline chunks cannot be evidence sources. Dropping this quote."
                        )
                        continue
                    if cid and cid not in valid_chunk_ids:
                        print(f"    WARNING: chunk_id '{cid}' not in input batch, dropping evidence quote.")
                        continue
                    g["evidence_quotes"].append({
                        "chunk_id": cid,
                        "quote": safe_str(q.get("quote", "")),
                    })

        if not g["evidence_quotes"]:
            print(f"    WARNING: gap '{g['gap_text'][:60]}...' has no valid evidence quotes. Dropping.")
            continue

        # Flag if counter_evidence came from baseline
        for ce in g["counter_evidence_chunks"]:
            if isinstance(ce, dict) and ce.get("chunk_id") in baseline_chunk_ids:
                g["_baseline_counter_evidence"] = True
                break

        # Capability tags
        raw_caps = gap.get("capability", [])
        if isinstance(raw_caps, list):
            normalized_caps = []
            for c in raw_caps:
                cap = normalize_capability_tag(c)
                if cap in CAPABILITY_VOCAB:
                    normalized_caps.append(cap)
                else:
                    if cap:
                        print(f"    WARNING: invalid capability '{cap}' not in controlled vocab. Dropping tag.")
            g["capability"] = sorted(set(normalized_caps))[:3]

        if not g["capability"]:
            print(f"    WARNING: gap '{g['gap_text'][:60]}...' has no valid capability tags. Dropping.")
            continue

        # Industry tags
        inds = gap.get("industry", [])
        if isinstance(inds, list):
            normalized_inds = [normalize_industry_tag(i) for i in inds if normalize_industry_tag(i)]
            g["industry"] = sorted(set(normalized_inds))[:3]
        if not g["industry"]:
            g["industry"] = ["GENERAL"]

        # Geography
        geos = gap.get("geography", [])
        if isinstance(geos, list):
            g["geography"] = [safe_str(geo) for geo in geos if safe_str(geo)]
        if not g["geography"]:
            g["geography"] = ["GLOBAL"]

        # Gap strength
        try:
            strength = int(gap.get("gap_strength", 1))
            g["gap_strength"] = max(1, min(5, strength))
        except (ValueError, TypeError):
            g["gap_strength"] = 1

        # Source types
        sources = gap.get("source_types", [])
        if isinstance(sources, list):
            g["source_types"] = [safe_str(s) for s in sources if safe_str(s)]

        validated.append(g)

    return validated


def call_model(system_prompt: str, user_prompt: str) -> Any:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    print(f"stop_reason: {response.choices[0].finish_reason}")
    raw_text = response.choices[0].message.content
    print(f"output length: {len(raw_text) if raw_text else 0}")
    if raw_text:
        print(f"output tail: {raw_text[-200:]}")
    return extract_json(raw_text)


# =========================
# Embedding & Deduplication
# =========================
def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    if not texts:
        return None
    try:
        result = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return np.array([d.embedding for d in result.data])
    except Exception as e:
        print(f"  WARNING: embedding failed: {e}. Falling back to no dedup.")
        return None


def cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = vecs / norms
    return normed @ normed.T


def _pick_better_gap(gaps: List[Dict], i: int, j: int) -> Tuple[int, int]:
    gi, gj = gaps[i], gaps[j]
    if gi["gap_strength"] != gj["gap_strength"]:
        return (i, j) if gi["gap_strength"] > gj["gap_strength"] else (j, i)
    if len(gi["evidence_quotes"]) != len(gj["evidence_quotes"]):
        return (i, j) if len(gi["evidence_quotes"]) > len(gj["evidence_quotes"]) else (j, i)
    return (i, j)


def deduplicate_gaps(
    gaps: List[Dict[str, Any]],
    threshold: float = GAP_DEDUP_SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    if len(gaps) <= 1:
        return gaps
    gap_texts = [g["gap_text"] for g in gaps]
    vecs = embed_texts(gap_texts)
    if vecs is not None:
        sim_matrix = cosine_similarity_matrix(vecs)
        to_remove = set()
        for i in range(len(gaps)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(gaps)):
                if j in to_remove:
                    continue
                if sim_matrix[i, j] >= threshold:
                    keep, drop = _pick_better_gap(gaps, i, j)
                    to_remove.add(drop)
                    print(
                        f"    DEDUP: gap {drop} ('{gaps[drop]['gap_text'][:50]}...') "
                        f"similar to gap {keep} (sim={sim_matrix[i, j]:.3f}). Dropping {drop}."
                    )
        return [g for idx, g in enumerate(gaps) if idx not in to_remove]
    else:
        seen = set()
        deduped = []
        for g in gaps:
            key = g["gap_text"].strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(g)
        return deduped


# =========================
# Multi-Batch Processing
# =========================
def split_into_batches(
    chunks: List[Dict[str, Any]],
    batch_size: int = MAX_CHUNKS_PER_BATCH,
    max_batches: Optional[int] = MAX_BATCHES_PER_COMPANY,
) -> List[List[Dict[str, Any]]]:
    sorted_chunks = sorted(chunks, key=lambda x: int(x.get("strength_1_5", 0)), reverse=True)
    batches = []
    for start in range(0, len(sorted_chunks), batch_size):
        if max_batches is not None and len(batches) >= max_batches:
            remaining = len(sorted_chunks) - start
            print(f"  Reached max batch limit ({max_batches}). Skipping remaining {remaining} chunks.")
            break
        batches.append(sorted_chunks[start:start + batch_size])
    return batches


def extract_gaps_for_company(
    chunks: List[Dict[str, Any]],
    company: str,
    baseline_chunks: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if baseline_chunks is None:
        baseline_chunks = []

    baseline_chunk_ids = get_chunk_ids(baseline_chunks)

    batches = split_into_batches(chunks)
    total_batches = len(batches)
    print(f"  Split into {total_batches} batch(es) of up to {MAX_CHUNKS_PER_BATCH} chunks each.")
    if baseline_chunks:
        print(f"  Using {len(baseline_chunks)} 10-K baseline chunks for counter-evidence (capped at {MAX_TENK_CHUNKS_PER_BATCH} per batch).")

    all_gaps: List[Dict[str, Any]] = []

    for batch_idx, batch_chunks in enumerate(batches):
        batch_num = batch_idx + 1
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")

        valid_chunk_ids = get_chunk_ids(batch_chunks)
        # Baseline chunk_ids are valid references for counter_evidence_chunks
        all_valid_ids = valid_chunk_ids | baseline_chunk_ids

        chunks_text = format_chunks_for_prompt(batch_chunks, baseline_chunks)

        prev_gaps_section = ""
        if all_gaps:
            # Cap to the most recent PREV_GAPS_CAP gaps to avoid blowing context window
            capped_prev = all_gaps[-PREV_GAPS_CAP:]
            prev_gaps_list = "\n".join(f"  - {g['gap_text']}" for g in capped_prev)
            prev_gaps_section = PREV_GAPS_TEMPLATE.format(prev_gaps_list=prev_gaps_list)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            company=company,
            chunk_count=len(batch_chunks),
            batch_num=batch_num,
            total_batches=total_batches,
            prev_gaps_section=prev_gaps_section,
            chunks_text=chunks_text,
        )

        last_error = None
        batch_gaps = []

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_output = call_model(SYSTEM_PROMPT, user_prompt)
                if isinstance(raw_output, dict):
                    raw_output = [raw_output]
                if not isinstance(raw_output, list):
                    raise ValueError(f"Expected list, got {type(raw_output)}")
                batch_gaps = validate_gaps(
                    raw_output, company, all_valid_ids, baseline_chunk_ids
                )
                break
            except Exception as e:
                last_error = str(e)
                print(f"    Attempt {attempt} failed: {last_error}")
                time.sleep(2.0 * attempt)
        else:
            error_record = {
                "company": company,
                "batch_num": batch_num,
                "chunk_count": len(batch_chunks),
                "_extraction_error": last_error,
            }
            append_jsonl(error_record, ERROR_FILE)

        # Attach evidence_dates: look up document_date for each evidence chunk
        # so downstream freshness checks can detect stale evidence.
        chunk_date_index = {
            safe_str(c.get("chunk_id", "")): safe_str(c.get("document_date", ""))
            for c in batch_chunks
            if safe_str(c.get("chunk_id", ""))
        }
        for gap in batch_gaps:
            dates = [
                chunk_date_index[cid]
                for cid in gap.get("gap_evidence_chunks", [])
                if cid in chunk_date_index and chunk_date_index[cid]
            ]
            gap["evidence_dates"] = dates

        # Log how many gaps had baseline counter-evidence
        baseline_ce_count = sum(1 for g in batch_gaps if g.get("_baseline_counter_evidence"))
        print(
            f"    Extracted {len(batch_gaps)} gaps from batch {batch_num} "
            f"({baseline_ce_count} with 10-K counter-evidence)"
        )

        all_gaps.extend(batch_gaps)
        time.sleep(SLEEP_BETWEEN_CALLS)

    pre_dedup_count = len(all_gaps)
    all_gaps = deduplicate_gaps(all_gaps)
    post_dedup_count = len(all_gaps)

    if pre_dedup_count != post_dedup_count:
        print(f"  Deduplication: {pre_dedup_count} → {post_dedup_count} gaps")

    return all_gaps


# =========================
# Main
# =========================
def load_done_companies(output_file: str) -> set:
    """Return the set of companies already written to the output file (for resume)."""
    done = set()
    path = Path(output_file)
    if not path.exists():
        return done
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    c = r.get("company", "")
                    if c:
                        done.add(c)
                except Exception:
                    pass
    return done


def main():
    records = load_records(INPUT_FILE)
    print(f"Loaded {len(records)} labeled chunks from {INPUT_FILE}")

    # Resume: skip companies already written to output
    done_companies = load_done_companies(OUTPUT_FILE)
    if done_companies:
        print(f"Resuming: already done companies = {sorted(done_companies)}")

    total_gaps_written = sum(
        1 for _ in open(OUTPUT_FILE, "r", encoding="utf-8")
    ) if Path(OUTPUT_FILE).exists() else 0

    for company in COMPANIES:
        if company in done_companies:
            print(f"\n{'='*50}")
            print(f"Company: {company} — already done, skipping.")
            continue

        chunks = filter_chunks(records, company)
        print(f"\n{'='*50}")
        print(f"Company: {company}")
        print(f"Qualified chunks (strength >= {MIN_STRENGTH}, usable): {len(chunks)}")

        if not chunks:
            print("  No qualifying chunks. Skipping.")
            continue

        # Load 10-K baseline if available for this company
        baseline_chunks = load_tenk_chunks(company)

        print("  Extracting gaps...")
        gaps = extract_gaps_for_company(chunks, company, baseline_chunks)
        print(f"  Final gap count for {company}: {len(gaps)}")

        for i, gap in enumerate(gaps, 1):
            baseline_flag = " [10K-CE]" if gap.get("_baseline_counter_evidence") else ""
            print(f"    Gap {i}: {gap['gap_title']}{baseline_flag} | {gap['gap_text'][:80]}...")
            print(
                f"           strength={gap['gap_strength']}, "
                f"capabilities={gap['capability']}, industries={gap['industry']}"
            )

        # Write immediately after each company (incremental, append mode)
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for gap in gaps:
                f.write(json.dumps(gap, ensure_ascii=False) + "\n")
        total_gaps_written += len(gaps)
        print(f"  Appended {len(gaps)} gaps to {OUTPUT_FILE} (total so far: {total_gaps_written})")

    print(f"\n{'='*50}")
    print(f"Done. Total gaps written: {total_gaps_written}")
    print(f"Output: {OUTPUT_FILE}")

    # Summary from output file
    all_gaps = load_records(OUTPUT_FILE)
    for company in COMPANIES:
        company_gaps = [g for g in all_gaps if g.get("company") == company]
        baseline_ce = sum(1 for g in company_gaps if g.get("_baseline_counter_evidence"))
        print(f"  {company}: {len(company_gaps)} gaps ({baseline_ce} with 10-K counter-evidence)")


if __name__ == "__main__":
    main()