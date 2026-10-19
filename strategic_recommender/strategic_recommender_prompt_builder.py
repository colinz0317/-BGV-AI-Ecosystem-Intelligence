#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Dict, List, Any


def format_chunk_for_prompt(chunk: Dict[str, Any], idx: int) -> str:
    text = str(chunk.get("chunk_text", ""))
    company = str(chunk.get("company", "Unknown"))
    theme = str(chunk.get("theme", "Unknown"))
    sub_theme = str(chunk.get("sub_theme", "Unknown"))

    source_type = str(chunk.get("source_type", "Unknown"))
    source_name = str(chunk.get("source_name", "Unknown"))
    source_title = str(chunk.get("source_title", "Unknown"))
    source_url = str(chunk.get("source_url", "Unknown"))
    document_date = str(chunk.get("document_date", "Unknown"))

    event_type = str(chunk.get("event_type", "Unknown"))
    strength = str(chunk.get("strength_1_5", "Unknown"))
    confidence = str(chunk.get("confidence", "Unknown"))
    evidence_span = str(chunk.get("evidence_span", ""))
    reason_short = str(chunk.get("reason_short", ""))

    return (
        f"[Evidence {idx}]\n"
        f"Company: {company}\n"
        f"Theme: {theme}\n"
        f"Sub-theme: {sub_theme}\n"
        f"Event Type: {event_type}\n"
        f"Strength: {strength}\n"
        f"Confidence: {confidence}\n"
        f"Source Type: {source_type}\n"
        f"Source Name: {source_name}\n"
        f"Source Title: {source_title}\n"
        f"Document Date: {document_date}\n"
        f"URL: {source_url}\n"
        f"Evidence Span: {evidence_span}\n"
        f"Label Reason: {reason_short}\n"
        f"Chunk Text: {text}\n"
    )

def build_recommendation_prompt(
    portfolio: Dict[str, Any],
    gap: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    portfolio_name = portfolio.get("company_name", gap.get("candidate_company", "Unknown Portfolio Company"))
    portfolio_desc = portfolio.get("description", "")
    portfolio_caps = portfolio.get("capability", portfolio.get("capabilities", []))
    portfolio_industry = portfolio.get("industry", [])
    portfolio_geo = portfolio.get("geography", portfolio.get("geography_all", []))

    target_company = gap.get("target_company", gap.get("company", "Unknown Target Company"))
    gap_title = gap.get("gap_title", "")
    gap_text = gap.get("gap_text", "")
    gap_capability = gap.get("capability", [])
    gap_industry = gap.get("industry", [])
    gap_geo = gap.get("geography", [])

    final_score_raw = gap.get("final_score_raw", "")
    display_score = gap.get("display_score", "")
    gms_reason = gap.get("gms_reason", "")

    evidence_text = "\n\n".join(
        format_chunk_for_prompt(chunk, idx + 1)
        for idx, chunk in enumerate(retrieved_chunks)
    )

    prompt = f"""
You are generating a strategic recommendation for BGV's AI ecosystem intelligence platform.

The system already produced a Gap Matching Score (GMS) between one target company gap and one BGV portfolio company.
Your job is NOT to recalculate the algorithm score. Your job is to synthesize the GMS result and supporting evidence into a clear investor-facing recommendation.

Portfolio Company:
- Name: {portfolio_name}
- Description: {portfolio_desc}
- Capabilities: {portfolio_caps}
- Industry: {portfolio_industry}
- Geography: {portfolio_geo}

Target Company Gap:
- Target Company: {target_company}
- Gap Title: {gap_title}
- Gap Description: {gap_text}
- Required / Matched Capabilities: {gap_capability}
- Gap Industry: {gap_industry}
- Gap Geography: {gap_geo}

Existing GMS Result:
- Raw Score: {final_score_raw}
- Display Score: {display_score}
- Algorithm Reason: {gms_reason}

Supporting Evidence Chunks:
{evidence_text}

Please produce:
1. Strategic Recommendation Summary: 2-4 sentences.
2. Recommendation Strength: Strong / Moderate / Weak.
3. Key Evidence: 3 bullets using the GMS result and supporting chunks.
4. Strategic Rationale: why this match matters from an investor or partnership perspective.
5. Risks / Missing Evidence: mention weak evidence, missing capability, industry mismatch, or geography issue.

Important rules:
- Use only the provided information.
- Do not invent facts.
- Do not claim a partnership exists unless directly supported.
- If evidence is weak, say so clearly.
- Be concise and specific.
- FORMAT AS PLAIN TEXT ONLY. DO NOT USE MARKDOWN (e.g. ### or **).
""".strip()

    return prompt


def preview_prompt(prompt: str, max_chars: int = 2500) -> None:
    print("\n" + "=" * 80)
    print("PROMPT PREVIEW")
    print("=" * 80)
    print(prompt[:max_chars])
    if len(prompt) > max_chars:
        print("\n...[truncated]")
    print("=" * 80 + "\n")

