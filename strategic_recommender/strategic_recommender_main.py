#!/usr/bin/env python
# coding: utf-8

import os

from strategic_recommender_data_retrieval import (
    load_chunks,
    load_portfolio,
    load_gaps,
    retrieve_chunks,
    find_first_portfolio,
    select_top_gms_results,
)

from strategic_recommender_prompt_builder import (
    build_recommendation_prompt,
    preview_prompt,
)


CHUNKS_FILE = "chunks_labeled_output_sample_10000.jsonl"
PORTFOLIO_FILE = "bgv_portfolio.json"
GMS_RESULTS_FILE = "gms_results.jsonl"

RETRIEVAL_MODE = "local"

# Top-k evidence chunks per gap
TOP_K = 5

# Top-N GMS gap matches to process
# Set to 3 if you want to match the live report exactly.
# Set to 5 if the current product goal is top-5 gaps.
TOP_GAPS = 5

# Optional: set to a portfolio company name like "Workhelix"
# If None, it will select the top GMS rows across all candidate companies.
SELECTED_PORTFOLIO_COMPANY = None

OPENAI_MODEL = "gpt-4.1-mini"


def call_openai_chat(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai package: pip install openai")

    client = OpenAI(api_key=api_key, timeout=30.0)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise strategic recommendation assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def convert_gms_to_gap(gms_row: dict) -> dict:
    tech_coverage = gms_row.get("tech_coverage") or {}
    industry_fit = gms_row.get("industry_fit") or {}
    geography_fit = gms_row.get("geography_fit") or {}

    return {
        "gap_id": gms_row.get("gap_id", ""),
        "gap_title": gms_row.get("gap_title", ""),
        "gap_text": gms_row.get("gap_text", ""),
        "company": gms_row.get("target_company", ""),
        "target_company": gms_row.get("target_company", ""),
        "candidate_company": gms_row.get("candidate_company", ""),
        "capability": tech_coverage.get("matched_capabilities", []),
        "industry": industry_fit.get("gap_industries_normalized", []),
        "geography": geography_fit.get("gap_geography_normalized", []),
        "final_score_raw": gms_row.get("final_score_raw", ""),
        "display_score": gms_row.get("display_score", ""),
        "gms_reason": gms_row.get("reason", "")
    }


def find_matching_portfolio(portfolios, candidate_name):
    if not candidate_name:
        return find_first_portfolio(portfolios)

    for p in portfolios:
        name = p.get("company_name", "")
        if name.lower().strip() == candidate_name.lower().strip():
            return p

    print(f"Warning: No exact portfolio match found for {candidate_name}. Using first portfolio record.")
    return find_first_portfolio(portfolios)


def print_retrieved_chunks(retrieved):
    print(f"\nRetrieved top {len(retrieved)} chunks")

    for i, chunk in enumerate(retrieved, start=1):
        print(f"\n--- Retrieved Chunk {i} ---")
        print(f"Retrieval Score: {chunk.get('retrieval_score')}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")
        print(f"Company: {chunk.get('company')}")
        print(f"Theme: {chunk.get('theme')}")
        print(f"Event Type: {chunk.get('event_type')}")
        print(f"Strength: {chunk.get('strength_1_5')}")
        print(f"Source: {chunk.get('source_name')}")
        print(f"Title: {chunk.get('source_title')}")
        print(f"Date: {chunk.get('document_date')}")
        text = chunk.get("chunk_text", "")
        print(f"Text Preview: {text[:300]}")


def main():
    chunks = load_chunks(CHUNKS_FILE)
    portfolios = load_portfolio(PORTFOLIO_FILE)
    gms_results = load_gaps(GMS_RESULTS_FILE)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loaded {len(portfolios)} portfolio records")
    print(f"Loaded {len(gms_results)} GMS result records")

    top_gms_rows = select_top_gms_results(
        gms_results,
        candidate_company=SELECTED_PORTFOLIO_COMPANY,
        top_n=TOP_GAPS
    )

    print(f"\nSelected top {len(top_gms_rows)} GMS gap matches")
    print(f"TOP_GAPS = {TOP_GAPS}")
    print(f"TOP_K = {TOP_K}")

    use_api = True
    all_outputs = []

    for gap_index, gms_row in enumerate(top_gms_rows, start=1):
        gap = convert_gms_to_gap(gms_row)

        candidate_name = gap.get("candidate_company", "")
        portfolio = find_matching_portfolio(portfolios, candidate_name)

        print("\n" + "=" * 80)
        print(f"Processing Gap Match {gap_index}/{len(top_gms_rows)}")
        print("=" * 80)
        print(f"Candidate Portfolio Company: {candidate_name}")
        print(f"Target Company: {gap.get('target_company')}")
        print(f"Gap Title: {gap.get('gap_title')}")
        print(f"Display Score: {gap.get('display_score')}")
        print(f"Raw Score: {gap.get('final_score_raw')}")

        retrieved = retrieve_chunks(
            chunks=chunks,
            portfolio=portfolio,
            gap=gap,
            mode=RETRIEVAL_MODE,
            top_k=TOP_K
        )

        print_retrieved_chunks(retrieved)

        prompt = build_recommendation_prompt(portfolio, gap, retrieved)
        preview_prompt(prompt)

        if use_api:
            print("Calling LLM...")
            output = call_openai_chat(prompt)

            print("\n" + "=" * 80)
            print(f"LLM OUTPUT FOR GAP MATCH {gap_index}")
            print("=" * 80)
            print(output)
            print("=" * 80)

            all_outputs.append({
                "gap_rank": gap_index,
                "candidate_company": candidate_name,
                "target_company": gap.get("target_company"),
                "gap_title": gap.get("gap_title"),
                "display_score": gap.get("display_score"),
                "final_score_raw": gap.get("final_score_raw"),
                "retrieved_chunk_count": len(retrieved),
                "top_retrieval_scores": [c.get("retrieval_score") for c in retrieved],
                "llm_output": output,
            })
        else:
            print("API call skipped. Set use_api = True when ready.")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed {len(top_gms_rows)} top GMS gap matches")
    print(f"Top-k retrieval setting: {TOP_K}")
    print(f"Generated {len(all_outputs)} LLM recommendation outputs")


if __name__ == "__main__":
    main()
