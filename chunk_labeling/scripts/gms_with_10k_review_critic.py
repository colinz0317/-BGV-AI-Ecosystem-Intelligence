import os
import json
import time
from pathlib import Path
from openai import OpenAI

from paths import (
    BASE_DIR,
    FILE_GAPS_EXTRACTED,
    FILE_GAPS_CRITIC_FILTERED,
    FILE_GAPS_CRITIC_REJECTED,
    FILE_GAPS_ECOSYSTEM_QUARANTINE
)

# Auto-load .env
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

client = OpenAI()
MODEL_NAME = "gpt-4o-mini"

INPUT_FILE = str(FILE_GAPS_EXTRACTED)
OUTPUT_ACCEPTED_FILE = str(FILE_GAPS_CRITIC_FILTERED)
OUTPUT_REJECTED_FILE = str(FILE_GAPS_CRITIC_REJECTED)
OUTPUT_ECOSYSTEM_QUARANTINE_FILE = str(FILE_GAPS_ECOSYSTEM_QUARANTINE)


# =============================================================================
# Prompt: Capability Deficit Critic
# Verifies that the gap reflects a genuine internal need the target company
# wants to fill — not just ecosystem activity built on their platform.
# =============================================================================
CAPABILITY_CRITIC_SYSTEM = """You are a strict, objective Verification AI.
Your job is to read the provided Evidence Quotes and classify whether they
confirm a genuine CAPABILITY DEFICIT for the target company.

You must be completely literal. Do not infer anything that is not explicitly
stated in the quotes.

A CAPABILITY DEFICIT means the target company itself needs a capability
but does not currently have it in production, and there is pressure
(from customers, product limitations, or explicit executive admission)
to address it.

Categories:
- TRUE: The quotes confirm a genuine capability deficit. Accept if ANY of:
  1. The company explicitly acknowledges a technical deficiency, limitation,
     or high error rate in their existing products (first-person language).
  2. The company is partnering with an external party to help THEIR OWN
     CUSTOMERS solve a problem the company cannot solve internally
     (subject is the target company, purpose is to serve their customers).
  3. The company is launching a research center or explicitly describes a
     field as "an active area of research" to tackle an unsolved internal
     challenge.
  4. The company's customers are explicitly quoted as requesting a capability
     that the company's current products do not provide.

- FALSE: The quotes show the company is already actively rolling out,
  launching, releasing, or testing a product/feature that directly addresses
  this gap. It is an existing capability or active rollout, NOT a deficit.

- UNCLEAR: The quotes discuss a broad industry trend, third-party competitor,
  general policy, vague social sentiment, or a third party using the target
  company's infrastructure — without clear proof the target company itself
  lacks the capability.

Output JSON exactly in this format:
{
  "classification": "TRUE" | "FALSE" | "UNCLEAR",
  "reason": "Brief explanation based strictly on the quotes."
}
"""


# =============================================================================
# Prompt: Ecosystem Opportunity Critic
# Verifies that the gap reflects a genuine ecosystem opportunity — a third
# party building a specific application on the target company's platform,
# filling an application-layer space the target company deliberately leaves
# open.
# =============================================================================
ECOSYSTEM_CRITIC_SYSTEM = """You are a strict, objective Verification AI.
Your job is to read the provided Evidence Quotes and classify whether they
confirm a genuine ECOSYSTEM OPPORTUNITY gap for the target company.

An ECOSYSTEM OPPORTUNITY means:
- The target company provides infrastructure or a platform.
- A third-party partner is explicitly building a specific application or
  capability on top of that platform, in a space the target company does
  NOT itself occupy.
- The target company benefits from more third parties building on their
  platform, and has no product of its own in this specific application space.

You must be completely literal. Do not infer anything not in the quotes.

Categories:
- TRUE: The quotes confirm a genuine ecosystem opportunity. Accept ONLY IF:
  1. A third-party (partner/startup) is the grammatical subject and is
     explicitly described as using the target company's specific technology,
     platform, or infrastructure to build a product or capability.
  2. The application or capability the third party is building is NOT
     something the target company itself offers as a standalone product
     in the quotes.
  Both conditions must be present. If only one is present, classify UNCLEAR.

- FALSE: REJECT if either of the following is true:
  1. The target company itself also offers a product or feature that
     directly competes with or duplicates what the third party is building
     (the gap does not actually exist at the application layer).
  2. The third party is only mentioned as a customer or data provider,
     not as a builder using the platform's technology.

- UNCLEAR: The connection between the third party and the target company's
  platform is vague, indirect, or not explicitly stated in the quotes.

Additionally, output an ecosystem_confidence score:
- "high": Both conditions are met with explicit, unambiguous language.
- "medium": Both conditions are met but the language is somewhat indirect.
- "low": The evidence supports the classification but leaves meaningful
  room for doubt.

Output JSON exactly in this format:
{
  "classification": "TRUE" | "FALSE" | "UNCLEAR",
  "ecosystem_confidence": "high" | "medium" | "low",
  "reason": "Brief explanation based strictly on the quotes."
}
"""


def build_quotes_text(quotes):
    lines = ""
    for i, q in enumerate(quotes):
        lines += f"Quote {i+1}:\n{q.get('quote', '')}\n---\n"
    return lines


def call_capability_critic(company, quotes):
    """Run the capability deficit critic. Returns dict with classification and reason."""
    quotes_text = build_quotes_text(quotes)
    user_prompt = f"Target Company: {company}\n\nEvidence Quotes:\n{quotes_text}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CAPABILITY_CRITIC_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    result = json.loads(response.choices[0].message.content)
    return {
        "classification": result.get("classification", "UNCLEAR"),
        "reason": result.get("reason", ""),
        "ecosystem_confidence": None,
    }


def call_ecosystem_critic(company, quotes):
    """Run the ecosystem opportunity critic. Returns dict with classification, confidence, reason."""
    quotes_text = build_quotes_text(quotes)
    user_prompt = f"Target Company: {company}\n\nEvidence Quotes:\n{quotes_text}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ECOSYSTEM_CRITIC_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    result = json.loads(response.choices[0].message.content)
    return {
        "classification": result.get("classification", "UNCLEAR"),
        "reason": result.get("reason", ""),
        "ecosystem_confidence": result.get("ecosystem_confidence", "low"),
    }


def load_records(filepath):
    records = []
    if not Path(filepath).exists():
        return records
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_done_titles(*filepaths):
    """Return set of gap_titles already written across output files (for resume)."""
    done = set()
    for fp in filepaths:
        for r in load_records(fp):
            t = r.get("gap_title", "")
            if t:
                done.add(t)
    return done


def main():
    gaps = load_records(INPUT_FILE)
    print(f"Loaded {len(gaps)} gaps for Critic review.")

    # Resume: skip gaps already written to any output file
    done_titles = load_done_titles(
        OUTPUT_ACCEPTED_FILE, OUTPUT_REJECTED_FILE, OUTPUT_ECOSYSTEM_QUARANTINE_FILE
    )
    if done_titles:
        print(f"Resuming: {len(done_titles)} gaps already processed, skipping.")
    print()

    accepted_count = len(load_records(OUTPUT_ACCEPTED_FILE))
    rejected_count = len(load_records(OUTPUT_REJECTED_FILE))
    quarantine_count = len(load_records(OUTPUT_ECOSYSTEM_QUARANTINE_FILE))

    # Open output files in append mode for incremental writing
    f_accepted = open(OUTPUT_ACCEPTED_FILE, "a", encoding="utf-8")
    f_rejected = open(OUTPUT_REJECTED_FILE, "a", encoding="utf-8")
    f_quarantine = open(OUTPUT_ECOSYSTEM_QUARANTINE_FILE, "a", encoding="utf-8")

    for i, gap in enumerate(gaps):
        comp = gap.get("company", "Unknown")
        title = gap.get("gap_title", "?")
        quotes = gap.get("evidence_quotes", [])
        gap_type = gap.get("gap_type", "ambiguous")
        partnership_signal = gap.get("partnership_signal_type", "n/a")

        # Resume: skip already-processed gaps
        if title in done_titles:
            print(f"[{i+1}/{len(gaps)}] SKIP (already done): {comp} | {title}")
            continue

        print(f"[{i+1}/{len(gaps)}] {comp} | {title}")
        print(f"  gap_type={gap_type}  partnership_signal={partnership_signal}")

        # ---------------------------------------------------------------
        # Guard: ecosystem_opportunity requires partnership_signal=not_building
        # If the signal doesn't match, downgrade to ambiguous for stricter review
        # ---------------------------------------------------------------
        effective_type = gap_type
        if gap_type == "ecosystem_opportunity" and partnership_signal != "not_building":
            effective_type = "ambiguous"
            print(f"  ⚠️  gap_type=ecosystem_opportunity but signal={partnership_signal} → treating as ambiguous")

        try:
            if effective_type == "capability_deficit":
                result = call_capability_critic(comp, quotes)
                critic_path = "capability"

            elif effective_type == "ecosystem_opportunity":
                result = call_ecosystem_critic(comp, quotes)
                critic_path = "ecosystem"

            else:
                # ambiguous: run both critics, take the stricter (capability) result
                # but also check ecosystem so we can preserve the signal if capability rejects
                cap_result = call_capability_critic(comp, quotes)
                time.sleep(0.3)
                eco_result = call_ecosystem_critic(comp, quotes)

                if cap_result["classification"] == "TRUE":
                    # capability critic is satisfied — use that result, stronger signal
                    result = cap_result
                    critic_path = "capability (ambiguous→capability accepted)"
                elif eco_result["classification"] == "TRUE":
                    # capability didn't accept but ecosystem did — use ecosystem result
                    result = eco_result
                    critic_path = "ecosystem (ambiguous→ecosystem accepted)"
                    # Downgrade confidence for ambiguous-origin ecosystem gaps
                    if result["ecosystem_confidence"] == "high":
                        result["ecosystem_confidence"] = "medium"
                else:
                    # Both rejected
                    result = cap_result  # use capability reason for logging
                    critic_path = "both_rejected"

            # ---------------------------------------------------------------
            # Write result fields back onto the gap
            # ---------------------------------------------------------------
            gap["critic_classification"] = result["classification"]
            gap["critic_reason"] = result["reason"]
            gap["critic_path"] = critic_path
            if result["ecosystem_confidence"] is not None:
                gap["ecosystem_confidence"] = result["ecosystem_confidence"]

            # ---------------------------------------------------------------
            # Routing decision — write immediately
            # ---------------------------------------------------------------
            if result["classification"] == "TRUE":
                if (
                    critic_path.startswith("ecosystem")
                    and result.get("ecosystem_confidence") == "low"
                ):
                    print(f"  🔶 QUARANTINE (ecosystem, low confidence): {result['reason']}")
                    f_quarantine.write(json.dumps(gap, ensure_ascii=False) + "\n")
                    f_quarantine.flush()
                    quarantine_count += 1
                else:
                    conf_tag = f" [eco_confidence={result['ecosystem_confidence']}]" if result.get("ecosystem_confidence") else ""
                    print(f"  ✅ ACCEPTED ({critic_path}){conf_tag}: {result['reason']}")
                    f_accepted.write(json.dumps(gap, ensure_ascii=False) + "\n")
                    f_accepted.flush()
                    accepted_count += 1
            else:
                print(f"  ❌ REJECTED ({result['classification']}, {critic_path}): {result['reason']}")
                f_rejected.write(json.dumps(gap, ensure_ascii=False) + "\n")
                f_rejected.flush()
                rejected_count += 1

            time.sleep(0.5)

        except Exception as e:
            print(f"  ⚠️  Error reviewing gap: {e}")

    f_accepted.close()
    f_rejected.close()
    f_quarantine.close()

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Review complete.")
    print(f"  ACCEPTED         : {accepted_count}")
    print(f"  REJECTED/UNCLEAR : {rejected_count}")
    print(f"  QUARANTINE (eco) : {quarantine_count}")
    print(f"\nOutputs:")
    print(f"  Accepted    → {OUTPUT_ACCEPTED_FILE}")
    print(f"  Rejected    → {OUTPUT_REJECTED_FILE}")
    print(f"  Quarantine  → {OUTPUT_ECOSYSTEM_QUARANTINE_FILE}")

    # Summary by company and gap_type (read from output files)
    all_gaps = load_records(INPUT_FILE)
    accepted_gaps = load_records(OUTPUT_ACCEPTED_FILE)
    rejected_gaps = load_records(OUTPUT_REJECTED_FILE)
    quarantine_gaps = load_records(OUTPUT_ECOSYSTEM_QUARANTINE_FILE)

    print("\nBreakdown by company:")
    companies = sorted({g.get("company", "?") for g in all_gaps})
    for co in companies:
        co_acc = [g for g in accepted_gaps if g.get("company") == co]
        co_rej = [g for g in rejected_gaps if g.get("company") == co]
        co_qua = [g for g in quarantine_gaps if g.get("company") == co]
        print(f"  {co}: accepted={len(co_acc)}  rejected={len(co_rej)}  quarantine={len(co_qua)}")

    print("\nBreakdown by gap_type (accepted only):")
    for gt in ("capability_deficit", "ecosystem_opportunity", "ambiguous"):
        count = sum(1 for g in accepted_gaps if g.get("gap_type") == gt)
        print(f"  {gt}: {count}")


if __name__ == "__main__":
    main()
