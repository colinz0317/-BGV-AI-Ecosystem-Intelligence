"""
BGV CVC Data Cleaner v2 (Optimized)
=====================================
Improvements over v1:
  1. Expanded event_type keywords (fixes 376 "Other" misclassifications)
  2. Noise filter removes stock analysis / opinion pieces (~300 records)
  3. Better category classification (reduces "General Technology" from 65%)
  4. Target company extraction from Google News title patterns
  5. Improved signal_strength heuristic

Usage:
    python cvc_cleaner_v2.py

Input:  cvc_raw_data/cvc_raw_combined.csv
Output: cvc_cleaned.csv
"""

import pandas as pd
import re
import os
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

COMPANY_STANDARD = {
    "nvidia": "NVIDIA", "nvidia corporation": "NVIDIA",
    "microsoft": "Microsoft", "microsoft corporation": "Microsoft",
    "google": "Google", "alphabet": "Google", "alphabet inc": "Google",
    "google llc": "Google", "google cloud": "Google",
    "openai": "OpenAI", "open ai": "OpenAI",
}

VALID_COMPANIES = ["NVIDIA", "Microsoft", "Google", "OpenAI"]
MIN_DESCRIPTION_LENGTH = 30

# ============================================================
# EXPANDED EVENT TYPE CLASSIFICATION
# ============================================================

# Each pattern: (compiled_regex, event_type)
# Using regex for word-boundary matching to avoid partial matches
EVENT_TYPE_PATTERNS = [
    # === Strategic Investment ===
    (re.compile(r'\b(invests?|investing|invested|investments?|investors?)\b', re.I), "Strategic Investment"),
    (re.compile(r'\b(back(s|ed|ing)?)\b.*\b(startup|startups|company|firm|venture)\b', re.I), "Strategic Investment"),
    (re.compile(r'\b-backed\b', re.I), "Strategic Investment"),
    (re.compile(r'\b(bets?|betting|wager)\b.*\b(on|big|billion|million|\$|startup|AI)', re.I), "Strategic Investment"),
    (re.compile(r'\b(pour(s|ed|ing)?|pump(s|ed|ing)?)\b.*\b(billion|million|\$|money|capital)', re.I), "Strategic Investment"),
    (re.compile(r'\b(commit(s|ted|ting)?|pledge(s|d)?)\b.*\b(billion|million|\$)', re.I), "Strategic Investment"),
    (re.compile(r'\b(venture\s+(capital|fund|arm|investment)|VC\s+arm)\b', re.I), "Strategic Investment"),
    (re.compile(r'\b(lead(s|ing)?)\s+(a\s+)?(round|investment|funding)', re.I), "Strategic Investment"),
    (re.compile(r'\b(corporate\s+venture|CVC|strategic\s+investor)\b', re.I), "Strategic Investment"),
    (re.compile(r'\b(stake|equity)\b.*\b(startup|company)', re.I), "Strategic Investment"),
    (re.compile(r'\bCapitalG\b|\bGV\b|\bM12\b|\bNVentures\b', re.I), "Strategic Investment"),
    (re.compile(r'\b(double(s|d)?\s+down)\b', re.I), "Strategic Investment"),
    (re.compile(r'\b(fund(s|ed|ing)?)\b.*\b(startup|startups|AI|round|company)\b', re.I), "Strategic Investment"),
    (re.compile(r'\b(startup|startups)\b.*\b(fund|capital|investor|backed)\b', re.I), "Strategic Investment"),

    # === Acquisition ===
    (re.compile(r'\b(acquir(es?|ed|ing)|acquisition)\b', re.I), "Acquisition"),
    (re.compile(r'\b(buys?|buying|bought|purchase[sd]?)\b', re.I), "Acquisition"),
    (re.compile(r'\b(takeover|take\s+over|merger|merge[sd]?)\b', re.I), "Acquisition"),
    (re.compile(r'\b(deal|bid)\b.*\b(billion|million|\$)', re.I), "Acquisition"),

    # === Strategic Partnership ===
    (re.compile(r'\b(partner(s|ship|ing|ed)?)\b', re.I), "Strategic Partnership"),
    (re.compile(r'\b(collaborat(e|es|ion|ing)|alliance|joint\s+venture)\b', re.I), "Strategic Partnership"),
    (re.compile(r'\b(team(s|ed|ing)?\s+up)\b', re.I), "Strategic Partnership"),
    (re.compile(r'\b(join(s|ed)?\s+forces)\b', re.I), "Strategic Partnership"),
    (re.compile(r'\b(deepen(s|ed)?\s+(alliance|ties|relationship))\b', re.I), "Strategic Partnership"),
    (re.compile(r'\b(expand(s|ed|ing)?\s+(partnership|collaboration))\b', re.I), "Strategic Partnership"),

    # === Funding Round ===
    (re.compile(r'\b(series\s+[A-F]|seed\s+(round|funding))\b', re.I), "Funding Round"),
    (re.compile(r'\b(funding\s+round|raises?|raised)\b', re.I), "Funding Round"),
    (re.compile(r'\b(valuation)\b.*\b(billion|million|\$)', re.I), "Funding Round"),
    (re.compile(r'\bIPO\b', re.I), "Funding Round"),

    # === Product/Platform Launch ===
    (re.compile(r'\b(launch(es|ed|ing)?|unveil(s|ed)?|introduc(es?|ed|ing))\b', re.I), "Product/Platform Launch"),
    (re.compile(r'\b(rolls?\s+out|ship(s|ped)?|release[sd]?)\b', re.I), "Product/Platform Launch"),
    (re.compile(r'\b(open\s*-?\s*source[sd]?|new\s+product|new\s+platform)\b', re.I), "Product/Platform Launch"),
]


def classify_event_type(text):
    """Classify event type using regex patterns with priority ordering."""
    if not isinstance(text, str):
        return "Other"

    # Score each event type
    scores = {}
    for pattern, event_type in EVENT_TYPE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            scores[event_type] = scores.get(event_type, 0) + len(matches)

    if scores:
        return max(scores, key=scores.get)
    return "Other"


# ============================================================
# EXPANDED CATEGORY CLASSIFICATION
# ============================================================

CATEGORY_KEYWORDS = {
    "AI Infrastructure": [
        "data center", "gpu", "chip", "compute", "infrastructure",
        "hardware", "semiconductor", "hpc", "accelerator", "server",
        "networking", "optics", "photonics", "nuclear", "energy",
        "power grid", "cooling", "rack", "cluster", "supercomputer",
        "data centre", "colocation", "neocloud",
    ],
    "Cloud Computing": [
        "cloud", "azure", "gcp", "aws", "saas", "iaas", "paas",
        "serverless", "multi-cloud", "hybrid cloud", "cloud native",
    ],
    "AI/ML Platform": [
        "machine learning", "deep learning", "llm", "foundation model",
        "generative ai", "ai platform", "mlops", "ai model",
        "large language", "chatgpt", "copilot", "gpt", "claude",
        "gemini", "ai agent", "agentic", "fine-tun", "inference",
        "training", "neural network", "transformer", "diffusion",
        "synthetic data", "ai safety", "alignment",
    ],
    "Cybersecurity": [
        "security", "cybersecurity", "threat", "zero trust",
        "encryption", "identity", "fraud", "firewall", "soc",
        "vulnerability", "malware", "ransomware", "crowdstrike",
    ],
    "Enterprise Software": [
        "enterprise", "erp", "crm", "workflow", "automation",
        "productivity", "office", "no-code", "low-code", "saas",
        "business application", "salesforce",
    ],
    "Healthcare AI": [
        "health", "biotech", "pharma", "medical", "clinical",
        "drug discovery", "genomic", "patient", "hospital",
        "diagnostic", "therapeutic", "life science",
    ],
    "Autonomous/Robotics": [
        "autonomous", "self-driving", "robotics", "drone", "robot",
        "humanoid", "lidar", "waymo", "wayve", "adas",
        "autonomous vehicle", "av ", "trucking",
    ],
    "Developer Tools": [
        "developer", "devops", "api", "sdk", "code", "github",
        "open source", "ide", "compiler", "vibe coding",
        "programming", "software development",
    ],
    "Data & Analytics": [
        "data", "analytics", "database", "warehouse", "lakehouse",
        "etl", "bi tool", "databricks", "snowflake", "observability",
    ],
    "Fintech": [
        "fintech", "banking", "payment", "finance", "trading",
        "insurtech", "blockchain", "crypto", "defi",
    ],
    "AI Startup Ecosystem": [
        "startup", "startups", "unicorn", "accelerator",
        "incubator", "venture", "founder", "entrepreneur",
    ],
    "Telecom & Connectivity": [
        "telecom", "5g", "6g", "satellite", "connectivity",
        "network", "spectrum", "spacemobile",
    ],
    "Climate & Energy Tech": [
        "climate", "clean energy", "solar", "wind", "battery",
        "carbon", "sustainability", "renewable", "fusion",
        "nuclear reactor", "modular reactor",
    ],
}


def classify_category(text):
    """Classify category with expanded keywords."""
    if not isinstance(text, str):
        return "General Technology"

    text_lower = text.lower()
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text_lower)
        if score > 0:
            scores[cat] = score

    if scores:
        # If "AI Startup Ecosystem" is the only match, try harder
        top = max(scores, key=scores.get)
        if top == "AI Startup Ecosystem" and len(scores) > 1:
            del scores["AI Startup Ecosystem"]
            return max(scores, key=scores.get)
        return top

    return "General Technology"


# ============================================================
# TARGET COMPANY EXTRACTION (from title)
# ============================================================

def extract_target_company(title, description, source_company):
    """
    Extract the target company (being invested in / acquired) from title.
    Google News titles follow patterns like:
      "X invests in Y"
      "X acquires Y"
      "X-backed Y"
      "Y raises $X from Z"
    """
    if not isinstance(title, str):
        return "Unknown"

    full_text = f"{title} {description}" if isinstance(description, str) else title

    # Pattern 1: "Company invests/backs/acquires [in] TargetName"
    patterns_after_verb = [
        rf'(?:NVIDIA|Nvidia|Microsoft|Google|Alphabet|OpenAI)\s+(?:to\s+)?(?:invests?\s+in|acquires?|backs?|leads?\s+(?:round\s+(?:in|for)|investment\s+in))\s+([A-Z][A-Za-z0-9\.\-\s]+?)(?:\s*[-,;|]|\s+at\s+|\s+for\s+|\s+as\s+|\s+with\s+|$)',
    ]

    # Pattern 2: "Company-backed TargetName"
    patterns_backed = [
        rf'(?:NVIDIA|Nvidia|Microsoft|Google|Alphabet|OpenAI)[- ]backed\s+([A-Z][A-Za-z0-9\.\-\s]+?)(?:\s*[-,;|]|\s+hits?\s+|\s+raises?\s+|\s+valued?\s+|\s+eyes?\s+|\s+files?\s+|$)',
    ]

    # Pattern 3: "TargetName raises $X from Company"
    patterns_raises = [
        r'([A-Z][A-Za-z0-9\.\-]+)\s+raises?\s+\$[\d\.]+\s*[BMK]',
        r'([A-Z][A-Za-z0-9\.\-]+)\s+(?:valued|hits?|reaches?)\s+.*\$[\d\.]+\s*[BMKb]',
    ]

    # Pattern 4: "invest in TargetName" (generic)
    patterns_invest_in = [
        r'invest(?:s|ing|ed)?\s+(?:in|into)\s+([A-Z][A-Za-z0-9\.\-\s]+?)(?:\s*[-,;|]|\s+at\s+|\s+as\s+|$)',
        r'acquir(?:e|es|ed|ing)\s+([A-Z][A-Za-z0-9\.\-\s]+?)(?:\s*[-,;|]|\s+for\s+|\s+to\s+|$)',
        r'backs?\s+([A-Z][A-Za-z0-9\.\-\s]+?)(?:\s*[-,;|]|\s+at\s+|\s+as\s+|$)',
    ]

    all_patterns = patterns_after_verb + patterns_backed + patterns_raises + patterns_invest_in

    for pattern in all_patterns:
        match = re.search(pattern, full_text)
        if match:
            target = match.group(1).strip()
            # Clean up
            target = re.sub(r'\s+', ' ', target).strip()
            # Remove trailing common words
            target = re.sub(r'\s+(is|are|was|were|has|had|in|at|to|for|the|a|an|and|or|its|with|from|by)$', '', target, flags=re.I)

            # Validate: not the source company, reasonable length
            if (target.lower() not in [c.lower() for c in VALID_COMPANIES]
                and target.lower() not in ['alphabet', 'unknown', 'the', 'a']
                and 2 < len(target) < 60
                and not target.startswith(('How', 'Why', 'What', 'When', 'Where', 'The '))):
                return target

    return "Unknown"


# ============================================================
# NOISE FILTER
# ============================================================

NOISE_PATTERNS = [
    # Stock analysis / market commentary
    re.compile(r'\b(stock\s+(analysis|forecast|prediction|price|target|rally|drop|surge))\b', re.I),
    re.compile(r'\b(NASDAQ|NYSE|BATS)\s*:', re.I),
    re.compile(r'\bshares\s+(fall|drop|rise|surge|slip|jump|tumble)\b', re.I),
    re.compile(r'\b(bull|bear)\s+case\b', re.I),
    re.compile(r'\b(buy|sell|hold)\s+rating\b', re.I),
    re.compile(r'\bbest\s+(stocks?|etfs?)\s+to\s+buy\b', re.I),

    # How-to / retail investor guides
    re.compile(r'^how\s+to\s+(invest|buy|get)\b', re.I),
    re.compile(r'\bwant\s+to\s+invest\s+in\b.*\b(before|IPO)\b', re.I),
    re.compile(r'\bshould\s+(investors?|you)\b', re.I),
    re.compile(r'\bretail\s+investors?\b', re.I),
    re.compile(r'\bhow\s+to\s+be\s+a\b', re.I),

    # Crunchbase-style listing pages (not real CVC events)
    re.compile(r'Funding Rounds & List of Investors', re.I),
    re.compile(r'List of \d+ (Acquisitions|Investments)', re.I),
    re.compile(r'Cap Table Breakdown', re.I),

    # Windows startup sound noise
    re.compile(r'\b(Windows|Vista)\b.*\bstartup\s+sound\b', re.I),
    re.compile(r'\bstartup\s+sound\b.*\b(Windows|Vista)\b', re.I),

    # "push back" / "rolls back" / non-investment "back" phrases
    re.compile(r'\b(push(es|ed)?\s+back|rolls?\s+back|fires?\s+back|fights?\s+back|brings?\s+back|bounce\s+back|pull(s|ed)?\s+back|set\s+back|cut(s)?\s+back|scale[sd]?\s+back)\b', re.I),
]

# Keywords that rescue a record from noise filtering
# (if these appear, it's likely a real CVC event even if noise patterns match)
RESCUE_KEYWORDS = re.compile(
    r'\b(acquires?|acquisition|funding\s+round|series\s+[A-F]|raises?\s+\$|'
    r'leads?\s+round|strategic\s+investment|venture\s+capital)\b', re.I
)


def is_noise(title, description=""):
    """Check if a record is noise (stock analysis, opinions, not real CVC)."""
    full_text = f"{title} {description}"

    # First check rescue keywords - if present, keep regardless
    if RESCUE_KEYWORDS.search(full_text):
        return False

    # Check noise patterns
    for pattern in NOISE_PATTERNS:
        if pattern.search(title):  # Only check title for noise (more precise)
            return True

    return False


# ============================================================
# SIGNAL STRENGTH HEURISTIC
# ============================================================

def assign_signal_strength(row):
    """
    Improved signal strength (1-5) based on multiple factors.
    """
    score = 0
    title = str(row.get('title', ''))
    desc = str(row.get('description', ''))
    full = f"{title} {desc}"

    # Factor 1: Has dollar amount (+2)
    if row.get('amount', 'Undisclosed') != 'Undisclosed':
        score += 2

    # Factor 2: Has target company (+1)
    if row.get('target_company', 'Unknown') != 'Unknown':
        score += 1

    # Factor 3: Clear event type (+1)
    if row.get('event_type', 'Other') not in ('Other', ''):
        score += 1

    # Factor 4: Specific category (+1)
    if row.get('category', 'General Technology') != 'General Technology':
        score += 1

    # Factor 5: Description richness
    if len(desc) > 200:
        score += 1

    # Factor 6: Contains specific evidence words
    evidence_words = re.compile(r'\b(confirmed|announced|closed|signed|completed|finalized)\b', re.I)
    if evidence_words.search(full):
        score += 1

    # Clamp to 1-5
    return max(1, min(score, 5))


# ============================================================
# MAIN PIPELINE
# ============================================================

def clean_cvc_data(input_path, output_path):
    print("\n" + "=" * 60)
    print("  BGV CVC DATA CLEANER v2 (Optimized)")
    print("=" * 60)

    # --- Load ---
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"Raw records: {initial_count}")

    # --- Step 1: Standardize company names ---
    print("\n[Step 1] Standardizing company names...")
    def std_company(name):
        if not isinstance(name, str): return "Unknown"
        return COMPANY_STANDARD.get(name.strip().lower(), name.strip())
    df['company'] = df['company'].apply(std_company)
    df = df[df['company'].isin(VALID_COMPANIES)]
    print(f"  Valid companies: {len(df)}")

    # --- Step 2: Clean text ---
    print("[Step 2] Cleaning text fields...")
    df['title'] = df['title'].fillna("").astype(str).apply(
        lambda t: re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', t)).strip()
    )
    df['description'] = df['description'].fillna("").astype(str).apply(
        lambda t: re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', t)).strip()
    )
    df = df[df['description'].str.len() >= MIN_DESCRIPTION_LENGTH]
    print(f"  After text cleanup: {len(df)}")

    # --- Step 3: Validate dates ---
    print("[Step 3] Validating dates...")
    def validate_date(d):
        try:
            dt = datetime.strptime(str(d).strip(), "%Y-%m-%d")
            if dt > datetime.now() or dt.year < 2020:
                return None
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return None
    df['date'] = df['date'].apply(validate_date)
    df = df.dropna(subset=['date'])
    print(f"  Valid dates: {len(df)}")

    # --- Step 4: Noise filter ---
    print("[Step 4] Filtering noise...")
    noise_mask = df.apply(lambda r: is_noise(r['title'], r['description']), axis=1)
    noise_count = noise_mask.sum()
    df = df[~noise_mask]
    print(f"  Removed {noise_count} noise records → {len(df)} remaining")

    # --- Step 5: Re-classify event types ---
    print("[Step 5] Re-classifying event types...")
    old_other = (df['event_type'] == 'Other').sum()
    df['event_type'] = df.apply(
        lambda r: classify_event_type(f"{r['title']} {r['description']}"), axis=1
    )
    new_other = (df['event_type'] == 'Other').sum()
    print(f"  'Other' reduced: {old_other} → {new_other}")

    # --- Step 6: Re-classify categories ---
    print("[Step 6] Re-classifying categories...")
    old_gen = (df['category'] == 'General Technology').sum()
    df['category'] = df.apply(
        lambda r: classify_category(f"{r['title']} {r['description']}"), axis=1
    )
    new_gen = (df['category'] == 'General Technology').sum()
    print(f"  'General Technology' reduced: {old_gen} → {new_gen}")

    # --- Step 7: Extract target companies ---
    print("[Step 7] Extracting target companies...")
    df['target_company'] = df.apply(
        lambda r: extract_target_company(r['title'], r['description'], r['company']), axis=1
    )
    known = (df['target_company'] != 'Unknown').sum()
    print(f"  Target companies identified: {known}/{len(df)}")

    # --- Step 8: Dedup ---
    print("[Step 8] Deduplicating...")
    before = len(df)
    df = df.drop_duplicates(subset=['source_url'], keep='first')
    df['_key'] = df['company'] + '|' + df['title'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['_key'], keep='first')
    df = df.drop(columns=['_key'])
    print(f"  {before} → {len(df)}")

    # --- Step 9: Signal strength ---
    print("[Step 9] Assigning signal strength...")
    df['signal_strength'] = df.apply(assign_signal_strength, axis=1)

    # --- Step 10: Sort & ID ---
    print("[Step 10] Finalizing...")
    df = df.sort_values(['company', 'date'], ascending=[True, False]).reset_index(drop=True)
    df['chunk_id'] = [f"cvc_{i+1:04d}" for i in range(len(df))]

    # Final columns
    cols = [
        'chunk_id', 'date', 'company', 'event_type', 'category',
        'target_company', 'amount', 'title', 'description',
        'signal_strength', 'source_url', 'source_type',
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df_out = df[cols]

    # --- Save ---
    df_out.to_csv(output_path, index=False)

    # --- Report ---
    print(f"\n{'=' * 60}")
    print("  CLEANING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Input:  {initial_count} raw records")
    print(f"  Output: {len(df_out)} clean records")
    print(f"  Removed: {initial_count - len(df_out)} (noise + invalid + dupes)")

    print(f"\n  By Company:")
    for c in VALID_COMPANIES:
        print(f"    {c}: {len(df_out[df_out['company'] == c])}")

    print(f"\n  By Event Type:")
    for et, cnt in df_out['event_type'].value_counts().items():
        print(f"    {et}: {cnt}")

    print(f"\n  By Category (top 10):")
    for cat, cnt in df_out['category'].value_counts().head(10).items():
        print(f"    {cat}: {cnt}")

    print(f"\n  Signal Strength:")
    for s in sorted(df_out['signal_strength'].unique()):
        print(f"    Strength {s}: {len(df_out[df_out['signal_strength'] == s])}")

    print(f"\n  Target Company: {(df_out['target_company'] != 'Unknown').sum()} identified / {len(df_out)} total")
    print(f"  Amount known: {(df_out['amount'] != 'Undisclosed').sum()} / {len(df_out)}")
    print(f"  Date range: {df_out['date'].min()} → {df_out['date'].max()}")

    print(f"\n  Saved to: {output_path}")
    print(f"  Next: python row_to_sentence_v2.py\n")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try both possible input locations
    input_path = os.path.join(current_dir, "cvc_raw_data", "cvc_raw_combined.csv")
    if not os.path.exists(input_path):
        input_path = os.path.join(current_dir, "cvc_raw_combined.csv")

    output_path = os.path.join(current_dir, "cvc_cleaned.csv")

    if os.path.exists(input_path):
        clean_cvc_data(input_path, output_path)
    else:
        print(f"Error: Raw data not found.")
        print(f"Tried: {input_path}")
        print("Run 'python cvc_scraper_v3.py' first.")
