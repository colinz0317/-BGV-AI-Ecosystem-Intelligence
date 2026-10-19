"""
BGV Portfolio Company Tagging Script
=====================================

Purpose
-------
Transforms raw BGV portfolio company descriptions into the structured
entity-level format required by the GMS (Gap Match Score) pipeline

Output schema (per company):
{
    "company_id":    str,     # Unique identifier, e.g. "c001"
    "company_name":  str,     # Company name (verbatim)
    "description":   str,     # Verbatim description for semantic embedding
                              # (traceable - no paraphrasing)
    "capability":    [str],   # Controlled-vocab capability tags (snake_case)
    "industry":      [str],   # Controlled-vocab industry tags (IND_XXX)
    "geography_all": [str],   # ALL operating markets (country/region names),
                              # ordered by first mention in the source text
}

Design decisions / caveats
--------------------------

1. GEOGRAPHY is a list of all operating markets (country/region names) that
   appear in the source text, ordered by first mention. Cities are
   mapped to country names (e.g. United States, Israel) via CITY_TO_COUNTRY.
   Regional aggregates like "Europe" and "Latin America" are preserved.

2. DESCRIPTION is preserved VERBATIM from the source document. Do not
   paraphrase - traceability is required for downstream evidence
   citation.

3. Tagging approach: rule-based (deterministic) here. Each company was
   manually tagged by reading the description. For scale-up, an
   LLM-based tagging pass with few-shot examples would replace the manual dicts below, but for 31 companies
   manual tagging is more accurate and auditable.

Raw source
----------
The 31 company descriptions below are copied verbatim from the BGV
portfolio document provided by the user. See COMPANY_DESCRIPTIONS dict.
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# CONTROLLED VOCABULARIES (provisional - pending BGV confirmation)
# ---------------------------------------------------------------------------

# Capability taxonomy. snake_case. Extends examples in Interim Report 4.3.1.
# Each tag represents a distinct technical capability a company can deliver.
CAPABILITY_VOCAB = {
    # AI / ML / data
    "ai_crm", "ai_observability", "ai_agents", "llm_fine_tuning",
    "llm_routing", "ml_in_database", "rag_tooling", "model_governance",
    "ai_strategy_analytics", "finops_ai", "genai_platform",
    "predictive_analytics", "computer_vision", "ai_accessibility_testing",
    # Security / identity / fraud
    "identity_risk_intelligence", "passwordless_auth", "client_side_security",
    "fraud_detection", "anti_money_laundering", "audit_logging",
    # Data / infrastructure
    "customer_data_platform", "cloud_cost_optimization", "multi_cloud_monitoring",
    "application_deployment", "paas", "mobile_app_deployment",
    "tech_intelligence",
    # Vertical / operational
    "regulatory_automation", "construction_management", "drone_delivery",
    "drone_inspection", "satellite_analytics", "recycling_automation",
    "food_robotics", "logistics_optimization", "route_optimization",
    "carbon_accounting",
    # GTM / revenue / CX
    "customer_success", "revenue_intelligence", "buyer_experience_platform",
    "marketing_automation",
    # Digital transformation services
    "digital_transformation_services",
    # E-commerce
    "ecommerce_platform",
}

# Industry taxonomy. IND_XXX. Extends examples in Interim Report 4.3.1.
# Note: this namespace mixes two kinds of tags:
#   (a) true verticals (IND_HEALTHCARE, IND_FINANCIAL_SERVICES, ...)
#   (b) horizontal functions (IND_SECURITY, IND_AI_INFRA, ...) for
#       companies that sell cross-industry tooling.
# Design decision: keep both in one field (not a separate
# horizontal_function field) so the GMS industry-match formula
# (|Ind_g cap Ind_p|) in Interim Report 4.3.2 does not have to change.
# IND_GENERAL is retained only as a last-resort fallback.
INDUSTRY_VOCAB = {
    # True verticals
    "IND_HEALTHCARE", "IND_MANUFACTURING", "IND_ENERGY_OIL_GAS",
    "IND_AUTOMOTIVE", "IND_INTERNET_PLATFORM",
    "IND_FINANCIAL_SERVICES", "IND_INSURANCE", "IND_CONSTRUCTION",
    "IND_LOGISTICS", "IND_RETAIL", "IND_FOOD_SERVICE",
    "IND_MEDIA_DIGITAL", "IND_UTILITIES", "IND_WASTE_MANAGEMENT",
    "IND_MEDICAL_DEVICES", "IND_INDUSTRIAL_INSPECTION",
    "IND_TELECOMMUNICATIONS", "IND_PUBLIC_SECTOR",
    "IND_SUSTAINABILITY", "IND_ECOMMERCE",
    # Horizontal functions (for cross-industry tooling)
    "IND_SECURITY",         # cybersecurity, identity, fraud protection
    "IND_CLOUD_INFRA",      # cloud/app deployment, hybrid-cloud ops
    "IND_AI_INFRA",         # AI/ML platform, observability, FinOps, LLM ops
    "IND_SALES_MARKETING",  # GTM / revenue / CX / customer data
    "IND_DEV_TOOLS",        # developer productivity, testing, accessibility
    # Last-resort fallback
    "IND_GENERAL",
}

# City -> Country name mapping for geography normalization.
CITY_TO_COUNTRY = {
    # US cities
    "san francisco": "United States", "los altos": "United States", "santa clara": "United States",
    "fremont": "United States", "los angeles": "United States", "palo alto": "United States",
    "san mateo": "United States", "redwood city": "United States", "menlo park": "United States",
    "bellevue": "United States", "santa barbara": "United States", "san ramon": "United States",
    "boulder": "United States", "new york": "United States", "new haven": "United States",
    "wilmington": "United States", "austin": "United States", "florida": "United States", "texas": "United States",
    "north carolina": "United States", "massachusetts": "United States",
    # Israel
    "tel aviv": "Israel", "israel": "Israel",
    # France
    "paris": "France", "france": "France",
    # UK
    "london": "United Kingdom", "uk": "United Kingdom",
    # Spain
    "madrid": "Spain", "spain": "Spain",
    # India
    "bengaluru": "India", "chennai": "India", "india": "India",
    # Ukraine
    "kiev": "Ukraine", "ukraine": "Ukraine",
    # Poland
    "warsaw": "Poland", "poland": "Poland",
    # Mexico
    "mexico": "Mexico",
    # Generic
    "us": "United States", "europe": "Europe", "latin america": "Latin America",
}


# ---------------------------------------------------------------------------
# RAW COMPANY DATA (verbatim from source document)
# ---------------------------------------------------------------------------

COMPANY_DESCRIPTIONS = [
    {
        "name": "Covu",
        "description": "Covu is a tech-enabled platform for insurance agencies that combines AI-powered CRM, marketing tools, and operational support to help agencies scale efficiently. The platform enables agencies to improve customer retention, drive new client acquisition, and streamline operations through automation and data-driven insights, while maintaining their independence.",
        "geography_raw": "San Francisco (US), Europe",
    },
    {
        "name": "Onymos",
        "description": "Onymos provides an enterprise-grade application development platform that enables companies to rapidly build, deploy, and manage web, mobile, and IoT applications. Its modular architecture and reusable components significantly accelerate development cycles, particularly for complex, regulated industries such as healthcare, while ensuring data security and compliance.",
        "geography_raw": "Los Altos (US), India",
    },
    {
        "name": "Blendid",
        "description": "Blendid is a robotics and AI company that automates food preparation through fully autonomous kiosks. Its platform integrates robotics, computer vision, and IoT to deliver consistent, scalable, and cost-efficient food service operations, reducing labor dependency while improving customer experience.",
        "geography_raw": "Santa Clara (US)",
    },
    {
        "name": "Outbuild",
        "description": "Outbuild is a construction management SaaS platform that optimizes project planning, scheduling, and execution. By combining real-time collaboration, analytics, and integrations with legacy tools, it enables contractors to improve productivity, reduce delays, and make data-driven decisions across construction workflows.",
        "geography_raw": "US, Latin America",
    },
    {
        "name": "Flytrex",
        "description": "Flytrex operates an autonomous drone delivery platform focused on last-mile logistics, particularly in suburban environments. Its end-to-end solution combines FAA-approved drones, cloud-based control systems, and operational software to enable faster, lower-cost, and more scalable delivery services for retailers and food chains.",
        "geography_raw": "North Carolina (US), Tel Aviv (Israel)",
    },
    {
        "name": "ThetaRay",
        "description": "ThetaRay provides an AI-powered transaction monitoring and anti-money laundering platform for banks and fintechs. Its solution leverages advanced machine learning to detect both known and unknown financial crime patterns in real time, improving compliance, reducing false positives, and enhancing risk management.",
        "geography_raw": "Israel, London, Spain, New York",
    },
    {
        "name": "Greenly",
        "description": "Greenly is a climate tech SaaS platform that enables companies to measure, manage, and reduce their carbon footprint. By automating data collection and providing actionable insights aligned with global standards, it helps organizations drive sustainability strategies and engage stakeholders across their value chain.",
        "geography_raw": "Paris (France), US, UK",
    },
    {
        "name": "Constella",
        "description": "Constella is an AI-driven identity risk intelligence platform that helps organizations detect and mitigate identity-based threats. Leveraging a large breach data lake and advanced analytics, it provides real-time alerts, threat intelligence, and protection against fraud, social engineering, and credential compromise.",
        "geography_raw": "Los Altos (US), Madrid (Spain)",
    },
    {
        "name": "Kardinal",
        "description": "Kardinal is an AI-powered logistics optimization platform that enables companies to improve route planning and delivery operations in real time. Using advanced mathematical modeling and machine learning, it helps logistics providers increase efficiency, reduce costs, and adapt dynamically to operational disruptions.",
        "geography_raw": "Paris (France)",
    },
    {
        "name": "AiDash",
        "description": "AiDash is a satellite analytics platform that leverages AI and geospatial data to optimize operations for utilities and infrastructure companies. Its solutions enable predictive maintenance, risk detection, and sustainability management, helping organizations improve resilience and reduce operational costs.",
        "geography_raw": "Santa Clara (US), India",
    },
    {
        "name": "EverestLabs",
        "description": "EverestLabs develops AI and robotics solutions for recycling facilities through its RecycleOS platform. By automating sorting processes and providing real-time analytics, it increases material recovery rates, reduces operational costs, and improves sustainability outcomes for waste management operators.",
        "geography_raw": "Fremont (US), Chennai (India)",
    },
    {
        "name": "Essenvia",
        "description": "Essenvia is a regulatory management platform for medical device companies that automates and streamlines pre- and post-market approval processes. Its software reduces time-to-market, improves compliance, and enhances collaboration across regulatory and product teams.",
        "geography_raw": "Los Angeles (US), Bengaluru (India)",
    },
    {
        "name": "Percepto",
        "description": "Percepto delivers autonomous drone and AI-powered inspection solutions for industrial sites. Its platform enables continuous monitoring, automated data collection, and real-time analytics, helping organizations improve safety, reduce manual inspections, and optimize asset management.",
        "geography_raw": "Israel, Austin (US), Florida (US), Texas (US)",
    },
    {
        "name": "Virtana",
        "description": "Virtana is an AI-powered platform for monitoring, optimizing, and managing hybrid and multi-cloud environments. It enables enterprises to improve performance, control costs, and make data-driven infrastructure decisions across complex IT ecosystems.",
        "geography_raw": "Palo Alto (US), London (UK), Warsaw (Poland)",
    },
    {
        "name": "MindsDB",
        "description": "MindsDB is a platform that enables developers to integrate machine learning directly into databases using SQL. By abstracting AI models into queryable structures, it simplifies the deployment of AI applications such as forecasting, chatbots, and predictive analytics.",
        "geography_raw": "US (remote-first)",
    },
    {
        "name": "Blue Cedar",
        "description": "Blue Cedar provides a mobile application deployment automation platform that streamlines secure app delivery for enterprises. It orchestrates security, compliance, and release workflows, significantly reducing development time and operational complexity.",
        "geography_raw": "San Francisco (US), Massachusetts (US)",
    },
    {
        "name": "Datawizz",
        "description": "Datawizz is an AI platform that enables enterprises to build and optimize specialized language models by capturing and leveraging their own LLM interaction data. It improves performance and reduces costs by intelligently routing and fine-tuning models based on use case requirements.",
        "geography_raw": "US, Tel Aviv (Israel)",
    },
    {
        "name": "Secret Double Octopus",
        "description": "Secret Double Octopus provides passwordless authentication solutions that enhance security and user experience across enterprise environments. Its platform eliminates reliance on passwords while ensuring strong, phishing-resistant multi-factor authentication across cloud and legacy systems.",
        "geography_raw": "Tel Aviv (Israel), Wilmington (US), London (UK), Paris (France)",
    },
    {
        "name": "Source Defense",
        "description": "Source Defense is a cybersecurity platform that protects websites from client-side attacks such as formjacking and malicious scripts. It provides real-time monitoring and control over third-party integrations to secure digital transactions and sensitive user data.",
        "geography_raw": "Tel Aviv (Israel), New Haven (US)",
    },
    {
        "name": "Folloze",
        "description": "Folloze is a B2B buyer experience platform that enables marketing and sales teams to deliver highly personalized, account-based engagement. Its AI-powered platform orchestrates content, campaigns, and interactions to drive pipeline growth and revenue conversion.",
        "geography_raw": "San Mateo (US), Tel Aviv (Israel)",
    },
    {
        "name": "Evinced",
        "description": "Evinced provides AI-driven digital accessibility testing and remediation tools for enterprise applications. Its platform integrates into development workflows to automatically detect and fix accessibility issues, helping organizations ensure compliance and inclusivity.",
        "geography_raw": "Los Altos (US), Tel Aviv (Israel)",
    },
    {
        "name": "Totango",
        "description": "Totango is a customer success platform that helps enterprises manage the full customer lifecycle. By unifying customer data and enabling proactive engagement, it drives retention, expansion, and overall customer value.",
        "geography_raw": "Redwood City (US), Tel Aviv (Israel), New York (US), Kiev (Ukraine)",
    },
    {
        "name": "Upsun (ex-Platform.sh)",
        "description": "Upsun is a platform-as-a-service that automates the deployment and management of web applications. It enables developers to focus on building products while abstracting infrastructure complexity across multi-cloud environments.",
        "geography_raw": "Paris (France), San Francisco (US)",
    },
    {
        "name": "HG Insights",
        "description": "HG Insights is a technology intelligence platform that uses AI and data science to provide insights into enterprise IT usage and spending. It enables go-to-market teams to identify high-propensity buyers and optimize sales strategies.",
        "geography_raw": "Santa Barbara (US), London (UK)",
    },
    {
        "name": "Grid Dynamics",
        "description": "Grid Dynamics is a digital transformation services company that helps enterprises modernize their technology stack using AI, cloud, and data engineering. It provides consulting and engineering services to build scalable digital products and platforms.",
        "geography_raw": "San Ramon (US), Europe, India, Mexico",
    },
    {
        "name": "FirstHive",
        "description": "FirstHive is a customer data platform that enables brands to unify and activate first-party data for personalized marketing campaigns. It provides real-time data orchestration and analytics to improve customer engagement and conversions.",
        "geography_raw": "India",
    },
    {
        "name": "Webscale",
        "description": "Webscale is an AI-powered cloud platform for e-commerce businesses that optimizes performance, security, and scalability. It combines infrastructure, application services, and AI-driven tools to enhance user experience and drive conversions.",
        "geography_raw": "Boulder (US), India",
    },
    {
        "name": "BoostUp",
        "description": "BoostUp is a revenue intelligence platform that leverages AI to provide real-time visibility into sales performance and pipeline health. It automates forecasting, identifies risks and opportunities, and improves decision-making for revenue teams.",
        "geography_raw": "Menlo Park (US)",
    },
    {
        "name": "Workhelix",
        "description": "Workhelix is a GenAI strategy and analytics platform that helps enterprises identify, prioritize, and measure the impact of AI initiatives. By mapping tasks and quantifying ROI, it enables organizations to deploy AI effectively and maximize productivity gains.",
        "geography_raw": "Palo Alto (US)",
    },
    {
        "name": "Pay-i",
        "description": "Pay-i is a FinOps platform for generative AI that connects model usage to cost and business outcomes. It enables organizations to track unit economics, optimize performance, and scale AI investments with financial discipline.",
        "geography_raw": "Bellevue (US)",
    },
    {
        "name": "Fiddler AI",
        "description": "Fiddler AI provides an enterprise AI observability platform that enables teams to monitor, explain, and govern machine learning and LLM systems. It helps organizations ensure performance, fairness, and compliance across the AI lifecycle.",
        "geography_raw": "Palo Alto (US)",
    },
]


# ---------------------------------------------------------------------------
# MANUAL TAGS (capability + industry per company)
# ---------------------------------------------------------------------------
# Keyed by company name. Each entry: (capability_list, industry_list).
# Tags chosen from the controlled vocabularies above.

COMPANY_TAGS = {
    "Covu": (
        ["ai_crm", "marketing_automation", "customer_success"],
        ["IND_INSURANCE"],
    ),
    "Onymos": (
        ["application_deployment", "paas"],
        ["IND_HEALTHCARE", "IND_CLOUD_INFRA"],
    ),
    "Blendid": (
        ["food_robotics", "computer_vision"],
        ["IND_FOOD_SERVICE"],
    ),
    "Outbuild": (
        ["construction_management", "predictive_analytics"],
        ["IND_CONSTRUCTION"],
    ),
    "Flytrex": (
        ["drone_delivery", "logistics_optimization"],
        ["IND_LOGISTICS", "IND_RETAIL", "IND_FOOD_SERVICE"],
    ),
    "ThetaRay": (
        ["anti_money_laundering", "fraud_detection", "audit_logging"],
        ["IND_FINANCIAL_SERVICES"],
    ),
    "Greenly": (
        ["carbon_accounting"],
        ["IND_SUSTAINABILITY"],
    ),
    "Constella": (
        ["identity_risk_intelligence", "fraud_detection"],
        ["IND_SECURITY"],
    ),
    "Kardinal": (
        ["route_optimization", "logistics_optimization"],
        ["IND_LOGISTICS"],
    ),
    "AiDash": (
        ["satellite_analytics", "predictive_analytics"],
        ["IND_UTILITIES", "IND_SUSTAINABILITY"],
    ),
    "EverestLabs": (
        ["recycling_automation", "computer_vision"],
        ["IND_WASTE_MANAGEMENT", "IND_SUSTAINABILITY"],
    ),
    "Essenvia": (
        ["regulatory_automation"],
        ["IND_MEDICAL_DEVICES"],
    ),
    "Percepto": (
        ["drone_inspection", "computer_vision"],
        ["IND_INDUSTRIAL_INSPECTION"],
    ),
    "Virtana": (
        ["multi_cloud_monitoring", "cloud_cost_optimization"],
        ["IND_CLOUD_INFRA"],
    ),
    "MindsDB": (
        ["ml_in_database", "predictive_analytics"],
        ["IND_AI_INFRA"],
    ),
    "Blue Cedar": (
        ["mobile_app_deployment"],
        ["IND_CLOUD_INFRA"],
    ),
    "Datawizz": (
        ["llm_fine_tuning", "llm_routing", "genai_platform"],
        ["IND_AI_INFRA"],
    ),
    "Secret Double Octopus": (
        ["passwordless_auth"],
        ["IND_SECURITY"],
    ),
    "Source Defense": (
        ["client_side_security"],
        ["IND_SECURITY"],
    ),
    "Folloze": (
        ["buyer_experience_platform", "marketing_automation", "ai_agents"],
        ["IND_SALES_MARKETING"],
    ),
    "Evinced": (
        ["ai_accessibility_testing"],
        ["IND_DEV_TOOLS"],
    ),
    "Totango": (
        ["customer_success", "customer_data_platform"],
        ["IND_SALES_MARKETING"],
    ),
    "Upsun (ex-Platform.sh)": (
        ["paas", "application_deployment"],
        ["IND_CLOUD_INFRA"],
    ),
    "HG Insights": (
        ["tech_intelligence", "predictive_analytics"],
        ["IND_SALES_MARKETING"],
    ),
    "Grid Dynamics": (
        ["digital_transformation_services"],
        ["IND_CLOUD_INFRA"],
    ),
    "FirstHive": (
        ["customer_data_platform", "marketing_automation"],
        ["IND_SALES_MARKETING"],
    ),
    "Webscale": (
        ["ecommerce_platform", "cloud_cost_optimization"],
        ["IND_ECOMMERCE","IND_CLOUD_INFRA"],
    ),
    "BoostUp": (
        ["revenue_intelligence", "predictive_analytics"],
        ["IND_SALES_MARKETING"],
    ),
    "Workhelix": (
        ["ai_strategy_analytics", "genai_platform"],
        ["IND_AI_INFRA"],
    ),
    "Pay-i": (
        ["finops_ai", "genai_platform"],
        ["IND_AI_INFRA"],
    ),
    "Fiddler AI": (
        ["ai_observability", "model_governance"],
        ["IND_AI_INFRA"],
    ),
}


# ---------------------------------------------------------------------------
# PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------

def parse_geography(raw: str) -> list[str]:
    """
    Parse a raw geography string like 'San Francisco (US), Europe' into
    a deduplicated list of ISO-2 country codes (plus EU / LATAM for
    regional aggregates), ordered by first mention in the source text.

    Returns:
        List of country codes. Empty list if nothing matches.
    """
    raw_lc = raw.lower()

    # Collect (position_in_text, country) for every key that matches.
    # We need longest-first matching to avoid "us" inside "austin"
    # stealing a hit, but we need the FINAL order to reflect the
    # left-to-right order of mentions in the source text (because the
    # user wrote HQ first, e.g. "San Francisco (US), Europe").
    keys_by_length = sorted(CITY_TO_COUNTRY.keys(), key=len, reverse=True)

    # scratch is masked as we consume matches to prevent double-counting
    # short keys that are substrings of longer ones.
    scratch = raw_lc
    positioned: list[tuple[int, str]] = []
    for key in keys_by_length:
        idx = scratch.find(key)
        while idx != -1:
            country = CITY_TO_COUNTRY[key]
            positioned.append((idx, country))
            # Mask this match so we don't re-find it AND so shorter
            # substrings inside it (like "us" in "austin") are hidden.
            scratch = scratch[:idx] + (" " * len(key)) + scratch[idx + len(key):]
            idx = scratch.find(key)

    if not positioned:
        return []

    # Sort by position in the text. Dedup while preserving first-seen order.
    positioned.sort(key=lambda p: p[0])
    found: list[str] = []
    for _, country in positioned:
        if country not in found:
            found.append(country)

    return found


def validate_tags(
    company: str,
    capabilities: list[str],
    industries: list[str],
) -> None:
    """Raise if any tag is not in the controlled vocabulary."""
    bad_cap = [c for c in capabilities if c not in CAPABILITY_VOCAB]
    bad_ind = [i for i in industries if i not in INDUSTRY_VOCAB]
    if bad_cap:
        raise ValueError(f"{company}: unknown capability tags {bad_cap}")
    if bad_ind:
        raise ValueError(f"{company}: unknown industry tags {bad_ind}")


def build_records() -> list[dict]:
    """Assemble the final list of tagged portfolio records."""
    records = []
    for idx, raw in enumerate(COMPANY_DESCRIPTIONS, start=1):
        name = raw["name"]
        if name not in COMPANY_TAGS:
            raise KeyError(f"No tags defined for {name}")

        capabilities, industries = COMPANY_TAGS[name]
        validate_tags(name, capabilities, industries)
        all_geo = parse_geography(raw["geography_raw"])
        if not all_geo:
            raise ValueError(
                f"{name}: could not parse any country from "
                f"geography_raw={raw['geography_raw']!r}"
            )

        record = {
            "company_id": f"c{idx:03d}",
            "company_name": name,
            "description": raw["description"],   # verbatim - traceable
            "capability": capabilities,
            "industry": industries,
            "geography_all": all_geo,   # all operating markets (country codes)
        }
        records.append(record)
    return records


def write_outputs(records: list[dict], outdir: Path) -> dict:
    """Write JSON array and JSONL. Return paths."""
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / "bgv_portfolio_tagged.json"
    jsonl_path = outdir / "bgv_portfolio_tagged.jsonl"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {"json": json_path, "jsonl": jsonl_path}


def print_summary(records: list[dict]) -> None:
    """Sanity-check summary: coverage + any UNKNOWN geographies."""
    from collections import Counter

    cap_counter = Counter(c for r in records for c in r["capability"])
    ind_counter = Counter(i for r in records for i in r["industry"])
    geo_counter = Counter(g for r in records for g in r["geography_all"])
    multi_geo = [r["company_name"] for r in records if len(r["geography_all"]) > 1]

    print(f"Total companies tagged: {len(records)}")
    print(f"Distinct capabilities used: {len(cap_counter)} / {len(CAPABILITY_VOCAB)} in vocab")
    print(f"Distinct industries used:  {len(ind_counter)} / {len(INDUSTRY_VOCAB)} in vocab")
    print(f"Geography mentions (all markets): {dict(geo_counter)}")
    print(f"Companies operating in >1 market: {len(multi_geo)} / {len(records)}")
    print()
    print("Top 10 capability tags:")
    for cap, n in cap_counter.most_common(10):
        print(f"  {cap:35s} {n}")
    print()
    print("Industry tag distribution:")
    for ind, n in ind_counter.most_common():
        print(f"  {ind:30s} {n}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    records = build_records()
    out = Path(__file__).parent
    paths = write_outputs(records, out)
    print_summary(records)
    print()
    print(f"Wrote: {paths['json']}")
    print(f"Wrote: {paths['jsonl']}")