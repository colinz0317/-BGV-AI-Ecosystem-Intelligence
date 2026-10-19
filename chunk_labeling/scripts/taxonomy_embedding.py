import os
import json
from typing import Dict, List, Any, Tuple

import numpy as np
from openai import OpenAI


from paths import (
    FILE_CAPABILITY_SCORES_JSON,
    FILE_CAPABILITY_SCORES_JSONL,
    FILE_INDUSTRY_SCORES_JSON,
    FILE_INDUSTRY_SCORES_JSONL,
    FILE_CAPABILITY_LOOKUP,
    FILE_INDUSTRY_LOOKUP
)

# ============================================================
# Configuration
# ============================================================
EMBEDDING_MODEL = "text-embedding-3-small"

OUT_CAP_JSON = str(FILE_CAPABILITY_SCORES_JSON)
OUT_CAP_JSONL = str(FILE_CAPABILITY_SCORES_JSONL)
OUT_IND_JSON = str(FILE_INDUSTRY_SCORES_JSON)
OUT_IND_JSONL = str(FILE_INDUSTRY_SCORES_JSONL)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# Taxonomy definitions
# These should stay aligned with your portfolio taxonomy.
# ============================================================

CAPABILITY_DEFINITIONS: Dict[str, str] = {
    "ai_crm": "AI-powered customer relationship management for lead management, customer engagement, retention, and growth.",
    "ai_observability": "Monitoring, explaining, and evaluating AI or machine learning systems for performance, drift, and reliability.",
    "ai_agents": "Autonomous or semi-autonomous AI agents that can plan, act, and coordinate tasks.",
    "llm_fine_tuning": "Customizing large language models with domain-specific or company-specific data.",
    "llm_routing": "Selecting, dispatching, or orchestrating requests across multiple language models based on task or cost.",
    "ml_in_database": "Running machine learning workflows directly inside database systems using SQL or native integrations.",
    "rag_tooling": "Retrieval-augmented generation infrastructure, tools, and retrieval pipelines for grounding model outputs.",
    "model_governance": "Policies, controls, explainability, fairness, and oversight for AI model lifecycle management.",
    "ai_strategy_analytics": "Analytics for identifying, prioritizing, and measuring business impact of AI initiatives.",
    "finops_ai": "Financial operations for AI usage, including cost tracking, unit economics, and AI spend optimization.",
    "genai_platform": "A platform for building, managing, deploying, or scaling generative AI applications.",
    "predictive_analytics": "Forecasting, prediction, or risk modeling using statistical or machine learning methods.",
    "computer_vision": "AI systems that analyze images or video for detection, classification, or automation.",
    "ai_accessibility_testing": "AI-assisted testing and remediation of accessibility issues in digital products.",
    "identity_risk_intelligence": "Detection of identity-based threats, breach exposure, and identity compromise risks.",
    "passwordless_auth": "Authentication without passwords using strong identity verification methods.",
    "client_side_security": "Security controls for browser-side or client-side threats such as script injection or formjacking.",
    "fraud_detection": "Detection of anomalous or suspicious patterns indicating fraud or financial crime.",
    "anti_money_laundering": "Monitoring and controls to detect money laundering activity and support AML compliance.",
    "audit_logging": "Recording system events, user actions, and traceable logs for compliance, review, and investigations.",
    "customer_data_platform": "Unified customer data infrastructure for activation, analytics, and lifecycle management.",
    "cloud_cost_optimization": "Monitoring and reducing cloud infrastructure cost while maintaining performance.",
    "multi_cloud_monitoring": "Observability and management across hybrid or multi-cloud infrastructure.",
    "application_deployment": "Deploying and managing applications across environments and infrastructure.",
    "paas": "Platform-as-a-service for abstracting infrastructure and simplifying app deployment and operations.",
    "mobile_app_deployment": "Secure delivery, release automation, and lifecycle management for mobile applications.",
    "tech_intelligence": "Technology usage, spending, and enterprise IT intelligence for commercial or strategy decisions.",
    "regulatory_automation": "Automation of regulatory processes, approval workflows, and compliance operations.",
    "construction_management": "Software for project planning, scheduling, and execution in construction workflows.",
    "drone_delivery": "Autonomous drone systems for delivering goods or packages.",
    "drone_inspection": "Drone-based inspection and monitoring of industrial assets or sites.",
    "satellite_analytics": "Using satellite or geospatial data for analytics, monitoring, and operational optimization.",
    "recycling_automation": "Automation and analytics for waste sorting, recycling, and material recovery.",
    "food_robotics": "Robotics systems for automating food preparation or food service operations.",
    "logistics_optimization": "Optimization of routing, delivery, and logistics workflows.",
    "route_optimization": "Optimization of routes, schedules, and dispatch decisions for transportation or delivery.",
    "carbon_accounting": "Measurement, reporting, and management of carbon emissions and sustainability metrics.",
    "customer_success": "Managing customer lifecycle, retention, expansion, and proactive engagement.",
    "revenue_intelligence": "AI-driven forecasting and analysis for revenue, pipeline, and sales performance.",
    "buyer_experience_platform": "Personalized buyer engagement experiences for B2B sales and marketing.",
    "marketing_automation": "Automation of campaigns, segmentation, messaging, and marketing workflows.",
    "digital_transformation_services": "Consulting and engineering services for enterprise modernization using cloud, AI, or data.",
    "ecommerce_platform": "Technology platform supporting e-commerce operations, scalability, and performance.",
}

INDUSTRY_DEFINITIONS: Dict[str, str] = {
    "IND_HEALTHCARE": "Healthcare providers, care delivery, clinical systems, and healthcare operations.",
    "IND_MANUFACTURING": "Manufacturing operations, factories, production systems, and industrial production.",
    "IND_ENERGY_OIL_GAS": "Energy, oil, gas, utilities infrastructure, and related industrial operations.",
    "IND_AUTOMOTIVE": "Automotive industry, vehicles, mobility systems, and automotive operations.",
    "IND_INTERNET_PLATFORM": "Internet-native platforms, digital consumer services, and large-scale online products.",
    "IND_FINANCIAL_SERVICES": "Banking, payments, fintech, capital markets, and financial operations.",
    "IND_INSURANCE": "Insurance carriers, brokers, underwriting, claims, and insurance operations.",
    "IND_CONSTRUCTION": "Construction firms, contractors, project planning, and built-environment workflows.",
    "IND_LOGISTICS": "Transportation, supply chain, delivery, warehousing, and logistics operations.",
    "IND_RETAIL": "Retail commerce, merchants, stores, and retail operations.",
    "IND_FOOD_SERVICE": "Restaurants, food service, food delivery, and food operations.",
    "IND_MEDIA_DIGITAL": "Media, digital publishing, content platforms, and audience engagement businesses.",
    "IND_UTILITIES": "Utilities, grid operations, infrastructure maintenance, and public utility services.",
    "IND_WASTE_MANAGEMENT": "Waste collection, recycling, sorting, and environmental recovery operations.",
    "IND_MEDICAL_DEVICES": "Medical devices, medtech products, and device regulatory workflows.",
    "IND_INDUSTRIAL_INSPECTION": "Inspection and monitoring of industrial assets, equipment, and facilities.",
    "IND_TELECOMMUNICATIONS": "Telecom, connectivity, broadband, carrier systems, and communications infrastructure.",
    "IND_PUBLIC_SECTOR": "Government, public services, and state-supported operational systems.",
    "IND_SUSTAINABILITY": "Climate, carbon reduction, ESG, and sustainability-related operations.",
    "IND_ECOMMERCE": "Online commerce, digital storefronts, and internet-based retail transactions.",
    "IND_SECURITY": "Cybersecurity, identity, fraud prevention, and digital trust solutions.",
    "IND_CLOUD_INFRA": "Cloud infrastructure, deployment, application hosting, and cloud operations.",
    "IND_AI_INFRA": "AI/ML platforms, model operations, observability, and AI infrastructure tooling.",
    "IND_SALES_MARKETING": "Sales, marketing, go-to-market operations, customer engagement, and revenue workflows.",
    "IND_DEV_TOOLS": "Developer tools, testing, accessibility tooling, and software engineering workflows.",
    "IND_GENERAL": "General cross-industry or industry-agnostic use cases.",
}


# ============================================================
# Helper functions
# ============================================================

def build_embedding_input(tag: str, definition: str, tag_type: str) -> str:
    """
    Build the text sent to the embedding model.
    """
    clean_tag = tag.replace("_", " ")
    return f"{tag_type} tag: {clean_tag}. Definition: {definition}"


def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Get embeddings in batch.
    """
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def similarity_to_relation_score(sim: float) -> float:
    """
    Map cosine similarity to a relation score.
    This is the same style as your updated scoring logic.
    """
    if sim >= 0.75:
        return 1.00
    elif sim >= 0.65:
        return 0.80
    elif sim >= 0.55:
        return 0.60
    elif sim >= 0.45:
        return 0.40
    elif sim >= 0.35:
        return 0.20
    else:
        return 0.00


def relation_label(score: float) -> str:
    """
    Convert numeric relation score to a readable label.
    """
    if score >= 1.0:
        return "very_strong"
    elif score >= 0.8:
        return "strong"
    elif score >= 0.6:
        return "moderate"
    elif score >= 0.4:
        return "weak"
    elif score >= 0.2:
        return "very_weak"
    else:
        return "none"


def save_json(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def save_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# Core function: build pairwise match score table
# ============================================================

def build_pairwise_match_scores(
    taxonomy: Dict[str, str],
    tag_type: str,
) -> List[Dict[str, Any]]:
    """
    Build a full pairwise match score table for one taxonomy.

    Output example:
    {
      "source_tag": "ai_observability",
      "target_tag": "model_governance",
      "cosine_similarity": 0.53,
      "relation_score": 0.4,
      "relation_label": "weak"
    }

    Interpretation:
    - source_tag and target_tag are two taxonomy labels
    - cosine_similarity is the raw embedding similarity
    - relation_score is the score you can later reuse in GMS
    """
    tags = list(taxonomy.keys())
    texts = [build_embedding_input(tag, taxonomy[tag], tag_type) for tag in tags]
    vectors = get_embeddings(texts)

    records = []

    for i, source_tag in enumerate(tags):
        for j, target_tag in enumerate(tags):
            sim = cosine_similarity(vectors[i], vectors[j])
            score = similarity_to_relation_score(sim)

            records.append({
                "tag_type": tag_type,
                "source_tag": source_tag,
                "target_tag": target_tag,
                "source_definition": taxonomy[source_tag],
                "target_definition": taxonomy[target_tag],
                "cosine_similarity": round(sim, 6),
                "relation_score": round(score, 4),
                "relation_label": relation_label(score),
                "is_exact_match": source_tag == target_tag,
            })

    return records


# ============================================================
# Optional helper: build a lookup dict for fast runtime use
# ============================================================

def build_lookup_dict(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Convert pairwise records into a nested lookup dictionary.

    Example:
    lookup["ai_observability"]["model_governance"] = 0.4
    """
    lookup: Dict[str, Dict[str, float]] = {}
    for r in records:
        source = r["source_tag"]
        target = r["target_tag"]
        score = r["relation_score"]

        if source not in lookup:
            lookup[source] = {}
        lookup[source][target] = score

    return lookup


# ============================================================
# Main
# ============================================================

def main():
    print("Building capability pairwise match scores...")
    capability_records = build_pairwise_match_scores(
        taxonomy=CAPABILITY_DEFINITIONS,
        tag_type="capability",
    )
    save_json(capability_records, OUT_CAP_JSON)
    save_jsonl(capability_records, OUT_CAP_JSONL)
    print(f"Wrote {OUT_CAP_JSON}")
    print(f"Wrote {OUT_CAP_JSONL}")

    print("Building industry pairwise match scores...")
    industry_records = build_pairwise_match_scores(
        taxonomy=INDUSTRY_DEFINITIONS,
        tag_type="industry",
    )
    save_json(industry_records, OUT_IND_JSON)
    save_jsonl(industry_records, OUT_IND_JSONL)
    print(f"Wrote {OUT_IND_JSON}")
    print(f"Wrote {OUT_IND_JSONL}")

    # Optional: also save compact lookup files for easy downstream scoring
    capability_lookup = build_lookup_dict(capability_records)
    industry_lookup = build_lookup_dict(industry_records)

    with open(FILE_CAPABILITY_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(capability_lookup, f, indent=2, ensure_ascii=False)

    with open(FILE_INDUSTRY_LOOKUP, "w", encoding="utf-8") as f:
        json.dump(industry_lookup, f, indent=2, ensure_ascii=False)

    print(f"Wrote {FILE_CAPABILITY_LOOKUP}")
    print(f"Wrote {FILE_INDUSTRY_LOOKUP}")


if __name__ == "__main__":
    main()