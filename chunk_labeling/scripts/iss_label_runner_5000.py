import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI


# Auto-load .env from project root (one level up from chunk_labeling/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
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

DEFAULT_INPUT_FILE = "chunks_sample_200.jsonl"
DEFAULT_OUTPUT_FILE = "chunks_labeled_output_400.jsonl"
ERROR_FILE = "chunks_labeling_errors.jsonl"
BATCH_STATE_FILE = "_batch_state.json"

MODEL_NAME = "gpt-4.1-mini"

client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Lazy-init OpenAI client."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
    return client


# =========================
# Controlled Vocabularies
# =========================

VERTICAL_INDUSTRY_VOCAB = {
    "healthcare",
    "life_sciences",
    "biotech_pharma",
    "materials_science",
    "manufacturing",
    "energy_oil_gas",
    "clean_energy",
    "automotive",
    "financial_services",
    "insurance",
    "construction",
    "logistics",
    "retail",
    "food_service",
    "utilities",
    "waste_management",
    "medical_devices",
    "industrial_inspection",
    "telecommunications",
    "public_sector",
    "education",
    "legal",
    "media_digital",
    "gaming",
    "ecommerce",
    "internet_platform",
    "agriculture",
    "cybersecurity",
    "sustainability",
    "unknown",
}

HORIZONTAL_FUNCTION_VOCAB = {
    "security",
    "cloud_infra",
    "ai_infra",
    "model_ops",
    "data_analytics",
    "sales_marketing",
    "advertising",
    "dev_tools",
    "productivity_office",
    "search_discovery",
    "content_generation",
    "creator_tools",
    "consumer_assistant",
    "customer_support",
    "enterprise_automation",
    "collaboration",
    "translation_localization",
    "knowledge_management",
    "platform_os",
    "startup_ecosystem",
    "unknown",
}

# Unified sub_theme vocab = VERTICAL_INDUSTRY ∪ HORIZONTAL_FUNCTION
SUB_THEME_VOCAB = VERTICAL_INDUSTRY_VOCAB | HORIZONTAL_FUNCTION_VOCAB
# Sub-theme definitions help the model avoid forcing a chunk into the closest wrong label.
# These definitions are included in the user prompt for every chunk.
SUBTHEME_DEFINITIONS = {
    # Vertical AI sub-themes
    "healthcare": "AI applied to patient care, clinical workflows, diagnosis, hospital operations, care delivery, or health services.",
    "life_sciences": "AI applied to genomics, biology, medical research, protein research, biomedical research, or clinical research.",
    "biotech_pharma": "AI applied to pharmaceutical R&D, biotech platforms, drug discovery, molecule design, or clinical trial optimization.",
    "materials_science": "AI applied to chemistry, materials discovery, molecular simulation, battery materials, or quantum chemistry.",
    "manufacturing": "AI applied to factory operations, production optimization, quality control, industrial automation, or manufacturing workflows.",
    "energy_oil_gas": "AI applied to oil, gas, energy production, exploration, refining, or traditional energy operations.",
    "clean_energy": "AI applied to solar, wind, grid optimization, carbon capture, geothermal, nuclear fusion, or energy storage.",
    "automotive": "AI applied to vehicles, autonomous driving, driver assistance, in-car assistants, mobility, or automotive manufacturing.",
    "financial_services": "AI applied to banking, payments, lending, trading, wealth management, fraud detection, or financial workflows.",
    "insurance": "AI applied to underwriting, claims, risk modeling, insurance operations, or policyholder workflows.",
    "construction": "AI applied to construction planning, jobsite monitoring, building design, safety, or project management.",
    "logistics": "AI applied to supply chain, transportation, routing, warehousing, freight, delivery, or logistics operations.",
    "retail": "AI applied to stores, retail operations, merchandising, inventory, shopping experiences, or retail analytics.",
    "food_service": "AI applied to restaurants, food preparation, food delivery, hospitality food operations, or food-service workflows.",
    "utilities": "AI applied to electric, water, gas utilities, grid reliability, utility inspection, or infrastructure operations.",
    "waste_management": "AI applied to waste collection, recycling, sorting, landfill operations, or circular economy operations.",
    "medical_devices": "AI applied to medical hardware, diagnostic devices, imaging devices, wearables, or regulated medical technology.",
    "industrial_inspection": "AI applied to visual inspection, defect detection, infrastructure inspection, robotics inspection, or asset monitoring.",
    "telecommunications": "AI applied to telecom networks, wireless infrastructure, network optimization, 5G/6G, or carrier operations.",
    "public_sector": "AI applied to government, public services, civic infrastructure, defense-adjacent public missions, or government digital transformation.",
    "education": "AI applied to learning, tutoring, classroom tools, assessment, student support, or education administration.",
    "legal": "AI applied to legal research, contract review, compliance workflows, e-discovery, or legal operations.",
    "media_digital": "AI applied to media, entertainment, creator tools, video platforms, music, publishing, or digital content businesses.",
    "gaming": "AI applied to game companions, NPCs, game development, gameplay assistance, player support, or interactive entertainment.",
    "ecommerce": "AI applied to online shopping, product discovery, checkout, recommendations, marketplaces, or e-commerce operations.",
    "internet_platform": "AI applied to large-scale consumer internet platforms, social platforms, search platforms, app stores, or online ecosystems.",
    "agriculture": "AI applied to farming, crop monitoring, precision agriculture, livestock, irrigation, or agricultural supply chains.",
    "cybersecurity": "AI applied to threat detection, SOC automation, vulnerability detection, identity security, or cyber defense.",
    "sustainability": "AI applied to ESG, climate analytics, environmental impact, emissions tracking, conservation, or sustainability reporting.",

    # Horizontal AI sub-themes
    "security": "AI used for general security workflows, access control, risk detection, safety, trust, or protection across industries.",
    "cloud_infra": "Cloud infrastructure, cloud services, compute/storage/network platforms, or cloud-native infrastructure supporting AI workloads.",
    "ai_infra": "AI infrastructure such as GPUs, accelerators, model training infrastructure, AI platforms, foundation model infrastructure, or AI compute.",
    "model_ops": "AI infrastructure for model deployment, inference serving, monitoring, evaluation, safety, guardrails, grounding, or model governance.",
    "data_analytics": "AI used for analytics, business intelligence, forecasting, data exploration, data processing, or insight generation.",
    "sales_marketing": "AI used for sales productivity, CRM workflows, lead generation, marketing operations, or go-to-market productivity.",
    "advertising": "AI used for ad targeting, ad creative generation, campaign optimization, ads measurement, or ad monetization.",
    "dev_tools": "AI used for coding, software development, developer productivity, code generation, testing, or DevOps workflows.",
    "productivity_office": "AI used in office productivity tools such as email, documents, spreadsheets, slides, calendars, or workplace productivity suites.",
    "search_discovery": "AI used for search, recommendations, information discovery, retrieval, ranking, or answer generation.",
    "content_generation": "AI used to generate or edit text, images, video, audio, design assets, or creative content.",
    "creator_tools": "AI used by creators, podcasters, video makers, streamers, or publishers for content production, editing, dubbing, likeness detection, channel insights, or creator workflow management.",
    "consumer_assistant": "AI assistants for general consumer tasks, personal productivity, mobile experiences, smart devices, or everyday help.",
    "customer_support": "AI used for support chatbots, contact center automation, ticket resolution, customer service, or helpdesk workflows.",
    "enterprise_automation": "AI used to automate business workflows across departments, enterprise applications, internal operations, or agentic business processes.",
    "collaboration": "AI used for meetings, team communication, workplace collaboration, shared documents, or coordination workflows.",
    "translation_localization": "AI used for real-time translation, dubbing, multilingual communication, localization, or cross-language access.",
    "knowledge_management": "AI used for document Q&A, enterprise search, note synthesis, research assistants, knowledge retrieval, or organizational memory.",
    "platform_os": "AI integrated into operating systems, app platforms, device platforms, browsers, or platform-level user experiences.",
    "startup_ecosystem": "Funding, accelerators, cloud credits, mentorship, technical support, or ecosystem programs for AI startups.",

    "unknown": "Use when no listed sub_theme accurately matches the core AI use case.",
}


VALID_DEVELOPMENT_SCOPES = {
    "Vertical AI",
    "Horizontal AI",
    "unknown",
}

VALID_AGENT_ARCHITECTURES = {
    "single-agent AI",
    "multi-agent AI",
    "unknown",
}

VALID_EVENT_TYPES = {
    "Strategic Commitment",
    "Resourcing",
    "Product Shipping & Iteration",
    "Commercial Traction",
    "Partnerships & GTM Motions",
    "unknown",
}

VALID_CONFIDENCE = {"high", "medium", "low"}

TARGET_FIELDS = [
    "development_scope",
    "agent_architecture",
    "sub_theme",
    "event_type",
    "strength_1_5",
    "evidence_span",
    "reason_short",
    "confidence",
    "is_usable_for_iss",
]


# =========================
# Deterministic Guard Config
# =========================

# Conservative explicit AI keywords.
# These are used after LLM labeling to reduce over-labeling.
AI_KEYWORDS = {
    "ai",
    "artificial intelligence",
    "generative ai",
    "genai",
    "gemini",
    "copilot",
    "chatgpt",
    "openai",
    "claude",
    "llama",
    "bard",
    "llm",
    "large language model",
    "foundation model",
    "agent",
    "agentic",
    "ai assistant",
    "chatbot",
    "machine learning",
    "deep learning",
    "neural network",
    "computer vision",
    "vision-language",
    "vlm",
    "autonomous",
    "robotics",
    "inference",
    "ai training",
    "training benchmark",
    "ai accelerator",
    "gpu for ai",
    "ai-powered",
    "ai powered",
    "model release",
    "model hosting",
    "quantum chemistry",
    "materials discovery",
    "drug discovery",
    "genomics",
    "protein",
    "dubbing",
    "translation",
    "guardrails",
    "grounding",
}
# Hard AI signal terms used by the deterministic ISS usability gate.
# Compared with AI_KEYWORDS, this list is intentionally stricter.
# Generic terms such as "smart", "platform", "cloud", "data", "search", or
# "recommendation" are intentionally excluded.
HARD_AI_SIGNAL_KEYWORDS = {
    " ai ",
    "artificial intelligence",
    "generative ai",
    "genai",
    "ai-powered",
    "ai powered",
    "ai feature",
    "ai platform",
    "ai infrastructure",
    "ai assistant",
    "gemini",
    "copilot",
    "chatgpt",
    "openai",
    "claude",
    "llama",
    "bard",
    "llm",
    "large language model",
    "foundation model",
    "machine learning",
    " ml ",
    "deep learning",
    "neural network",
    "agentic",
    " ai agent",
    "ai agents",
    "chatbot",
    "computer vision",
    "vision-language",
    "vlm",
    "automated reasoning",
    "deepfake detection",
    "deepfakes",
    "inference",
    "ai training",
    "training benchmark",
    "ai accelerator",
    "gpu for ai",
    "model release",
    "model hosting",
    "model deployment",
    "inference serving",
    "guardrails",
    "grounding",
    "quantum chemistry",
    "materials discovery",
    "drug discovery",
    "genomics",
    "h100",
    "a100",
    "blackwell",
    "hopper",
    "rtx",
    "cuda",
    "tensorrt",
    "triton",
    "nim",
    "nemo",
    "dgx",
    "mlperf",
    "gpu training benchmark",
    "training benchmark",
    "ai infrastructure",
    "autonomous mobile robot",
    "autonomous mobile robots",
    "autonomy",
    "robotics",
    "physical ai",
    "isaac amr",
    "robot simulation",
}

INVALID_AI_SIGNAL_ONLY_TERMS = {
    "clean energy",
    "smart grid",
    "smarter grid",
    "sustainability",
    "climate",
    "carbon capture",
    "renewable energy",
    "cloud",
    "digital transformation",
    "technology",
    "platform",
    "app",
    "security",
    "cybersecurity",
    "data",
    "analytics",
    "recommendation",
    "personalized",
    "search",
    "ads",
    "productivity",
    "automation",
    "hardware",
    "device",
    "satellite",
    "camera",
    "music",
    "content",
    "media",
    "entertainment",
}


# Aliases for the target companies in your ISS system.
# IMPORTANT:
# OpenAI is intentionally NOT included as a Microsoft alias to avoid
# incorrectly counting OpenAI events into Microsoft ISS.
KNOWN_COMPANY_ALIASES = {
    "Google": {
        "google",
        "alphabet",
        "deepmind",
        "google cloud",
        "vertex ai",
        "gemini",
        "youtube",
        "android",
        "pixel",
        "nest",
        "fitbit",
        "google play",
        "google search",
    },
    "Microsoft": {
        "microsoft",
        "azure",
        "github",
        "copilot",
        "xbox",
        "linkedin",
    },
    "NVIDIA": {
        "nvidia",
        "geforce",
        "rtx",
        "cuda",
        "h100",
        "hopper",
        "blackwell",
        "dgx",
    },
    "OpenAI": {
        "openai",
        "chatgpt",
        "gpt",
        "sora",
        "dall-e",
    },
}

WEAK_TARGET_ALIASES = {
    "Google": {"android", "google play"},
}

OTHER_MAJOR_COMPANIES = {
    "amazon",
    "aws",
    "amazon web services",
    "meta",
    "facebook",
    "llama",
    "anthropic",
    "claude",
    "apple",
    "oracle",
    "salesforce",
    "databricks",
    "snowflake",
    "tesla",
    "xai",
    "mistral",
    "cohere",
    "perplexity",
}

# Keywords for deterministic strength calibration.
# These do not replace the LLM label; they cap inflated scores.
RESEARCH_OR_DEMO_KEYWORDS = {
    "research",
    "researchers",
    "paper",
    "study",
    "benchmark",
    "benchmarks",
    "mlperf",
    "cvpr",
    "conference",
    "showcase",
    "demo",
    "prototype",
    "preview",
    "early access",
    "technical report",
}

FEATURE_RELEASE_KEYWORDS = {
    "feature",
    "features",
    "roll out",
    "rolled out",
    "release",
    "released",
    "launch",
    "launched",
    "update",
    "updated",
    "integration",
    "integrated",
    "available",
    "app",
}

STRONG_TRACTION_KEYWORDS = {
    "revenue",
    "revenues",
    "sales",
    "customers",
    "customer",
    "contracts",
    "adoption",
    "used by",
    "users",
    "usage",
    "growth",
    "grew",
    "increase",
    "increased",
    "improvement",
    "conversion",
    "run rate",
    "annual",
    "billions",
    "millions",
    "hundreds of thousands",
    "thousands of customers",
}

MAJOR_RESOURCE_KEYWORDS = {
    "acquisition",
    "acquire",
    "acquired",
    "investment",
    "invest",
    "capex",
    "capital expenditure",
    "data center",
    "data centers",
    "infrastructure buildout",
    "funding",
    "billion",
    "$",
}
MARKET_FORECAST_KEYWORDS = {
    "market size",
    "market opportunity",
    "global economy",
    "gdp",
    "projecting",
    "projects",
    "estimate",
    "estimated",
    "forecast",
    "forecasts",
    "expected to contribute",
    "could contribute",
    "could reach",
    "region's economy",
    "regional economy",
    "economic impact",
    "analyst",
    "pwc",
    "mckinsey",
}

NEGATIVE_RISK_EVENT_TERMS = {
    "lawsuit",
    "lawsuits",
    " suit ",
    "suits",
    "complaint",
    "privacy complaint",
    "defamatory",
    "hallucination",
    "hallucinations",
    "delusional",
    "spiral",
    "spirals",
    "tragedy",
    "tragedies",
    "suicide",
    "self-harm",
    "harm",
    "harmful",
    "manipulative",
    "sycophancy",
    "sycophantic",
    "internal warnings",
    "distressed users",
    "under investigation",
    "regulator",
    "regulators",
}

SEVERE_NEGATIVE_RISK_EVENT_TERMS = {
    "lawsuit",
    "lawsuits",
    " suit ",
    "suits",
    "complaint",
    "privacy complaint",
    "defamatory",
    "hallucination",
    "hallucinations",
    "delusional",
    "spiral",
    "spirals",
    "tragedy",
    "tragedies",
    "suicide",
    "self-harm",
    "manipulative",
    "sycophancy",
    "sycophantic",
    "internal warnings",
    "distressed users",
}

POSITIVE_ISS_BUSINESS_TERMS = {
    "launched",
    "launches",
    "launch",
    "released",
    "releases",
    "release",
    "rolled out",
    "rolls out",
    "introduced",
    "introduces",
    "announced",
    "announces",
    "available",
    "added",
    "adds",
    "revenue",
    "generated",
    "customers",
    "customer",
    "enterprise",
    "adoption",
    "uses over",
    "partnership",
    "partnered",
    "acquisition",
    "acquired",
    "investment",
    "invested",
    "funding",
    "unified several engineering",
    "product, and research teams",
}

POSITIVE_COMMERCIAL_OR_RESOURCE_TERMS = {
    "revenue",
    "generated",
    "customers",
    "customer",
    "enterprise",
    "adoption",
    "uses over",
    "partnership",
    "partnered",
    "acquisition",
    "acquired",
    "investment",
    "invested",
    "funding",
    "unified several engineering",
    "product, and research teams",
}

RANKING_TRACTION_TERMS = {
    "topped",
    "tops",
    "ranked",
    "ranks",
    "most downloaded",
    "downloads",
    "installs",
    "downloaded app",
    "top app",
    "top apps",
}

STARTUP_ECOSYSTEM_TERMS = {
    "accelerator",
    "startup",
    "startups",
    "cloud credits",
    "cloud credit",
    "mentorship",
    "mentor",
    "cohort",
    "equity-free",
    "support fund",
    "startup support",
    "seed to series a",
}

GENERAL_ASSISTANT_TERMS = {
    "chatgpt",
    "gemini",
    "copilot",
    "assistant",
    "chatbot",
    "general-purpose",
    "search engine",
    "office suite",
    "workspace",
    "microsoft 365",
    "google workspace",
}

SHOPPING_FEATURE_TERMS = {
    "shopping feature",
    "shopping features",
    "shopping assistant",
    "shopping assistants",
    "research potential purchases",
    "help users research",
    "product discovery",
    "recommendations",
    "finding a",
    "find a",
    "ask chatgpt for help finding",
}



def contains_any_keyword(text: str, keywords: Set[str]) -> bool:
    """Return True if text contains any keyword from the given set."""
    text_l = safe_str(text).lower()
    return any(keyword in text_l for keyword in keywords)


def has_market_forecast_only_signal(text: str) -> bool:
    """
    Return True if the text contains market-size/forecast language that should
    not be treated as target-company-specific strength evidence.
    """
    return contains_any_keyword(text, MARKET_FORECAST_KEYWORDS)


def is_general_assistant_shopping_feature(validated: Dict[str, Any]) -> bool:
    """
    Detect cases like ChatGPT shopping feature:
    a shopping/ecommerce feature embedded in a general-purpose assistant/chatbot.

    IMPORTANT:
    This uses evidence_span + source_title only, not reason_short, to avoid
    self-reinforcing template leakage.
    """
    evidence_and_title = " ".join([
        safe_str(validated.get("evidence_span", "")),
        safe_str(validated.get("source_title", "")),
    ]).lower()

    has_general_assistant = contains_any_keyword(evidence_and_title, GENERAL_ASSISTANT_TERMS)
    has_shopping_feature = contains_any_keyword(evidence_and_title, SHOPPING_FEATURE_TERMS) or (
        "shopping" in evidence_and_title and "chatbot" in evidence_and_title
    )

    return has_general_assistant and has_shopping_feature


def normalize_general_assistant_scope(validated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply the general_assistant_scope rule deterministically.
    """
    if is_general_assistant_shopping_feature(validated):
        validated["development_scope"] = "Horizontal AI"

        # Prefer consumer_assistant + search_discovery for general assistant shopping.
        subthemes = []
        for tag in ["consumer_assistant", "search_discovery"]:
            if tag in HORIZONTAL_FUNCTION_VOCAB:
                subthemes.append(tag)
        validated["sub_theme"] = subthemes or ["consumer_assistant"]

        reason = validated.get("reason_short", "")
        if validated.get("is_usable_for_iss", False):
            validated["reason_short"] = (
                "The chunk describes a shopping feature embedded in a general-purpose AI assistant, "
                "so it is classified as Horizontal AI rather than a dedicated ecommerce vertical solution."
            )

    return validated


ADVERTISING_EVIDENCE_TERMS = {
    "ai max",
    "search ads",
    "advertisers",
    "advertiser",
    "campaign",
    "campaigns",
    "conversion",
    "conversions",
    "roas",
    "pmax",
    "demand gen",
    "asset studio",
    "product studio",
    "marketing",
    "meridian",
}

SECURITY_AUTHENTICITY_TERMS = {
    "prompt injection",
    "indirect prompt injection",
    "threat intelligence",
    "threat",
    "scam",
    "scammers",
    "deepfake",
    "deepfakes",
    "watermark",
    "watermarking",
    "synthid",
    "content authenticity",
    "content credentials",
    "policy-violating ads",
    "bad ads",
    "fraud",
    "abuse",
    "safety",
    "security",
    "protect",
    "protections",
}

CREATOR_TOOL_EVIDENCE_TERMS = {
    "youtube studio",
    "ask studio",
    "creator",
    "creators",
    "likeness",
    "dubbing",
    "auto dubbing",
    "lip-synced",
    "podcasters",
    "veo",
    "speech to song",
    "channel insights",
    "studio updates",
}

TRANSLATION_EVIDENCE_TERMS = {
    "voice translate",
    "translate",
    "translation",
    "translates",
    "real time",
    "language barriers",
    "dubbing",
    "localization",
}

PIXEL_ASSISTANT_EVIDENCE_TERMS = {
    "magic cue",
    "daily hub",
    "gemini nano",
    "tensor g5",
    "pixel",
    "on-device ai",
}

OPENAI_SINGLE_AGENT_TERMS = {
    "chatgpt",
    "operator",
    "deep research",
    "codex",
    "chatgpt voice",
    "voice mode",
    "chatgpt tasks",
    "chatgpt connectors",
    "chatgpt memory",
    "chatgpt desktop app",
    "openai's ai-powered assistant",
    "openai ai assistant",
    "custom gpt",
    "custom gpts",
    "gpt store",
}

OPENAI_MULTI_AGENT_TERMS = {
    "multi-agent",
    "multi agent",
    "agent swarm",
    "agent swarms",
    "swarm of agents",
    "agents coordinate",
    "agents collaborate",
    "orchestrate agents",
    "orchestrates agents",
    "agent orchestration",
}


def normalize_openai_agent_architecture(validated: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI-specific agent architecture recovery.

    The general model often labels OpenAI ChatGPT / Operator / deep research /
    Codex product events as agent_architecture=unknown because it avoids
    inferring agent structure. For ISS, these are single-assistant or
    single-agent product forms unless the evidence explicitly says multi-agent
    orchestration, swarms, or agent collaboration.
    """
    if (
        safe_str(validated.get("company", "")) != "OpenAI"
        or not validated.get("is_usable_for_iss", False)
        or validated.get("agent_architecture") != "unknown"
        or validated.get("event_type") == "unknown"
    ):
        return validated

    combined = " ".join([
        safe_str(validated.get("evidence_span", "")),
        safe_str(validated.get("source_title", "")),
    ]).lower()

    if contains_any_keyword(combined, OPENAI_MULTI_AGENT_TERMS):
        validated["agent_architecture"] = "multi-agent AI"
        validated["reason_short"] = (
            safe_str(validated.get("reason_short", ""))
            + " OpenAI agent architecture was normalized to multi-agent AI because the evidence explicitly describes agent swarms, orchestration, or collaboration."
        ).strip()
        return validated

    if contains_any_keyword(combined, OPENAI_SINGLE_AGENT_TERMS):
        validated["agent_architecture"] = "single-agent AI"
        validated["reason_short"] = (
            safe_str(validated.get("reason_short", ""))
            + " OpenAI agent architecture was normalized to single-agent AI because the evidence describes ChatGPT, Operator, deep research, Codex, voice mode, Tasks, Connectors, Memory, or another single-assistant product form."
        ).strip()

    return validated


def normalize_labels_to_evidence_span(validated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep sub_theme and reason aligned to the chosen evidence_span.
    This prevents long chunks from mixing events and prevents copied template reasons.
    """
    if not validated.get("is_usable_for_iss", False):
        return validated

    evidence = safe_str(validated.get("evidence_span", "")).lower()
    title = safe_str(validated.get("source_title", "")).lower()
    combined = f"{evidence} {title}"

    # Security / authenticity evidence must win over broad ad-related language.
    if contains_any_keyword(combined, SECURITY_AUTHENTICITY_TERMS):
        validated["development_scope"] = "Horizontal AI"
        validated["sub_theme"] = ["security"]
        validated["reason_short"] = (
            "The selected evidence describes AI security, abuse prevention, authenticity, or safety controls, "
            "so the labels are aligned to security rather than advertising."
        )

    # Advertising / AI Search Ads evidence.
    elif contains_any_keyword(combined, ADVERTISING_EVIDENCE_TERMS):
        validated["development_scope"] = "Horizontal AI"
        subthemes = []
        if "advertising" in HORIZONTAL_FUNCTION_VOCAB:
            subthemes.append("advertising")
        if "search" in combined or "search ads" in combined or "ai max" in combined:
            subthemes.append("search_discovery")
        validated["sub_theme"] = subthemes or ["advertising"]
        validated["reason_short"] = (
            "The selected evidence describes an AI advertising/search ads product with advertiser adoption or measurable marketing impact, "
            "so the labels are aligned to advertising and search_discovery rather than a consumer assistant event."
        )

    # YouTube / creator AI tools evidence.
    elif contains_any_keyword(combined, CREATOR_TOOL_EVIDENCE_TERMS):
        validated["development_scope"] = "Horizontal AI"
        subthemes = []
        if "creator_tools" in HORIZONTAL_FUNCTION_VOCAB:
            subthemes.append("creator_tools")
        if any(term in combined for term in ["veo", "dubbing", "speech to song", "content", "creative"]):
            subthemes.append("content_generation")
        validated["sub_theme"] = subthemes or ["creator_tools"]
        validated["reason_short"] = (
            "The selected evidence describes AI creator tools or studio features, so the labels are aligned to creator_tools/content_generation."
        )

    # Pixel / translation / on-device assistant evidence.
    elif contains_any_keyword(combined, PIXEL_ASSISTANT_EVIDENCE_TERMS) or contains_any_keyword(combined, TRANSLATION_EVIDENCE_TERMS):
        validated["development_scope"] = "Horizontal AI"
        subthemes = []
        if "consumer_assistant" in HORIZONTAL_FUNCTION_VOCAB:
            subthemes.append("consumer_assistant")
        if contains_any_keyword(combined, TRANSLATION_EVIDENCE_TERMS):
            subthemes.append("translation_localization")
        validated["sub_theme"] = subthemes or ["consumer_assistant"]
        validated["reason_short"] = (
            "The selected evidence describes on-device assistant or translation AI features, so the labels are aligned to consumer_assistant and translation_localization."
        )

    return validated



NVIDIA_INFRA_TERMS = {
    "gpu",
    "gpus",
    "a100",
    "h100",
    "hopper",
    "blackwell",
    "rtx",
    "cuda",
    "tensorrt",
    "triton",
    "nim",
    "nemo",
    "dgx",
    "mlperf",
    "deepstream",
    "holoscan",
    "isaac",
    "isaac amr",
    "physical ai",
    "robotics",
    "autonomous mobile robot",
    "autonomous mobile robots",
    "autonomy",
    "robot simulation",
    "ai blueprints",
    "accelerated computing",
    "inference serving",
    "model deployment",
    "benchmarking",
    "ai infrastructure",
    "ai platform",
}

NVIDIA_MODELOPS_TERMS = {
    "triton",
    "tensorrt",
    "inference serving",
    "model deployment",
    "benchmarking",
    "mlperf",
    "optimiz",
    "throughput",
    "latency",
    "deployment",
    "deepstream",
    "holoscan",
}

NVIDIA_AUTONOMY_TERMS = {
    "isaac",
    "isaac amr",
    "autonomous mobile robot",
    "autonomous mobile robots",
    "amr",
    "autonomy",
    "robotics",
    "robot simulation",
    "physical ai",
    "mapping",
    "simulation",
    "mobile robots",
}



def normalize_nvidia_provider_infra_scope(validated: Dict[str, Any]) -> Dict[str, Any]:
    """
    For NVIDIA, provider infrastructure should be Horizontal AI even when the
    customer/application domain is media, retail, finance, healthcare, etc.

    NVIDIA robotics/autonomy platform launches such as Isaac AMR are still
    considered AI/physical-AI events, but can be Vertical AI when the product is
    specifically built for logistics/manufacturing robot workflows.
    """
    company = safe_str(validated.get("company", ""))
    if company != "NVIDIA" or not validated.get("is_usable_for_iss", False):
        return validated

    combined = " ".join([
        safe_str(validated.get("chunk_text", "")),
        safe_str(validated.get("evidence_span", "")),
        safe_str(validated.get("source_title", "")),
    ]).lower()

    if any(x in combined for x in ["ai factory", "ai factories", "dsx", "omniverse dsx", "reference design", "blueprint"]):
        validated["development_scope"] = "Horizontal AI"

        if contains_any_keyword(combined, NVIDIA_MODELOPS_TERMS):
            validated["sub_theme"] = ["model_ops"]
        else:
            validated["sub_theme"] = ["ai_infra"]

        validated["reason_short"] = (
            "The chunk describes NVIDIA AI factory, DSX, reference-design, or blueprint infrastructure, "
            "so it is classified as Horizontal AI rather than by the customer/application vertical."
        )
        return validated

    # Isaac AMR / physical AI autonomy platform is a product event. Classify as vertical
    # if it is specifically about autonomous mobile robot fleets in warehouses/factories/logistics.
    if contains_any_keyword(combined, NVIDIA_AUTONOMY_TERMS):
        if "isaac amr" in combined or "autonomous mobile robot" in combined or "mobile robots" in combined:
            validated["development_scope"] = "Vertical AI"
            if "logistics" in VERTICAL_INDUSTRY_VOCAB:
                validated["sub_theme"] = ["logistics"]
            validated["reason_short"] = (
                "The chunk describes NVIDIA's Isaac AMR autonomy platform for autonomous mobile robot fleets, "
                "so it is treated as a vertical AI robotics/logistics product event."
            )
            return validated

    if contains_any_keyword(combined, NVIDIA_INFRA_TERMS):
        validated["development_scope"] = "Horizontal AI"

        if contains_any_keyword(combined, NVIDIA_MODELOPS_TERMS):
            validated["sub_theme"] = ["model_ops"]
        else:
            validated["sub_theme"] = ["ai_infra"]

        validated["reason_short"] = (
            "The chunk describes NVIDIA provider-side AI infrastructure or model deployment capability, "
            "so it is classified as Horizontal AI rather than by the customer/application vertical."
        )

    return validated


def has_strong_quantified_or_strategic_evidence(text: str, event_type: str) -> bool:
    """
    Return True if the chunk has evidence strong enough to justify strength 5 or high strength.

    Market-size forecasts, regional GDP projections, and third-party opportunity estimates
    do not count as strong evidence because they are not target-company-specific.
    """
    text_l = safe_str(text).lower()

    has_number_or_money = bool(re.search(r"(\$?\d+(\.\d+)?\s?(billion|million|%|percent|x|k|m|b))", text_l))
    has_traction = contains_any_keyword(text_l, STRONG_TRACTION_KEYWORDS)
    has_major_resource = contains_any_keyword(text_l, MAJOR_RESOURCE_KEYWORDS)
    has_market_forecast = has_market_forecast_only_signal(text_l)

    # If the number mainly appears in market-forecast language, don't treat it as strong evidence.
    if has_market_forecast:
        target_specific_indicators = {
            "we launched",
            "we are announcing",
            "we announced",
            "we invested",
            "we acquired",
            "this fund",
            "support fund",
            "customer",
            "customers",
            "users",
            "monthly active users",
            "revenue",
            "cost reduction",
            "time savings",
            "throughput",
        }
        if not contains_any_keyword(text_l, target_specific_indicators):
            return False

    if event_type == "Commercial Traction" and has_number_or_money and has_traction:
        return True

    if event_type == "Resourcing" and has_number_or_money and has_major_resource and not has_market_forecast:
        return True

    return False


def is_negative_risk_event_without_positive_signal(validated: Dict[str, Any]) -> bool:
    """Detect risk/legal/harm coverage that should not count as positive ISS."""
    if not validated.get("is_usable_for_iss", False):
        return False

    combined = " ".join([
        safe_str(validated.get("source_title", "")),
        safe_str(validated.get("evidence_span", "")),
    ]).lower()

    if not contains_any_keyword(combined, NEGATIVE_RISK_EVENT_TERMS):
        return False

    if contains_any_keyword(combined, SEVERE_NEGATIVE_RISK_EVENT_TERMS):
        return not (
            contains_any_keyword(combined, POSITIVE_COMMERCIAL_OR_RESOURCE_TERMS)
            or has_strong_quantified_or_strategic_evidence(
                combined,
                safe_str(validated.get("event_type", "unknown")),
            )
        )

    has_positive_business_signal = contains_any_keyword(combined, POSITIVE_ISS_BUSINESS_TERMS)
    has_quantified_positive_signal = has_strong_quantified_or_strategic_evidence(
        combined,
        safe_str(validated.get("event_type", "unknown")),
    )

    # Legal/harm reports are not positive investment-intent events unless the same
    # evidence also contains a concrete commercial, product-launch, partnership, or
    # resourcing signal from the target company.
    return not (has_positive_business_signal or has_quantified_positive_signal)


def calibrate_strength(validated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically cap inflated strength scores.

    Main goal:
    - unknown / non-usable => 1
    - research/demo/blog/product description => usually <= 3
    - normal feature release without adoption/revenue/customer proof => <= 3
    - strength 5 only for major quantified traction or major resource events
    """
    if not validated.get("is_usable_for_iss", False):
        validated["strength_1_5"] = 1
        return validated

    chunk_text = validated.get("chunk_text", "")
    evidence_span = validated.get("evidence_span", "")
    combined_text = f"{chunk_text} {evidence_span}"
    event_type = validated.get("event_type", "unknown")

    strength = int(validated.get("strength_1_5", 1))

    if is_negative_risk_event_without_positive_signal(validated):
        return force_not_usable(
            validated,
            "Excluded from ISS because the evidence is primarily legal, harm, safety-risk, or negative incident coverage rather than a positive target-company investment-intent event."
        )

    # No clear AI signal or no clear scope/event should never score above 1.
    if (
        not has_explicit_ai_signal(chunk_text)
        or validated.get("development_scope") == "unknown"
        or event_type == "unknown"
    ):
        validated["strength_1_5"] = 1
        validated["is_usable_for_iss"] = False
        return validated

    # Strength 5 is rare and must be backed by quantified traction or major resourcing.
    if strength >= 5 and not has_strong_quantified_or_strategic_evidence(combined_text, event_type):
        strength = 4

    # Research/demo/showcase/benchmark content should usually not exceed 3
    # unless there is commercial traction or major resourcing evidence.
    if contains_any_keyword(combined_text, RESEARCH_OR_DEMO_KEYWORDS):
        if event_type not in {"Commercial Traction", "Resourcing"}:
            strength = min(strength, 3)

    # Product feature releases without strong traction/resource evidence should usually cap at 3.
    if event_type == "Product Shipping & Iteration":
        has_traction_or_resource = (
            contains_any_keyword(combined_text, STRONG_TRACTION_KEYWORDS)
            or contains_any_keyword(combined_text, MAJOR_RESOURCE_KEYWORDS)
        )
        if contains_any_keyword(combined_text, FEATURE_RELEASE_KEYWORDS) and not has_traction_or_resource:
            strength = min(strength, 3)

    # Strategic statements without concrete action should not exceed 3.
    if event_type == "Strategic Commitment":
        if not contains_any_keyword(combined_text, MAJOR_RESOURCE_KEYWORDS):
            strength = min(strength, 3)

    # Startup ecosystem programs are usually moderate signals unless there is target-company-specific
    # major funding, acquisition, or quantified large-scale impact. Market-size estimates do not count.
    if "startup_ecosystem" in validated.get("sub_theme", []):
        has_specific_program_amount = contains_any_keyword(combined_text, {"this fund", "support fund", "committed", "investment from google", "google invested"})
        if not has_specific_program_amount:
            strength = min(strength, 3)
        elif has_market_forecast_only_signal(combined_text) and not has_strong_quantified_or_strategic_evidence(combined_text, event_type):
            strength = min(strength, 3)
        elif not has_strong_quantified_or_strategic_evidence(combined_text, event_type):
            strength = min(strength, 3)

    # CVC one-liners are sparse; cap strength conservatively unless clear major details exist.
    if validated.get("source_type") == "CVC_from_google_news":
        cvc_text = f"{validated.get('chunk_text', '')} {validated.get('source_title', '')}"
        has_amount = bool(re.search(r"(\$\s?\d|\d+(\.\d+)?\s?(billion|million|m|b))", cvc_text.lower()))
        has_direct_ai = has_explicit_ai_signal(cvc_text)
        has_major_action = any(term in cvc_text.lower() for term in ["acquisition", "funding round", "strategic investment", "investment"])
        if not (has_amount and has_direct_ai and has_major_action):
            strength = min(strength, 2)
        elif has_amount and has_direct_ai and has_major_action:
            strength = min(strength, 3)

    validated["strength_1_5"] = max(1, min(5, strength))
    return validated



# =========================
# Prompts
# =========================

TARGET_COMPANY_GATE = """
================================================================
CRITICAL GATE 0: target_company_relevance
================================================================
Before assigning any ISS labels, first determine whether the main AI/business event
in the chunk is about the target company provided in metadata.

The target company is the `company` field in the user message.

Set is_usable_for_iss = false and return unknown labels if:
- the main event is about another company, competitor, customer, startup, VC, government, or general market trend;
- the target company is only mentioned as a comparison, background example, partner list item, investor list item, cloud host list item, or passing reference;
- the article is assigned to the target company only because the target company name appears somewhere in the text, but the main action belongs to another entity.

Only set is_usable_for_iss = true if:
- the target company is the main actor performing the event; OR
- the target company is clearly one of the main actors in a partnership, investment, acquisition, product launch, customer deployment, or commercial traction event.

Examples:
- company = Microsoft, but the chunk mainly says AWS launched Automated Reasoning checks. Microsoft is only mentioned as a similar product provider.
  => is_usable_for_iss = false, development_scope = unknown, event_type = unknown, strength_1_5 = 1.

- company = Google, but the chunk mainly discusses Meta's Llama revenue-sharing agreements and only lists Google Cloud as one of many hosts.
  => is_usable_for_iss = false.

- company = NVIDIA, and the chunk says NVIDIA and Oracle announced a collaboration for sovereign AI.
  => usable, because NVIDIA is a main actor.
""".strip()


AI_SIGNAL_GATE = """
================================================================
CRITICAL GATE 1: explicit_ai_signal
================================================================
Do not assign Vertical AI or Horizontal AI unless the evidence_span itself contains an explicit AI-related signal.

Valid AI signals include explicit mentions of:
- AI, artificial intelligence, generative AI, GenAI
- Gemini, Copilot, ChatGPT, OpenAI, Claude, Llama, Bard
- LLM, large language model, foundation model, model release
- agent, agentic AI, AI assistant, chatbot
- machine learning, ML, deep learning, neural network
- computer vision, vision-language model, VLM
- autonomous AI system, robotics AI
- inference, AI training, AI accelerator, GPU for AI workloads
- AI-powered product, AI feature, AI platform, AI infrastructure
- automated reasoning, deepfake detection
- for NVIDIA: H100, A100, Blackwell, Hopper, RTX, CUDA, TensorRT, Triton, NIM, NeMo, DGX, MLPerf, AI infrastructure, inference platform, GPU training benchmark
- robotics/physical AI signals: autonomous robots, autonomous mobile robots, autonomy stack, Isaac AMR, robot simulation, physical AI

Set development_scope = unknown and is_usable_for_iss = false if:
- the chunk is only about general software, cloud, platform, hardware, office expansion, policy, legal settlement, culture, marketing, or business operations without explicit AI relevance;
- the chunk discusses technology broadly but does not clearly connect it to AI;
- the chunk only mentions data, analytics, automation, digital transformation, or platform improvement without explicit AI/model/agent evidence.

Invalid AI signals by themselves:
- clean energy, smart grid, smarter grid, sustainability, climate, carbon capture, renewable energy
- cloud, digital transformation, technology, platform, app, security, cybersecurity
- data, analytics, recommendation, personalized, search, ads, productivity, automation
- hardware, device, satellite, camera, music, content, media, entertainment

Important:
- Do not classify a chunk as Horizontal AI just because it is about a broad platform.
- Do not classify a chunk as Vertical AI just because it mentions an industry.
- Industry words like healthcare, retail, education, finance, gaming, clean energy, sustainability, or public sector are not enough by themselves.
- The supporting evidence_span must include an explicit AI/ML/model/agent signal or a clearly AI-powered capability.
- If the evidence_span only describes clean energy, smart grid, sustainability, cloud, digital, platform, data, recommendation, search, or automation without explicit AI language, return unknown labels and is_usable_for_iss=false.
""".strip()


SOURCE_QUALITY_GATE = """
================================================================
CRITICAL GATE 2: source_quality
================================================================
Exclude low-quality source pages that are not standalone business events.

Set is_usable_for_iss = false and return unknown labels if:
- source_url contains /author/, /tag/, /category/, /search/, /page/, /topics/, /archive/, or similar index-page paths;
- source_title appears to be a person name, author page, tag page, archive page, or topic collection rather than a specific article or announcement;
- the chunk is mainly an author bio, list of post titles, archive snippets, search results, navigation text, or collection page;
- the text combines multiple unrelated post summaries rather than describing one specific event.

These pages should be excluded even if they mention AI terms, because they are not one auditable ISS event.
""".strip()


CONCRETE_EVENT_OWNER_GATE = """
================================================================
CRITICAL GATE 3: concrete_event_owner
================================================================
The evidence_span must describe an action, business signal, product, partnership,
investment, customer deployment, or strategic commitment by the target company.

Set is_usable_for_iss = false if:
- the concrete event in the evidence_span is owned by another company, startup, customer, investor, or competitor;
- the target company is mentioned only in background, market context, comparison, or a broad list;
- the chunk starts with broad market context about the target company, but the specific evidence_span describes another company's funding, product, launch, or traction;
- the funding/acquisition/product/customer event is about another startup or company, even if the target company operates in the same market.

Example:
- company = OpenAI, but the evidence_span says Onton raised $7.5M.
  => exclude, because the concrete funding event is about Onton, not OpenAI.

- company = Google, but the chunk describes OpenAI Study Mode.
  => exclude, because the product event is about OpenAI, not Google.

- company = NVIDIA, and the evidence_span says NVIDIA Triton helped Snap serve AI recommendations.
  => usable, because NVIDIA's product capability is part of the concrete event.
""".strip()


PROVIDER_CUSTOMER_SCOPE_RULE = """
================================================================
CRITICAL RULE: provider_vs_customer_use_case
================================================================
When the target company provides reusable AI infrastructure, chips, models, APIs,
cloud tools, inference serving, developer tooling, or AI platforms that a customer uses
in one specific industry, classify based on the target company's provider capability,
not the customer's vertical industry.

Use Horizontal AI when the target company's role is:
- AI infrastructure, chips, GPUs, accelerators, inference serving, model deployment, model APIs;
- cloud AI platforms, AI developer tools, AI model operations, guardrails, evaluation, grounding;
- general-purpose platforms used across many industries.

Only use Vertical AI when the target company itself is building, selling, or strategically
committing to an industry-specific AI solution or vertical workflow.

Special NVIDIA rule:
- If the target company is NVIDIA and the evidence is about GPUs, accelerators, Blackwell, H100, RTX, CUDA, TensorRT, Triton, NIM, DGX, AI Blueprints, DeepStream, Holoscan, or accelerated computing, classify as Horizontal AI.
- Use ai_infra for chips, GPUs, accelerators, AI compute platforms, Blackwell, H100, RTX, DGX, NIM, or AI Blueprints.
- Use model_ops for Triton, TensorRT, inference serving, deployment, optimization, benchmarking, DeepStream, or Holoscan.

Examples:
- NVIDIA Triton / TensorRT serving Snap shopping or recommendation use case
  => Horizontal AI, sub_theme = model_ops or ai_infra, not Vertical AI / retail.
- Google Cloud Vertex AI used by a financial customer
  => usually Horizontal AI / ai_infra, model_ops, or enterprise_automation, not Vertical AI / financial_services.
- Microsoft Copilot used inside a bank
  => usually Horizontal AI / productivity_office, enterprise_automation, or sales_marketing.
- A company building a visual inspection solution specifically for utilities or factories
  => Vertical AI / industrial_inspection or utilities.
""".strip()


EXCLUSION_REASON_PRIORITY_RULE = """
================================================================
CRITICAL RULE: exclusion_reason_priority
================================================================
When excluding a chunk from ISS, choose the most specific and accurate exclusion reason.

Use this priority order:
1. source_quality: author page, tag page, archive page, search page, index page, or post-list page.
2. concrete_event_owner: the concrete event in evidence_span is owned by another company, not the target company.
3. explicit_ai_signal: the chunk is about the target company but does not contain an explicit AI-related business signal.
4. no_clear_event: the chunk has AI terms but no clear ISS event type or meaningful business action.
5. no_clear_scope: the chunk has AI terms/event but cannot support Vertical AI or Horizontal AI.

Important:
- Do not say the target company is not mentioned if the real issue is that the chunk lacks an AI signal.
- For official company blog pages, if the content is clearly from the target company but non-AI, say it lacks an explicit AI-related business signal.
""".strip()


GENERAL_ASSISTANT_SCOPE_RULE = """
================================================================
CRITICAL RULE: general_assistant_scope
================================================================
If the AI feature is embedded in a general-purpose chatbot, assistant, search engine,
office suite, operating system, browser, cloud platform, or developer platform,
classify it as Horizontal AI even if the example use case relates to one industry.

Use Horizontal AI when:
- a general chatbot/assistant adds shopping, education, travel, finance, coding, or search features;
- an office suite assistant supports a department or customer workflow;
- a developer platform or cloud AI platform is used by a customer in a specific industry.

Use Vertical AI only when:
- the product itself is purpose-built for one industry workflow;
- the target company is selling or building a dedicated vertical solution.

Examples:
- ChatGPT shopping feature inside a general-purpose assistant
  => Horizontal AI, sub_theme = consumer_assistant or search_discovery, not Vertical AI / ecommerce.
- Gemini or Copilot inside a general office or enterprise suite
  => Horizontal AI, sub_theme = productivity_office or enterprise_automation.
- A dedicated AI shopping platform like Onton
  => Vertical AI, sub_theme = ecommerce.
""".strip()


LONG_CHUNK_EVENT_CONSISTENCY_RULE = """
================================================================
CRITICAL RULE: long_chunk_event_consistency
================================================================
Long chunks may contain multiple AI events. You must choose exactly ONE primary event,
and all labels must describe the same event supported by evidence_span.

Rules:
- evidence_span is the anchor. development_scope, sub_theme, event_type, strength_1_5,
  agent_architecture, confidence, and reason_short must all match the selected evidence_span.
- Do not mix evidence from different parts of the chunk.
- Do not use a reason template from another event in the same chunk.
- Do not describe shopping, assistant, education, ads, creator tools, or AI infrastructure unless
  evidence_span actually describes that topic.
- For earnings call chunks, prefer the strongest concrete AI business signal with measurable traction,
  but ensure sub_theme and reason_short describe that exact signal.

Examples:
- If evidence_span is about AI Max in Search Ads used by advertisers, use advertising/search_discovery,
  not consumer_assistant.
- If evidence_span is about Magic Cue and Voice Translate, use consumer_assistant and translation_localization,
  not shopping/search_discovery unless shopping/search is explicitly in evidence_span.
- If evidence_span is about YouTube creator AI tools, use content_generation or creator_tools, not consumer_assistant
  unless the evidence describes an assistant answering user questions.
""".strip()


CVC_ONE_LINER_RULE = """
================================================================
CRITICAL RULE: cvc_one_liner_strength_cap
================================================================
For source_type = CVC_from_google_news, the text is often a short structured one-line summary.
Be conservative.

Rules:
- Default max strength is 2 for CVC one-liners.
- Allow strength 3 only if the chunk contains clear target-company-specific details such as:
  amount, investee/acquiree, product/capability, strategic rationale, or direct AI-specific relevance.
- Allow strength 4 or 5 only for major funding/acquisition/capex with clear amount and direct AI strategic relevance.
- If the one-liner only says "engaged in Strategic Investment activity" or "within Data & Analytics sector"
  without details, use strength 2 at most.
- If the sector is not explicitly AI-related and no AI signal appears, exclude from ISS.
""".strip()


SYSTEM_PROMPT = f"""
You are an expert information extraction model for a venture-investment intelligence pipeline.

Your task is to read ONE text chunk and assign chunk-level labels for the Investment Intent Score (ISS) system.

You must classify the chunk using only the information explicitly supported by the chunk text and the provided metadata.
Do not infer beyond the evidence. If the text does not provide enough support for a field, output "unknown" for that field.

{TARGET_COMPANY_GATE}

{AI_SIGNAL_GATE}

{SOURCE_QUALITY_GATE}

{CONCRETE_EVENT_OWNER_GATE}

{PROVIDER_CUSTOMER_SCOPE_RULE}

{EXCLUSION_REASON_PRIORITY_RULE}

{GENERAL_ASSISTANT_SCOPE_RULE}

{LONG_CHUNK_EVENT_CONSISTENCY_RULE}

{CVC_ONE_LINER_RULE}

You must return valid JSON only. Do not include markdown, explanations, or extra text outside the JSON.

================================================================
FIELD 1: development_scope
================================================================
Choose exactly ONE of the following:
- Vertical AI
- Horizontal AI
- unknown

Definitions:
- Vertical AI: AI solutions tailored to specific industries or domain-specific workflows, such as healthcare, banking, logistics, manufacturing, legal, insurance, education, energy, and other sector-specific use cases.
- Horizontal AI: AI solutions that support broad, cross-industry, or cross-functional use cases across many enterprises, such as coding assistants, generic copilots, search, customer support, sales productivity, or general enterprise automation tools.
- unknown: use when neither Vertical AI nor Horizontal AI can be supported confidently.

Important rules:
- A chunk cannot be both Vertical AI and Horizontal AI.
- Do not assign Vertical AI or Horizontal AI unless the chunk passes the explicit_ai_signal gate.
- If the chunk emphasizes industry-specific workflows, domain specialization, or regulated/sector-specific use cases, prefer Vertical AI.
- If the chunk emphasizes broad enterprise workflows, cross-industry functions, or general-purpose copilots/assistants, prefer Horizontal AI.
- Shopping/ecommerce boundary rule: if shopping is one feature inside a general-purpose chatbot or assistant, prefer Horizontal AI with consumer_assistant or search_discovery; if the product is specifically designed as an ecommerce/shopping platform or retail workflow solution, prefer Vertical AI with ecommerce or retail.
- If there is insufficient evidence, return unknown.
- If the chunk is not about the target company, return unknown.

================================================================
FIELD 2: agent_architecture
================================================================
Choose exactly ONE of the following:
- single-agent AI
- multi-agent AI
- unknown

Definitions:
- single-agent AI: one primary agent or assistant performs the workflow, even if it uses tools or APIs.
- multi-agent AI: multiple specialized agents coordinate, collaborate, delegate, or communicate to complete a workflow.
- unknown: use when the chunk does not clearly describe agent structure.

Important rules:
- A chunk cannot be both single-agent AI and multi-agent AI.
- Do not infer agent architecture merely because something is AI-native, model-native, or agentic.
- Assign single-agent AI or multi-agent AI only when the chunk explicitly supports that structure.
- development_scope and agent_architecture are independent dimensions:
  a chunk may have both a development_scope and an agent_architecture.
- Example:
  a healthcare workflow assistant can be Vertical AI and single-agent AI at the same time.

================================================================
FIELD 3: sub_theme  (conditional — list, tied to development_scope)
================================================================
Return a JSON list of one or more tags from the allowed sub_theme values provided in the user message.

sub_theme describes the specific vertical industry or horizontal function within the selected development_scope.

Rules:
- If development_scope is "Vertical AI", choose one or more tags ONLY from the allowed sub_theme values for Vertical AI.
- If development_scope is "Horizontal AI", choose one or more tags ONLY from the allowed sub_theme values for Horizontal AI.
- Use the provided sub_theme definitions to choose the most accurate tag.
- If development_scope is "unknown", return ["unknown"].
- If development_scope is Vertical AI or Horizontal AI but the chunk does not accurately match any listed value, return ["unknown"].
- Do not force a chunk into the closest available sub_theme if the match is weak or only partially related.
- Do not invent any new sub_theme values beyond those provided.
- Do not add a sub_theme just because an industry or function is mentioned in passing.
- Prefer 1 sub_theme. Use multiple sub_themes only when the chunk provides strong, direct evidence for multiple core use cases.
- sub_theme is independent of agent_architecture.

Examples:
- If the chunk describes an AI assistant for hospitals, return development_scope = "Vertical AI" and sub_theme = ["healthcare"].
- If the chunk describes AI coding tools for developers, return development_scope = "Horizontal AI" and sub_theme = ["dev_tools"].
- If the chunk describes AI infrastructure used across industries, return development_scope = "Horizontal AI" and sub_theme = ["ai_infra"].

================================================================
FIELD 4: event_type
================================================================
Choose exactly ONE of the following:
- Strategic Commitment
- Resourcing
- Product Shipping & Iteration
- Commercial Traction
- Partnerships & GTM Motions
- unknown

Event type definitions:
- Strategic Commitment: explicit strategic prioritization, executive commitment, declared long-term direction, top-level emphasis on an area.
- Resourcing: tangible allocation of resources, such as hiring, headcount expansion, budget allocation, capex, infrastructure buildout, acquisitions for capability, or dedicated teams.
- Product Shipping & Iteration: product launch, feature release, model release, integration launch, platform update, product improvement, or repeated shipping activity.
- Commercial Traction: evidence of monetization or adoption, such as customers, contracts, revenue, growth, usage, enterprise adoption, or demand signals clearly tied to the theme.
- Partnerships & GTM Motions: partnerships, integrations, alliances, marketplaces, channel motions, co-sell programs, ecosystem expansion, developer or go-to-market collaboration.
- unknown: use when the event type cannot be supported confidently.

Important event-type boundary rules:
- If the chunk mainly says the company "believes", "focuses on", or "sees AI as strategic", classify as Strategic Commitment.
- If the chunk mainly shows actual resource deployment, hiring, spending, infrastructure, or acquisition, classify as Resourcing.
- If the chunk mainly describes a launched or updated product/capability, classify as Product Shipping & Iteration.
- If the chunk mainly provides adoption, customers, contracts, sales, revenue, or usage evidence, classify as Commercial Traction.
- If the chunk mainly describes partners, integrations, marketplaces, or co-selling, classify as Partnerships & GTM Motions.
- If the main event is not about the target company, classify as unknown.
- If there is no explicit AI-related business event, classify as unknown.

Event-type priority when multiple signals appear:
1. Resourcing: investment, acquisition, hiring, capex, infrastructure buildout.
2. Commercial Traction: revenue, customer adoption, usage, contracts, quantified demand.
3. Product Shipping & Iteration: release, launch, preview, model/API/feature update.
4. Partnerships & GTM Motions: partnership, integration, alliance, marketplace, co-sell.
5. Strategic Commitment: direction, strategy, executive priority.
6. unknown: no clear target-company AI event.

================================================================
FIELD 5: strength_1_5
================================================================
Assign an integer from 1 to 5 based on how strong and explicit the evidence is.

Use a conservative scoring standard. Do not inflate strength just because the chunk mentions
a well-known company, product, or AI term.

Scoring rubric:
- 1 = no clear AI signal, wrong-company event, not usable for ISS, or no clear business action.
- 2 = weak mention only; conceptual, background, research discussion, general blog description, or directional AI statement without a concrete company action.
- 3 = clear AI product, feature, partnership, investment direction, or launch signal, but limited scale, weak proof, no customer proof, no revenue/adoption metric, or early preview/research/demo stage.
- 4 = strong evidence of a concrete AI launch, customer deployment, partnership, infrastructure investment, or capability rollout, with named initiative and clear support. Use 4 only when the event is more than a generic product description.
- 5 = very strong strategic or commercial evidence, such as large-scale commercialization, major capex, major acquisition, large funding/investment, strong quantified traction, revenue impact, customer adoption at scale, or clearly strategic executive-backed event.

Important strength rules:
- If is_usable_for_iss = false, strength_1_5 must be 1.
- If there is no explicit AI signal, strength_1_5 must be 1.
- If the main event is not about the target company, strength_1_5 must be 1.
- Research showcases, academic papers, benchmark demos, conference presentations, or technical blog posts should usually be 2 or 3 unless there is clear commercial deployment or major resource commitment.
- Product feature releases without adoption, revenue, customer proof, or scale should usually be 3, not automatically 4.
- General product descriptions, marketing blogs, or broad platform updates should not exceed 3 unless there is concrete evidence of customer deployment, partnership, or measurable traction.
- Startup accelerator, startup support, cloud-credit, mentorship, or ecosystem-program announcements should usually be 3 unless there is a target-company-specific large committed funding amount, major strategic investment, national-scale partnership, or large-scale quantified impact.
- Only assign strength 4 when there is strong evidence of a major product launch, named enterprise customer deployment, significant measurable impact, major funding/capex/acquisition, or large-scale commercial traction.
- Do not use market-size forecasts, regional GDP projections, analyst estimates, or third-party market opportunity numbers to raise strength. Strength must be based on target-company-specific evidence.
- Use 5 rarely. Do not assign 5 unless the chunk contains major quantified scale, revenue/adoption evidence, major capex/acquisition/funding, or strategic-level commitment with strong evidence directly tied to the target company.
- For CVC_from_google_news one-liners, strength should usually be 2 or lower unless clear amount + AI relevance + strategic details are present.

================================================================
FIELD 6: evidence_span
================================================================
Extract the shortest possible span from the chunk that best supports your classification.

Requirements:
- must be copied exactly from the chunk text
- should usually be 1–2 short consecutive sentences
- should directly support development_scope, agent_architecture, sub_theme, and event_type when applicable
- do not paraphrase
- do not fix spacing, punctuation, capitalization, or glued words from the original text
- do not combine distant sentences
- if is_usable_for_iss = false, provide the shortest span showing why the chunk is irrelevant, non-AI, from a low-quality/index page, or about another company

================================================================
FIELD 7: reason_short
================================================================
Write 1 concise sentence explaining why the labels were assigned.

Requirements:
- if is_usable_for_iss = true, must mention why the chosen event_type fits
- if is_usable_for_iss = true, should mention why development_scope fits if not unknown
- if is_usable_for_iss = true, should mention why agent_architecture fits if not unknown
- if is_usable_for_iss = true, should mention why sub_theme fits if not unknown
- must be grounded only in the given chunk
- if is_usable_for_iss = false, reason_short must ONLY explain why the chunk was excluded
- if excluded, do not say the chunk fits Vertical AI, Horizontal AI, or any event type
- if excluded, clearly state whether the chunk was excluded because it lacks explicit AI signal, is about another company, or is a low-quality author/tag/archive/index page

================================================================
FIELD 8: confidence
================================================================
Choose exactly ONE:
- high
- medium
- low

Confidence guidelines:
- high = strong direct evidence and low ambiguity
- medium = reasonably supported but some ambiguity remains
- low = weak evidence or multiple plausible labels

================================================================
FIELD 9: is_usable_for_iss
================================================================
Choose true or false.

Set to true only if:
- the chunk contains a meaningful AI-related business signal
- the main AI/business event is about the target company
- the concrete event in evidence_span is owned by or directly involves the target company
- event_type is not unknown
- evidence_span can clearly support the classification
- strength_1_5 is assigned with reasonable support

Set to false if:
- the chunk is mainly about another company
- the concrete funding/product/traction/partnership event is owned by another company, startup, customer, or competitor
- the target company is only mentioned in passing, comparison, broad market context, or background
- the chunk has no explicit AI signal
- the chunk is too vague, generic, promotional, repetitive, or ambiguous
- no clear AI business event is described
- event_type is unknown
- evidence is too weak to be useful in downstream ISS scoring

================================================================
OUTPUT RULES
================================================================
- Return exactly one JSON object containing ONLY the 9 target fields listed below.
- Do NOT return the known input fields (chunk_id, source_type, etc.) — they are merged back in code.
- development_scope must be one of: Vertical AI, Horizontal AI, unknown
- agent_architecture must be one of: single-agent AI, multi-agent AI, unknown
- sub_theme is tied to development_scope, not agent_architecture.
- If development_scope is Vertical AI, sub_theme must come from the allowed sub_theme values for Vertical AI.
- If development_scope is Horizontal AI, sub_theme must come from the allowed sub_theme values for Horizontal AI.
- If development_scope is unknown, sub_theme must be ["unknown"].
- For confidence, choose only: high, medium, low.
- For strength_1_5, return only an integer 1, 2, 3, 4, or 5.
- evidence_span must come from the chunk.
- If uncertain, prefer unknown rather than over-claiming.
- If multiple event types appear, choose the single most central event type based on the main business action emphasized in the chunk.
- If is_usable_for_iss = false, development_scope must be "unknown", agent_architecture must be "unknown", sub_theme must be ["unknown"], event_type must be "unknown", and strength_1_5 must be 1.
- If is_usable_for_iss = false, reason_short must only explain the exclusion reason and must not describe the chunk as fitting any AI scope, sub_theme, or event_type.

================================================================
OUTPUT SCHEMA
================================================================
{{
  "development_scope": "",
  "agent_architecture": "",
  "sub_theme": ["unknown"],
  "event_type": "",
  "strength_1_5": 0,
  "evidence_span": "",
  "reason_short": "",
  "confidence": "",
  "is_usable_for_iss": true
}}
""".strip()


_VERTICAL_INDUSTRY_LIST = ", ".join(sorted(VERTICAL_INDUSTRY_VOCAB - {"unknown"}))
_HORIZONTAL_FUNCTION_LIST = ", ".join(sorted(HORIZONTAL_FUNCTION_VOCAB - {"unknown"}))
_SUB_THEME_LIST = ", ".join(sorted(SUB_THEME_VOCAB - {"unknown"}))


def format_subtheme_definitions(vocab: Set[str]) -> str:
    """Format sub-theme definitions for prompt injection."""
    lines = []
    for tag in sorted(vocab - {"unknown"}):
        definition = SUBTHEME_DEFINITIONS.get(tag, "")
        lines.append(f"- {tag}: {definition}")
    return "\n".join(lines)


_VERTICAL_SUBTHEME_DEFINITIONS = format_subtheme_definitions(VERTICAL_INDUSTRY_VOCAB)
_HORIZONTAL_SUBTHEME_DEFINITIONS = format_subtheme_definitions(HORIZONTAL_FUNCTION_VOCAB)

USER_PROMPT_TEMPLATE = """
chunk_id: {chunk_id}
source_type: {source_type}
source_name: {source_name}
source_title: {source_title}
source_url: {source_url}
document_date: {document_date}
company: {company}

chunk_text:
{chunk_text}

---
Allowed sub_theme values for Vertical AI:
""" + _VERTICAL_INDUSTRY_LIST + """

Vertical AI sub_theme definitions:
""" + _VERTICAL_SUBTHEME_DEFINITIONS + """

Allowed sub_theme values for Horizontal AI:
""" + _HORIZONTAL_FUNCTION_LIST + """

Horizontal AI sub_theme definitions:
""" + _HORIZONTAL_SUBTHEME_DEFINITIONS + """

All allowed sub_theme values:
""" + _SUB_THEME_LIST + """

Important sub_theme rule:
If no sub_theme accurately matches the core AI use case, choose ["unknown"] instead of forcing the closest available label.
"""

USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE.strip()


# =========================
# Helpers
# =========================

def safe_str(value: Any) -> str:
    """Convert value to string, handling None and NaN."""
    if value is None:
        return ""
    if isinstance(value, float):
        import math
        if math.isnan(value):
            return ""
    return str(value)


def load_records(file_path: str) -> List[Dict[str, Any]]:
    """Load records from JSONL."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record: Dict[str, Any], file_path: str) -> None:
    """Append one record to JSONL."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_chunk_ids(output_file: str) -> Set[str]:
    """Load chunk_ids that are already in the output file for resume."""
    done = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        cid = rec.get("chunk_id", "")
                        if cid:
                            done.add(cid)
                    except json.JSONDecodeError:
                        continue
    return done


# =========================
# Evidence span verification
# =========================

def verify_evidence_span(evidence_span: str, chunk_text: str) -> bool:
    """
    Check if evidence_span is a substring of chunk_text.

    Strategy:
    1. Whitespace-normalized, case-insensitive exact substring match.
    2. Sliding-window fuzzy match with SequenceMatcher ratio > 0.85
       to tolerate minor model formatting differences.

    Returns True if the span is verified.
    """
    if not evidence_span or not chunk_text:
        return False

    norm_span = " ".join(evidence_span.lower().split())
    norm_chunk = " ".join(chunk_text.lower().split())

    if norm_span in norm_chunk:
        return True

    span_len = len(norm_span)
    if span_len == 0:
        return False

    step = max(1, span_len // 5)
    for i in range(0, max(1, len(norm_chunk) - span_len + 1), step):
        window = norm_chunk[i: i + span_len + 20]
        if SequenceMatcher(None, norm_span, window).ratio() > 0.85:
            return True

    return False


# =========================
# Deterministic Post-processing Guards
# =========================

def has_explicit_ai_signal(text: str) -> bool:
    """
    Return True if text contains an explicit AI-related signal.

    This intentionally ignores generic technology language such as smart, cloud,
    platform, data, digital, sustainability, clean energy, or automation unless
    explicit AI/ML/model/agent language is present.
    """
    text_l = f" {safe_str(text).lower()} "
    return any(keyword in text_l for keyword in HARD_AI_SIGNAL_KEYWORDS)


def evidence_span_has_explicit_ai_signal(validated: Dict[str, Any]) -> bool:
    """
    Require the selected evidence_span itself to contain a hard AI signal.
    This prevents clean energy, smart grid, sustainability, generic cloud/platform,
    and other non-AI business events from entering ISS.

    Exceptions:
    - NVIDIA infra/autonomy terms such as H100, Blackwell, Triton, TensorRT,
      MLPerf, Isaac AMR, autonomy, robotics are treated as AI/physical-AI signals.
    """
    evidence = safe_str(validated.get("evidence_span", ""))
    chunk_text = safe_str(validated.get("chunk_text", ""))
    source_title = safe_str(validated.get("source_title", ""))
    company = safe_str(validated.get("company", ""))

    evidence_title = f"{evidence} {source_title}"

    if company == "NVIDIA" and (
        contains_any_keyword(evidence_title, NVIDIA_INFRA_TERMS)
        or contains_any_keyword(evidence_title, NVIDIA_AUTONOMY_TERMS)
    ):
        return True

    # Prefer evidence_span. If it is empty or very short, fall back to chunk/title.
    if len(evidence.strip()) >= 20:
        return has_explicit_ai_signal(evidence)

    return has_explicit_ai_signal(f"{source_title} {chunk_text}")


def mentions_target_company(text: str, company: str) -> bool:
    """Check whether text mentions the target company or its aliases."""
    text_l = safe_str(text).lower()
    company = safe_str(company).strip()

    aliases = KNOWN_COMPANY_ALIASES.get(company, {company.lower()})
    return any(alias.lower() in text_l for alias in aliases)


def likely_other_company_main_actor(text: str, company: str) -> bool:
    """
    Conservative heuristic for company/source mismatch.

    If another major company appears early in the chunk while the target company
    does not appear early, the main event is likely about another company.

    Example:
    company = Microsoft
    chunk starts with AWS launching a hallucination-checking service
    Microsoft appears later only as a comparison
    => likely mismatch.
    """
    text_l = safe_str(text).lower()
    company = safe_str(company).strip()

    company_aliases = KNOWN_COMPANY_ALIASES.get(company, {company.lower()})
    first_500 = text_l[:500]

    target_in_first_500 = any(alias.lower() in first_500 for alias in company_aliases)
    other_in_first_500 = any(other in first_500 for other in OTHER_MAJOR_COMPANIES)

    if other_in_first_500 and not target_in_first_500:
        return True

    return False


LOW_QUALITY_URL_PATTERNS = {
    "/author/",
    "/tag/",
    "/tags/",
    "/category/",
    "/categories/",
    "/search/",
    "/page/",
    "/topic/",
    "/topics/",
    "/archive/",
    "/archives/",
}

POST_LIST_SIGNALS = {
    "author",
    "archives",
    "archive",
    "tagged",
    "posts by",
    "all posts",
    "latest posts",
    "read more",
    "previous post",
    "next post",
    "browse",
    "category",
}

def is_low_quality_source_page(record: Dict[str, Any]) -> bool:
    """
    Detect author/tag/category/archive/search/index pages.

    These pages often contain multiple post snippets or author bios, so they are
    not a single auditable ISS event.
    """
    source_url = safe_str(record.get("source_url", "")).lower()
    source_title = safe_str(record.get("source_title", "")).lower()
    chunk_text = safe_str(record.get("chunk_text", "")).lower()

    # URL-based exclusion is the strongest signal.
    if any(pattern in source_url for pattern in LOW_QUALITY_URL_PATTERNS):
        return True

    # Common title patterns for non-article pages.
    if source_title.startswith("author:") or source_title.startswith("tag:"):
        return True

    # Person-name author pages often have short titles and URL author paths,
    # already covered by URL; keep this conservative to avoid false positives.
    first_500 = chunk_text[:500]
    if any(signal in first_500 for signal in POST_LIST_SIGNALS):
        # Require list-like behavior too.
        list_like_count = sum(chunk_text.count(marker) for marker in ["...", "read more", "posted", "by "])
        if list_like_count >= 3:
            return True

    return False


# Extra company/startup names that frequently appear as non-target concrete event owners.
# This list is intentionally editable and conservative.
OTHER_EVENT_OWNER_NAMES = {
    "onton",
    "openai",
    "google",
    "alphabet",
    "microsoft",
    "azure",
    "nvidia",
    "amazon",
    "aws",
    "amazon web services",
    "meta",
    "facebook",
    "anthropic",
    "apple",
    "oracle",
    "salesforce",
    "databricks",
    "snowflake",
    "mistral",
    "cohere",
    "perplexity",
    "xai",
    "tesla",
    "snap",
    "revolut",
}

CONCRETE_EVENT_VERBS = {
    "raised",
    "raises",
    "raise",
    "funding",
    "fundraise",
    "launched",
    "launches",
    "launch",
    "released",
    "releases",
    "announced",
    "announces",
    "acquired",
    "acquires",
    "acquisition",
    "invested",
    "invests",
    "investment",
    "partnered",
    "partners",
    "partnership",
    "introduced",
    "introduces",
    "rolls out",
    "rolled out",
    "is launching",
    "is releasing",
    "unveiled",
    "debuts",
    "topped",
    "tops",
    "ranked",
    "ranks",
    "led",
    "leads",
}

def get_target_aliases(company: str) -> Set[str]:
    """Return aliases for a target company."""
    company = safe_str(company).strip()
    return {a.lower() for a in KNOWN_COMPANY_ALIASES.get(company, {company.lower()})}


def get_non_target_company_aliases(company: str) -> Set[str]:
    """Aliases for known companies other than the current target company."""
    aliases = set(OTHER_EVENT_OWNER_NAMES) | set(OTHER_MAJOR_COMPANIES)
    for known_company, known_aliases in KNOWN_COMPANY_ALIASES.items():
        if known_company != company:
            aliases.update(known_aliases)
    return {safe_str(a).lower() for a in aliases if safe_str(a).strip()}


def contains_concrete_event_verb(text: str) -> bool:
    """Return True if text contains a concrete event/action verb."""
    text_l = safe_str(text).lower()
    return any(v in text_l for v in CONCRETE_EVENT_VERBS)


def likely_other_company_owns_evidence(validated: Dict[str, Any]) -> bool:
    """
    Detect cases where the selected evidence_span describes another company's concrete event.

    This catches cases like:
    company = OpenAI, but evidence_span = Onton raised $7.5M...
    """
    company = safe_str(validated.get("company", "")).strip()
    title = safe_str(validated.get("source_title", ""))
    evidence = safe_str(validated.get("evidence_span", ""))
    reason = safe_str(validated.get("reason_short", ""))
    combined = f"{title} {evidence} {reason}".lower()

    if not company or not contains_concrete_event_verb(combined):
        return False

    target_aliases = get_target_aliases(company)
    target_hits = {alias for alias in target_aliases if alias in combined}
    target_in_evidence = bool(target_hits)
    weak_target_hits = WEAK_TARGET_ALIASES.get(company, set()) & target_hits
    strong_target_in_evidence = bool(target_hits - weak_target_hits)

    # "NVIDIA-backed startup" / "backed by OpenAI" describes the portfolio company
    # as the event owner, not the target company. Do not count these as target-company
    # product or traction signals.
    for alias in target_aliases:
        if f"{alias}-backed" in combined or f"backed by {alias}" in combined:
            return True

    # Find other company names in evidence/reason.
    other_names = set()
    for name in get_non_target_company_aliases(company):
        n = name.lower()
        if n in combined and n not in target_aliases:
            other_names.add(n)

    if other_names and target_in_evidence and not strong_target_in_evidence:
        return True

    # If another company is mentioned in the concrete evidence and the target is not,
    # the event is likely owned by the other company.
    if other_names and not target_in_evidence:
        return True

    earliest_target = min([combined.find(alias) for alias in target_aliases if alias in combined] or [-1])
    earliest_other = min([combined.find(other) for other in other_names if other in combined] or [-1])
    if (
        earliest_other >= 0
        and earliest_target >= 0
        and earliest_other + 80 < earliest_target
        and contains_any_keyword(combined, RANKING_TRACTION_TERMS)
    ):
        return True

    # Special case: evidence starts with another company/entity before mentioning target.
    first_160 = combined[:160]
    if other_names and not any(alias in first_160 for alias in target_aliases):
        for other in other_names:
            if other in first_160:
                return True

    return False


def infer_best_exclusion_reason(validated: Dict[str, Any]) -> str:
    """
    Create a consistent exclusion reason based on deterministic checks.
    Used to avoid reason_short contradicting is_usable_for_iss=false.
    """
    chunk_text = validated.get("chunk_text", "")
    company = validated.get("company", "")

    if is_low_quality_source_page(validated):
        return "Excluded from ISS because the source appears to be an author, tag, archive, search, or index page rather than a standalone business event."

    if company and likely_other_company_owns_evidence(validated):
        return f"Excluded from ISS because the concrete event in the evidence appears to be owned by another company, not the target company {company}."

    if company and likely_other_company_main_actor(chunk_text, company):
        return f"Excluded from ISS because the main AI/business event appears to be about another company, not the target company {company}."

    if not has_explicit_ai_signal(chunk_text):
        return "Excluded from ISS because the chunk is about the target company or its products but does not contain an explicit AI-related business signal."

    if not evidence_span_has_explicit_ai_signal(validated):
        return "Excluded from ISS because the supporting evidence does not contain an explicit AI/ML/model/agent signal."

    if company and not mentions_target_company(chunk_text, company):
        return f"Excluded from ISS because the chunk does not clearly mention the target company {company} as an actor."

    if validated.get("event_type") == "unknown":
        return "Excluded from ISS because no clear ISS event type is supported."

    if validated.get("development_scope") == "unknown":
        return "Excluded from ISS because no clear Vertical AI or Horizontal AI scope is supported."

    return "Excluded from ISS because the chunk is not usable for ISS scoring."



def force_not_usable(validated: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Force a record to be excluded from ISS scoring."""
    validated["development_scope"] = "unknown"
    validated["agent_architecture"] = "unknown"
    validated["sub_theme"] = ["unknown"]
    validated["event_type"] = "unknown"
    validated["strength_1_5"] = 1
    validated["is_usable_for_iss"] = False
    validated["confidence"] = "high"
    validated["reason_short"] = reason
    return validated


def postprocess_guard(validated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply deterministic guards after LLM labeling.

    This reduces:
    1. over-labeling non-AI/general-tech chunks as Horizontal or Vertical AI;
    2. company/source mismatch, where another company's event is counted
       under the target company.
    """
    chunk_text = validated.get("chunk_text", "")
    company = validated.get("company", "")

    # Guard 0: exclude low-quality/index pages such as author, tag, category, archive, search, or post-list pages.
    if is_low_quality_source_page(validated):
        return force_not_usable(
            validated,
            "Excluded from ISS because the source appears to be an author, tag, archive, search, or index page rather than a standalone business event."
        )

    # Guard 1: require explicit AI signal in the chunk.
    if not has_explicit_ai_signal(chunk_text):
        return force_not_usable(
            validated,
            "Excluded from ISS because the chunk does not contain an explicit AI-related business signal."
        )

    # Guard 1b: require explicit AI signal in the evidence_span itself.
    # This prevents clean energy / smart grid / sustainability / generic platform events from entering ISS.
    if not evidence_span_has_explicit_ai_signal(validated):
        return force_not_usable(
            validated,
            "Excluded from ISS because the supporting evidence does not contain an explicit AI/ML/model/agent signal."
        )

    # Guard 2: require target company presence.
    # This is conservative: if the chunk does not mention the target company or aliases,
    # it should not contribute to that company's ISS.
    if company and not mentions_target_company(chunk_text, company):
        return force_not_usable(
            validated,
            f"Excluded from ISS because the chunk does not clearly mention the target company {company} as an actor."
        )

    # Guard 3: evidence/concrete event owner must be the target company.
    if company and likely_other_company_owns_evidence(validated):
        return force_not_usable(
            validated,
            f"Excluded from ISS because the concrete event in the evidence appears to be owned by another company, not the target company {company}."
        )

    # Guard 4: likely another-company main actor.
    if company and likely_other_company_main_actor(chunk_text, company):
        return force_not_usable(
            validated,
            f"Excluded from ISS because the main AI/business event appears to be about another company, not the target company {company}."
        )

    # Guard 5: if LLM says not usable, make the label fields and reason consistent.
    if not validated.get("is_usable_for_iss", False):
        return force_not_usable(
            validated,
            infer_best_exclusion_reason(validated)
        )

    # Guard 5: if event_type is unknown, do not use it for ISS.
    if validated.get("event_type") == "unknown":
        return force_not_usable(
            validated,
            "Excluded from ISS because no clear ISS event type is supported."
        )

    # Guard 6: if development scope is unknown, do not use it for ISS.
    if validated.get("development_scope") == "unknown":
        return force_not_usable(
            validated,
            "Excluded from ISS because no clear Vertical AI or Horizontal AI scope is supported."
        )

    # Guard 7: normalize general assistant features such as ChatGPT shopping to Horizontal AI.
    validated = normalize_general_assistant_scope(validated)

    # Guard 8: align labels/reason to the selected evidence_span in long chunks.
    validated = normalize_labels_to_evidence_span(validated)

    # Guard 9: recover OpenAI single-agent architecture for clear ChatGPT / Operator / Codex product forms.
    validated = normalize_openai_agent_architecture(validated)

    # Guard 10: normalize NVIDIA provider infrastructure / autonomy.
    validated = normalize_nvidia_provider_infra_scope(validated)

    # Guard 11: conservative strength calibration.
    validated = calibrate_strength(validated)

    return validated


# =========================
# Validation
# =========================

def normalize_vocab_value(raw: str) -> str:
    """Normalize a vocab value: lowercase, strip IND_ prefix."""
    v = raw.strip().lower()
    if v.startswith("ind_"):
        v = v[4:]
    return v


def validate_list_field(raw_value: Any, vocab: Set[str]) -> List[str]:
    """Validate a list-type field against a controlled vocabulary."""
    if raw_value is None:
        return ["unknown"]

    if isinstance(raw_value, str):
        raw_value = [raw_value]

    if not isinstance(raw_value, list):
        return ["unknown"]

    validated = []
    for item in raw_value:
        normed = normalize_vocab_value(str(item))
        if normed in vocab:
            validated.append(normed)

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for item in validated:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped if deduped else ["unknown"]


def validate_output(output: Dict[str, Any], original: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize model output, then merge with original known fields.
    Includes chunk_text for traceability.
    """
    validated = {
        "chunk_id": safe_str(original.get("chunk_id", "")),
        "source_type": safe_str(original.get("source_type", "")),
        "source_name": safe_str(original.get("source_name", "")),
        "source_title": safe_str(original.get("source_title", "")),
        "source_url": safe_str(original.get("source_url", "")),
        "document_date": safe_str(original.get("document_date", "")),
        "company": safe_str(original.get("company", "")),
        "chunk_text": safe_str(original.get("chunk_text", "")),
    }

    # development_scope
    development_scope = safe_str(output.get("development_scope", "unknown"))
    validated["development_scope"] = (
        development_scope if development_scope in VALID_DEVELOPMENT_SCOPES else "unknown"
    )

    # agent_architecture
    agent_architecture = safe_str(output.get("agent_architecture", "unknown"))
    validated["agent_architecture"] = (
        agent_architecture if agent_architecture in VALID_AGENT_ARCHITECTURES else "unknown"
    )

    # sub_theme is tied to development_scope, not agent_architecture.
    if validated["development_scope"] == "Vertical AI":
        validated["sub_theme"] = validate_list_field(
            output.get("sub_theme"),
            VERTICAL_INDUSTRY_VOCAB,
        )
    elif validated["development_scope"] == "Horizontal AI":
        validated["sub_theme"] = validate_list_field(
            output.get("sub_theme"),
            HORIZONTAL_FUNCTION_VOCAB,
        )
    else:
        validated["sub_theme"] = ["unknown"]

    # event_type
    event_type = safe_str(output.get("event_type", "unknown"))
    validated["event_type"] = event_type if event_type in VALID_EVENT_TYPES else "unknown"

    # strength_1_5
    try:
        strength = int(output.get("strength_1_5", 1))
        validated["strength_1_5"] = max(1, min(5, strength))
    except (ValueError, TypeError):
        validated["strength_1_5"] = 1

    # evidence_span + verification
    evidence_span = safe_str(output.get("evidence_span", ""))
    validated["evidence_span"] = evidence_span
    validated["evidence_span_verified"] = verify_evidence_span(
        evidence_span,
        validated["chunk_text"],
    )

    # reason_short
    validated["reason_short"] = safe_str(output.get("reason_short", ""))

    # confidence
    confidence = safe_str(output.get("confidence", "low"))
    validated["confidence"] = confidence if confidence in VALID_CONFIDENCE else "low"

    # is_usable_for_iss
    raw_usable = output.get("is_usable_for_iss", False)
    if isinstance(raw_usable, bool):
        validated["is_usable_for_iss"] = raw_usable
    elif isinstance(raw_usable, str):
        validated["is_usable_for_iss"] = raw_usable.strip().lower() == "true"
    else:
        validated["is_usable_for_iss"] = bool(raw_usable)

    # If LLM says not usable, force consistency before postprocess guard.
    if not validated["is_usable_for_iss"]:
        validated["development_scope"] = "unknown"
        validated["agent_architecture"] = "unknown"
        validated["sub_theme"] = ["unknown"]
        validated["event_type"] = "unknown"
        validated["strength_1_5"] = 1

    # Deterministic post-processing guard.
    validated = postprocess_guard(validated)

    return validated


# =========================
# Batch API helpers
# =========================

def build_user_prompt(record: Dict[str, Any]) -> str:
    """Build the user message with one record's metadata + chunk text + vocab lists."""
    return USER_PROMPT_TEMPLATE.format(
        chunk_id=safe_str(record.get("chunk_id", "")),
        source_type=safe_str(record.get("source_type", "")),
        source_name=safe_str(record.get("source_name", "")),
        source_title=safe_str(record.get("source_title", "")),
        source_url=safe_str(record.get("source_url", "")),
        document_date=safe_str(record.get("document_date", "")),
        company=safe_str(record.get("company", "")),
        chunk_text=safe_str(record.get("chunk_text", "")),
    )


def build_batch_request(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build one line of the batch JSONL in OpenAI Batch API format."""
    user_prompt = build_user_prompt(record)

    return {
        "custom_id": record["chunk_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_NAME,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model output.
    Handles cases where the model wraps JSON in extra text.
    """
    import re

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")

    return json.loads(match.group(0))


# =========================
# Submit subcommand
# =========================

def cmd_submit(args):
    """Submit a batch of labeling requests to OpenAI."""
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    state_path = script_dir / BATCH_STATE_FILE

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    records = load_records(str(input_path))
    print(f"Loaded {len(records)} records from {input_path.name}")

    done_ids = load_done_chunk_ids(str(output_path))
    if done_ids:
        print(f"Resuming: {len(done_ids)} chunk_ids already in {output_path.name}, skipping.")

    to_label = [r for r in records if r.get("chunk_id", "") not in done_ids]
    print(f"Records to label: {len(to_label)}")

    if not to_label:
        print("Nothing to submit — all records already labeled.")
        return

    RECORDS_PER_SHARD = 100
    shards = []
    for shard_start in range(0, len(to_label), RECORDS_PER_SHARD):
        shard = to_label[shard_start:shard_start + RECORDS_PER_SHARD]
        shards.append(shard)

    print(f"Split into {len(shards)} shard(s) of up to {RECORDS_PER_SHARD} records each.")

    oai = get_client()
    batch_ids = []

    for shard_idx, shard in enumerate(shards):
        shard_num = shard_idx + 1
        shard_file = script_dir / f"_batch_requests_shard_{shard_num}.jsonl"

        with open(shard_file, "w", encoding="utf-8") as f:
            for record in shard:
                req = build_batch_request(record)
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        file_size_mb = shard_file.stat().st_size / (1024 * 1024)
        print(f"\nShard {shard_num}/{len(shards)}: {len(shard)} records, {file_size_mb:.1f} MB")

        with open(shard_file, "rb") as f:
            uploaded = oai.files.create(file=f, purpose="batch")
        print(f"  Uploaded file: {uploaded.id}")

        batch = oai.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_ids.append(batch.id)

        print(f"  Batch ID:   {batch.id}")
        print(f"  Status:     {batch.status}")

        shard_file.unlink(missing_ok=True)

    print(f"\n{'=' * 50}")
    print(f"All {len(shards)} batch(es) submitted successfully!")
    print(f"Total records: {len(to_label)}")

    state = {
        "batch_ids": batch_ids,
        "input_file": args.input,
        "output_file": args.output,
        "records_submitted": len(to_label),
        "shard_count": len(shards),
    }

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"\nBatch state saved to {state_path.name}")

    if len(batch_ids) == 1:
        print(f"\nNext step: python3 iss_label.py collect {batch_ids[0]}")
    else:
        print("\nNext step: collect each batch separately:")
        for bid in batch_ids:
            print(f"  python3 iss_label.py collect {bid} --input {args.input} --output {args.output}")


# =========================
# Collect subcommand
# =========================

def cmd_collect(args):
    """Poll for batch completion and collect results."""
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    error_path = script_dir / ERROR_FILE
    batch_id = args.batch_id

    oai = get_client()

    print(f"Polling batch {batch_id}...")
    while True:
        batch = oai.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts

        print(
            f"  Status: {status}  |  "
            f"completed: {counts.completed}  "
            f"failed: {counts.failed}  "
            f"total: {counts.total}"
        )

        if status in ("completed", "failed", "expired", "cancelled"):
            break

        time.sleep(30)

    if status != "completed":
        print(f"\nBatch ended with status: {status}")
        if batch.errors and batch.errors.data:
            for err in batch.errors.data:
                print(f"  Error: {err.message}")
        sys.exit(1)

    input_records = {}
    if input_path.exists():
        for rec in load_records(str(input_path)):
            cid = rec.get("chunk_id", "")
            if cid:
                input_records[cid] = rec

    result_file_id = batch.output_file_id
    if not result_file_id:
        print("Error: No output file ID in batch response.")
        sys.exit(1)

    print(f"\nDownloading results from {result_file_id}...")
    result_content = oai.files.content(result_file_id).content
    result_lines = result_content.decode("utf-8").strip().split("\n")
    print(f"Received {len(result_lines)} result lines.")

    success_count = 0
    error_count = 0
    span_verified_count = 0
    span_unverified_count = 0
    not_usable_count = 0

    for line in result_lines:
        if not line.strip():
            continue

        result = json.loads(line)
        custom_id = result.get("custom_id", "")
        result_body = result.get("response", {}).get("body", {})
        result_status = result.get("response", {}).get("status_code", 0)

        if result.get("error") or result_status != 200:
            error_record = {
                "chunk_id": custom_id,
                "_labeling_error": result.get("error", {}).get("message", f"HTTP {result_status}"),
                "_raw_response": result,
            }

            if custom_id in input_records:
                error_record.update(input_records[custom_id])
                error_record["_labeling_error"] = result.get("error", {}).get(
                    "message",
                    f"HTTP {result_status}",
                )

            append_jsonl(error_record, str(error_path))
            error_count += 1
            continue

        raw_text = ""
        try:
            choices = result_body.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            raw_text = choices[0].get("message", {}).get("content", "")
            model_output = extract_json(raw_text)
        except Exception as e:
            error_record = {
                "chunk_id": custom_id,
                "_labeling_error": f"JSON parse failed: {e}",
                "_raw_text": raw_text,
            }

            if custom_id in input_records:
                error_record.update(input_records[custom_id])
                error_record["_labeling_error"] = f"JSON parse failed: {e}"

            append_jsonl(error_record, str(error_path))
            error_count += 1
            continue

        original = input_records.get(custom_id, {"chunk_id": custom_id})
        validated = validate_output(model_output, original)

        if validated.get("evidence_span_verified"):
            span_verified_count += 1
        else:
            span_unverified_count += 1

        if not validated.get("is_usable_for_iss", False):
            not_usable_count += 1

        append_jsonl(validated, str(output_path))
        success_count += 1

    print(f"\n{'=' * 50}")
    print("COLLECTION SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Succeeded:   {success_count}")
    print(f"  Errors:      {error_count}")
    print(f"  Not usable for ISS:       {not_usable_count}")
    print(f"  Evidence span verified:   {span_verified_count}")
    print(f"  Evidence span unverified: {span_unverified_count}")
    print(f"  Output: {output_path}")

    if error_count > 0:
        print(f"  Errors: {error_path}")

    print()


# =========================
# Collect-all subcommand
# =========================

def cmd_collect_all(args):
    """Collect results from ALL batches saved in _batch_state.json."""
    script_dir = Path(__file__).parent
    state_path = script_dir / BATCH_STATE_FILE

    if not state_path.exists():
        print(f"Error: {state_path.name} not found. Run 'submit' first.")
        sys.exit(1)

    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    batch_ids = state.get("batch_ids", [])
    if not batch_ids and state.get("batch_id"):
        batch_ids = [state["batch_id"]]

    if not batch_ids:
        print("Error: No batch IDs found in state file.")
        sys.exit(1)

    input_file = args.input or state.get("input_file", DEFAULT_INPUT_FILE)
    output_file = args.output or state.get("output_file", DEFAULT_OUTPUT_FILE)

    print(f"Found {len(batch_ids)} batch(es) in {state_path.name}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()

    total_success = 0
    total_errors = 0

    for i, batch_id in enumerate(batch_ids, 1):
        print(f"{'=' * 50}")
        print(f"Collecting batch {i}/{len(batch_ids)}: {batch_id}")
        print(f"{'=' * 50}")

        collect_args = argparse.Namespace(
            batch_id=batch_id,
            input=input_file,
            output=output_file,
        )

        try:
            cmd_collect(collect_args)
        except SystemExit:
            print(f"  WARNING: Batch {batch_id} failed. Continuing with next batch...")
            total_errors += 1
            continue

        total_success += 1

    print(f"\n{'=' * 50}")
    print("ALL BATCHES COLLECTION COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Batches succeeded: {total_success}/{len(batch_ids)}")

    if total_errors > 0:
        print(f"  Batches failed:    {total_errors}")

    print(f"  Output: {output_file}")


# =========================
# Run-all subcommand
# =========================

def cmd_run_all(args):
    """Submit, wait, and collect one shard at a time to avoid token limits."""
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    error_path = script_dir / ERROR_FILE

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    records = load_records(str(input_path))
    print(f"Loaded {len(records)} records from {input_path.name}")

    done_ids = load_done_chunk_ids(str(output_path))
    if done_ids:
        print(f"Resuming: {len(done_ids)} chunk_ids already in {output_path.name}, skipping.")

    to_label = [r for r in records if r.get("chunk_id", "") not in done_ids]
    print(f"Records to label: {len(to_label)}")

    if not to_label:
        print("Nothing to do — all records already labeled.")
        return

    RECORDS_PER_SHARD = 100
    shards = []
    for start in range(0, len(to_label), RECORDS_PER_SHARD):
        shards.append(to_label[start:start + RECORDS_PER_SHARD])

    print(f"Split into {len(shards)} shard(s) of up to {RECORDS_PER_SHARD} records each.\n")

    oai = get_client()

    input_records = {}
    for rec in records:
        cid = rec.get("chunk_id", "")
        if cid:
            input_records[cid] = rec

    grand_success = 0
    grand_errors = 0
    grand_not_usable = 0

    for shard_idx, shard in enumerate(shards):
        shard_num = shard_idx + 1
        shard_file = script_dir / f"_batch_requests_shard_{shard_num}.jsonl"

        print(f"{'=' * 60}")
        print(f"SHARD {shard_num}/{len(shards)}  ({len(shard)} records)")
        print(f"{'=' * 60}")

        with open(shard_file, "w", encoding="utf-8") as f:
            for record in shard:
                req = build_batch_request(record)
                f.write(json.dumps(req, ensure_ascii=False) + "\n")

        file_size_mb = shard_file.stat().st_size / (1024 * 1024)
        print(f"  Wrote shard file: {file_size_mb:.1f} MB")

        with open(shard_file, "rb") as f:
            uploaded = oai.files.create(file=f, purpose="batch")
        print(f"  Uploaded: {uploaded.id}")

        batch = oai.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Batch ID: {batch.id}  Status: {batch.status}")

        shard_file.unlink(missing_ok=True)

        print("  Waiting for completion...")
        while True:
            batch = oai.batches.retrieve(batch.id)
            status = batch.status
            counts = batch.request_counts

            print(
                f"    {status}  |  "
                f"done={counts.completed}  "
                f"failed={counts.failed}  "
                f"total={counts.total}"
            )

            if status in ("completed", "failed", "expired", "cancelled"):
                break

            time.sleep(30)

        if status != "completed":
            print(f"  ERROR: Shard {shard_num} ended with status: {status}")
            if batch.errors and batch.errors.data:
                for err in batch.errors.data:
                    print(f"    {err.code}: {err.message}")
            grand_errors += len(shard)
            continue

        result_file_id = batch.output_file_id
        if not result_file_id:
            print(f"  ERROR: No output file ID. Skipping shard {shard_num}.")
            grand_errors += len(shard)
            continue

        print(f"  Downloading results from {result_file_id}...")
        result_content = oai.files.content(result_file_id).content
        result_lines = result_content.decode("utf-8").strip().split("\n")

        shard_success = 0
        shard_errors = 0
        shard_not_usable = 0

        for line in result_lines:
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            result_body = result.get("response", {}).get("body", {})
            result_status = result.get("response", {}).get("status_code", 0)

            if result.get("error") or result_status != 200:
                error_record = {
                    "chunk_id": custom_id,
                    "_labeling_error": result.get("error", {}).get("message", f"HTTP {result_status}"),
                }

                if custom_id in input_records:
                    error_record.update(input_records[custom_id])
                    error_record["_labeling_error"] = result.get("error", {}).get(
                        "message",
                        f"HTTP {result_status}",
                    )

                append_jsonl(error_record, str(error_path))
                shard_errors += 1
                continue

            raw_text = ""
            try:
                choices = result_body.get("choices", [])
                if not choices:
                    raise ValueError("No choices in response")
                raw_text = choices[0].get("message", {}).get("content", "")
                model_output = extract_json(raw_text)
            except Exception as e:
                error_record = {
                    "chunk_id": custom_id,
                    "_labeling_error": f"JSON parse failed: {e}",
                    "_raw_text": raw_text,
                }

                if custom_id in input_records:
                    error_record.update(input_records[custom_id])
                    error_record["_labeling_error"] = f"JSON parse failed: {e}"

                append_jsonl(error_record, str(error_path))
                shard_errors += 1
                continue

            original = input_records.get(custom_id, {"chunk_id": custom_id})
            validated = validate_output(model_output, original)

            if not validated.get("is_usable_for_iss", False):
                shard_not_usable += 1

            append_jsonl(validated, str(output_path))
            shard_success += 1

        print(
            f"  Shard {shard_num} done: "
            f"{shard_success} succeeded, "
            f"{shard_errors} errors, "
            f"{shard_not_usable} not usable for ISS"
        )

        grand_success += shard_success
        grand_errors += shard_errors
        grand_not_usable += shard_not_usable

    print(f"\n{'=' * 60}")
    print("ALL SHARDS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total succeeded:        {grand_success}")
    print(f"  Total errors:           {grand_errors}")
    print(f"  Total not usable ISS:   {grand_not_usable}")
    print(f"  Output: {output_path}")


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="BGV ISS Chunk Labeling with OpenAI Batch API",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    sub = subparsers.add_parser("submit", help="Submit a labeling batch to OpenAI")
    sub.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSONL file")
    sub.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSONL file for resume check")
    sub.set_defaults(func=cmd_submit)

    col = subparsers.add_parser("collect", help="Collect results from a completed batch")
    col.add_argument("batch_id", help="Batch ID from submit step")
    col.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSONL file for merging")
    col.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSONL file")
    col.set_defaults(func=cmd_collect)

    ca = subparsers.add_parser("collect-all", help="Collect ALL batches from _batch_state.json")
    ca.add_argument("--input", default=None, help="Input JSONL file, overrides state")
    ca.add_argument("--output", default=None, help="Output JSONL file, overrides state")
    ca.set_defaults(func=cmd_collect_all)

    ra = subparsers.add_parser(
        "run-all",
        help="Submit and collect all shards one at a time",
    )
    ra.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSONL file")
    ra.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSONL file")
    ra.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
