import argparse
import ast
import json
import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from openai import OpenAI

from paths import (
    BASE_DIR,
    FILE_ALL_CHUNKS,
    FILE_GAPS_EXTRACTION_ERRORS
)


# Auto-load .env from project root
_env_path = BASE_DIR / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())


# =========================
# Configuration
# =========================

DEFAULT_INPUT_FILE = str(FILE_ALL_CHUNKS)
DEFAULT_OUTPUT_FILE = str(BASE_DIR / "data" / "input" / "chunks_labeled_output.jsonl")
ERROR_FILE = str(FILE_GAPS_EXTRACTION_ERRORS)
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
    "healthcare", "life_sciences", "biotech_pharma", "materials_science",
    "manufacturing", "energy_oil_gas", "clean_energy", "automotive",
    "financial_services", "insurance", "construction", "logistics", "retail",
    "food_service", "utilities", "waste_management", "medical_devices",
    "industrial_inspection", "telecommunications", "public_sector", "education",
    "legal", "media_digital", "gaming", "ecommerce", "internet_platform",
    "agriculture", "cybersecurity", "sustainability", "unknown",
}

HORIZONTAL_FUNCTION_VOCAB = {
    "security", "cloud_infra", "ai_infra", "model_ops", "data_analytics",
    "sales_marketing", "advertising", "dev_tools", "productivity_office",
    "search_discovery", "content_generation", "creator_tools", "consumer_assistant",
    "customer_support", "enterprise_automation", "collaboration",
    "translation_localization", "knowledge_management", "platform_os",
    "startup_ecosystem", "unknown",
}

SUB_THEME_VOCAB = VERTICAL_INDUSTRY_VOCAB | HORIZONTAL_FUNCTION_VOCAB

SUBTHEME_DEFINITIONS = {
    # Vertical AI sub-themes
    "healthcare": "AI applied to patient care, clinical workflows, diagnosis, hospital operations, care delivery, or health services.",
    "life_sciences": "AI applied to genomics, biology, medical research, protein research, biomedical research, or clinical research.",
    "biotech_pharma": "AI applied to pharmaceutical R&D, biotech platforms, drug discovery, molecule design, or clinical trial optimization.",
    "materials_science": "AI applied to chemistry, materials discovery, molecular simulation, battery materials, or quantum chemistry.",
    "manufacturing": "AI applied to factory operations, production optimization, quality control, industrial automation, or manufacturing workflows.",
    "energy_oil_gas": "AI applied to oil, gas, energy production, exploration, refining, or traditional energy operations.",
    "clean_energy": "AI applied to solar, wind, grid optimization, carbon capture, geothermal, nuclear fusion, or energy storage.",
    "automotive": "AI applied to vehicles, autonomous driving, driver assistance, in-car assistants, mobility, robotaxis, autonomous vehicle platforms, or automotive manufacturing.",
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
    "security": "AI used for general security workflows, access control, risk detection, safety, trust, provenance, watermarking, content authenticity, or protection across industries.",
    "cloud_infra": "Cloud infrastructure, cloud services, compute/storage/network platforms, or cloud-native infrastructure supporting AI workloads.",
    "ai_infra": "AI infrastructure such as GPUs, accelerators, model training infrastructure, AI platforms, foundation model infrastructure, or AI compute.",
    "model_ops": "AI infrastructure for model deployment, inference serving, monitoring, evaluation, safety, guardrails, grounding, optimization, or model governance.",
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

VALID_DEVELOPMENT_SCOPES = {"Vertical AI", "Horizontal AI", "unknown"}
VALID_AGENT_ARCHITECTURES = {"single-agent AI", "multi-agent AI", "unknown"}
VALID_EVENT_TYPES = {
    "Strategic Commitment", "Resourcing", "Product Shipping & Iteration",
    "Commercial Traction", "Partnerships & GTM Motions", "unknown",
}
VALID_CONFIDENCE = {"high", "medium", "low"}
TARGET_FIELDS = [
    "development_scope", "agent_architecture", "sub_theme", "event_type",
    "strength_1_5", "evidence_span", "reason_short", "confidence", "is_usable_for_iss",
]


# =========================
# Keyword / heuristic config
# =========================

# Hard AI signal terms. Expanded to fix over-filtering of autonomous driving,
# robotics, physical AI, inference infrastructure, and reasoning/model-platform events.
HARD_AI_SIGNAL_KEYWORDS = {
    " ai ", "artificial intelligence", "generative ai", "genai", "ai-powered", "ai powered",
    "ai feature", "ai platform", "ai infrastructure", "ai assistant", "ai model", "ai models",
    "gemini", "copilot", "chatgpt", "openai", "claude", "llama", "bard",
    "llm", "large language model", "foundation model", "foundation models",
    "machine learning", " ml ", "deep learning", "neural network", "neural networks",
    "agentic", " ai agent", "ai agents", "autonomous agent", "autonomous agents", "chatbot",
    "computer vision", "vision-language", "vlm", "multimodal", "reasoning model", "reasoning models",
    "automated reasoning", "deepfake detection", "deepfakes", "inference", "inference serving",
    "ai training", "training benchmark", "ai accelerator", "gpu for ai", "model release",
    "model hosting", "model deployment", "guardrails", "grounding", "rag", "retrieval-augmented",
    "quantum chemistry", "materials discovery", "drug discovery", "genomics", "protein",
    "h100", "a100", "h200", "blackwell", "hopper", "rtx", "cuda", "tensorrt", "triton",
    "nim", "nemo", "dgx", "mlperf", "gpu training benchmark", "ai factory", "ai factories",
    "autonomous vehicle", "autonomous vehicles", "autonomous driving", "driver assistance",
    "robotaxi", "robotaxis", "autonomous mobile robot", "autonomous mobile robots",
    "autonomy", "robotics", "physical ai", "isaac amr", "robot simulation", "world foundation model",
    "synthid", "watermark", "watermarking", "content authenticity", "ai-generated",
}

INVALID_AI_SIGNAL_ONLY_TERMS = {
    "clean energy", "smart grid", "sustainability", "climate", "carbon capture", "renewable energy",
    "cloud", "digital transformation", "technology", "platform", "app", "security", "cybersecurity",
    "data", "analytics", "recommendation", "personalized", "search", "ads", "productivity",
    "automation", "hardware", "device", "satellite", "camera", "music", "content", "media", "entertainment",
}

KNOWN_COMPANY_ALIASES = {
    "Google": {"google", "alphabet", "deepmind", "google cloud", "vertex ai", "gemini", "youtube", "android", "pixel", "nest", "fitbit", "google play", "google search", "waymo"},
    "Microsoft": {"microsoft", "azure", "github", "copilot", "xbox", "linkedin", "m365", "microsoft 365"},
    "NVIDIA": {"nvidia", "geforce", "rtx", "cuda", "h100", "h200", "hopper", "blackwell", "dgx", "drive hyperion", "nvidia drive", "jetson", "isaac"},
    "OpenAI": {"openai", "chatgpt", "gpt", "sora", "dall-e", "dalle"},
}

WEAK_TARGET_ALIASES = {
    "Google": {"android", "google play"},
}

OTHER_MAJOR_COMPANIES = {
    "amazon", "aws", "amazon web services", "meta", "facebook", "llama", "anthropic", "claude",
    "apple", "oracle", "salesforce", "databricks", "snowflake", "tesla", "xai", "mistral",
    "cohere", "perplexity", "petco", "doordash", "asml", "mozilla", "periodic labs",
}

OTHER_EVENT_OWNER_NAMES = OTHER_MAJOR_COMPANIES | {
    "onton", "scale ai", "imbue", "devstral", "mistral ocr", "hubspot", "thrive ai health",
    "linkup", "koyeb", "tenstorrent", "openstack", "petco", "vetco", "doordash", "zesty",
}

CONCRETE_EVENT_VERBS = {
    "raised", "raises", "raise", "funding", "fundraise", "launched", "launches", "launch",
    "released", "releases", "announced", "announces", "acquired", "acquires", "acquisition",
    "invested", "invests", "investment", "partnered", "partners", "partnership", "introduced",
    "introduces", "rolls out", "rolled out", "is launching", "is releasing", "unveiled", "debuts",
    "expanded", "expands", "made available", "available", "adopted", "uses", "using",
    "topped", "tops", "ranked", "ranks", "led", "leads",
}

RESEARCH_OR_DEMO_KEYWORDS = {
    "research", "researchers", "paper", "study", "benchmark", "benchmarks", "mlperf",
    "cvpr", "conference", "showcase", "demo", "prototype", "preview", "early access", "technical report",
}

FEATURE_RELEASE_KEYWORDS = {
    "feature", "features", "roll out", "rolled out", "release", "released", "launch", "launched",
    "update", "updated", "integration", "integrated", "available", "app", "introduced", "introducing",
}

STRONG_TRACTION_KEYWORDS = {
    "revenue", "revenues", "sales", "customers", "customer", "contracts", "adoption", "used by",
    "users", "usage", "growth", "grew", "increase", "increased", "improvement", "conversion",
    "run rate", "annual", "billions", "millions", "hundreds of thousands", "thousands of customers",
    "save millions", "cost reduction", "time savings",
}

MAJOR_RESOURCE_KEYWORDS = {
    "acquisition", "acquire", "acquired", "investment", "invest", "capex", "capital expenditure",
    "data center", "data centers", "infrastructure buildout", "funding", "billion", "$", "procurement",
}

MARKET_FORECAST_KEYWORDS = {
    "market size", "market opportunity", "global economy", "gdp", "projecting", "projects",
    "estimate", "estimated", "forecast", "forecasts", "expected to contribute", "could contribute",
    "could reach", "economic impact", "analyst", "pwc", "mckinsey",
}

NEGATIVE_RISK_EVENT_TERMS = {
    "lawsuit", "lawsuits", " suit ", "suits", "complaint", "privacy complaint",
    "defamatory", "hallucination", "hallucinations", "delusional", "spiral",
    "spirals", "tragedy", "tragedies", "suicide", "self-harm", "harm",
    "harmful", "manipulative", "sycophancy", "sycophantic", "internal warnings",
    "distressed users", "under investigation", "regulator", "regulators",
}

SEVERE_NEGATIVE_RISK_EVENT_TERMS = {
    "lawsuit", "lawsuits", " suit ", "suits", "complaint", "privacy complaint",
    "defamatory", "hallucination", "hallucinations", "delusional", "spiral",
    "spirals", "tragedy", "tragedies", "suicide", "self-harm", "manipulative",
    "sycophancy", "sycophantic", "internal warnings", "distressed users",
}

POSITIVE_ISS_BUSINESS_TERMS = {
    "launched", "launches", "launch", "released", "releases", "release",
    "rolled out", "rolls out", "introduced", "introduces", "announced",
    "announces", "available", "added", "adds", "revenue", "generated",
    "customers", "customer", "enterprise", "adoption", "uses over",
    "partnership", "partnered", "acquisition", "acquired", "investment",
    "invested", "funding", "unified several engineering",
    "product, and research teams",
}

POSITIVE_COMMERCIAL_OR_RESOURCE_TERMS = {
    "revenue", "generated", "customers", "customer", "enterprise", "adoption",
    "uses over", "partnership", "partnered", "acquisition", "acquired",
    "investment", "invested", "funding", "unified several engineering",
    "product, and research teams",
}

RANKING_TRACTION_TERMS = {
    "topped", "tops", "ranked", "ranks", "most downloaded", "downloads",
    "installs", "downloaded app", "top app", "top apps",
}

GENERAL_ASSISTANT_TERMS = {
    "chatgpt", "gemini", "copilot", "assistant", "chatbot", "general-purpose", "search engine",
    "office suite", "workspace", "microsoft 365", "google workspace",
}

SHOPPING_FEATURE_TERMS = {
    "shopping feature", "shopping features", "shopping assistant", "research potential purchases",
    "help users research", "product discovery", "recommendations", "ask chatgpt for help finding",
}

ADVERTISING_EVIDENCE_TERMS = {
    "ai max", "search ads", "advertisers", "advertiser", "campaign", "campaigns",
    "conversion", "conversions", "roas", "pmax", "demand gen", "asset studio", "product studio",
    "marketing", "meridian",
}

CREATOR_TOOL_EVIDENCE_TERMS = {
    "youtube studio", "ask studio", "creator", "creators", "likeness", "dubbing", "auto dubbing",
    "lip-synced", "podcasters", "veo", "speech to song", "channel insights", "studio updates",
}

TRANSLATION_EVIDENCE_TERMS = {"voice translate", "translate", "translation", "translates", "real time", "language barriers", "dubbing", "localization"}
PIXEL_ASSISTANT_EVIDENCE_TERMS = {"magic cue", "daily hub", "gemini nano", "tensor g5", "pixel", "on-device ai"}
SECURITY_AUTHENTICITY_TERMS = {
    "synthid", "watermark", "watermarking", "content credentials", "content authenticity",
    "deepfake", "deepfakes", "generated using google ai", "verify", "verification",
    "prompt injection", "indirect prompt injection", "threat intelligence", "threat",
    "scam", "scammers", "policy-violating ads", "bad ads", "fraud", "abuse",
    "safety", "security", "protect", "protections",
}

NVIDIA_INFRA_TERMS = {
    "gpu", "gpus", "a100", "h100", "h200", "hopper", "blackwell", "rtx", "cuda", "tensorrt",
    "triton", "nim", "nemo", "dgx", "mlperf", "deepstream", "holoscan", "ai blueprints",
    "accelerated computing", "inference serving", "model deployment", "benchmarking", "ai infrastructure",
    "ai platform", "ai factory", "ai factories", "jetson", "geforce",
}
NVIDIA_MODELOPS_TERMS = {"triton", "tensorrt", "inference serving", "model deployment", "benchmarking", "mlperf", "optimiz", "throughput", "latency", "deployment", "deepstream", "holoscan"}
NVIDIA_AUTOMOTIVE_TERMS = {"nvidia drive", "drive hyperion", "autonomous vehicle", "autonomous vehicles", "av platform", "autonomous driving", "driver assistance", "robotaxi", "automotive"}
NVIDIA_ROBOTICS_LOGISTICS_TERMS = {"isaac", "isaac amr", "autonomous mobile robot", "autonomous mobile robots", "amr", "robotics", "mobile robots", "warehouse", "warehouses", "logistics", "factory", "factories"}


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            if value != value:
                return ""
        except Exception:
            pass
    return str(value)


def contains_any_keyword(text: str, keywords: Set[str]) -> bool:
    text_l = safe_str(text).lower()
    return any(keyword in text_l for keyword in keywords)


def normalize_for_match(text: str) -> str:
    """Aggressive normalization for evidence-span verification."""
    text = unicodedata.normalize("NFKC", safe_str(text)).lower()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9$%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_explicit_ai_signal(text: str) -> bool:
    text_l = f" {safe_str(text).lower()} "
    return any(keyword in text_l for keyword in HARD_AI_SIGNAL_KEYWORDS)


def load_records(file_path: str) -> List[Dict[str, Any]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record: Dict[str, Any], file_path: str) -> None:
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_chunk_ids(output_file: str) -> Set[str]:
    done = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    cid = rec.get("chunk_id", "")
                    if cid:
                        done.add(cid)
                except Exception:
                    continue
    return done


# =========================
# Evidence span verification
# =========================


def verify_evidence_span(evidence_span: str, chunk_text: str) -> bool:
    """
    More tolerant evidence verification.

    Fixes false negatives caused by glued HTML text, punctuation changes, line breaks,
    smart quotes, and minor model spacing differences.
    """
    if not evidence_span or not chunk_text:
        return False

    norm_span = normalize_for_match(evidence_span)
    norm_chunk = normalize_for_match(chunk_text)

    if not norm_span:
        return False

    if norm_span in norm_chunk:
        return True

    # If the model chose a long span, verifying key fragments is often more robust.
    span_tokens = norm_span.split()
    if len(span_tokens) >= 12:
        first_part = " ".join(span_tokens[:12])
        last_part = " ".join(span_tokens[-12:])
        if first_part in norm_chunk and last_part in norm_chunk:
            return True

    # Sliding window fuzzy check.
    span_len = len(norm_span)
    step = max(1, span_len // 6)
    best = 0.0
    for i in range(0, max(1, len(norm_chunk) - span_len + 1), step):
        window = norm_chunk[i: i + span_len + 40]
        score = SequenceMatcher(None, norm_span, window).ratio()
        best = max(best, score)
        if score >= 0.82:
            return True

    # Token-overlap fallback: useful when original chunk has missing spaces.
    span_set = set(span_tokens)
    chunk_set = set(norm_chunk.split())
    if len(span_set) >= 8:
        overlap = len(span_set & chunk_set) / max(1, len(span_set))
        if overlap >= 0.88:
            return True

    return False


# =========================
# Prompt text
# =========================

TARGET_COMPANY_GATE = """
================================================================
CRITICAL GATE 0: target_company_relevance
================================================================
Before assigning ISS labels, determine whether the main AI/business event is about the target company provided in metadata.

Set is_usable_for_iss = false and return unknown labels if:
- the main event is about another company, competitor, startup, VC, government, or general market trend;
- the target company is only a comparison, background example, partner-list item, cloud-host list item, or passing reference;
- the article is assigned to the target company only because the company name appears somewhere in the text, but the main action belongs to another entity.

Set is_usable_for_iss = true only if:
- the target company is the main actor performing the event; OR
- the target company is clearly one of the main actors in a partnership, investment, acquisition, product launch, customer deployment, or commercial traction event; OR
- the source is the target company's official blog/newsroom and the chunk discusses the target company's own product, platform, research, ecosystem, strategic direction, or customer deployment.

Official-source exception:
- For official company blog/newsroom pages, do not require the target company name to repeat in every sentence.
- If the official source is from Google/Microsoft/NVIDIA/OpenAI and the chunk describes that company's own AI product/platform/action, it can pass target_company_relevance.
""".strip()

AI_SIGNAL_GATE = """
================================================================
CRITICAL GATE 1: explicit_ai_signal
================================================================
Do not assign Vertical AI or Horizontal AI unless the chunk/evidence contains a meaningful AI-related signal.

Valid AI signals include:
- AI, artificial intelligence, generative AI, GenAI, AI-powered, AI-generated
- Gemini, Copilot, ChatGPT, OpenAI, Claude, Llama, Bard, Sora, DALL-E
- LLM, large language model, foundation model, model release, reasoning model, multimodal model
- agent, AI agent, agentic AI, AI assistant, chatbot
- machine learning, ML, deep learning, neural network, computer vision, VLM
- inference, model serving, AI training, AI accelerator, GPU for AI workloads, AI factory
- automated reasoning, guardrails, grounding, RAG, retrieval-augmented generation
- content authenticity, watermarking, SynthID, deepfake detection
- autonomous vehicle, autonomous driving, robotaxi, robotics, physical AI, world foundation model, robot simulation
- for NVIDIA: H100, H200, A100, Blackwell, Hopper, RTX, CUDA, TensorRT, Triton, NIM, NeMo, DGX, MLPerf, NVIDIA DRIVE, DRIVE Hyperion, Jetson, Isaac

Important correction:
- Do NOT exclude autonomous vehicle, autonomous driving, robotics, physical AI, AI factory, inference platform, or model deployment just because the exact words "AI" or "ML" are not repeated in the evidence_span.
- Domain-specific AI terms above are valid AI signals.

Set is_usable_for_iss = false if the chunk only discusses general software, cloud, platform, hardware, business operations, clean energy, policy, marketing, search, ads, productivity, data, or automation without a clear AI/model/agent/autonomy/inference signal.
""".strip()

SOURCE_QUALITY_GATE = """
================================================================
CRITICAL GATE 2: source_quality
================================================================
Exclude source pages that are not standalone business events: author pages, tag pages, archive pages, search-result pages, category pages, topic collection pages, navigation pages, or pages combining many unrelated snippets.

Important correction:
- Do NOT exclude a normal product article merely because the URL contains product words such as /products/search/ or because the product is Google Search.
- Exclude only when the page is clearly an index/search/archive/list page, not a specific announcement or article.
""".strip()

CONCRETE_EVENT_OWNER_GATE = """
================================================================
CRITICAL GATE 3: concrete_event_owner
================================================================
The evidence_span must describe an action, business signal, product, partnership, investment, customer deployment, or strategic commitment by or directly involving the target company.

Exclude if the concrete event is owned by another company/startup/customer and the target company is only mentioned in passing.

Examples:
- company = Google, evidence says DoorDash launches Zesty => exclude.
- company = Microsoft, evidence says Mistral launched an OCR API => exclude.
- company = NVIDIA, evidence says NVIDIA DRIVE passed safety assessments => usable.
- company = NVIDIA, evidence says NVIDIA GPUs power a named customer deployment => usable as NVIDIA provider-side traction.
""".strip()

PROVIDER_CUSTOMER_SCOPE_RULE = """
================================================================
CRITICAL RULE: provider_vs_customer_use_case
================================================================
When the target company provides reusable AI infrastructure, chips, models, APIs, cloud tools, inference serving, developer tooling, or AI platforms that a customer uses in one industry, classify based on the target company's provider capability, not the customer's vertical.

Use Horizontal AI when the target company's role is:
- AI infrastructure, chips, GPUs, accelerators, inference serving, model deployment, model APIs;
- cloud AI platforms, developer tools, model operations, guardrails, evaluation, grounding;
- general-purpose platforms used across industries.

Use Vertical AI when the target company itself is building, selling, or strategically committing to a dedicated industry-specific AI solution or vertical workflow.

NVIDIA nuance:
- NVIDIA Blackwell/H100/RTX/CUDA/TensorRT/Triton/NIM/DGX/AI factory/model deployment => usually Horizontal AI / ai_infra or model_ops.
- NVIDIA DRIVE / DRIVE Hyperion / autonomous vehicle platform => Vertical AI / automotive.
- NVIDIA Isaac AMR / autonomous mobile robots for warehouses/factories/logistics => Vertical AI / logistics or manufacturing.
""".strip()

AGENT_ARCHITECTURE_RULE = """
================================================================
CRITICAL RULE: agent_architecture_boundary
================================================================
Use agent labels conservatively.

single-agent AI:
- one assistant, copilot, chatbot, coding agent, or task-specific agent performs a workflow;
- one agent may use tools/APIs/retrieval, but there is no explicit coordination among multiple agents.

multi-agent AI:
- only when the evidence explicitly describes multiple agents, separate specialized agents, agent collaboration, planner-executor architecture, orchestration, delegation, agent teams, or multiple agents working together.

Do NOT label multi-agent AI merely because the chunk says "agentic", "workflow", "autonomous", "assistant", or "agent".
If evidence only says "coding assistant", "AI assistant", "copilot", or one named agent, use single-agent AI or unknown, not multi-agent AI.
""".strip()

LONG_CHUNK_EVENT_CONSISTENCY_RULE = """
================================================================
CRITICAL RULE: long_chunk_event_consistency
================================================================
Long chunks may contain multiple events. Choose exactly ONE primary event.
All labels must describe the same event supported by evidence_span.

- evidence_span is the anchor.
- Do not mix labels from different parts of the chunk.
- Do not use a reason template from another event.
- For earnings calls, prefer the strongest concrete AI business signal with measurable traction.
""".strip()

FUNDING_ACQUISITION_RULE = """
================================================================
CRITICAL RULE: funding_acquisition_resourcing
================================================================
Funding, investment, acquisition, and capex should not always be excluded.

Use Resourcing when:
- the target company allocates capital, acquires a capability, builds infrastructure, hires, expands an AI team, procures compute, or raises/uses funding clearly tied to AI compute, model development, AI infrastructure, data centers, or a specific AI theme.

Exclude when:
- funding/acquisition is generic corporate finance with no identifiable AI theme;
- a CVC one-liner says only "General Technology" or a broad sector without AI-specific relevance;
- the investment is actually by another company into the target company and does not reveal target-company thematic intent.

CVC one-liners:
- be conservative;
- strength is usually 2;
- allow strength 3 only with clear AI relevance + amount + target-company-specific strategic detail;
- allow 4+ only for major AI-specific capex/acquisition/infrastructure with strong detail.
""".strip()

SUB_THEME_RULE = """
================================================================
CRITICAL RULE: sub_theme_list_output
================================================================
Return sub_theme as a JSON list of one or more allowed tags.
Prefer one primary sub_theme. Use multiple tags only when the same evidence_span directly supports multiple core use cases.
Do not invent tags. Do not output stringified lists. Correct examples:
- ["search_discovery"]
- ["advertising", "search_discovery"]
- ["ai_infra"]
Wrong examples:
- "['advertising', 'search_discovery']"
- "advertising, search_discovery"
""".strip()

SYSTEM_PROMPT = f"""
You are an expert information extraction model for a venture-investment intelligence pipeline.

Your task is to read ONE text chunk and assign chunk-level labels for the Investment Intent Score (ISS) system.
Use only the chunk text and metadata. Do not infer beyond the evidence. Return valid JSON only.

{TARGET_COMPANY_GATE}

{AI_SIGNAL_GATE}

{SOURCE_QUALITY_GATE}

{CONCRETE_EVENT_OWNER_GATE}

{PROVIDER_CUSTOMER_SCOPE_RULE}

{AGENT_ARCHITECTURE_RULE}

{LONG_CHUNK_EVENT_CONSISTENCY_RULE}

{FUNDING_ACQUISITION_RULE}

{SUB_THEME_RULE}

================================================================
FIELD 1: development_scope
================================================================
Choose exactly ONE: Vertical AI, Horizontal AI, unknown.

Vertical AI: AI solution tailored to a specific industry or domain workflow.
Horizontal AI: broad cross-industry or cross-functional AI capability.
unknown: insufficient support.

Rules:
- General assistant/search/office/browser/cloud/developer-platform features are Horizontal AI unless purpose-built for a vertical workflow.
- Dedicated industry solutions such as autonomous driving platforms, clinical copilots, radiology AI, logistics robots, insurance underwriting, or banking AI workflows are Vertical AI.
- Provider-side AI infrastructure used by a vertical customer is usually Horizontal AI.

================================================================
FIELD 2: agent_architecture
================================================================
Choose exactly ONE: single-agent AI, multi-agent AI, unknown.

Use single-agent AI only when one assistant/copilot/agent is explicitly described.
Use multi-agent AI only when multiple agents or explicit orchestration/collaboration are described.
Otherwise use unknown.

================================================================
FIELD 3: sub_theme
================================================================
Return a JSON list of one or more allowed tags.
- If development_scope = Vertical AI, tags must come from Vertical AI vocab.
- If development_scope = Horizontal AI, tags must come from Horizontal AI vocab.
- If development_scope = unknown, return ["unknown"].
- Prefer one tag; use multiple only with strong direct support.

================================================================
FIELD 4: event_type
================================================================
Choose exactly ONE:
- Strategic Commitment
- Resourcing
- Product Shipping & Iteration
- Commercial Traction
- Partnerships & GTM Motions
- unknown

Priority:
1. Resourcing: investment, acquisition, hiring, capex, data center, infrastructure buildout.
2. Commercial Traction: revenue, customer adoption, usage, contracts, quantified demand.
3. Product Shipping & Iteration: release, launch, preview, model/API/feature update.
4. Partnerships & GTM Motions: partnership, integration, alliance, marketplace, co-sell.
5. Strategic Commitment: direction, strategy, executive priority.
6. unknown.

================================================================
FIELD 5: strength_1_5
================================================================
1 = not usable / no AI / wrong company / no business action.
2 = weak, conceptual, sparse one-liner, research/demo, or directional statement.
3 = clear product/feature/partnership/investment signal but limited scale or no traction proof.
4 = strong concrete launch, customer deployment, partnership, infrastructure investment, or rollout.
5 = major quantified traction, revenue, capex, acquisition/funding, large-scale adoption, or strategic commitment.

Be conservative. Product feature releases without adoption/revenue/customer proof are usually 3.

================================================================
FIELD 6: evidence_span
================================================================
Copy the shortest possible span from the chunk that supports all non-unknown labels.
Use 1-2 consecutive sentences when possible. Do not paraphrase.

================================================================
FIELD 7: reason_short
================================================================
One concise sentence explaining why the labels were assigned or why the chunk was excluded.

================================================================
FIELD 8: confidence
================================================================
Choose exactly ONE: high, medium, low.

================================================================
FIELD 9: is_usable_for_iss
================================================================
true only if the target company has a meaningful AI business signal with clear event_type, scope, evidence, and strength.
false otherwise.

If false: development_scope="unknown", agent_architecture="unknown", sub_theme=["unknown"], event_type="unknown", strength_1_5=1.

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
    lines = []
    for tag in sorted(vocab - {"unknown"}):
        definition = SUBTHEME_DEFINITIONS.get(tag, "")
        lines.append(f"- {tag}: {definition}")
    return "\n".join(lines)


_VERTICAL_SUBTHEME_DEFINITIONS = format_subtheme_definitions(VERTICAL_INDUSTRY_VOCAB)
_HORIZONTAL_SUBTHEME_DEFINITIONS = format_subtheme_definitions(HORIZONTAL_FUNCTION_VOCAB)

USER_PROMPT_TEMPLATE = f"""
chunk_id: {{chunk_id}}
source_type: {{source_type}}
source_name: {{source_name}}
source_title: {{source_title}}
source_url: {{source_url}}
document_date: {{document_date}}
company: {{company}}

chunk_text:
{{chunk_text}}

---
Allowed sub_theme values for Vertical AI:
{_VERTICAL_INDUSTRY_LIST}

Vertical AI sub_theme definitions:
{_VERTICAL_SUBTHEME_DEFINITIONS}

Allowed sub_theme values for Horizontal AI:
{_HORIZONTAL_FUNCTION_LIST}

Horizontal AI sub_theme definitions:
{_HORIZONTAL_SUBTHEME_DEFINITIONS}

All allowed sub_theme values:
{_SUB_THEME_LIST}

Important:
- Return sub_theme as a JSON list, not a string.
- Prefer one primary tag; use multiple only when directly supported by the same evidence_span.
- If no sub_theme accurately matches, choose ["unknown"].
""".strip()


# =========================
# Deterministic post-processing guards
# =========================

LOW_QUALITY_URL_SEGMENTS = {"author", "tag", "tags", "category", "categories", "topics", "topic", "archive", "archives"}
LOW_QUALITY_QUERY_KEYS = {"s", "q", "query", "search"}
POST_LIST_SIGNALS = {"author", "archives", "archive", "tagged", "posts by", "all posts", "latest posts", "read more", "previous post", "next post", "category"}


def is_low_quality_source_page(record: Dict[str, Any]) -> bool:
    """
    Relaxed source-quality check.
    Fix: /products/search/... is NOT treated as low-quality search page.
    """
    source_url = safe_str(record.get("source_url", "")).lower()
    source_title = safe_str(record.get("source_title", "")).lower()
    chunk_text = safe_str(record.get("chunk_text", "")).lower()

    parsed = urlparse(source_url)
    path_segments = [seg for seg in parsed.path.split("/") if seg]

    if any(seg in LOW_QUALITY_URL_SEGMENTS for seg in path_segments):
        return True

    # True search result pages generally have /search as a path segment or query params.
    if path_segments and path_segments[-1] == "search":
        return True
    if parsed.query:
        query_keys = {part.split("=")[0].lower() for part in parsed.query.split("&") if part}
        if query_keys & LOW_QUALITY_QUERY_KEYS and "google.com/search" in source_url:
            return True

    if source_title.startswith(("author:", "tag:", "category:", "archive:")):
        return True

    first_700 = chunk_text[:700]
    if any(signal in first_700 for signal in POST_LIST_SIGNALS):
        list_like_count = sum(chunk_text.count(marker) for marker in ["read more", "posted", " by ", "previous", "next"])
        if list_like_count >= 4:
            return True

    return False


def get_target_aliases(company: str) -> Set[str]:
    company = safe_str(company).strip()
    return {a.lower() for a in KNOWN_COMPANY_ALIASES.get(company, {company.lower()})}


def get_non_target_company_aliases(company: str) -> Set[str]:
    """Aliases for known companies other than the current target company."""
    aliases = set(OTHER_EVENT_OWNER_NAMES)
    for known_company, known_aliases in KNOWN_COMPANY_ALIASES.items():
        if known_company != company:
            aliases.update(known_aliases)
    return {safe_str(a).lower() for a in aliases if safe_str(a).strip()}


def mentions_target_company(text: str, company: str) -> bool:
    text_l = safe_str(text).lower()
    return any(alias in text_l for alias in get_target_aliases(company))


def is_official_target_source(record: Dict[str, Any]) -> bool:
    company = safe_str(record.get("company", ""))
    source_type = safe_str(record.get("source_type", ""))
    source_name = safe_str(record.get("source_name", "")).lower()
    source_url = safe_str(record.get("source_url", "")).lower()

    if source_type != "official_website_announcement":
        return False
    if company == "Google":
        return "google" in source_name or "blog.google" in source_url
    if company == "Microsoft":
        return "microsoft" in source_name or "microsoft.com" in source_url
    if company == "NVIDIA":
        return "nvidia" in source_name or "nvidia.com" in source_url or "blogs.nvidia.com" in source_url
    if company == "OpenAI":
        return "openai" in source_name or "openai.com" in source_url
    return False


def contains_concrete_event_verb(text: str) -> bool:
    text_l = safe_str(text).lower()
    return any(v in text_l for v in CONCRETE_EVENT_VERBS)


def likely_other_company_main_actor(text: str, company: str) -> bool:
    text_l = safe_str(text).lower()
    first_500 = text_l[:500]
    target_aliases = get_target_aliases(company)
    target_in_first_500 = any(alias in first_500 for alias in target_aliases)
    other_in_first_500 = any(other in first_500 for other in OTHER_MAJOR_COMPANIES if other not in target_aliases)
    return other_in_first_500 and not target_in_first_500


def has_provider_pattern(text: str, company: str) -> bool:
    """Return True when another actor is clearly using/building on the target company's platform.

    Uses company aliases such as Azure, CUDA, Vertex AI, Gemini, ChatGPT, etc.,
    not just the canonical company name.
    """
    text_l = safe_str(text).lower()
    aliases = get_target_aliases(company)
    provider_verbs = [
        "uses", "using", "powered by", "runs on", "built on", "built with",
        "deployed on", "accelerated by", "trained on", "hosted on", "with",
    ]

    for alias in aliases:
        for verb in provider_verbs:
            if f"{verb} {alias}" in text_l:
                return True

    return False


def likely_other_company_owns_evidence(validated: Dict[str, Any]) -> bool:
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

    other_names = set()
    for name in get_non_target_company_aliases(company):
        n = name.lower()
        if n in combined and n not in target_aliases:
            other_names.add(n)

    if other_names and target_in_evidence and not strong_target_in_evidence:
        return True

    # Customer/provider exception: if evidence says another company uses target company's platform,
    # this can be valid traction for the target company. Check aliases too, e.g., Azure/CUDA/Vertex AI.
    if has_provider_pattern(combined, company):
        return False

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

    first_180 = combined[:180]
    if other_names and not any(alias in first_180 for alias in target_aliases):
        return True

    return False


def evidence_span_has_explicit_ai_signal(validated: Dict[str, Any]) -> bool:
    evidence = safe_str(validated.get("evidence_span", ""))
    chunk_text = safe_str(validated.get("chunk_text", ""))
    source_title = safe_str(validated.get("source_title", ""))
    company = safe_str(validated.get("company", ""))
    combined = f"{evidence} {source_title}"

    if has_explicit_ai_signal(evidence):
        return True

    # Allow title/chunk support if evidence is a concrete product sentence but title/chunk carries the AI context.
    if has_explicit_ai_signal(combined) and contains_concrete_event_verb(evidence):
        return True

    # NVIDIA domain-specific exceptions.
    if company == "NVIDIA" and contains_any_keyword(combined, NVIDIA_INFRA_TERMS | NVIDIA_AUTOMOTIVE_TERMS | NVIDIA_ROBOTICS_LOGISTICS_TERMS):
        return True

    # Official page exception: the chunk/title can establish the AI context if evidence is a rollout/action sentence.
    if is_official_target_source(validated) and has_explicit_ai_signal(f"{source_title} {chunk_text}") and contains_concrete_event_verb(evidence):
        return True

    return False


def normalize_subthemes(value: Any) -> List[str]:
    """Normalize sub_theme into a clean list of strings.

    Handles native lists, JSON list strings, Python literal list strings,
    comma-separated strings, and scalar tags without corrupting apostrophes.
    """
    if value is None:
        return ["unknown"]

    if isinstance(value, list):
        raw = value

    elif isinstance(value, str):
        s = value.strip()

        # Fix accidental stringified list: "['a', 'b']" or '["a", "b"]'.
        # Use json.loads first, then ast.literal_eval for Python-style lists.
        # Avoid global single-quote replacement because it can corrupt values.
        if s.startswith("[") and s.endswith("]"):
            parsed = None
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = None

            if isinstance(parsed, list):
                raw = parsed
            else:
                raw = [x.strip(" '\"") for x in s.strip("[]").split(",")]

        elif "," in s:
            raw = [x.strip() for x in s.split(",")]
        else:
            raw = [s]

    else:
        raw = [safe_str(value)]

    cleaned = []
    for tag in raw:
        t = safe_str(tag).strip().strip("'\"")
        if t:
            cleaned.append(t)

    # De-dup preserving order.
    out = []
    for t in cleaned:
        if t not in out:
            out.append(t)

    return out or ["unknown"]

def validate_subthemes_for_scope(subthemes: List[str], scope: str) -> List[str]:
    if scope == "Vertical AI":
        allowed = VERTICAL_INDUSTRY_VOCAB
    elif scope == "Horizontal AI":
        allowed = HORIZONTAL_FUNCTION_VOCAB
    else:
        return ["unknown"]

    valid = [t for t in subthemes if t in allowed and t != "unknown"]
    return valid or ["unknown"]


def force_not_usable(validated: Dict[str, Any], reason: str) -> Dict[str, Any]:
    validated["development_scope"] = "unknown"
    validated["agent_architecture"] = "unknown"
    validated["sub_theme"] = ["unknown"]
    validated["event_type"] = "unknown"
    validated["strength_1_5"] = 1
    validated["is_usable_for_iss"] = False
    validated["reason_short"] = reason
    validated["confidence"] = validated.get("confidence") if validated.get("confidence") in VALID_CONFIDENCE else "high"
    return validated


def infer_best_exclusion_reason(validated: Dict[str, Any]) -> str:
    chunk_text = safe_str(validated.get("chunk_text", ""))
    company = safe_str(validated.get("company", ""))

    if is_low_quality_source_page(validated):
        return "Excluded from ISS because the source appears to be an author, tag, archive, search, or index page rather than a standalone business event."
    if company and likely_other_company_owns_evidence(validated):
        return f"Excluded from ISS because the concrete event in the evidence appears to be owned by another company, not the target company {company}."
    if company and likely_other_company_main_actor(chunk_text, company):
        return f"Excluded from ISS because the main AI/business event appears to be about another company, not the target company {company}."
    if not has_explicit_ai_signal(chunk_text):
        return "Excluded from ISS because the chunk does not contain an explicit AI-related business signal."
    if not evidence_span_has_explicit_ai_signal(validated):
        return "Excluded from ISS because the supporting evidence does not contain an explicit AI/model/agent/autonomy/inference signal."
    return "Excluded from ISS because no clear target-company AI business event is supported."


def is_general_assistant_shopping_feature(validated: Dict[str, Any]) -> bool:
    text = " ".join([
        safe_str(validated.get("evidence_span", "")),
        safe_str(validated.get("source_title", "")),
        safe_str(validated.get("chunk_text", ""))[:500],
    ]).lower()
    return contains_any_keyword(text, GENERAL_ASSISTANT_TERMS) and (
        contains_any_keyword(text, SHOPPING_FEATURE_TERMS) or ("shopping" in text and "assistant" in text)
    )


def normalize_general_assistant_scope(validated: Dict[str, Any]) -> Dict[str, Any]:
    if validated.get("is_usable_for_iss") and is_general_assistant_shopping_feature(validated):
        validated["development_scope"] = "Horizontal AI"
        validated["sub_theme"] = ["consumer_assistant", "search_discovery"]
        validated["reason_short"] = "The chunk describes a shopping feature inside a general-purpose AI assistant, so it is classified as Horizontal AI rather than a dedicated ecommerce vertical solution."
    return validated


def normalize_labels_to_evidence_span(validated: Dict[str, Any]) -> Dict[str, Any]:
    """Align sub_theme to evidence to fix copied template mistakes."""
    if not validated.get("is_usable_for_iss", False):
        return validated

    evidence = safe_str(validated.get("evidence_span", "")).lower()
    title = safe_str(validated.get("source_title", "")).lower()
    combined = f" {evidence} {title} "

    if contains_any_keyword(combined, SECURITY_AUTHENTICITY_TERMS):
        validated["development_scope"] = "Horizontal AI"
        validated["sub_theme"] = ["security"]
        validated["reason_short"] = "The selected evidence describes AI content authenticity, watermarking, or generated-media verification, so the labels are aligned to Horizontal AI security."
        return validated

    if contains_any_keyword(combined, ADVERTISING_EVIDENCE_TERMS):
        validated["development_scope"] = "Horizontal AI"
        sub = ["advertising"]
        if "search" in combined or "search ads" in combined or "ai max" in combined:
            sub.append("search_discovery")
        validated["sub_theme"] = sub
        validated["reason_short"] = "The selected evidence describes an AI advertising or search-ads product, so the labels are aligned to advertising/search_discovery."
        return validated

    if contains_any_keyword(combined, CREATOR_TOOL_EVIDENCE_TERMS):
        validated["development_scope"] = "Horizontal AI"
        sub = ["creator_tools"]
        if any(term in combined for term in ["veo", "dubbing", "speech to song", "content", "creative", "video"]):
            sub.append("content_generation")
        validated["sub_theme"] = sub
        validated["reason_short"] = "The selected evidence describes AI creator tools or studio features, so the labels are aligned to creator_tools/content_generation."
        return validated

    if contains_any_keyword(combined, PIXEL_ASSISTANT_EVIDENCE_TERMS) or contains_any_keyword(combined, TRANSLATION_EVIDENCE_TERMS):
        validated["development_scope"] = "Horizontal AI"
        sub = ["consumer_assistant"]
        if contains_any_keyword(combined, TRANSLATION_EVIDENCE_TERMS):
            sub.append("translation_localization")
        validated["sub_theme"] = sub
        validated["reason_short"] = "The selected evidence describes on-device assistant or translation AI features, so the labels are aligned to consumer_assistant/translation_localization."
        return validated

    return validated


def normalize_nvidia_scope(validated: Dict[str, Any]) -> Dict[str, Any]:
    if validated.get("company") != "NVIDIA" or not validated.get("is_usable_for_iss", False):
        return validated

    combined = " ".join([
        safe_str(validated.get("evidence_span", "")),
        safe_str(validated.get("source_title", "")),
        safe_str(validated.get("chunk_text", ""))[:1200],
    ]).lower()

    if any(x in combined for x in ["ai factory", "ai factories", "dsx", "omniverse dsx", "reference design", "blueprint"]):
        validated["development_scope"] = "Horizontal AI"
        if contains_any_keyword(combined, NVIDIA_MODELOPS_TERMS):
            validated["sub_theme"] = ["model_ops"]
        else:
            validated["sub_theme"] = ["ai_infra"]
        validated["reason_short"] = "The selected evidence describes NVIDIA AI factory, DSX, reference-design, or blueprint infrastructure, so it is classified as Horizontal AI."
        return validated

    # 1) Dedicated robotics / AMR / warehouse / logistics workflows first.
    # This prevents broad chunks that mention both AV and robotics from always falling into automotive.
    robotics_specific = any(x in combined for x in [
        "isaac amr", "autonomous mobile robot", "autonomous mobile robots",
        "mobile robots", "robot fleet", "warehouse", "warehouses", "logistics",
        "supply chain", "industrial inspection", "robotics workflow",
    ])
    if robotics_specific:
        validated["development_scope"] = "Vertical AI"
        if any(x in combined for x in ["warehouse", "warehouses", "logistics", "supply chain", "freight"]):
            validated["sub_theme"] = ["logistics"]
        elif any(x in combined for x in ["factory", "factories", "manufacturing", "production line", "quality control"]):
            validated["sub_theme"] = ["manufacturing"]
        else:
            validated["sub_theme"] = ["industrial_inspection"]
        validated["reason_short"] = "The selected evidence describes NVIDIA robotics, AMR, logistics, or industrial workflow AI, so it is classified as Vertical AI."
        return validated

    # 2) Dedicated autonomous driving / NVIDIA DRIVE platform.
    if contains_any_keyword(combined, NVIDIA_AUTOMOTIVE_TERMS):
        validated["development_scope"] = "Vertical AI"
        validated["sub_theme"] = ["automotive"]
        validated["reason_short"] = "The selected evidence describes NVIDIA's autonomous vehicle or DRIVE platform, so it is classified as Vertical AI in automotive."
        return validated

    # 3) Provider-side infrastructure / model deployment. Treat AI factories and compute buildouts as infra,
    # not manufacturing, unless the evidence clearly describes factory operations or robotics workflows above.
    if contains_any_keyword(combined, NVIDIA_INFRA_TERMS):
        validated["development_scope"] = "Horizontal AI"
        if contains_any_keyword(combined, NVIDIA_MODELOPS_TERMS):
            validated["sub_theme"] = ["model_ops"]
        else:
            validated["sub_theme"] = ["ai_infra"]
        validated["reason_short"] = "The selected evidence describes NVIDIA provider-side AI infrastructure or model deployment capability, so it is classified as Horizontal AI."

    return validated

def normalize_agent_architecture(validated: Dict[str, Any]) -> Dict[str, Any]:
    if not validated.get("is_usable_for_iss", False):
        return validated

    text = " ".join([
        safe_str(validated.get("evidence_span", "")),
        safe_str(validated.get("source_title", "")),
    ]).lower()

    multi_markers = {
        "multiple agents", "multi-agent", "multi agent", "agent teams", "agent team", "planner-executor",
        "planner executor", "orchestrates agents", "orchestration of agents", "two agents", "second agent",
        "specialized agents", "agents coordinate", "agents collaborate", "agent collaboration",
    }
    single_markers = {
        "assistant", "copilot", "chatbot", "coding assistant", "ai agent", "agent", "gemini", "chatgpt", "jules",
    }

    if contains_any_keyword(text, multi_markers):
        validated["agent_architecture"] = "multi-agent AI"
    elif validated.get("agent_architecture") == "multi-agent AI" and not contains_any_keyword(text, multi_markers):
        # Fix over-labeling: agentic/workflow alone is not enough for multi-agent.
        if contains_any_keyword(text, single_markers):
            validated["agent_architecture"] = "single-agent AI"
        else:
            validated["agent_architecture"] = "unknown"
    elif validated.get("agent_architecture") == "unknown" and contains_any_keyword(text, single_markers):
        # Only auto-promote if the evidence is actually an assistant/agent feature.
        if any(x in text for x in ["assistant", "copilot", "chatbot", "coding assistant", "ai agent"]):
            validated["agent_architecture"] = "single-agent AI"

    return validated


def has_market_forecast_only_signal(text: str) -> bool:
    return contains_any_keyword(text, MARKET_FORECAST_KEYWORDS)


def has_strong_quantified_or_strategic_evidence(text: str, event_type: str) -> bool:
    text_l = safe_str(text).lower()
    has_number_or_money = bool(re.search(r"(\$\s?\d|\d+(\.\d+)?\s?(billion|million|%|percent|x|k|m|b))", text_l))
    has_traction = contains_any_keyword(text_l, STRONG_TRACTION_KEYWORDS)
    has_major_resource = contains_any_keyword(text_l, MAJOR_RESOURCE_KEYWORDS)
    has_market_forecast = has_market_forecast_only_signal(text_l)

    if has_market_forecast:
        target_specific_indicators = {"we launched", "we announced", "we invested", "we acquired", "customer", "customers", "users", "revenue", "cost reduction", "time savings", "throughput"}
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
    return not (has_positive_business_signal or has_quantified_positive_signal)


def normalize_funding_acquisition(validated: Dict[str, Any]) -> Dict[str, Any]:
    if not validated.get("is_usable_for_iss", False):
        return validated

    source_type = safe_str(validated.get("source_type", ""))
    text = " ".join([
        safe_str(validated.get("chunk_text", "")),
        safe_str(validated.get("source_title", "")),
        safe_str(validated.get("evidence_span", "")),
    ]).lower()

    has_funding_or_acq = any(t in text for t in ["funding round", "strategic investment", "investment", "acquisition", "acquires", "acquired", "capex", "data center", "data centers"])
    has_ai_specific = has_explicit_ai_signal(text) or any(t in text for t in ["ai infrastructure", "ai/ml platform", "ai compute", "data centers", "model training", "gpu"])
    generic_sector = any(t in text for t in ["general technology sector", "general technology"])

    if has_funding_or_acq:
        if not has_ai_specific or generic_sector:
            return force_not_usable(validated, "Excluded from ISS because the funding/acquisition signal is generic or not tied to a clear AI theme.")
        if validated.get("event_type") in {"unknown", "Strategic Commitment"}:
            validated["event_type"] = "Resourcing"

    if source_type == "CVC_from_google_news":
        # Conservative cap for sparse CVC one-liners.
        strength = int(validated.get("strength_1_5", 1))
        has_amount = bool(re.search(r"(\$\s?\d|\d+(\.\d+)?\s?(billion|million|m|b))", text))
        has_major_action = any(t in text for t in ["acquisition", "funding round", "strategic investment", "investment", "data center"])
        if has_amount and has_ai_specific and has_major_action:
            validated["strength_1_5"] = min(strength, 3)
        else:
            validated["strength_1_5"] = min(strength, 2)

    return validated


def calibrate_strength(validated: Dict[str, Any]) -> Dict[str, Any]:
    if not validated.get("is_usable_for_iss", False):
        validated["strength_1_5"] = 1
        return validated

    chunk_text = safe_str(validated.get("chunk_text", ""))
    evidence_span = safe_str(validated.get("evidence_span", ""))
    combined_text = f"{chunk_text} {evidence_span}"
    event_type = validated.get("event_type", "unknown")
    strength = int(validated.get("strength_1_5", 1))

    if is_negative_risk_event_without_positive_signal(validated):
        return force_not_usable(
            validated,
            "Excluded from ISS because the evidence is primarily legal, harm, safety-risk, or negative incident coverage rather than a positive target-company investment-intent event."
        )

    if validated.get("development_scope") == "unknown" or event_type == "unknown":
        return force_not_usable(validated, "Excluded from ISS because no clear ISS scope or event type is supported.")

    if strength >= 5 and not has_strong_quantified_or_strategic_evidence(combined_text, event_type):
        strength = 4

    is_nvidia_mlperf_signal = (
        validated.get("company") == "NVIDIA"
        and "mlperf" in combined_text.lower()
        and validated.get("development_scope") == "Horizontal AI"
    )
    if (
        contains_any_keyword(combined_text, RESEARCH_OR_DEMO_KEYWORDS)
        and not is_nvidia_mlperf_signal
        and event_type not in {"Commercial Traction", "Resourcing"}
    ):
        strength = min(strength, 3)

    if event_type == "Product Shipping & Iteration":
        has_traction_or_resource = contains_any_keyword(combined_text, STRONG_TRACTION_KEYWORDS) or contains_any_keyword(combined_text, MAJOR_RESOURCE_KEYWORDS)
        if contains_any_keyword(combined_text, FEATURE_RELEASE_KEYWORDS) and not has_traction_or_resource:
            strength = min(strength, 3)

    if event_type == "Strategic Commitment" and not contains_any_keyword(combined_text, MAJOR_RESOURCE_KEYWORDS):
        strength = min(strength, 3)

    if "startup_ecosystem" in validated.get("sub_theme", []):
        if not has_strong_quantified_or_strategic_evidence(combined_text, event_type):
            strength = min(strength, 3)

    validated["strength_1_5"] = max(1, min(5, strength))
    return validated


def postprocess_guard(validated: Dict[str, Any]) -> Dict[str, Any]:
    """Final deterministic cleanup after LLM output."""
    chunk_text = safe_str(validated.get("chunk_text", ""))
    company = safe_str(validated.get("company", ""))

    # Guard 0: low-quality source pages.
    if is_low_quality_source_page(validated):
        return force_not_usable(validated, "Excluded from ISS because the source appears to be an author, tag, archive, search, or index page rather than a standalone business event.")

    # Guard 1: require AI/autonomy/model signal in chunk.
    if not has_explicit_ai_signal(chunk_text):
        return force_not_usable(validated, "Excluded from ISS because the chunk does not contain an explicit AI-related business signal.")

    # Guard 1b: require evidence/title/chunk support for AI signal.
    if not evidence_span_has_explicit_ai_signal(validated):
        return force_not_usable(validated, "Excluded from ISS because the supporting evidence does not contain an explicit AI/model/agent/autonomy/inference signal.")

    # Guard 2: require target company, with official-source exception.
    if company and not mentions_target_company(chunk_text, company) and not is_official_target_source(validated):
        return force_not_usable(validated, f"Excluded from ISS because the chunk does not clearly mention the target company {company} as an actor.")

    # Guard 3: evidence owner mismatch.
    if company and likely_other_company_owns_evidence(validated):
        return force_not_usable(validated, f"Excluded from ISS because the concrete event in the evidence appears to be owned by another company, not the target company {company}.")

    # Guard 4: chunk main actor mismatch. Skip for official target sources.
    if company and not is_official_target_source(validated) and likely_other_company_main_actor(chunk_text, company):
        return force_not_usable(validated, f"Excluded from ISS because the main AI/business event appears to be about another company, not the target company {company}.")

    # Guard 5: user/LLM not usable.
    if not validated.get("is_usable_for_iss", False):
        return force_not_usable(validated, infer_best_exclusion_reason(validated))

    if validated.get("event_type") == "unknown":
        return force_not_usable(validated, "Excluded from ISS because no clear ISS event type is supported.")

    if validated.get("development_scope") == "unknown":
        return force_not_usable(validated, "Excluded from ISS because no clear Vertical AI or Horizontal AI scope is supported.")

    # Normalizers.
    validated = normalize_general_assistant_scope(validated)
    validated = normalize_labels_to_evidence_span(validated)
    validated = normalize_nvidia_scope(validated)
    validated = normalize_agent_architecture(validated)
    validated = normalize_funding_acquisition(validated)

    # Re-validate subthemes after normalizers.
    validated["sub_theme"] = validate_subthemes_for_scope(
        normalize_subthemes(validated.get("sub_theme", ["unknown"])),
        validated.get("development_scope", "unknown"),
    )

    # Final strength calibration.
    validated = calibrate_strength(validated)

    # Keep fields consistent if a later normalizer excluded it.
    if not validated.get("is_usable_for_iss", False):
        return force_not_usable(validated, validated.get("reason_short") or infer_best_exclusion_reason(validated))

    return validated


# =========================
# Validation / model output merge
# =========================


def validate_model_output(output: Dict[str, Any], original_record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate model output, merge original fields, then apply deterministic guards."""
    validated = dict(original_record)

    development_scope = safe_str(output.get("development_scope", "unknown"))
    validated["development_scope"] = development_scope if development_scope in VALID_DEVELOPMENT_SCOPES else "unknown"

    agent_architecture = safe_str(output.get("agent_architecture", "unknown"))
    validated["agent_architecture"] = agent_architecture if agent_architecture in VALID_AGENT_ARCHITECTURES else "unknown"

    event_type = safe_str(output.get("event_type", "unknown"))
    validated["event_type"] = event_type if event_type in VALID_EVENT_TYPES else "unknown"

    subthemes = normalize_subthemes(output.get("sub_theme", ["unknown"]))
    validated["sub_theme"] = validate_subthemes_for_scope(subthemes, validated["development_scope"])

    try:
        strength = int(output.get("strength_1_5", 1))
    except Exception:
        strength = 1
    validated["strength_1_5"] = max(1, min(5, strength))

    evidence_span = safe_str(output.get("evidence_span", ""))
    validated["evidence_span"] = evidence_span
    validated["evidence_span_verified"] = verify_evidence_span(evidence_span, validated.get("chunk_text", ""))

    validated["reason_short"] = safe_str(output.get("reason_short", ""))

    confidence = safe_str(output.get("confidence", "low"))
    validated["confidence"] = confidence if confidence in VALID_CONFIDENCE else "low"

    raw_usable = output.get("is_usable_for_iss", False)
    if isinstance(raw_usable, bool):
        validated["is_usable_for_iss"] = raw_usable
    elif isinstance(raw_usable, str):
        validated["is_usable_for_iss"] = raw_usable.strip().lower() == "true"
    else:
        validated["is_usable_for_iss"] = bool(raw_usable)

    if not validated["is_usable_for_iss"]:
        validated["development_scope"] = "unknown"
        validated["agent_architecture"] = "unknown"
        validated["sub_theme"] = ["unknown"]
        validated["event_type"] = "unknown"
        validated["strength_1_5"] = 1

    validated = postprocess_guard(validated)
    return validated


# =========================
# Batch API helpers
# =========================


def build_user_prompt(record: Dict[str, Any]) -> str:
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
                {"role": "user", "content": build_user_prompt(record)},
            ],
        },
    }


def extract_json(text: str) -> Dict[str, Any]:
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
# Submit / Collect commands
# =========================


def cmd_submit(args):
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

    records_per_shard = args.records_per_shard
    shards = [to_label[i:i + records_per_shard] for i in range(0, len(to_label), records_per_shard)]
    print(f"Split into {len(shards)} shard(s) of up to {records_per_shard} records each.")

    oai = get_client()
    batch_ids = []

    for shard_idx, shard in enumerate(shards, 1):
        shard_file = script_dir / f"_batch_requests_shard_{shard_idx}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for record in shard:
                f.write(json.dumps(build_batch_request(record), ensure_ascii=False) + "\n")

        file_size_mb = shard_file.stat().st_size / (1024 * 1024)
        print(f"\nShard {shard_idx}/{len(shards)}: {len(shard)} records, {file_size_mb:.1f} MB")

        with open(shard_file, "rb") as f:
            uploaded = oai.files.create(file=f, purpose="batch")
        print(f"  Uploaded file: {uploaded.id}")

        batch = oai.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"  Batch ID: {batch.id}  Status: {batch.status}")
        batch_ids.append(batch.id)

    state = {
        "input_file": args.input,
        "output_file": args.output,
        "batch_ids": batch_ids,
        "submitted_at": int(time.time()),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"\nSaved batch state to: {state_path}")


def build_input_lookup(input_file: str) -> Dict[str, Dict[str, Any]]:
    records = load_records(input_file)
    return {safe_str(r.get("chunk_id", "")): r for r in records if r.get("chunk_id", "")}


def cmd_collect(args):
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    error_path = script_dir / ERROR_FILE

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    input_lookup = build_input_lookup(str(input_path))
    oai = get_client()
    batch = oai.batches.retrieve(args.batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status:   {batch.status}")

    if batch.status != "completed":
        print("Batch is not completed yet. Try again later.")
        sys.exit(1)

    if not batch.output_file_id:
        print("Error: completed batch has no output_file_id.")
        sys.exit(1)

    content = oai.files.content(batch.output_file_id).read().decode("utf-8")

    success_count = 0
    error_count = 0
    not_usable_count = 0

    with open(output_path, "a", encoding="utf-8") as out_f, open(error_path, "a", encoding="utf-8") as err_f:
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                custom_id = row.get("custom_id", "")
                original = input_lookup.get(custom_id)
                if not original:
                    raise ValueError(f"custom_id not found in input: {custom_id}")

                if row.get("error"):
                    raise ValueError(f"Batch row error: {row['error']}")

                message_content = row["response"]["body"]["choices"][0]["message"]["content"]
                model_json = extract_json(message_content)
                validated = validate_model_output(model_json, original)
                out_f.write(json.dumps(validated, ensure_ascii=False) + "\n")
                success_count += 1
                if not validated.get("is_usable_for_iss", False):
                    not_usable_count += 1
            except Exception as e:
                error_count += 1
                err_f.write(json.dumps({"error": str(e), "raw_line": line}, ensure_ascii=False) + "\n")

    print("Collection complete:")
    print(f"  Success:        {success_count}")
    print(f"  Errors:         {error_count}")
    print(f"  Not usable ISS: {not_usable_count}")
    print(f"  Output:         {output_path}")
    if error_count:
        print(f"  Error file:     {error_path}")


def cmd_collect_all(args):
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

    print(f"Found {len(batch_ids)} batch(es).")
    for i, batch_id in enumerate(batch_ids, 1):
        print(f"\n{'=' * 50}\nCollecting batch {i}/{len(batch_ids)}: {batch_id}\n{'=' * 50}")
        collect_args = argparse.Namespace(batch_id=batch_id, input=input_file, output=output_file)
        try:
            cmd_collect(collect_args)
        except SystemExit:
            print(f"WARNING: Batch {batch_id} not collected. Continuing.")


def wait_for_batch(oai: OpenAI, batch_id: str, poll_seconds: int = 30):
    while True:
        batch = oai.batches.retrieve(batch_id)
        status = batch.status
        counts = getattr(batch, "request_counts", None)
        if counts:
            print(f"    {status} | done={counts.completed} failed={counts.failed} total={counts.total}")
        else:
            print(f"    {status}")
        if status in {"completed", "failed", "cancelled", "expired"}:
            return batch
        time.sleep(poll_seconds)


def cmd_run_all(args):
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output

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

    records_per_shard = args.records_per_shard
    shards = [to_label[i:i + records_per_shard] for i in range(0, len(to_label), records_per_shard)]
    print(f"Split into {len(shards)} shard(s) of up to {records_per_shard} records each.")

    oai = get_client()
    grand_success = 0
    grand_errors = 0

    for shard_idx, shard in enumerate(shards, 1):
        print(f"\n{'=' * 60}\nSHARD {shard_idx}/{len(shards)} ({len(shard)} records)\n{'=' * 60}")
        shard_file = script_dir / f"_batch_requests_runall_shard_{shard_idx}.jsonl"
        with open(shard_file, "w", encoding="utf-8") as f:
            for record in shard:
                f.write(json.dumps(build_batch_request(record), ensure_ascii=False) + "\n")

        print(f"  Wrote shard file: {shard_file.stat().st_size / (1024 * 1024):.1f} MB")
        with open(shard_file, "rb") as f:
            uploaded = oai.files.create(file=f, purpose="batch")
        print(f"  Uploaded: {uploaded.id}")

        batch = oai.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h")
        print(f"  Batch ID: {batch.id}  Status: {batch.status}")
        print("  Waiting for completion...")
        final_batch = wait_for_batch(oai, batch.id, poll_seconds=args.poll_seconds)

        if final_batch.status != "completed":
            print(f"  ERROR: Batch ended with status {final_batch.status}. Skipping collect.")
            grand_errors += len(shard)
            continue

        before_done = load_done_chunk_ids(str(output_path))
        collect_args = argparse.Namespace(batch_id=batch.id, input=args.input, output=args.output)
        cmd_collect(collect_args)
        after_done = load_done_chunk_ids(str(output_path))
        grand_success += len(after_done - before_done)

    print(f"\n{'=' * 60}\nALL SHARDS COMPLETE\n{'=' * 60}")
    print(f"  Total newly succeeded: {grand_success}")
    print(f"  Total errors:          {grand_errors}")
    print(f"  Output:                {output_path}")


# =========================
# Optional helper for ISS scoring script,方案 B
# =========================

SCORING_SUBTHEME_EXPANSION_HELPER = r'''
# Add this logic to your ISS scoring script when reading labeled records.
# It keeps sub_theme as a list in labeling output, but expands it during scoring.

def normalize_subtheme_list(raw_sub_theme):
    if raw_sub_theme is None:
        return ["unknown"]
    if isinstance(raw_sub_theme, list):
        tags = [str(x).strip() for x in raw_sub_theme if str(x).strip()]
    elif isinstance(raw_sub_theme, str):
        s = raw_sub_theme.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                tags = json.loads(s.replace("'", '"'))
            except Exception:
                tags = [x.strip(" '\"") for x in s.strip("[]").split(",")]
        else:
            tags = [s]
    else:
        tags = [str(raw_sub_theme).strip()]
    tags = [t for t in tags if t]
    return tags or ["unknown"]

# In expand_base_record_to_scoring_events(base), replace the single raw_sub_theme handling with:
# subthemes = normalize_subtheme_list(base["raw_sub_theme"])
# real_subthemes = [s for s in subthemes if s != "unknown"] or ["unknown"]
# subtheme_fraction = 1.0 / len(real_subthemes)
# for one_sub_theme in real_subthemes:
#     create one scoring event for one_sub_theme
#     add event["subtheme_fraction"] = subtheme_fraction
#     optionally scale event_weight = event_weight * subtheme_fraction
'''


def cmd_print_scoring_patch(_args):
    print(SCORING_SUBTHEME_EXPANSION_HELPER)


# =========================
# CLI
# =========================


def main():
    parser = argparse.ArgumentParser(description="BGV ISS Chunk Labeling with OpenAI Batch API")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub = subparsers.add_parser("submit", help="Submit a labeling batch to OpenAI")
    sub.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSONL file")
    sub.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSONL file for resume check")
    sub.add_argument("--records-per-shard", type=int, default=100, help="Records per batch shard")
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

    ra = subparsers.add_parser("run-all", help="Submit and collect all shards one at a time")
    ra.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSONL file")
    ra.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSONL file")
    ra.add_argument("--records-per-shard", type=int, default=650, help="Records per shard")
    ra.add_argument("--poll-seconds", type=int, default=30, help="Batch polling interval")
    ra.set_defaults(func=cmd_run_all)

    sp = subparsers.add_parser("print-scoring-patch", help="Print the ISS scoring patch for sub_theme list expansion")
    sp.set_defaults(func=cmd_print_scoring_patch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
