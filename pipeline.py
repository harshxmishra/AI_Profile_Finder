from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from apify_client import ApifyClient
from openai import OpenAI


DEFAULT_ACTOR_ID = "bebity/linkedin-premium-actor"
DEFAULT_MAX_PROFILES_PER_RUN = 50
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_RAW_DIRNAME = "raw_profiles"
DEFAULT_INTERMEDIATE_CSV = "intermediate_profiles.csv"
DEFAULT_FINAL_CSV = "final_candidates_llm.csv"
DEFAULT_PHASE1_CSV = "1_profiles_phase.csv"
DEFAULT_PHASE2_CSV = "2_normalized_profiles.csv"
DEFAULT_PHASE3_CSV = "3_profiles_with_dossier.csv"
DEFAULT_PHASE3_FILTERED_CSV = "3_relevance_profiles.csv"
DEFAULT_PHASE4_CSV = "4_scored_collaborators_llm.csv"
DEFAULT_PHASE5_CSV = "5_all_candidates_llm.csv"

DASHBOARD_REQUIRED_COLUMNS = [
    "full_name",
    "url",
    "headline",
    "tier",
    "best_fit",
    "confidence",
    "risk_flags",
    "blogs_score",
    "blogs_reasoning",
    "blogs_evidence",
    "courses_score",
    "courses_reasoning",
    "courses_evidence",
    "hack_session_score",
    "hack_session_reasoning",
    "hack_session_evidence",
    "powertalk_score",
    "powertalk_reasoning",
    "powertalk_evidence",
    "total_score",
    "influence_score",
    "adjusted_score",
]

SCORING_KEYS = [
    "blogs",
    "courses",
    "hack_session",
    "powertalk",
    "best_fit",
    "confidence",
    "risk_flags",
]
CATEGORY_KEYS = ["blogs", "courses", "hack_session", "powertalk"]


@dataclass
class PipelineArtifacts:
    phase1_df: pd.DataFrame
    phase2_df: pd.DataFrame
    phase3_df: pd.DataFrame
    phase3_filtered_df: pd.DataFrame
    phase4_df: pd.DataFrame
    phase5_df: pd.DataFrame
    phase1_csv_path: str
    phase2_csv_path: str
    phase3_csv_path: str
    phase3_filtered_csv_path: str
    phase4_csv_path: str
    phase5_csv_path: str
    intermediate_csv_path: str
    final_csv_path: str
    output_dir: str
    raw_profiles_dir: str
    blog_campaign_csv_path: str
    course_campaign_csv_path: str
    hack_campaign_csv_path: str
    powertalk_campaign_csv_path: str
    a_tier_csv_path: str
    high_conf_csv_path: str


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_logger(message: str) -> None:
    print(message, flush=True)


class PipelineLogger:
    def __init__(self, logger: Callable[[str], None] = default_logger):
        self.logger = logger

    def log(self, message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.logger(f"[{ts} UTC] {message}")



def clean_text(text: Any, max_chars: Optional[int] = None) -> str:
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if max_chars is not None:
        return text[:max_chars]
    return text


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []



def try_parse_maybe_json(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    s = x.strip()
    if not s:
        return x
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(s)
            except Exception:
                continue
    return x



def dataframe_parse_structured_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(try_parse_maybe_json)
    return out



def get_apify_client(apify_api_token: Optional[str] = None) -> ApifyClient:
    token = apify_api_token or os.getenv("APIFY_API_TOKEN")
    if not token:
        raise ValueError("Missing APIFY_API_TOKEN. Set it in your environment or pass apify_api_token.")
    return ApifyClient(token)



def get_openai_client(openai_api_key: Optional[str] = None) -> OpenAI:
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY. Set it in your environment or pass openai_api_key.")
    return OpenAI(api_key=api_key, timeout=60.0, max_retries=2)



def build_actor_input(
    keywords: List[str] | str,
    location: Optional[List[str] | str] = None,
    limit: int = 100,
    is_name: bool = True,
    is_url: bool = False,
    max_profiles_per_run: int = DEFAULT_MAX_PROFILES_PER_RUN,
) -> Dict[str, Any]:
    if isinstance(keywords, str):
        keywords = [keywords]
    if isinstance(location, str):
        location = [location]
    return {
        "action": "get-profiles",
        "isName": is_name,
        "isUrl": is_url,
        "keywords": keywords,
        "limit": min(limit, max_profiles_per_run),
        "location": location or [],
    }



def fetch_profiles_from_apify(
    run_input: Dict[str, Any],
    client: ApifyClient,
    actor_id: str = DEFAULT_ACTOR_ID,
    logger: Callable[[str], None] = default_logger,
) -> List[Dict[str, Any]]:
    logger(f"Starting Apify actor run with actor_id={actor_id}")
    logger(f"Apify actor input: {json.dumps(run_input, ensure_ascii=False)}")
    run = client.actor(actor_id).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]
    logger(f"Apify actor finished. Dataset ID: {dataset_id}")
    logger("Downloading dataset items from Apify...")

    dataset_client = client.dataset(dataset_id)
    items: List[Dict[str, Any]] = []
    offset = 0
    page_limit = 100
    while True:
        logger(f"Requesting dataset page with offset={offset}, limit={page_limit}")
        page = dataset_client.list_items(offset=offset, limit=page_limit, clean=True)
        batch = list(page.items)
        logger(f"Received dataset batch size: {len(batch)}")
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        logger(f"Downloaded {len(items)} profiles from dataset so far")
        if len(batch) < page_limit:
            break

    logger(f"Total profiles fetched: {len(items)}")
    return items



def sanitize_profile_filename(value: str) -> str:
    value = value.replace("https://", "").replace("http://", "")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value[:180] or "profile"



def save_raw_profiles(items: List[Dict[str, Any]], raw_dir: str | Path, logger: Callable[[str], None] = default_logger) -> str:
    raw_path = ensure_dir(raw_dir)
    logger(f"Saving {len(items)} raw profile JSON files into {raw_path}")
    for idx, item in enumerate(items):
        profile_id = item.get("url") or f"profile_{idx}"
        filename = sanitize_profile_filename(profile_id)
        with open(raw_path / f"{filename}.json", "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
    logger("Raw profile JSON save complete")
    return str(raw_path)



def validate_profile_schema(profile: Dict[str, Any]) -> bool:
    required_fields = ["firstName", "lastName", "headline", "url"]
    return all(field in profile for field in required_fields)



def convert_to_dataframe(items: List[Dict[str, Any]], logger: Callable[[str], None] = default_logger) -> pd.DataFrame:
    total = len(items)
    valid_profiles = [item for item in items if validate_profile_schema(item)]
    invalid_count = total - len(valid_profiles)
    logger(f"Schema validation complete. Valid profiles: {len(valid_profiles)} | Invalid profiles: {invalid_count}")
    if invalid_count:
        logger("Some profiles were dropped because required fields were missing")
    df = pd.DataFrame(valid_profiles)
    logger(f"Phase 1 dataframe created with shape {df.shape}")
    return df



def extract_child_text(node: Any) -> List[str]:
    texts: List[str] = []
    if not isinstance(node, dict):
        return texts
    if node.get("text"):
        texts.append(str(node["text"]))
    for child in node.get("child", []):
        texts.extend(extract_child_text(child))
    return texts



def extract_about(profile: Dict[str, Any]) -> Dict[str, str]:
    about_section = profile.get("ABOUT")
    if not isinstance(about_section, list):
        return {"about_text": "", "top_skills": ""}
    about_text = ""
    top_skills = ""
    for item in about_section:
        if not isinstance(item, dict):
            continue
        if "text" in item:
            about_text += f" {item['text']}"
        if item.get("title") == "Top skills":
            subtitle = item.get("subtitle")
            if isinstance(subtitle, dict):
                top_skills = subtitle.get("text", "")
    return {
        "about_text": clean_text(about_text, 2000),
        "top_skills": clean_text(top_skills, 500),
    }



def extract_experience(profile: Dict[str, Any]) -> List[Dict[str, str]]:
    experience_section = profile.get("EXPERIENCE")
    if not isinstance(experience_section, list):
        return []

    experiences: List[Dict[str, str]] = []
    for company_block in experience_section:
        if not isinstance(company_block, dict):
            continue
        company_name = company_block.get("title")
        company_duration = company_block.get("subtitle")
        child_roles = company_block.get("child")
        if isinstance(child_roles, list) and child_roles:
            for role in child_roles:
                if not isinstance(role, dict):
                    continue
                description_texts: List[str] = []
                for desc_node in role.get("child", []):
                    description_texts.extend(extract_child_text(desc_node))
                experiences.append(
                    {
                        "company": clean_text(company_name, 200),
                        "company_duration": clean_text(company_duration, 100),
                        "title": clean_text(role.get("title"), 200),
                        "employment_type": clean_text(role.get("subtitle"), 100),
                        "location": clean_text(role.get("meta"), 100),
                        "duration": clean_text(role.get("caption"), 100),
                        "description": clean_text(" ".join(description_texts), 1500),
                    }
                )
        else:
            description_texts = extract_child_text(company_block)
            experiences.append(
                {
                    "company": "",
                    "company_duration": "",
                    "title": clean_text(company_block.get("title"), 200),
                    "employment_type": clean_text(company_block.get("subtitle"), 100),
                    "location": "",
                    "duration": clean_text(company_block.get("caption"), 100),
                    "description": clean_text(" ".join(description_texts), 1500),
                }
            )
    return experiences[:15]



def extract_education(profile: Dict[str, Any]) -> List[Dict[str, str]]:
    section = profile.get("EDUCATION", [])
    if not isinstance(section, list):
        return []
    return [
        {"school": clean_text(edu.get("title"), 200), "degree": clean_text(edu.get("subtitle"), 200)}
        for edu in section[:5]
        if isinstance(edu, dict)
    ]



def extract_projects(profile: Dict[str, Any]) -> List[Dict[str, str]]:
    section = profile.get("PROJECTS", [])
    if not isinstance(section, list):
        return []
    projects: List[Dict[str, str]] = []
    for proj in section[:10]:
        if not isinstance(proj, dict):
            continue
        description_texts: List[str] = []
        for child in proj.get("child", []):
            description_texts.extend(extract_child_text(child))
        projects.append(
            {
                "title": clean_text(proj.get("title"), 200),
                "duration": clean_text(proj.get("subtitle"), 100),
                "description": clean_text(" ".join(description_texts), 1000),
            }
        )
    return projects



def extract_simple_section(profile: Dict[str, Any], section_name: str, limit: int = 10) -> List[str]:
    section = profile.get(section_name)
    if not isinstance(section, list):
        return []
    results: List[str] = []
    for item in section[:limit]:
        if isinstance(item, dict):
            title = item.get("title") or item.get("text")
            if title:
                results.append(clean_text(title, 300))
    return results



def extract_volunteer_causes(profile: Dict[str, Any]) -> List[str]:
    section = profile.get("VOLUNTEER_CAUSES")
    if not isinstance(section, list):
        return []
    out: List[str] = []
    for item in section[:10]:
        if isinstance(item, dict):
            txt = item.get("text") or item.get("title")
            if txt:
                out.append(clean_text(txt, 200))
    return out



def normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    about_data = extract_about(profile)
    return {
        "full_name": f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
        "headline": clean_text(profile.get("headline")),
        "url": profile.get("url"),
        "about_text": about_data["about_text"],
        "top_skills": about_data["top_skills"],
        "experience": extract_experience(profile),
        "education": extract_education(profile),
        "certifications": extract_simple_section(profile, "LICENSES_AND_CERTIFICATIONS"),
        "publications": extract_simple_section(profile, "PUBLICATIONS"),
        "projects": extract_projects(profile),
        "awards": extract_simple_section(profile, "HONORS_AND_AWARDS"),
        "languages": extract_simple_section(profile, "LANGUAGES"),
        "organizations": extract_simple_section(profile, "ORGANIZATIONS"),
        "volunteer_causes": extract_volunteer_causes(profile),
    }



def normalize_profiles_dataframe(phase1_df: pd.DataFrame, logger: Callable[[str], None] = default_logger) -> pd.DataFrame:
    logger(f"Normalizing {len(phase1_df)} profiles")
    normalized_profiles = []
    total = len(phase1_df)
    for idx, (_, row) in enumerate(phase1_df.iterrows(), start=1):
        logger(f"Normalizing profile {idx}/{total}: {clean_text(row.get('firstName', ''))} {clean_text(row.get('lastName', ''))}".strip())
        normalized_profiles.append(normalize_profile(row.to_dict()))
    df = pd.DataFrame(normalized_profiles)
    logger(f"Normalization complete with shape {df.shape}")
    return df



def parse_month_year(text: Any) -> Optional[tuple[int, int]]:
    if not text:
        return None
    text = str(text)
    m = re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s+(\d{4})", text)
    if m:
        month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        return int(m.group(2)), month_map[m.group(1)]
    y = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if y:
        return int(y.group(1)), 1
    return None



def estimate_years_from_experience(experience_list: Any) -> float:
    exp = safe_list(experience_list)
    years = 0.0
    for e in exp:
        if not isinstance(e, dict):
            continue
        dur = ((e.get("duration") or "") + " " + (e.get("company_duration") or "")).lower()
        matched = re.search(r"(\d+(?:\.\d+)?)\s*(?:yr|yrs|year|years)", dur)
        if matched:
            years += float(matched.group(1))
            continue
        months = re.search(r"(\d+)\s*(?:mo|mos|month|months)", dur)
        if months:
            years += round(int(months.group(1)) / 12.0, 2)
            continue
        start = parse_month_year(dur)
        years_found = re.findall(r"\b(19\d{2}|20\d{2})\b", dur)
        if start and len(years_found) >= 2:
            start_year = int(years_found[0])
            end_year = int(years_found[-1])
            years += max(end_year - start_year, 0.5)
        elif "present" in dur or dur.strip():
            years += 0.5
    return round(min(years, 25.0), 1)



def detect_recency_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    exp = safe_list(row.get("experience"))
    now_year = datetime.utcnow().year
    recent_speaking = False
    recent_genai = False
    for e in exp:
        if not isinstance(e, dict):
            continue
        title = (e.get("title") or "").lower()
        desc = (e.get("description") or "").lower()
        dur = (e.get("duration") or "").lower()
        year_hit = False
        y = re.search(r"\b(20\d{2})\b", dur)
        if y and int(y.group(1)) >= now_year - 2:
            year_hit = True
        if "present" in dur:
            year_hit = True
        if any(k in title for k in ["speaker", "keynote", "panel", "guest speaker", "lecturer", "talk", "ted"]):
            if year_hit:
                recent_speaking = True
        if any(k in desc for k in ["genai", "generative ai", "llm", "rag", "agent", "agents", "prompt", "openai", "azure ai", "foundry"]):
            if year_hit:
                recent_genai = True
    return {"recent_speaking": recent_speaking, "recent_genai": recent_genai}



def build_dossier_json(
    row: Dict[str, Any],
    max_exp: int = 10,
    max_exp_desc_chars: int = 800,
    max_about_chars: int = 2000,
    max_project_desc_chars: int = 600,
) -> Dict[str, Any]:
    dossier = {
        "identity": {
            "full_name": row.get("full_name", ""),
            "headline": row.get("headline", ""),
            "url": row.get("url", ""),
        },
        "about": {
            "about_text": (row.get("about_text") or "")[:max_about_chars],
            "top_skills": row.get("top_skills") or "",
        },
        "experience": [],
        "education": safe_list(row.get("education")),
        "certifications": safe_list(row.get("certifications")),
        "publications": safe_list(row.get("publications")),
        "projects": [],
        "awards": safe_list(row.get("awards")),
        "languages": safe_list(row.get("languages")),
        "organizations": safe_list(row.get("organizations")),
        "volunteer_causes": safe_list(row.get("volunteer_causes")),
        "meta_counts": {},
    }
    exp_list = safe_list(row.get("experience"))

    def is_speaking_role(e: Dict[str, Any]) -> bool:
        t = (e.get("title") or "").lower()
        return any(k in t for k in ["speaker", "keynote", "panel", "guest speaker", "lecturer", "ted"])

    speaking = [e for e in exp_list if isinstance(e, dict) and is_speaking_role(e)]
    non_speaking = [e for e in exp_list if isinstance(e, dict) and not is_speaking_role(e)]
    ordered_exp = (speaking + non_speaking)[:max_exp]

    for e in ordered_exp:
        dossier["experience"].append(
            {
                "company": e.get("company", ""),
                "company_duration": e.get("company_duration", ""),
                "title": e.get("title", ""),
                "employment_type": e.get("employment_type", ""),
                "location": e.get("location", ""),
                "duration": e.get("duration", ""),
                "description": (e.get("description") or "")[:max_exp_desc_chars],
            }
        )

    proj_list = safe_list(row.get("projects"))
    for p in proj_list[:10]:
        if isinstance(p, dict):
            dossier["projects"].append(
                {
                    "title": p.get("title", ""),
                    "duration": p.get("duration", ""),
                    "description": (p.get("description") or "")[:max_project_desc_chars],
                }
            )
        else:
            dossier["projects"].append({"title": str(p)[:200], "duration": "", "description": ""})

    dossier["meta_counts"] = {
        "experience_count": len(exp_list),
        "speaking_role_count": len(speaking),
        "education_count": len(safe_list(row.get("education"))),
        "certifications_count": len(safe_list(row.get("certifications"))),
        "publications_count": len(safe_list(row.get("publications"))),
        "projects_count": len(proj_list),
        "awards_count": len(safe_list(row.get("awards"))),
        "organizations_count": len(safe_list(row.get("organizations"))),
        "volunteer_causes_count": len(safe_list(row.get("volunteer_causes"))),
    }
    dossier["derived"] = {
        "estimated_years_experience": estimate_years_from_experience(exp_list),
        **detect_recency_flags({"experience": exp_list}),
    }
    return dossier



def build_dossier_text(dossier: Dict[str, Any]) -> str:
    ident = dossier["identity"]
    about = dossier["about"]
    derived = dossier.get("derived", {})
    counts = dossier.get("meta_counts", {})

    lines: List[str] = []
    lines.append(f"NAME: {ident.get('full_name', '')}")
    lines.append(f"HEADLINE: {ident.get('headline', '')}")
    lines.append(f"URL: {ident.get('url', '')}")
    lines.append("")
    lines.append("ABOUT:")
    lines.append(about.get("about_text", ""))
    lines.append("")
    lines.append("TOP SKILLS:")
    lines.append(about.get("top_skills", ""))
    lines.append("")
    lines.append("DERIVED SUMMARY:")
    lines.append(json.dumps(derived, ensure_ascii=False))
    lines.append("")
    lines.append("COUNTS:")
    lines.append(json.dumps(counts, ensure_ascii=False))
    lines.append("")
    lines.append("EXPERIENCE (most relevant first):")
    for e in dossier.get("experience", []):
        lines.append(
            f"- {e.get('title', '')} | {e.get('company', '')} | {e.get('duration', '')} | {e.get('location', '')}\n"
            f"  Type: {e.get('employment_type', '')} | Company tenure: {e.get('company_duration', '')}\n"
            f"  Description: {e.get('description', '')}"
        )
    lines.append("")
    lines.append("EDUCATION:")
    for edu in dossier.get("education", []):
        if isinstance(edu, dict):
            lines.append(f"- {edu.get('degree', '')} | {edu.get('school', '')}")
        else:
            lines.append(f"- {str(edu)[:200]}")
    lines.append("")
    lines.append("CERTIFICATIONS:")
    for c in dossier.get("certifications", []):
        lines.append(f"- {str(c)[:250]}")
    lines.append("")
    lines.append("PUBLICATIONS:")
    for p in dossier.get("publications", []):
        lines.append(f"- {str(p)[:250]}")
    lines.append("")
    lines.append("PROJECTS:")
    for p in dossier.get("projects", []):
        if isinstance(p, dict):
            lines.append(f"- {p.get('title', '')} | {p.get('duration', '')}\n  {p.get('description', '')}")
        else:
            lines.append(f"- {str(p)[:250]}")
    lines.append("")
    lines.append("AWARDS:")
    for a in dossier.get("awards", []):
        lines.append(f"- {str(a)[:250]}")
    lines.append("")
    lines.append("ORGANIZATIONS:")
    for o in dossier.get("organizations", []):
        lines.append(f"- {str(o)[:250]}")
    lines.append("")
    lines.append("VOLUNTEER CAUSES:")
    for v in dossier.get("volunteer_causes", []):
        lines.append(f"- {str(v)[:250]}")
    lines.append("")
    lines.append("LANGUAGES:")
    for l in dossier.get("languages", []):
        lines.append(f"- {str(l)[:120]}")
    return "\n".join(lines).strip()



def build_dossiers_dataframe(phase2_df: pd.DataFrame, logger: Callable[[str], None] = default_logger) -> pd.DataFrame:
    logger(f"Building dossiers for {len(phase2_df)} normalized profiles")
    phase3_df = phase2_df.copy()
    phase3_df = dataframe_parse_structured_columns(phase3_df)
    total = len(phase3_df)
    dossier_jsons: List[Dict[str, Any]] = []
    dossier_texts: List[str] = []
    for idx, (_, row) in enumerate(phase3_df.iterrows(), start=1):
        logger(f"Building dossier {idx}/{total}: {row.get('full_name', '')}")
        dossier_json = build_dossier_json(row.to_dict())
        dossier_jsons.append(dossier_json)
        dossier_texts.append(build_dossier_text(dossier_json))
    phase3_df["dossier_json"] = dossier_jsons
    phase3_df["dossier_text"] = dossier_texts
    logger("Dossier building complete")
    return phase3_df



def build_relevance_prompt(dossier_text: str) -> str:
    return f"""
You are screening a LinkedIn profile dossier.
Your job is to decide whether this person is meaningfully active in:
- AI
- Machine Learning
- Data Science
- Data Engineering
- GenAI
- LLM systems
- Applied AI infrastructure
Important:
- They must have technical involvement.
- Speaking alone is not enough.
- Management without AI depth is not enough.
- Generic tech roles without AI exposure are not enough.
- Vague buzzwords are not enough.
Only approve if there is clear, credible AI or data technical depth.
Return ONLY valid JSON:
{{
  "ai_relevant": true or false,
  "confidence": 0-100,
  "reason": "one short sentence explaining decision"
}}
Dossier:
{dossier_text}
""".strip()



def llm_ai_relevance_gate(
    dossier_text: str,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
    max_retries: int = 3,
    logger: Callable[[str], None] = default_logger,
) -> Dict[str, Any]:
    prompt = build_relevance_prompt(dossier_text)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger(f"Relevance LLM attempt {attempt}/{max_retries} - sending request")
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0,
            )
            logger(f"Relevance LLM attempt {attempt}/{max_retries} - response received")
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            obj = json.loads(raw)
            if isinstance(obj, dict) and all(k in obj for k in ["ai_relevant", "confidence", "reason"]):
                return obj
            last_error = "Missing required keys in relevance JSON"
            logger(f"Relevance LLM attempt {attempt}/{max_retries} - invalid JSON schema")
        except Exception as e:
            last_error = str(e)
            logger(f"Relevance LLM attempt {attempt}/{max_retries} failed: {e}")
        time.sleep(1.0)
    logger(f"Relevance classification failed after retries: {last_error}")
    return {"ai_relevant": False, "confidence": 0, "reason": f"Relevance classification failed: {last_error or 'unknown'}"}



def apply_relevance_filter(
    phase3_df: pd.DataFrame,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
    logger: Callable[[str], None] = default_logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = phase3_df.copy()
    total = len(out)
    relevance_results = []
    logger(f"Starting relevance filter for {total} dossiers")
    for idx, text in enumerate(out["dossier_text"].tolist(), start=1):
        logger(f"Relevance check {idx}/{total} - starting")
        relevance_results.append(llm_ai_relevance_gate(text, client=client, model=model, logger=logger))
        logger(f"Relevance check {idx}/{total} - completed")
    out["relevance_json"] = relevance_results
    out["ai_relevant"] = out["relevance_json"].apply(lambda x: x["ai_relevant"])
    out["relevance_confidence"] = out["relevance_json"].apply(lambda x: x["confidence"])
    out["relevance_reason"] = out["relevance_json"].apply(lambda x: x["reason"])
    filtered = out[out["ai_relevant"] == True].reset_index(drop=True)
    logger(f"Relevance filter complete. Retained {len(filtered)} of {len(out)} profiles")
    return out, filtered



def build_scoring_prompt(dossier_text: str) -> str:
    return f"""
You are evaluating a potential collaborator based on a LinkedIn profile dossier.
Hard rules:
- Use ONLY the dossier text. Do not guess or invent.
- Every score must cite evidence quotes copied from the dossier text.
- If evidence is missing, score lower and include a risk_flag "missing_evidence".
- Keep reasoning short and specific (no generic fluff).
- Output ONLY valid JSON. No markdown. No extra text.
Scoring scale (0–5):
0 = no evidence / irrelevant
1 = weak evidence
2 = some evidence but shallow
3 = solid and credible
4 = strong, clearly proven
5 = exceptional, world-class (rare)
Collaboration areas:
1) blogs
Score technical blog contribution potential:
- technical depth in GenAI/data/ML/cloud
- demonstrated writing/publications/blogs
- clarity and real-world implementation
- bonus for publications or repeated content creation
Penalize buzzwords, vague claims, no writing evidence.
2) courses
Score course development potential:
- teaching/instructor/trainer evidence
- structured curriculum building signals
- ability to explain complex topics
- bonus for workshops, mentoring, certifications relevant to GenAI
Penalize no teaching signals.
3) hack_session
Score GenAI hack session facilitation potential:
- hands-on GenAI build experience (agents, RAG, LLM workflows, frameworks)
- real implementation evidence, not just interest
- demo/workshop/hackathon signals
- bonus for open-source/github/projects and recent GenAI
Penalize theoretical-only.
4) powertalk
Score leadership / strategy talk potential:
- leadership scope, seniority, strategy, impact
- strong speaking record, community talks, keynote/panel roles
- real-world business impact and storytelling potential
Penalize junior scope with no leadership.
Return JSON with this exact structure:
{{
  "blogs": {{"score": 0, "reasoning": "", "evidence": ["", ""]}},
  "courses": {{"score": 0, "reasoning": "", "evidence": ["", ""]}},
  "hack_session": {{"score": 0, "reasoning": "", "evidence": ["", ""]}},
  "powertalk": {{"score": 0, "reasoning": "", "evidence": ["", ""]}},
  "best_fit": "Blogs|Courses|Hack Session|PowerTalk",
  "confidence": 0,
  "risk_flags": ["..."]
}}
Dossier:
{dossier_text}
""".strip()



def validate_scoring_json(obj: Dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Not a dict"
    for k in SCORING_KEYS:
        if k not in obj:
            return False, f"Missing key: {k}"
    for cat in CATEGORY_KEYS:
        c = obj.get(cat)
        if not isinstance(c, dict):
            return False, f"{cat} is not a dict"
        if "score" not in c or "reasoning" not in c or "evidence" not in c:
            return False, f"{cat} missing score/reasoning/evidence"
        if not isinstance(c["score"], int) or not (0 <= c["score"] <= 5):
            return False, f"{cat}.score invalid"
        if not isinstance(c["reasoning"], str):
            return False, f"{cat}.reasoning not str"
        ev = c["evidence"]
        if not isinstance(ev, list) or len(ev) < 2:
            return False, f"{cat}.evidence must be list with >=2 items"
        for e in ev:
            if not isinstance(e, str) or not e.strip():
                return False, f"{cat}.evidence contains empty"
    if obj["best_fit"] not in ["Blogs", "Courses", "Hack Session", "PowerTalk"]:
        return False, "best_fit invalid"
    if not isinstance(obj["confidence"], int) or not (0 <= obj["confidence"] <= 100):
        return False, "confidence invalid"
    if not isinstance(obj["risk_flags"], list):
        return False, "risk_flags not list"
    return True, ""



def score_profile_with_llm(
    dossier_text: str,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
    max_retries: int = 3,
    sleep_s: float = 1.0,
    logger: Callable[[str], None] = default_logger,
) -> Dict[str, Any]:
    prompt = build_scoring_prompt(dossier_text)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger(f"Scoring LLM attempt {attempt}/{max_retries} - sending request")
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                timeout=60.0,
            )
            logger(f"Scoring LLM attempt {attempt}/{max_retries} - response received")
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.replace("json", "", 1).strip()
            obj = json.loads(raw)
            ok, err = validate_scoring_json(obj)
            if ok:
                return obj
            last_error = f"Validation failed: {err}"
            logger(f"Scoring LLM attempt {attempt}/{max_retries} invalid schema: {err}")
            prompt = build_scoring_prompt(dossier_text) + "\n\nYour previous JSON was invalid. Output ONLY valid JSON with the exact schema and correct types."
        except Exception as e:
            last_error = str(e)
            logger(f"Scoring LLM attempt {attempt}/{max_retries} failed: {e}")
            prompt = build_scoring_prompt(dossier_text) + "\n\nYour previous output was not valid JSON. Output ONLY valid JSON with the exact schema."
        time.sleep(sleep_s)
    fallback_reason = "Scoring failed after retries."
    fallback_evidence = ["No valid model response received.", "Proceeding with zero score fallback."]
    logger(f"Scoring failed after retries. Using fallback response. Last error: {last_error}")
    return {
        "blogs": {"score": 0, "reasoning": fallback_reason, "evidence": fallback_evidence},
        "courses": {"score": 0, "reasoning": fallback_reason, "evidence": fallback_evidence},
        "hack_session": {"score": 0, "reasoning": fallback_reason, "evidence": fallback_evidence},
        "powertalk": {"score": 0, "reasoning": fallback_reason, "evidence": fallback_evidence},
        "best_fit": "Blogs",
        "confidence": 0,
        "risk_flags": ["scoring_failed", f"error:{last_error}" if last_error else "error:unknown"],
    }



def flatten_scoring(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "blogs_score": obj["blogs"]["score"],
        "blogs_reasoning": obj["blogs"]["reasoning"],
        "blogs_evidence": json.dumps(obj["blogs"]["evidence"], ensure_ascii=False),
        "courses_score": obj["courses"]["score"],
        "courses_reasoning": obj["courses"]["reasoning"],
        "courses_evidence": json.dumps(obj["courses"]["evidence"], ensure_ascii=False),
        "hack_session_score": obj["hack_session"]["score"],
        "hack_session_reasoning": obj["hack_session"]["reasoning"],
        "hack_session_evidence": json.dumps(obj["hack_session"]["evidence"], ensure_ascii=False),
        "powertalk_score": obj["powertalk"]["score"],
        "powertalk_reasoning": obj["powertalk"]["reasoning"],
        "powertalk_evidence": json.dumps(obj["powertalk"]["evidence"], ensure_ascii=False),
        "best_fit": obj["best_fit"],
        "confidence": obj["confidence"],
        "risk_flags": json.dumps(obj["risk_flags"], ensure_ascii=False),
    }



def assign_tier_llm(row: pd.Series) -> str:
    if row["total_score"] >= 16 and row["confidence"] >= 70:
        return "A"
    if row["total_score"] >= 12:
        return "B"
    return "C"



def score_profiles_dataframe(
    phase3_filtered_df: pd.DataFrame,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
    logger: Callable[[str], None] = default_logger,
) -> pd.DataFrame:
    phase4_df = phase3_filtered_df.copy()
    total = len(phase4_df)
    logger(f"Starting Phase 4 scoring for {total} relevant profiles")
    scoring_results = []
    for idx, row in enumerate(phase4_df.to_dict(orient="records"), start=1):
        name = row.get("full_name", "")
        logger(f"LLM scoring {idx}/{total} - candidate: {name} - starting")
        result = score_profile_with_llm(row.get("dossier_text", ""), client=client, model=model, logger=logger)
        logger(f"LLM scoring {idx}/{total} - candidate: {name} - completed")
        scoring_results.append(result)
    phase4_df["llm_scoring_json"] = scoring_results
    scoring_flat = pd.json_normalize(phase4_df["llm_scoring_json"].apply(flatten_scoring))
    phase4_df = pd.concat([phase4_df.drop(columns=["llm_scoring_json"]), scoring_flat], axis=1)
    score_cols = ["blogs_score", "courses_score", "hack_session_score", "powertalk_score"]
    phase4_df["total_score"] = phase4_df[score_cols].sum(axis=1)
    phase4_df["tier"] = phase4_df.apply(assign_tier_llm, axis=1)
    phase4_df = phase4_df.sort_values(by=["tier", "total_score", "confidence"], ascending=[True, False, False]).reset_index(drop=True)
    logger("Phase 4 scoring dataframe preparation complete")
    return phase4_df



def compute_influence_score_from_row(row: pd.Series) -> int:
    dossier = row.get("dossier_json")
    if isinstance(dossier, str):
        try:
            dossier = json.loads(dossier)
        except Exception:
            dossier = None
    if isinstance(dossier, dict):
        counts = dossier.get("meta_counts", {})
        speaking = counts.get("speaking_role_count", 0)
        pubs = counts.get("publications_count", 0)
        certs = counts.get("certifications_count", 0)
        awards = counts.get("awards_count", 0)
        orgs = counts.get("organizations_count", 0)
        projects = counts.get("projects_count", 0)
    else:
        exp = safe_list(row.get("experience"))
        speaking = sum(1 for e in exp if isinstance(e, dict) and "speaker" in (e.get("title", "").lower()))
        pubs = len(safe_list(row.get("publications")))
        certs = len(safe_list(row.get("certifications")))
        awards = len(safe_list(row.get("awards")))
        orgs = len(safe_list(row.get("organizations")))
        projects = len(safe_list(row.get("projects")))
    score = 0
    score += min(speaking, 3)
    score += min(pubs, 2)
    score += min(certs, 2)
    score += min(awards, 1)
    score += min(orgs, 1)
    score += min(projects, 1)
    return int(score)



def build_outreach_snapshot_llm(row: pd.Series) -> str:
    return (
        f"Tier {row.get('tier')} | Best fit: {row.get('best_fit')} | "
        f"Total: {row.get('total_score')} | Conf: {row.get('confidence')} | "
        f"Influence: {row.get('influence_score')}"
    )



def finalize_rankings(phase4_df: pd.DataFrame, logger: Callable[[str], None] = default_logger) -> Dict[str, pd.DataFrame]:
    logger(f"Finalizing rankings for {len(phase4_df)} scored profiles")
    phase5_df = phase4_df.copy()
    phase5_df["influence_score"] = phase5_df.apply(compute_influence_score_from_row, axis=1)
    phase5_df["adjusted_score"] = (
        phase5_df["total_score"] + 0.25 * phase5_df["influence_score"] + 0.02 * phase5_df["confidence"]
    )
    phase5_df["outreach_snapshot"] = phase5_df.apply(build_outreach_snapshot_llm, axis=1)
    blog_campaign = phase5_df.sort_values(by=["blogs_score", "confidence", "adjusted_score"], ascending=[False, False, False]).reset_index(drop=True)
    course_campaign = phase5_df.sort_values(by=["courses_score", "confidence", "adjusted_score"], ascending=[False, False, False]).reset_index(drop=True)
    hack_campaign = phase5_df.sort_values(by=["hack_session_score", "confidence", "adjusted_score"], ascending=[False, False, False]).reset_index(drop=True)
    powertalk_campaign = phase5_df.sort_values(by=["powertalk_score", "confidence", "adjusted_score"], ascending=[False, False, False]).reset_index(drop=True)
    a_tier = phase5_df[phase5_df["tier"] == "A"].sort_values(by=["adjusted_score"], ascending=False).reset_index(drop=True)
    high_conf = phase5_df[phase5_df["confidence"] >= 75].sort_values(by=["adjusted_score"], ascending=False).reset_index(drop=True)
    logger("Ranking outputs created")
    return {
        "phase5_df": phase5_df,
        "blog_campaign": blog_campaign,
        "course_campaign": course_campaign,
        "hack_campaign": hack_campaign,
        "powertalk_campaign": powertalk_campaign,
        "a_tier": a_tier,
        "high_conf": high_conf,
    }



def prepare_dashboard_output(phase5_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = DASHBOARD_REQUIRED_COLUMNS + ["outreach_snapshot"]
    output_columns = [c for c in output_columns if c in phase5_df.columns]
    return phase5_df[output_columns].copy()



def save_dataframe(df: pd.DataFrame, path: str | Path, logger: Callable[[str], None] = default_logger) -> str:
    target = Path(path)
    ensure_dir(target.parent)
    df.to_csv(target, index=False)
    logger(f"Saved CSV: {target} | rows={len(df)} | cols={len(df.columns)}")
    return str(target)



def run_full_pipeline(
    keywords: List[str] | str,
    location: Optional[List[str] | str] = None,
    limit: int = 50,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    actor_id: str = DEFAULT_ACTOR_ID,
    apify_api_token: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    relevance_model: str = "gpt-4.1-mini",
    scoring_model: str = "gpt-4.1-mini",
    logger: Callable[[str], None] = default_logger,
    on_phase_complete: Optional[Callable[[str, pd.DataFrame, str], None]] = None,
) -> PipelineArtifacts:
    plog = PipelineLogger(logger)
    base_dir = ensure_dir(output_dir)
    raw_dir = ensure_dir(base_dir / DEFAULT_RAW_DIRNAME)

    plog.log(f"Pipeline starting. Output dir: {base_dir}")
    plog.log(f"Keywords: {keywords}")
    plog.log(f"Locations: {location}")
    plog.log(f"Limit: {limit}")
    plog.log("Initializing clients...")
    apify_client = get_apify_client(apify_api_token)
    openai_client = get_openai_client(openai_api_key)
    plog.log("Clients initialized successfully")

    plog.log("Running Phase 1: fetch profiles from Apify")
    run_input = build_actor_input(keywords=keywords, location=location, limit=limit)
    raw_items = fetch_profiles_from_apify(run_input=run_input, client=apify_client, actor_id=actor_id, logger=plog.log)
    save_raw_profiles(raw_items, raw_dir, logger=plog.log)
    phase1_df = convert_to_dataframe(raw_items, logger=plog.log)
    phase1_csv_path = save_dataframe(phase1_df, base_dir / DEFAULT_PHASE1_CSV, logger=plog.log)
    plog.log(f"Phase 1 complete. Rows: {len(phase1_df)}")
    if on_phase_complete is not None:
        on_phase_complete("phase1", phase1_df.copy(), phase1_csv_path)

    plog.log("Running Phase 2: normalize profiles")
    phase2_df = normalize_profiles_dataframe(phase1_df, logger=plog.log)
    phase2_csv_path = save_dataframe(phase2_df, base_dir / DEFAULT_PHASE2_CSV, logger=plog.log)
    plog.log(f"Phase 2 complete. Rows: {len(phase2_df)}")
    if on_phase_complete is not None:
        on_phase_complete("phase2", phase2_df.copy(), phase2_csv_path)

    plog.log("Running Phase 3: build dossiers")
    phase3_df = build_dossiers_dataframe(phase2_df, logger=plog.log)
    phase3_for_save = phase3_df.copy()
    phase3_for_save["dossier_json"] = phase3_for_save["dossier_json"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    phase3_csv_path = save_dataframe(phase3_for_save, base_dir / DEFAULT_PHASE3_CSV, logger=plog.log)
    plog.log(f"Phase 3 complete. Rows: {len(phase3_df)}")
    if on_phase_complete is not None:
        on_phase_complete("phase3", phase3_for_save.copy(), phase3_csv_path)

    plog.log("Running Phase 3B: relevance filter")
    phase3_full_df, phase3_filtered_df = apply_relevance_filter(phase3_df, client=openai_client, model=relevance_model, logger=plog.log)
    phase3_filtered_for_save = phase3_filtered_df.copy()
    phase3_filtered_for_save["dossier_json"] = phase3_filtered_for_save["dossier_json"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
    phase3_filtered_for_save["relevance_json"] = phase3_filtered_for_save["relevance_json"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
    phase3_filtered_csv_path = save_dataframe(phase3_filtered_for_save, base_dir / DEFAULT_PHASE3_FILTERED_CSV, logger=plog.log)

    intermediate_df = phase3_full_df.copy()
    intermediate_df["dossier_json"] = intermediate_df["dossier_json"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
    intermediate_df["relevance_json"] = intermediate_df["relevance_json"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
    intermediate_csv_path = save_dataframe(intermediate_df, base_dir / DEFAULT_INTERMEDIATE_CSV, logger=plog.log)
    plog.log(f"Phase 3B complete. Relevant rows: {len(phase3_filtered_df)}")
    if on_phase_complete is not None:
        on_phase_complete("phase3_filtered", phase3_filtered_for_save.copy(), phase3_filtered_csv_path)

    plog.log("Running Phase 4: LLM scoring")
    phase4_df = score_profiles_dataframe(phase3_filtered_df, client=openai_client, model=scoring_model, logger=plog.log)
    phase4_for_save = phase4_df.copy()
    phase4_for_save["dossier_json"] = phase4_for_save["dossier_json"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
    if "relevance_json" in phase4_for_save.columns:
        phase4_for_save["relevance_json"] = phase4_for_save["relevance_json"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)
    phase4_csv_path = save_dataframe(phase4_for_save, base_dir / DEFAULT_PHASE4_CSV, logger=plog.log)
    plog.log(f"Phase 4 complete. Rows: {len(phase4_df)}")
    if on_phase_complete is not None:
        on_phase_complete("phase4", phase4_for_save.copy(), phase4_csv_path)

    plog.log("Running Phase 5: ranking and campaign outputs")
    ranking_outputs = finalize_rankings(phase4_df, logger=plog.log)
    phase5_df = ranking_outputs["phase5_df"]
    dashboard_df = prepare_dashboard_output(phase5_df)
    final_csv_path = save_dataframe(dashboard_df, base_dir / DEFAULT_FINAL_CSV, logger=plog.log)
    phase5_csv_path = save_dataframe(dashboard_df, base_dir / DEFAULT_PHASE5_CSV, logger=plog.log)
    blog_campaign_csv_path = save_dataframe(ranking_outputs["blog_campaign"], base_dir / "blog_campaign.csv", logger=plog.log)
    course_campaign_csv_path = save_dataframe(ranking_outputs["course_campaign"], base_dir / "course_campaign.csv", logger=plog.log)
    hack_campaign_csv_path = save_dataframe(ranking_outputs["hack_campaign"], base_dir / "hack_campaign.csv", logger=plog.log)
    powertalk_campaign_csv_path = save_dataframe(ranking_outputs["powertalk_campaign"], base_dir / "powertalk_campaign.csv", logger=plog.log)
    a_tier_csv_path = save_dataframe(ranking_outputs["a_tier"], base_dir / "a_tier_shortlist.csv", logger=plog.log)
    high_conf_csv_path = save_dataframe(ranking_outputs["high_conf"], base_dir / "high_conf_shortlist.csv", logger=plog.log)
    plog.log(f"Phase 5 complete. Rows: {len(dashboard_df)}")
    if on_phase_complete is not None:
        on_phase_complete("phase5", dashboard_df.copy(), final_csv_path)

    plog.log(f"Pipeline complete. Final rows: {len(dashboard_df)}")
    return PipelineArtifacts(
        phase1_df=phase1_df,
        phase2_df=phase2_df,
        phase3_df=phase3_full_df,
        phase3_filtered_df=phase3_filtered_df,
        phase4_df=phase4_df,
        phase5_df=dashboard_df,
        phase1_csv_path=phase1_csv_path,
        phase2_csv_path=phase2_csv_path,
        phase3_csv_path=phase3_csv_path,
        phase3_filtered_csv_path=phase3_filtered_csv_path,
        phase4_csv_path=phase4_csv_path,
        phase5_csv_path=phase5_csv_path,
        intermediate_csv_path=intermediate_csv_path,
        final_csv_path=final_csv_path,
        output_dir=str(base_dir),
        raw_profiles_dir=str(raw_dir),
        blog_campaign_csv_path=blog_campaign_csv_path,
        course_campaign_csv_path=course_campaign_csv_path,
        hack_campaign_csv_path=hack_campaign_csv_path,
        powertalk_campaign_csv_path=powertalk_campaign_csv_path,
        a_tier_csv_path=a_tier_csv_path,
        high_conf_csv_path=high_conf_csv_path,
    )


__all__ = [
    "PipelineArtifacts",
    "DASHBOARD_REQUIRED_COLUMNS",
    "build_actor_input",
    "fetch_profiles_from_apify",
    "save_raw_profiles",
    "convert_to_dataframe",
    "normalize_profiles_dataframe",
    "build_dossiers_dataframe",
    "apply_relevance_filter",
    "score_profiles_dataframe",
    "finalize_rankings",
    "prepare_dashboard_output",
    "run_full_pipeline",
]

