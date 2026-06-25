import re
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any
import os

import pdfplumber
import spacy

from utils import (
    SPACY_MODEL_PATH,
    extract_skills_from_text, get_enriched_skills,
    extract_experience_years, extract_year_mentions,
    file_hash, load_json_cache, save_json_cache,
)
from backend.user_context import get_resumes_dir, get_user_dir

logger = logging.getLogger(__name__)

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        model_path = str(SPACY_MODEL_PATH)
        try:
            _nlp = spacy.load(model_path)
            logger.info(f"Loaded spaCy model from {model_path}")
        except OSError:
            try:
                _nlp = spacy.load("en_core_web_sm")
                logger.warning("Loaded spaCy model from venv (not local path)")
            except OSError:
                raise RuntimeError(
                    f"spaCy model not found at {model_path}. "
                    "Run setup.bat to download models."
                )
    return _nlp

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return ""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text.strip())
                except Exception as e:
                    continue
    except Exception as e:
        return ""
    full_text = "\n".join(text_parts)
    return full_text

_TITLE_PATTERNS = [
    re.compile(r'(?:title|role|position|designation)\s*[:;]\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'^((?:senior|junior|lead|principal|staff|chief|head of|vp of|director of)\s+(?:software|data|ml|ai|frontend|backend|fullstack|full-stack|devops|cloud|product|project|program|engineering|machine learning|research)\s*(?:engineer|developer|scientist|analyst|architect|manager|lead|consultant)?)', re.IGNORECASE | re.MULTILINE),
]
_COMMON_TITLES = [
    "software engineer", "software developer", "data scientist", "data analyst",
    "data engineer", "machine learning engineer", "ml engineer", "ai engineer",
    "backend developer", "backend engineer", "frontend developer", "frontend engineer",
    "full stack developer", "fullstack developer", "full-stack engineer",
    "devops engineer", "cloud engineer", "sre", "site reliability engineer",
    "product manager", "project manager", "program manager", "engineering manager",
    "tech lead", "technical lead", "team lead", "architect", "solutions architect",
    "principal engineer", "staff engineer", "senior engineer", "junior developer",
    "research scientist", "research engineer", "qa engineer", "test engineer",
    "mobile developer", "ios developer", "android developer", "web developer",
    "ui developer", "ux designer", "ui/ux designer", "business analyst",
]

def extract_titles(text: str) -> list[str]:
    if not text:
        return []
    titles = []
    text_lower = text.lower()
    for pattern in _TITLE_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            title = m.strip().rstrip(".,;:")
            if 3 < len(title) < 80:
                titles.append(title)
    for title in _COMMON_TITLES:
        if title in text_lower:
            titles.append(title.title())
    seen = set()
    unique = []
    for t in titles:
        t_lower = t.lower().strip()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)
    return unique[:5]

_DEGREE_PATTERNS = re.compile(
    r'\b(ph\.?d|doctorate|master(?:\'s)?|m\.?s\.?|m\.?b\.?a\.?|'
    r'bachelor(?:\'s)?|b\.?s\.?|b\.?e\.?|b\.?tech|m\.?tech|'
    r'associate(?:\'s)?|diploma|certification|certified)\b',
    re.IGNORECASE
)

def extract_education(text: str) -> list[str]:
    if not text:
        return []
    matches = _DEGREE_PATTERNS.findall(text)
    return list(set(m.strip() for m in matches))

def parse_resume(text: str, filename: str = "") -> dict[str, Any]:
    if not text or not text.strip():
        return {
            "filename": filename, "raw_text": "", "skills": [], "enriched_skills": [],
            "experience_years": 0.0, "titles": [], "education": [], "year_mentions": [], "text_length": 0,
        }
    skills = extract_skills_from_text(text)
    enriched = get_enriched_skills(skills)
    exp_years = extract_experience_years(text)
    titles = extract_titles(text)
    education = extract_education(text)
    year_mentions = extract_year_mentions(text)
    
    try:
        nlp = _get_nlp()
        doc = nlp(text[:100000])
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        orgs = list(dict.fromkeys(orgs))[:10]
    except Exception:
        orgs = []

    return {
        "filename": filename, "raw_text": text, "skills": sorted(skills),
        "enriched_skills": sorted(enriched), "experience_years": exp_years,
        "titles": titles, "education": education, "organizations": orgs,
        "year_mentions": year_mentions, "text_length": len(text),
    }

def _process_single_resume(pdf_path: str) -> dict[str, Any] | None:
    try:
        path = Path(pdf_path)
        text = extract_text_from_pdf(path)
        if not text.strip():
            return None
        parsed = parse_resume(text, filename=path.name)
        parsed["file_hash"] = file_hash(path)
        parsed["file_path"] = str(path)
        return parsed
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return None

def batch_parse_resumes(user_id: str, max_workers: int = 1, use_cache: bool = True) -> list[dict[str, Any]]:
    resume_dir = Path(get_resumes_dir(user_id))
    pdf_files = sorted(resume_dir.glob("*.pdf"))
    if not pdf_files:
        return []

    cache_path = os.path.join(get_user_dir(user_id), "parsed_cache.json")
    cache = load_json_cache(cache_path) if use_cache else {}

    to_process = []
    cached_results = []

    for pdf_path in pdf_files:
        fhash = file_hash(pdf_path)
        cache_key = pdf_path.name
        if use_cache and cache_key in cache and cache.get(cache_key, {}).get("file_hash") == fhash:
            cached_data = cache[cache_key]
            text = extract_text_from_pdf(pdf_path)
            cached_data["raw_text"] = text
            cached_results.append(cached_data)
        else:
            to_process.append(str(pdf_path))

    new_results = []
    if to_process:
        if len(to_process) > 1 and max_workers > 1:
            with ProcessPoolExecutor(max_workers=min(max_workers, len(to_process))) as executor:
                futures = {executor.submit(_process_single_resume, path): path for path in to_process}
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        new_results.append(result)
        else:
            for path in to_process:
                result = _process_single_resume(path)
                if result:
                    new_results.append(result)

    if new_results and use_cache:
        for r in new_results:
            cache_entry = {k: v for k, v in r.items() if k != "raw_text"}
            cache_entry["raw_text_preview"] = r.get("raw_text", "")[:500]
            cache[r["filename"]] = cache_entry
        save_json_cache(cache, cache_path)

    all_results = cached_results + new_results
    all_results.sort(key=lambda x: x.get("filename", ""))
    return all_results

def parse_job_description(jd_text: str) -> dict[str, Any]:
    skills = extract_skills_from_text(jd_text)
    enriched = get_enriched_skills(skills)
    exp_years = extract_experience_years(jd_text)
    titles = extract_titles(jd_text)
    return {
        "raw_text": jd_text, "skills": sorted(skills), "enriched_skills": sorted(enriched),
        "required_experience": exp_years, "titles": titles,
    }
