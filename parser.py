"""
parser.py — Resume Ingestion and Parsing Module
Extracts text from PDFs, parses skills/experience using spaCy + rule-based extraction.
Supports batch processing with multiprocessing and caching.
"""

import re
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pdfplumber
import spacy

from utils import (
    RESUMES_DIR, SPACY_MODEL_PATH, PARSED_CACHE_PATH,
    extract_skills_from_text, get_enriched_skills,
    extract_experience_years, extract_year_mentions,
    file_hash, load_json_cache, save_json_cache,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# spaCy model loading (from local path only)
# ---------------------------------------------------------------------------
_nlp = None


def _get_nlp():
    """Load spaCy model from local /models directory. Cached after first call."""
    global _nlp
    if _nlp is None:
        model_path = str(SPACY_MODEL_PATH)
        try:
            _nlp = spacy.load(model_path)
            logger.info(f"Loaded spaCy model from {model_path}")
        except OSError:
            # Fallback: try loading by name (if installed in venv)
            try:
                _nlp = spacy.load("en_core_web_sm")
                logger.warning("Loaded spaCy model from venv (not local path)")
            except OSError:
                raise RuntimeError(
                    f"spaCy model not found at {model_path}. "
                    "Run setup.bat to download models."
                )
    return _nlp


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    Handles corrupt pages, empty pages, and extraction errors gracefully.
    """
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
                    logger.warning(f"Error extracting page {i + 1} from {pdf_path.name}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path.name}: {e}")
        return ""

    full_text = "\n".join(text_parts)

    if not full_text.strip():
        logger.warning(f"No text extracted from {pdf_path.name} (may be image-based)")

    return full_text


# ---------------------------------------------------------------------------
# Job title extraction (rule-based)
# ---------------------------------------------------------------------------
_TITLE_PATTERNS = [
    re.compile(
        r'(?:title|role|position|designation)\s*[:;]\s*(.+?)(?:\n|$)',
        re.IGNORECASE
    ),
    re.compile(
        r'^((?:senior|junior|lead|principal|staff|chief|head of|vp of|director of)\s+'
        r'(?:software|data|ml|ai|frontend|backend|fullstack|full-stack|devops|cloud|'
        r'product|project|program|engineering|machine learning|research)\s*'
        r'(?:engineer|developer|scientist|analyst|architect|manager|lead|consultant)?)',
        re.IGNORECASE | re.MULTILINE
    ),
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
    "systems administrator", "database administrator", "network engineer",
    "security engineer", "cybersecurity analyst", "consultant", "cto", "cio", "vp engineering",
]


def extract_titles(text: str) -> list[str]:
    """Extract job titles from resume text."""
    if not text:
        return []

    titles = []
    text_lower = text.lower()

    # Pattern-based extraction
    for pattern in _TITLE_PATTERNS:
        matches = pattern.findall(text)
        for m in matches:
            title = m.strip().rstrip(".,;:")
            if 3 < len(title) < 80:
                titles.append(title)

    # Common title matching
    for title in _COMMON_TITLES:
        if title in text_lower:
            titles.append(title.title())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in titles:
        t_lower = t.lower().strip()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)

    return unique[:5]  # Return top 5 titles


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------
_DEGREE_PATTERNS = re.compile(
    r'\b(ph\.?d|doctorate|master(?:\'s)?|m\.?s\.?|m\.?b\.?a\.?|'
    r'bachelor(?:\'s)?|b\.?s\.?|b\.?e\.?|b\.?tech|m\.?tech|'
    r'associate(?:\'s)?|diploma|certification|certified)\b',
    re.IGNORECASE
)


def extract_education(text: str) -> list[str]:
    """Extract education qualifications from resume text."""
    if not text:
        return []
    matches = _DEGREE_PATTERNS.findall(text)
    return list(set(m.strip() for m in matches))


# ---------------------------------------------------------------------------
# Full resume parsing
# ---------------------------------------------------------------------------
def parse_resume(text: str, filename: str = "") -> dict[str, Any]:
    """
    Parse resume text into structured data:
    - skills (set of canonical skill names)
    - enriched_skills (skills + broader categories)
    - experience_years (float)
    - titles (list of job titles)
    - education (list of degrees)
    - year_mentions (list of years found in text)
    - text_length (int)
    """
    if not text or not text.strip():
        return {
            "filename": filename,
            "raw_text": "",
            "skills": [],
            "enriched_skills": [],
            "experience_years": 0.0,
            "titles": [],
            "education": [],
            "year_mentions": [],
            "text_length": 0,
        }

    # Extract skills using taxonomy
    skills = extract_skills_from_text(text)
    enriched = get_enriched_skills(skills)

    # Extract experience
    exp_years = extract_experience_years(text)

    # Extract titles
    titles = extract_titles(text)

    # Extract education
    education = extract_education(text)

    # Extract year mentions for recency
    year_mentions = extract_year_mentions(text)

    # Use spaCy for additional entity extraction (if needed)
    try:
        nlp = _get_nlp()
        # Only process first 100K chars to avoid memory issues
        doc = nlp(text[:100000])

        # Extract organization names as potential employers
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        # Keep unique, top 10
        orgs = list(dict.fromkeys(orgs))[:10]
    except Exception as e:
        logger.warning(f"spaCy processing failed for {filename}: {e}")
        orgs = []

    return {
        "filename": filename,
        "raw_text": text,
        "skills": sorted(skills),
        "enriched_skills": sorted(enriched),
        "experience_years": exp_years,
        "titles": titles,
        "education": education,
        "organizations": orgs,
        "year_mentions": year_mentions,
        "text_length": len(text),
    }


# ---------------------------------------------------------------------------
# Single-file processing (for multiprocessing)
# ---------------------------------------------------------------------------
def _process_single_resume(pdf_path: str) -> dict[str, Any] | None:
    """Process a single resume PDF → parsed data. Used by batch processor."""
    try:
        path = Path(pdf_path)
        text = extract_text_from_pdf(path)
        if not text.strip():
            logger.warning(f"Empty text from {path.name}, skipping")
            return None
        parsed = parse_resume(text, filename=path.name)
        parsed["file_hash"] = file_hash(path)
        parsed["file_path"] = str(path)
        # Don't store raw_text in cache (too large)
        cache_entry = {k: v for k, v in parsed.items() if k != "raw_text"}
        cache_entry["raw_text_preview"] = text[:500]
        return parsed
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch processing with caching
# ---------------------------------------------------------------------------
def batch_parse_resumes(
    resume_dir: Path | None = None,
    max_workers: int = 4,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """
    Parse all PDF resumes in a directory.
    Uses caching to skip already-processed files.
    Uses multiprocessing for parallel processing.
    
    Returns list of parsed resume dicts.
    """
    if resume_dir is None:
        resume_dir = RESUMES_DIR

    # Discover PDFs
    pdf_files = sorted(resume_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {resume_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files in {resume_dir}")

    # Load cache
    cache = load_json_cache(PARSED_CACHE_PATH) if use_cache else {}

    # Determine which files need processing
    to_process = []
    cached_results = []

    for pdf_path in pdf_files:
        fhash = file_hash(pdf_path)
        cache_key = pdf_path.name

        if use_cache and cache_key in cache and cache.get(cache_key, {}).get("file_hash") == fhash:
            # Use cached result
            cached_data = cache[cache_key]
            # Restore raw_text by re-extracting (needed for embeddings)
            text = extract_text_from_pdf(pdf_path)
            cached_data["raw_text"] = text
            cached_results.append(cached_data)
            logger.debug(f"Cache hit: {pdf_path.name}")
        else:
            to_process.append(str(pdf_path))

    if cached_results:
        logger.info(f"Using cached data for {len(cached_results)} resumes")

    # Process new/changed files
    new_results = []
    if to_process:
        logger.info(f"Processing {len(to_process)} new/changed resumes...")

        # Use multiprocessing for CPU-bound parsing
        if len(to_process) > 1 and max_workers > 1:
            with ProcessPoolExecutor(max_workers=min(max_workers, len(to_process))) as executor:
                futures = {
                    executor.submit(_process_single_resume, path): path
                    for path in to_process
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        new_results.append(result)
        else:
            # Single-threaded for 1 file or max_workers=1
            for path in to_process:
                result = _process_single_resume(path)
                if result:
                    new_results.append(result)

    # Update cache with new results
    if new_results and use_cache:
        for r in new_results:
            cache_entry = {k: v for k, v in r.items() if k != "raw_text"}
            cache_entry["raw_text_preview"] = r.get("raw_text", "")[:500]
            cache[r["filename"]] = cache_entry
        save_json_cache(cache, PARSED_CACHE_PATH)
        logger.info(f"Cache updated with {len(new_results)} new entries")

    all_results = cached_results + new_results

    # Sort by filename for consistent ordering
    all_results.sort(key=lambda x: x.get("filename", ""))

    logger.info(f"Total parsed resumes: {len(all_results)}")
    return all_results


# ---------------------------------------------------------------------------
# JD Parsing
# ---------------------------------------------------------------------------
def parse_job_description(jd_text: str) -> dict[str, Any]:
    """
    Parse a job description to extract required skills, experience, etc.
    """
    skills = extract_skills_from_text(jd_text)
    enriched = get_enriched_skills(skills)
    exp_years = extract_experience_years(jd_text)
    titles = extract_titles(jd_text)

    return {
        "raw_text": jd_text,
        "skills": sorted(skills),
        "enriched_skills": sorted(enriched),
        "required_experience": exp_years,
        "titles": titles,
    }
