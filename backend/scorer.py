import math
import re
import logging
from collections import Counter
from datetime import datetime
from typing import Any

from utils import normalize_skill, get_enriched_skills, extract_skills_from_text

logger = logging.getLogger(__name__)

WEIGHTS = {
    "semantic_similarity": 0.35,
    "skill_match": 0.25,
    "experience_relevance": 0.20,
    "recency": 0.10,
    "keyword_precision": 0.10,
}
RECENCY_LAMBDA = 0.15

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can", "need", "must", "it", "its",
    "this", "that", "these", "those", "i", "we", "you", "he", "she", "they", "me", "us", "him", "her",
    "them", "my", "our", "your", "his", "their", "what", "which", "who", "whom", "where", "when", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "about", "above", "after", "again",
    "also", "any", "because", "before", "between", "during", "into", "through", "under", "up", "down",
    "out", "over", "then", "once", "here", "there", "if", "while", "per", "etc", "able", "work",
    "working", "worked", "using", "used", "use", "including", "include", "includes", "experience",
    "experienced", "responsible", "responsibilities", "responsibility", "role", "team", "new", "well",
    "strong", "good", "knowledge", "skills", "skill", "years", "year", "company", "project", "projects",
    "development", "developed", "developing", "manage", "managed", "management",
})

def compute_skill_match(resume_skills: set[str], jd_skills: set[str], use_enriched: bool = True) -> tuple[float, set[str], set[str]]:
    if not jd_skills:
        return 1.0, set(), set()
    resume_enriched = get_enriched_skills(resume_skills) if use_enriched else resume_skills
    jd_enriched = get_enriched_skills(jd_skills) if use_enriched else jd_skills

    matched = resume_enriched & jd_enriched
    missing = jd_skills - resume_enriched

    score = len(matched) / len(jd_enriched) if jd_enriched else 1.0
    extra_relevant = resume_enriched - jd_enriched
    if extra_relevant and len(jd_enriched) > 0:
        score = min(1.0, score + min(0.1, len(extra_relevant) * 0.01))
    return score, matched, missing

def compute_experience_relevance(candidate_years: float, required_years: float) -> float:
    if required_years <= 0:
        return min(1.0, 0.5 + candidate_years * 0.05) if candidate_years > 0 else 0.5
    ratio = candidate_years / required_years
    if 0.8 <= ratio <= 1.5:
        score = 0.8 + 0.2 * (1 - abs(ratio - 1.0) / 0.5)
    elif ratio < 0.8:
        score = 1.0 / (1.0 + math.exp(-5 * (ratio - 0.4)))
    else:
        score = max(0.6, 1.0 - 0.05 * (ratio - 1.5))
    return max(0.0, min(1.0, score))

def compute_recency(year_mentions: list[int]) -> float:
    if not year_mentions: return 0.3
    current_year = datetime.now().year
    valid_years = [y for y in year_mentions if 1990 <= y <= current_year + 1]
    if not valid_years: return 0.3
    scores = [math.exp(-RECENCY_LAMBDA * max(0, current_year - y)) for y in valid_years]
    scores.sort(reverse=True)
    top_scores = scores[:3]
    return sum(top_scores) / len(top_scores)

def _extract_keywords(text: str) -> Counter:
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return Counter(w for w in words if w not in _STOP_WORDS)

def compute_keyword_precision(resume_text: str, jd_text: str) -> float:
    if not jd_text or not resume_text: return 0.0
    jd_keywords = _extract_keywords(jd_text)
    resume_keywords = _extract_keywords(resume_text)
    if not jd_keywords: return 0.5
    top_jd = set(k for k, _ in jd_keywords.most_common(30))
    if not top_jd: return 0.5
    return len(top_jd & set(resume_keywords.keys())) / len(top_jd)

def apply_hard_filters(resume_skills: set[str], must_have_skills: set[str]) -> tuple[bool, set[str]]:
    if not must_have_skills: return True, set()
    enriched = get_enriched_skills(resume_skills)
    missing = must_have_skills - enriched
    return len(missing) == 0, missing

def score_candidate(resume_data: dict[str, Any], jd_data: dict[str, Any], semantic_score: float, must_have_skills: set[str] | None = None) -> dict[str, Any]:
    resume_skills = set(resume_data.get("enriched_skills", []))
    jd_skills = set(jd_data.get("enriched_skills", []))
    sem_score = max(0.0, min(1.0, semantic_score))
    skill_score, matched_skills, missing_skills = compute_skill_match(resume_skills, jd_skills)
    exp_years = resume_data.get("experience_years", 0.0)
    required_exp = jd_data.get("required_experience", 0.0)
    exp_score = compute_experience_relevance(exp_years, required_exp)
    recency_score = compute_recency(resume_data.get("year_mentions", []))
    kw_score = compute_keyword_precision(resume_data.get("raw_text", ""), jd_data.get("raw_text", ""))

    final_score = (WEIGHTS["semantic_similarity"] * sem_score + WEIGHTS["skill_match"] * skill_score + 
                   WEIGHTS["experience_relevance"] * exp_score + WEIGHTS["recency"] * recency_score + 
                   WEIGHTS["keyword_precision"] * kw_score)

    filtered = False
    missing_must_haves = set()
    if must_have_skills:
        passes, missing_must_haves = apply_hard_filters(resume_skills, must_have_skills)
        if not passes:
            filtered = True
            final_score = 0.0

    original_jd_skills = set(jd_data.get("skills", []))
    original_resume_skills = set(resume_data.get("skills", []))
    skill_match_pct = (len(original_resume_skills & original_jd_skills) / len(original_jd_skills) * 100) if original_jd_skills else 0.0

    parts = []
    if filtered:
        parts.append(f"Missing must-have: {', '.join(sorted(missing_must_haves)[:3])}")
    else:
        if skill_score >= 0.8: parts.append(f"Strong skill match ({skill_score:.0%})")
        elif skill_score >= 0.5: parts.append(f"Moderate skill match ({skill_score:.0%})")
        else: parts.append(f"Low skill match ({skill_score:.0%})")
        if exp_years > 0:
            parts.append(f"{exp_years:.0f}yr exp")
            if required_exp > 0: parts.append("meets exp req" if exp_years >= required_exp * 0.8 else f"needs {required_exp:.0f}yr")
        if sem_score >= 0.7: parts.append("highly relevant profile")
        elif sem_score >= 0.4: parts.append("relevant profile")
        if missing_skills: parts.append(f"missing: {', '.join(sorted(missing_skills)[:3])}")

    return {
        "filename": resume_data.get("filename", "unknown"),
        "final_score": round(final_score, 4),
        "semantic_score": round(sem_score, 4),
        "skill_score": round(skill_score, 4),
        "experience_score": round(exp_score, 4),
        "recency_score": round(recency_score, 4),
        "keyword_score": round(kw_score, 4),
        "skill_match_pct": round(skill_match_pct, 1),
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills | missing_must_haves),
        "experience_years": exp_years,
        "explanation": " | ".join(parts),
        "filtered": filtered,
    }

def rerank_candidates(faiss_results: list[dict[str, Any]], parsed_resumes: dict[str, dict], jd_data: dict[str, Any], must_have_skills: set[str] | None = None, top_k: int = 50) -> list[dict[str, Any]]:
    scored_candidates = []
    for result in faiss_results:
        filename = result["filename"]
        if resume_data := parsed_resumes.get(filename):
            candidate_score = score_candidate(resume_data, jd_data, result["score"], must_have_skills)
            scored_candidates.append(candidate_score)

    active = sorted([c for c in scored_candidates if not c["filtered"]], key=lambda x: x["final_score"], reverse=True)
    filtered = sorted([c for c in scored_candidates if c["filtered"]], key=lambda x: x["filename"])
    ranked = active + filtered
    for i, candidate in enumerate(ranked, 1): candidate["rank"] = i
    return ranked[:top_k]
