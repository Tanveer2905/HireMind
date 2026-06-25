"""
scorer.py — Composite Scoring Engine and Reranking Layer
Implements multi-factor candidate evaluation that mimics recruiter judgment.

Scoring Formula:
    final_score = 0.35 * semantic_similarity
                + 0.25 * skill_match
                + 0.20 * experience_relevance
                + 0.10 * recency
                + 0.10 * keyword_precision
"""

import math
import re
import logging
from collections import Counter
from datetime import datetime
from typing import Any

from utils import normalize_skill, get_enriched_skills, extract_skills_from_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights (configurable)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "semantic_similarity": 0.35,
    "skill_match": 0.25,
    "experience_relevance": 0.20,
    "recency": 0.10,
    "keyword_precision": 0.10,
}

# Recency decay parameter (higher = faster decay for old skills)
RECENCY_LAMBDA = 0.15

# Stop words for keyword precision (filtered out)
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "must", "it", "its", "this", "that", "these", "those", "i", "we",
    "you", "he", "she", "they", "me", "us", "him", "her", "them", "my",
    "our", "your", "his", "their", "what", "which", "who", "whom",
    "where", "when", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "no", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "about", "above",
    "after", "again", "also", "any", "because", "before", "between",
    "during", "into", "through", "under", "up", "down", "out", "over",
    "then", "once", "here", "there", "if", "while", "per", "etc",
    "able", "work", "working", "worked", "using", "used", "use",
    "including", "include", "includes", "experience", "experienced",
    "responsible", "responsibilities", "responsibility", "role", "team",
    "new", "well", "strong", "good", "knowledge", "skills", "skill",
    "years", "year", "company", "project", "projects", "development",
    "developed", "developing", "manage", "managed", "management",
})


# ---------------------------------------------------------------------------
# Sub-score: Skill Match
# ---------------------------------------------------------------------------
def compute_skill_match(
    resume_skills: set[str],
    jd_skills: set[str],
    use_enriched: bool = True,
) -> tuple[float, set[str], set[str]]:
    """
    Compute skill match score between resume and job description.
    
    Args:
        resume_skills: Set of canonical skill names from resume
        jd_skills: Set of canonical skill names from JD
        use_enriched: If True, also consider broader skill categories
    
    Returns:
        Tuple of (match_score, matched_skills, missing_skills)
    """
    if not jd_skills:
        return 1.0, set(), set()

    # Enrich both skill sets
    if use_enriched:
        resume_enriched = get_enriched_skills(resume_skills)
        jd_enriched = get_enriched_skills(jd_skills)
    else:
        resume_enriched = resume_skills
        jd_enriched = jd_skills

    # Compute intersection using enriched sets
    matched = resume_enriched & jd_enriched
    # Missing = JD skills not found in resume (use original JD skills for clarity)
    missing = jd_skills - resume_enriched

    # Score based on what fraction of JD skills are matched
    score = len(matched) / len(jd_enriched) if jd_enriched else 1.0

    # Boost slightly if candidate has extra relevant skills
    extra_relevant = resume_enriched - jd_enriched
    if extra_relevant and len(jd_enriched) > 0:
        bonus = min(0.1, len(extra_relevant) * 0.01)
        score = min(1.0, score + bonus)

    return score, matched, missing


# ---------------------------------------------------------------------------
# Sub-score: Experience Relevance
# ---------------------------------------------------------------------------
def compute_experience_relevance(
    candidate_years: float,
    required_years: float,
) -> float:
    """
    Score experience relevance using a sigmoid-based function.
    - Exact match or slight over-qualification → high score
    - Under-qualified → lower score (steep drop)
    - Heavily over-qualified → slight diminishing returns
    
    Returns float in [0, 1]
    """
    if required_years <= 0:
        # No requirement specified — give a moderate boost for experience
        if candidate_years <= 0:
            return 0.5
        return min(1.0, 0.5 + candidate_years * 0.05)

    ratio = candidate_years / required_years if required_years > 0 else 0

    if ratio >= 0.8 and ratio <= 1.5:
        # Sweet spot: 80%-150% of required experience
        score = 0.8 + 0.2 * (1 - abs(ratio - 1.0) / 0.5)
    elif ratio < 0.8:
        # Under-qualified: sigmoid drop
        score = 1.0 / (1.0 + math.exp(-5 * (ratio - 0.4)))
    else:
        # Over-qualified: gentle decay
        score = max(0.6, 1.0 - 0.05 * (ratio - 1.5))

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Sub-score: Recency
# ---------------------------------------------------------------------------
def compute_recency(year_mentions: list[int]) -> float:
    """
    Compute recency score based on year mentions in resume.
    More recent years → higher score.
    Uses exponential decay: score = exp(-lambda * years_ago)
    
    Returns float in [0, 1]
    """
    if not year_mentions:
        return 0.3  # Unknown recency — neutral score

    current_year = datetime.now().year
    # Filter to reasonable years
    valid_years = [y for y in year_mentions if 1990 <= y <= current_year + 1]

    if not valid_years:
        return 0.3

    # Weight recent years more heavily
    scores = []
    for year in valid_years:
        years_ago = max(0, current_year - year)
        score = math.exp(-RECENCY_LAMBDA * years_ago)
        scores.append(score)

    # Use the average of top 3 most recent
    scores.sort(reverse=True)
    top_scores = scores[:3]

    return sum(top_scores) / len(top_scores)


# ---------------------------------------------------------------------------
# Sub-score: Keyword Precision
# ---------------------------------------------------------------------------
def _extract_keywords(text: str) -> Counter:
    """Extract meaningful keywords from text, filtering stop words."""
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return Counter(w for w in words if w not in _STOP_WORDS)


def compute_keyword_precision(
    resume_text: str,
    jd_text: str,
) -> float:
    """
    Compute keyword precision: what fraction of meaningful JD keywords
    appear in the resume.
    
    Returns float in [0, 1]
    """
    if not jd_text or not resume_text:
        return 0.0

    jd_keywords = _extract_keywords(jd_text)
    resume_keywords = _extract_keywords(resume_text)

    if not jd_keywords:
        return 0.5

    # Get top 30 JD keywords by frequency
    top_jd = set(k for k, _ in jd_keywords.most_common(30))

    if not top_jd:
        return 0.5

    # How many of the top JD keywords appear in the resume?
    matched = top_jd & set(resume_keywords.keys())

    return len(matched) / len(top_jd)


# ---------------------------------------------------------------------------
# Hard Filters
# ---------------------------------------------------------------------------
def apply_hard_filters(
    resume_skills: set[str],
    must_have_skills: set[str],
) -> tuple[bool, set[str]]:
    """
    Check if candidate meets must-have requirements.
    
    Args:
        resume_skills: Candidate's enriched skill set
        must_have_skills: Skills that are non-negotiable
    
    Returns:
        Tuple of (passes_filter, missing_must_haves)
    """
    if not must_have_skills:
        return True, set()

    # Enrich resume skills for more lenient matching
    enriched = get_enriched_skills(resume_skills)
    missing = must_have_skills - enriched

    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------
def score_candidate(
    resume_data: dict[str, Any],
    jd_data: dict[str, Any],
    semantic_score: float,
    must_have_skills: set[str] | None = None,
    feedback_engine=None,
) -> dict[str, Any]:
    """
    Compute the full composite score for a single candidate.
    
    Args:
        resume_data: Parsed resume dict
        jd_data: Parsed job description dict
        semantic_score: Cosine similarity from FAISS (0-1)
        must_have_skills: Set of must-have skills for hard filtering
    
    Returns:
        Dict with all scores, missing skills, explanation, and filtered status.
    """
    resume_skills = set(resume_data.get("enriched_skills", []))
    jd_skills = set(jd_data.get("enriched_skills", []))
    resume_text = resume_data.get("raw_text", "")
    jd_text = jd_data.get("raw_text", "")
    exp_years = resume_data.get("experience_years", 0.0)
    required_exp = jd_data.get("required_experience", 0.0)
    year_mentions = resume_data.get("year_mentions", [])
    filename = resume_data.get("filename", "unknown")

    # --- Sub-scores ---
    # 1. Semantic similarity (already computed by FAISS)
    sem_score = max(0.0, min(1.0, semantic_score))

    # 2. Skill match
    skill_score, matched_skills, missing_skills = compute_skill_match(
        resume_skills, jd_skills
    )

    # 3. Experience relevance
    exp_score = compute_experience_relevance(exp_years, required_exp)

    # 4. Recency
    recency_score = compute_recency(year_mentions)

    # 5. Keyword precision
    kw_score = compute_keyword_precision(resume_text, jd_text)

    # --- Composite score ---
    final_score = (
        WEIGHTS["semantic_similarity"] * sem_score
        + WEIGHTS["skill_match"] * skill_score
        + WEIGHTS["experience_relevance"] * exp_score
        + WEIGHTS["recency"] * recency_score
        + WEIGHTS["keyword_precision"] * kw_score
    )

    # --- Feedback adjustment (learned from recruiter decisions) ---
    feedback_adj = 0.0
    if feedback_engine is not None:
        scores_for_feedback = {
            "semantic_score": sem_score,
            "skill_score": skill_score,
            "experience_score": exp_score,
            "recency_score": recency_score,
            "keyword_score": kw_score,
            "skill_match_pct": 0.0,  # computed below
        }
        feedback_adj = feedback_engine.predict_adjustment(scores_for_feedback)
        final_score = max(0.0, min(1.0, final_score + feedback_adj))

    # --- Hard filters ---
    filtered = False
    missing_must_haves = set()
    if must_have_skills:
        passes, missing_must_haves = apply_hard_filters(
            resume_skills, must_have_skills
        )
        if not passes:
            filtered = True
            final_score = 0.0

    # --- Explanation ---
    explanation = _generate_explanation(
        skill_score, exp_years, required_exp, matched_skills,
        missing_skills, missing_must_haves, sem_score, filtered
    )

    # Skill match percentage (using original JD skills for user clarity)
    original_jd_skills = set(jd_data.get("skills", []))
    original_resume_skills = set(resume_data.get("skills", []))
    if original_jd_skills:
        skill_match_pct = (
            len(original_resume_skills & original_jd_skills)
            / len(original_jd_skills) * 100
        )
    else:
        skill_match_pct = 0.0

    return {
        "filename": filename,
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
        "explanation": explanation,
        "filtered": filtered,
        "feedback_adjustment": round(feedback_adj, 4),
    }


def _generate_explanation(
    skill_score: float,
    exp_years: float,
    required_exp: float,
    matched_skills: set[str],
    missing_skills: set[str],
    missing_must_haves: set[str],
    sem_score: float,
    filtered: bool,
) -> str:
    """Generate a human-readable explanation for the candidate's ranking."""
    parts = []

    if filtered:
        must_have_list = ", ".join(sorted(missing_must_haves)[:3])
        parts.append(f"Missing must-have: {must_have_list}")
        return " | ".join(parts)

    # Skill assessment
    if skill_score >= 0.8:
        parts.append(f"Strong skill match ({skill_score:.0%})")
    elif skill_score >= 0.5:
        parts.append(f"Moderate skill match ({skill_score:.0%})")
    else:
        parts.append(f"Low skill match ({skill_score:.0%})")

    # Experience assessment
    if exp_years > 0:
        parts.append(f"{exp_years:.0f}yr exp")
        if required_exp > 0:
            if exp_years >= required_exp * 0.8:
                parts.append("meets exp req")
            else:
                parts.append(f"needs {required_exp:.0f}yr")

    # Semantic relevance
    if sem_score >= 0.7:
        parts.append("highly relevant profile")
    elif sem_score >= 0.4:
        parts.append("relevant profile")

    # Missing skills (top 3)
    if missing_skills:
        top_missing = sorted(missing_skills)[:3]
        parts.append(f"missing: {', '.join(top_missing)}")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Reranking Layer
# ---------------------------------------------------------------------------
def rerank_candidates(
    faiss_results: list[dict[str, Any]],
    parsed_resumes: dict[str, dict],
    jd_data: dict[str, Any],
    must_have_skills: set[str] | None = None,
    top_k: int = 100,
    feedback_engine=None,
) -> list[dict[str, Any]]:
    """
    Rerank FAISS results using the full composite scoring system.
    
    Args:
        faiss_results: List of dicts from EmbeddingEngine.search()
                       with 'filename' and 'score' (semantic similarity)
        parsed_resumes: Dict mapping filename → parsed resume data
        jd_data: Parsed job description
        must_have_skills: Optional set of must-have skills
        top_k: Number of results to return after reranking
    
    Returns:
        List of scored candidate dicts, sorted by final_score descending.
        Filtered candidates are placed at the end.
    """
    scored_candidates = []

    for result in faiss_results:
        filename = result["filename"]
        semantic_score = result["score"]

        resume_data = parsed_resumes.get(filename)
        if not resume_data:
            logger.warning(f"No parsed data for {filename}, skipping")
            continue

        candidate_score = score_candidate(
            resume_data=resume_data,
            jd_data=jd_data,
            semantic_score=semantic_score,
            must_have_skills=must_have_skills,
            feedback_engine=feedback_engine,
        )
        scored_candidates.append(candidate_score)

    # Sort: non-filtered first by score, then filtered
    active = [c for c in scored_candidates if not c["filtered"]]
    filtered = [c for c in scored_candidates if c["filtered"]]

    active.sort(key=lambda x: x["final_score"], reverse=True)
    filtered.sort(key=lambda x: x["filename"])

    # Assign ranks
    ranked = active + filtered
    for i, candidate in enumerate(ranked, 1):
        candidate["rank"] = i

    return ranked[:top_k]
