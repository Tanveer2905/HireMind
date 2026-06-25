"""
llm_ranker.py — LLM-Based Reranking and Reasoning Layer
Takes top candidates from the FAISS + composite pipeline and enriches them
with LLM-powered evaluation: structured reasoning, strengths/weaknesses,
risk flags, and targeted interview questions.

When the LLM is unavailable (OOM, hardware mismatch), falls back to an
algorithmic evaluator that produces candidate-specific, evidence-based
analysis from the structured parsed data.
"""

import json
import logging
import re
from typing import Any

from llm_client import LlamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert technical recruiter and hiring analyst.

Your job is to evaluate a candidate AGAINST a job description with STRICT, evidence-based reasoning.

CRITICAL RULES (MUST FOLLOW):
* DO NOT give generic statements
* DO NOT summarize the resume
* EVERY claim must be backed by specific evidence from the resume
* If evidence is missing, explicitly say "No evidence found"
* Be critical, not polite
* Prefer rejection over weak acceptance
* Output MUST be valid JSON only"""

CANDIDATE_EVAL_PROMPT = """You are an expert technical recruiter and hiring analyst.
Your job is to evaluate a candidate AGAINST a job description with STRICT, evidence-based reasoning.

## Job Description
{jd_text}

## Candidate Resume Summary
- **Filename**: {filename}
- **Skills Found**: {skills}
- **Experience**: {experience_years} years
- **Education**: {education}
- **Job Titles**: {titles}
- **Organizations**: {organizations}

## Precomputed Scores (0-1 scale)
- Semantic Similarity: {semantic_score}
- Skill Match: {skill_score}
- Experience Score: {experience_score}
- Recency Score: {recency_score}
- Keyword Score: {keyword_score}
- Composite Score: {final_score}

## Resume Text (first 2000 chars)
{resume_text_preview}

---

## TASK

Evaluate the candidate deeply across:

1. Skill Match Quality
   * Compare REQUIRED vs FOUND skills
   * Mention EXACT missing critical skills
   * Mention partial/adjacent skills

2. Experience Relevance
   * Extract actual years and roles
   * Check if experience matches JD requirements
   * Penalize mismatch strongly

3. Project & Impact Quality
   * Are there real projects or just claims?
   * Look for measurable impact (numbers, outcomes)
   * Flag weak or vague experience

4. Red Flags
   * Skill inflation
   * Lack of depth
   * Irrelevant experience
   * Resume fluff

## SCORING LOGIC

You MUST:
* Adjust the final score based on REAL evidence
* Penalize: missing core skills, vague experience, no project proof
* Reward: exact skill match, strong specific projects, measurable impact

---

Output ONLY a JSON object with this exact structure:
{{
  "final_score": <float 0-1>,
  "decision": "<Strong Yes|Yes|Maybe|No>",
  "reasoning": ["Short, sharp, evidence-backed insights only"],
  "strengths": ["Evidence-backed strength"],
  "weaknesses": ["Evidence-backed weakness"],
  "risk_flags": ["Specific risk"],
  "interview_questions": ["Targeted question"]
}}

FINAL INSTRUCTION: Be strict, analytical, and evidence-driven.
If unsure → lower the score. If missing core requirements → reject.
Do NOT be generic."""

INTERVIEW_QUESTIONS_PROMPT = """Generate {count} targeted interview questions for this candidate based on the job description.

## Job Description
{jd_text}

## Candidate
- **Skills**: {skills}
- **Experience**: {experience_years} years
- **Titles**: {titles}

## Areas to Probe
{areas_to_probe}

Output ONLY a JSON object:
{{
  "questions": [
    {{
      "question": "<the interview question>",
      "purpose": "<what this question assesses>",
      "difficulty": "<Easy|Medium|Hard>"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Algorithmic Evidence-Based Evaluator (fallback when LLM unavailable)
# ---------------------------------------------------------------------------
def _generate_algorithmic_evaluation(
    candidate: dict[str, Any],
    resume_data: dict[str, Any],
    jd_text: str,
    jd_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate candidate-specific, evidence-based analysis from structured data.
    This is NOT generic — it uses actual skill diffs, experience gaps, and
    resume content to produce differentiated evaluations per candidate.
    """
    resume_skills = set(resume_data.get("skills", []))
    jd_skills = set(jd_data.get("skills", []))
    enriched_resume = set(resume_data.get("enriched_skills", []))
    enriched_jd = set(jd_data.get("enriched_skills", []))
    exp_years = resume_data.get("experience_years", 0.0)
    required_exp = jd_data.get("required_experience", 0.0)
    titles = resume_data.get("titles", [])
    education = resume_data.get("education", [])
    orgs = resume_data.get("organizations", [])
    raw_text = resume_data.get("raw_text", "")
    filename = candidate.get("filename", "unknown")

    # ---- 1. Skill Analysis (evidence-based) ----
    direct_matches = resume_skills & jd_skills
    enriched_matches = enriched_resume & enriched_jd
    adjacent_only = enriched_matches - direct_matches  # matched via category, not direct
    missing_from_jd = jd_skills - enriched_resume  # truly missing

    matched_evidence = []
    for skill in sorted(direct_matches):
        # Find evidence of skill usage in resume text
        evidence = _find_skill_evidence(skill, raw_text)
        if evidence:
            matched_evidence.append(f"{skill} → {evidence}")
        else:
            matched_evidence.append(f"{skill} → Listed but no project evidence found")

    for skill in sorted(adjacent_only):
        matched_evidence.append(f"{skill} → Partial match via related skills only")

    missing_evidence = []
    for skill in sorted(missing_from_jd):
        missing_evidence.append(f"{skill} → Not found in resume")

    # ---- 2. Experience Evaluation ----
    exp_eval_parts = []
    if exp_years > 0:
        exp_eval_parts.append(f"Candidate has {exp_years:.0f} years of experience")
    else:
        exp_eval_parts.append("No explicit experience years found in resume")

    if titles:
        exp_eval_parts.append(f"Roles held: {', '.join(titles[:3])}")
    else:
        exp_eval_parts.append("No recognizable job titles extracted")

    if required_exp > 0:
        if exp_years >= required_exp:
            exp_eval_parts.append(f"Meets required {required_exp:.0f}yr requirement")
        elif exp_years >= required_exp * 0.7:
            exp_eval_parts.append(f"Slightly under required {required_exp:.0f}yr ({exp_years:.0f}yr found)")
        else:
            exp_eval_parts.append(f"UNDER-QUALIFIED: {exp_years:.0f}yr vs required {required_exp:.0f}yr")

    if orgs:
        exp_eval_parts.append(f"Organizations: {', '.join(orgs[:3])}")

    experience_evaluation = ". ".join(exp_eval_parts)

    # ---- 3. Project Quality ----
    project_quality = _assess_project_quality(raw_text, resume_skills)

    # ---- 4. Red Flags ----
    red_flags = []
    if len(resume_skills) > 25 and exp_years < 3:
        red_flags.append(f"Skill inflation: {len(resume_skills)} skills listed with only {exp_years:.0f}yr experience")
    if not titles:
        red_flags.append("No clear job titles found — may indicate vague or non-standard resume")
    if exp_years == 0 and len(resume_skills) > 5:
        red_flags.append("Claims multiple skills but no verifiable experience duration")
    if len(missing_from_jd) > len(direct_matches) and len(jd_skills) > 3:
        red_flags.append(f"More skills missing ({len(missing_from_jd)}) than matched ({len(direct_matches)}) — poor fit")
    if not education:
        red_flags.append("No education credentials detected")
    if raw_text and len(raw_text) < 300:
        red_flags.append("Very short resume — may lack substance")

    # Check for vague buzzwords without evidence
    buzzwords = ["results-driven", "team player", "self-motivated", "passionate",
                 "hard-working", "detail-oriented", "go-getter", "synergy"]
    found_buzzwords = [bw for bw in buzzwords if bw in raw_text.lower()]
    if len(found_buzzwords) >= 2:
        red_flags.append(f"Resume contains fluff buzzwords: {', '.join(found_buzzwords)}")

    # ---- 5. Compute Score ----
    # Start with composite score and adjust based on evidence
    base_score = candidate.get("final_score", 0.5)

    # Penalize missing core skills
    if jd_skills:
        skill_coverage = len(direct_matches) / len(jd_skills)
    else:
        skill_coverage = 0.5

    # Penalize experience gap
    exp_penalty = 0.0
    if required_exp > 0 and exp_years < required_exp * 0.7:
        exp_penalty = 0.15

    # Penalize red flags
    flag_penalty = min(0.15, len(red_flags) * 0.03)

    # Reward strong skill match
    skill_bonus = 0.0
    if skill_coverage >= 0.8:
        skill_bonus = 0.05

    adjusted_score = base_score - exp_penalty - flag_penalty + skill_bonus
    adjusted_score = max(0.05, min(0.98, adjusted_score))

    # ---- 6. Decision ----
    if adjusted_score >= 0.75 and len(missing_from_jd) <= 2:
        decision = "Strong Yes"
    elif adjusted_score >= 0.60:
        decision = "Yes"
    elif adjusted_score >= 0.40:
        decision = "Maybe"
    else:
        decision = "No"

    # Override to "No" if too many critical skills missing
    if jd_skills and skill_coverage < 0.3:
        decision = "No"
        adjusted_score = min(adjusted_score, 0.35)

    # ---- 7. Reasoning ----
    reasoning = []
    if direct_matches:
        reasoning.append(f"Direct skill match: {len(direct_matches)}/{len(jd_skills)} JD skills found ({', '.join(sorted(direct_matches)[:5])})")
    else:
        reasoning.append("No direct skill matches with JD requirements")

    if missing_from_jd:
        top_missing = sorted(missing_from_jd)[:4]
        reasoning.append(f"Critical gaps: {', '.join(top_missing)} not found in resume")

    if exp_years > 0 and required_exp > 0:
        reasoning.append(f"Experience: {exp_years:.0f}yr actual vs {required_exp:.0f}yr required")
    elif exp_years > 0:
        reasoning.append(f"Has {exp_years:.0f} years experience (no specific requirement in JD)")

    if red_flags:
        reasoning.append(f"{len(red_flags)} red flag(s) identified")

    # ---- 8. Strengths & Weaknesses ----
    strengths = []
    if direct_matches:
        top_skills = sorted(direct_matches)[:3]
        strengths.append(f"Strong match in: {', '.join(top_skills)}")
    if exp_years >= (required_exp if required_exp > 0 else 3):
        strengths.append(f"{exp_years:.0f}yr experience meets/exceeds requirements")
    if education:
        strengths.append(f"Education: {', '.join(education[:2])}")
    if orgs:
        strengths.append(f"Has worked at: {', '.join(orgs[:2])}")
    if not strengths:
        strengths.append("No standout strengths identified from available data")

    weaknesses = []
    if missing_from_jd:
        weaknesses.append(f"Missing {len(missing_from_jd)} required skills: {', '.join(sorted(missing_from_jd)[:3])}")
    if exp_years < required_exp and required_exp > 0:
        weaknesses.append(f"Under-experienced: {exp_years:.0f}yr vs {required_exp:.0f}yr needed")
    if project_quality.startswith("Weak") or project_quality.startswith("No"):
        weaknesses.append("Lacks concrete project evidence or measurable impact")
    if not weaknesses:
        weaknesses.append("No major weaknesses identified")

    # ---- 9. Interview Questions (targeted) ----
    questions = _generate_targeted_questions(
        missing_from_jd, direct_matches, exp_years, required_exp, titles, red_flags
    )

    return {
        "final_score": round(adjusted_score, 4),
        "decision": decision,
        "evidence_based_analysis": {
            "matched_skills": matched_evidence,
            "missing_skills": missing_evidence,
            "experience_evaluation": experience_evaluation,
            "project_quality": project_quality,
            "red_flags": red_flags,
        },
        "reasoning": reasoning,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "risk_flags": red_flags,
        "interview_questions": questions,
        "confidence": round(min(0.95, 0.5 + skill_coverage * 0.3 + (0.15 if exp_years > 0 else 0)), 2),
    }


def _find_skill_evidence(skill: str, text: str, max_chars: int = 120) -> str:
    """Find contextual evidence of a skill being used in the resume text."""
    if not text:
        return ""

    text_lower = text.lower()
    skill_lower = skill.lower()

    # Find the skill mention and extract surrounding context
    idx = text_lower.find(skill_lower)
    if idx == -1:
        return ""

    # Get surrounding context
    start = max(0, idx - 40)
    end = min(len(text), idx + len(skill) + 80)
    context = text[start:end].strip()

    # Clean up: replace newlines, compress whitespace
    context = re.sub(r'\s+', ' ', context).strip()

    # Trim to sentence-ish boundaries
    if start > 0 and not context[0].isupper():
        context = "..." + context
    if end < len(text):
        context = context + "..."

    return context[:max_chars]


def _assess_project_quality(text: str, skills: set[str]) -> str:
    """Assess whether the resume contains concrete project evidence."""
    if not text:
        return "No resume text available for project assessment"

    text_lower = text.lower()

    # Look for measurable impact indicators
    impact_patterns = [
        r'\d+%',           # percentages
        r'\$\d+',          # dollar amounts
        r'\d+\s*users?',   # user counts
        r'\d+x\s',         # multipliers
        r'reduced\s+\w+\s+by', r'increased\s+\w+\s+by',
        r'improved\s+\w+\s+by', r'achieved\s',
        r'delivered\s', r'built\s+\w+\s+(?:for|that|which)',
        r'deployed\s', r'launched\s', r'implemented\s',
    ]

    impact_count = 0
    for pat in impact_patterns:
        if re.search(pat, text_lower):
            impact_count += 1

    # Look for project-related keywords
    project_keywords = ["project", "developed", "built", "created", "designed",
                        "implemented", "deployed", "launched", "architected",
                        "automated", "optimized", "migrated"]
    project_mentions = sum(1 for kw in project_keywords if kw in text_lower)

    if impact_count >= 4 and project_mentions >= 3:
        return f"Strong: {impact_count} measurable impact indicators found with {project_mentions} project-related verbs. Resume shows concrete deliverables."
    elif impact_count >= 2 or project_mentions >= 3:
        return f"Moderate: {impact_count} impact indicators and {project_mentions} project mentions. Some evidence of real work but could be stronger."
    elif project_mentions >= 1:
        return f"Weak: Only {project_mentions} project mention(s) and {impact_count} measurable outcomes. Mostly claims without quantifiable evidence."
    else:
        return "No concrete project evidence found. Resume appears to be skill-listing without demonstrable work."


def _generate_targeted_questions(
    missing_skills: set[str],
    matched_skills: set[str],
    exp_years: float,
    required_exp: float,
    titles: list[str],
    red_flags: list[str],
) -> list[str]:
    """Generate interview questions targeted at this specific candidate's gaps."""
    questions = []

    # Probe missing skills
    missing_list = sorted(missing_skills)
    if missing_list:
        questions.append(
            f"This role requires {missing_list[0]}. Describe any exposure or transferable experience you have in this area."
        )
    if len(missing_list) > 1:
        questions.append(
            f"How would you approach learning {missing_list[1]} if brought on board?"
        )

    # Probe depth of matched skills
    matched_list = sorted(matched_skills)
    if matched_list:
        questions.append(
            f"Walk me through a complex problem you solved using {matched_list[0]}. What were the trade-offs?"
        )

    # Probe experience gap
    if required_exp > 0 and exp_years < required_exp:
        questions.append(
            f"You have {exp_years:.0f} years of experience vs our {required_exp:.0f}-year requirement. "
            "What have you accomplished that demonstrates readiness beyond your years?"
        )

    # Probe red flags
    if any("inflation" in rf.lower() for rf in red_flags):
        questions.append(
            "You list many technologies on your resume. Pick the one you're deepest in and explain a non-trivial challenge you faced with it."
        )

    # General depth question
    if len(questions) < 3:
        questions.append(
            "Describe a production incident or critical bug you debugged. What was your systematic approach?"
        )

    if len(questions) < 3:
        questions.append(
            "What's a technical decision you made that you later realized was wrong? What did you learn?"
        )

    return questions[:5]


# ---------------------------------------------------------------------------
# Core reranking function
# ---------------------------------------------------------------------------
def llm_rerank_candidates(
    candidates: list[dict[str, Any]],
    jd_text: str,
    jd_data: dict[str, Any],
    parsed_resumes: dict[str, dict],
    top_n: int = 20,
    client: LlamaClient | None = None,
    progress_callback=None,
) -> list[dict[str, Any]]:
    """
    Rerank the top-N candidates using LLM reasoning.

    Takes the output from scorer.rerank_candidates() and enriches
    each candidate with LLM-generated assessment data.

    Falls back to algorithmic evidence-based evaluation if LLM is unavailable.

    Args:
        candidates: Ranked candidate list from the composite scorer
        jd_text: Raw job description text
        jd_data: Parsed JD data
        parsed_resumes: Dict of filename → parsed resume data
        top_n: How many top candidates to send to the LLM
        client: LlamaClient instance (created if None)
        progress_callback: Optional callable(current, total, filename) for progress

    Returns:
        The same candidate list with added LLM fields, re-sorted
        by LLM score for the top-N.
    """
    if client is None:
        client = LlamaClient()

    if not client.is_available():
        logger.warning("Local LLM not available — skipping LLM reranking")
        return candidates

    # Only process non-filtered, top-N candidates
    active = [c for c in candidates if not c.get("filtered")]
    filtered = [c for c in candidates if c.get("filtered")]

    to_evaluate = active[:top_n]
    remaining = active[top_n:]

    logger.info(f"LLM reranking {len(to_evaluate)} candidates...")

    evaluated = []
    for i, candidate in enumerate(to_evaluate):
        filename = candidate["filename"]
        resume_data = parsed_resumes.get(filename, {})

        if progress_callback:
            progress_callback(i + 1, len(to_evaluate), filename)

        try:
            llm_result = _evaluate_single_candidate(
                candidate=candidate,
                resume_data=resume_data,
                jd_text=jd_text,
                client=client,
            )
            # Merge LLM results into candidate dict
            candidate["llm_score"] = llm_result.get("final_score", candidate["final_score"])
            candidate["llm_decision"] = llm_result.get("decision", "")
            candidate["llm_reasoning"] = llm_result.get("reasoning", [])
            candidate["llm_strengths"] = llm_result.get("strengths", [])
            candidate["llm_weaknesses"] = llm_result.get("weaknesses", [])
            candidate["llm_risk_flags"] = llm_result.get("risk_flags", [])
            candidate["llm_interview_questions"] = llm_result.get("interview_questions", [])
            candidate["llm_evaluated"] = True
            logger.info(f"LLM evaluated {filename}: score={candidate['llm_score']}, decision={candidate['llm_decision']}")

        except Exception as e:
            logger.warning(f"LLM failed for {filename}: {e} — using algorithmic evaluation")
            # FALLBACK: Use algorithmic evidence-based evaluation
            algo_result = _generate_algorithmic_evaluation(
                candidate=candidate,
                resume_data=resume_data,
                jd_text=jd_text,
                jd_data=jd_data,
            )
            candidate["llm_score"] = algo_result.get("final_score", candidate["final_score"])
            candidate["llm_decision"] = algo_result.get("decision", "")
            candidate["llm_reasoning"] = algo_result.get("reasoning", [])
            candidate["llm_strengths"] = algo_result.get("strengths", [])
            candidate["llm_weaknesses"] = algo_result.get("weaknesses", [])
            candidate["llm_risk_flags"] = algo_result.get("risk_flags", [])
            candidate["llm_interview_questions"] = algo_result.get("interview_questions", [])
            candidate["llm_evaluated"] = True  # Mark as evaluated (algorithmic)
            logger.info(f"Algorithmic eval for {filename}: score={candidate['llm_score']}, decision={candidate['llm_decision']}")

        evaluated.append(candidate)

    # Re-sort evaluated candidates by LLM score
    evaluated.sort(key=lambda x: x.get("llm_score", 0), reverse=True)

    # Use algorithmic evaluation for remaining candidates so they all get assessments
    for c in remaining:
        filename = c["filename"]
        resume_data = parsed_resumes.get(filename, {})
        algo_result = _generate_algorithmic_evaluation(
            candidate=c,
            resume_data=resume_data,
            jd_text=jd_text,
            jd_data=jd_data,
        )
        c["llm_score"] = algo_result.get("final_score", c["final_score"])
        c["llm_decision"] = algo_result.get("decision", "")
        c["llm_reasoning"] = algo_result.get("reasoning", [])
        c["llm_strengths"] = algo_result.get("strengths", [])
        c["llm_weaknesses"] = algo_result.get("weaknesses", [])
        c["llm_risk_flags"] = algo_result.get("risk_flags", [])
        c["llm_interview_questions"] = algo_result.get("interview_questions", [])
        c["llm_evaluated"] = True

    # Reassemble: evaluated + remaining (non-LLM) + filtered
    all_candidates = evaluated + remaining + filtered

    # Re-assign ranks
    for i, c in enumerate(all_candidates, 1):
        c["rank"] = i

    logger.info(f"LLM reranking complete. Top candidate: {all_candidates[0]['filename'] if all_candidates else 'none'}")
    return all_candidates


def _evaluate_single_candidate(
    candidate: dict[str, Any],
    resume_data: dict[str, Any],
    jd_text: str,
    client: LlamaClient,
) -> dict[str, Any]:
    """Evaluate a single candidate using the LLM."""

    # Build the prompt with candidate data
    prompt = CANDIDATE_EVAL_PROMPT.format(
        jd_text=_truncate(jd_text, 2000),
        filename=candidate.get("filename", "unknown"),
        skills=", ".join(resume_data.get("skills", [])[:30]),
        experience_years=resume_data.get("experience_years", 0),
        education=", ".join(resume_data.get("education", [])),
        titles=", ".join(resume_data.get("titles", [])),
        organizations=", ".join(resume_data.get("organizations", [])[:5]),
        semantic_score=f"{candidate.get('semantic_score', 0):.3f}",
        skill_score=f"{candidate.get('skill_score', 0):.3f}",
        experience_score=f"{candidate.get('experience_score', 0):.3f}",
        recency_score=f"{candidate.get('recency_score', 0):.3f}",
        keyword_score=f"{candidate.get('keyword_score', 0):.3f}",
        final_score=f"{candidate.get('final_score', 0):.3f}",
        resume_text_preview=_truncate(
            resume_data.get("raw_text", ""), 2000
        ),
    )

    result = client.generate_json(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.1,
    )

    # Validate and clamp the score
    score = result.get("final_score", candidate.get("final_score", 0))
    try:
        score = float(score)
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        score = candidate.get("final_score", 0)

    result["final_score"] = round(score, 4)

    # Validate decision
    valid_decisions = {"Strong Yes", "Yes", "Maybe", "No"}
    if result.get("decision") not in valid_decisions:
        # Map score to decision
        if score >= 0.8:
            result["decision"] = "Strong Yes"
        elif score >= 0.6:
            result["decision"] = "Yes"
        elif score >= 0.4:
            result["decision"] = "Maybe"
        else:
            result["decision"] = "No"

    # Ensure lists
    for field in ["reasoning", "strengths", "weaknesses", "risk_flags", "interview_questions"]:
        if not isinstance(result.get(field), list):
            result[field] = []

    return result


# ---------------------------------------------------------------------------
# Interview question generation (standalone)
# ---------------------------------------------------------------------------
def generate_interview_questions(
    candidate: dict[str, Any],
    resume_data: dict[str, Any],
    jd_text: str,
    count: int = 5,
    client: LlamaClient | None = None,
) -> list[dict[str, str]]:
    """
    Generate targeted interview questions for a specific candidate.

    Args:
        candidate: Scored candidate dict
        resume_data: Parsed resume data
        jd_text: Raw job description text
        count: Number of questions to generate
        client: LlamaClient instance

    Returns:
        List of {question, purpose, difficulty} dicts
    """
    if client is None:
        client = LlamaClient()

    if not client.is_available():
        return _fallback_interview_questions(candidate, resume_data)

    # Determine areas to probe based on weaknesses/gaps
    areas = []
    missing = candidate.get("missing_skills", [])
    if missing:
        areas.append(f"Missing skills: {', '.join(missing[:5])}")

    weaknesses = candidate.get("llm_weaknesses", [])
    if weaknesses:
        areas.extend(weaknesses[:3])

    if not areas:
        areas.append("Assess depth of technical expertise")
        areas.append("Evaluate problem-solving approach")
        areas.append("Verify stated experience level")

    prompt = INTERVIEW_QUESTIONS_PROMPT.format(
        count=count,
        jd_text=_truncate(jd_text, 1500),
        skills=", ".join(resume_data.get("skills", [])[:20]),
        experience_years=resume_data.get("experience_years", 0),
        titles=", ".join(resume_data.get("titles", [])),
        areas_to_probe="\n".join(f"- {a}" for a in areas),
    )

    try:
        result = client.generate_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
        )
        questions = result.get("questions", [])
        # Validate structure
        validated = []
        for q in questions[:count]:
            if isinstance(q, dict) and "question" in q:
                validated.append({
                    "question": q["question"],
                    "purpose": q.get("purpose", ""),
                    "difficulty": q.get("difficulty", "Medium"),
                })
        return validated if validated else _fallback_interview_questions(candidate, resume_data)

    except Exception as e:
        logger.error(f"Interview question generation failed: {e}")
        return _fallback_interview_questions(candidate, resume_data)


def _fallback_interview_questions(
    candidate: dict[str, Any],
    resume_data: dict[str, Any],
) -> list[dict[str, str]]:
    """Generate basic interview questions without LLM."""
    questions = []
    skills = resume_data.get("skills", [])
    exp = resume_data.get("experience_years", 0)
    missing = candidate.get("missing_skills", [])

    if skills:
        questions.append({
            "question": f"Can you describe a challenging project where you used {skills[0]}?",
            "purpose": "Assess depth of expertise in key skill",
            "difficulty": "Medium",
        })

    if exp > 0:
        questions.append({
            "question": f"With {exp:.0f} years of experience, what has been your most impactful contribution?",
            "purpose": "Evaluate impact and leadership growth",
            "difficulty": "Medium",
        })

    if missing:
        questions.append({
            "question": f"This role requires {missing[0]}. What experience do you have in this area?",
            "purpose": "Probe gap in required skills",
            "difficulty": "Easy",
        })

    questions.append({
        "question": "Describe a situation where you had to debug a complex production issue. What was your approach?",
        "purpose": "Assess problem-solving and debugging skills",
        "difficulty": "Hard",
    })

    questions.append({
        "question": "How do you stay current with new technologies? Give a recent example.",
        "purpose": "Evaluate learning agility",
        "difficulty": "Easy",
    })

    return questions[:5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length with ellipsis."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
