import logging
from typing import Any
import json

from backend.llm_engine import LlamaEngine
from backend.memory_engine import get_preferences, get_feedback_summary
from llm_ranker import _generate_algorithmic_evaluation

logger = logging.getLogger(__name__)

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

## Recruiter Preferences
{preferences_json}

## Past Decisions Summary
{feedback_summary}

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
1. Skill Match Quality (Compare REQUIRED vs FOUND skills)
2. Experience Relevance (Extract actual years and roles)
3. Project & Impact Quality (Concrete deliverables)
4. Red Flags (Skill inflation, vague fluff)

## SCORING LOGIC
You MUST:
* Align evaluation with recruiter preferences while still being critical.
* Provide highly thorough, specific, and evidence-backed points.
* If the candidate is a complete mismatch or lacks critical skills, their decision MUST be "No".
* Boost candidates matching preferred_skills
* Penalize candidates matching rejected_patterns
* Adjust the final score based on REAL evidence

Output ONLY a JSON object with this exact structure:
{{
  "final_score": <float 0-1>,
  "decision": "<Strong Yes|Yes|Maybe|No>",
  "reasoning": ["Thorough, specific, evidence-backed insights detailing exactly why they fit or fail"],
  "strengths": ["Specific strength with resume evidence"],
  "weaknesses": ["Specific weakness or missing requirement"],
  "risk_flags": ["Specific risk or red flag"],
  "interview_questions": ["Targeted question based on their resume"]
}}
"""

def _truncate(text: str, max_len: int) -> str:
    if not text: return ""
    return text if len(text) <= max_len else text[:max_len] + "..."

def llm_rerank_candidates(user_id: str, candidates: list[dict[str, Any]], jd_text: str, jd_data: dict[str, Any], parsed_resumes: dict[str, dict], top_n: int = 20) -> list[dict[str, Any]]:
    client = LlamaEngine(user_id)
    if not client.is_available():
        logger.warning("Local LLM not available — skipping LLM reranking")
        return candidates

    to_evaluate = candidates[:top_n]
    remaining = candidates[top_n:]

    prefs = get_preferences(user_id)
    feedback_summary = get_feedback_summary(user_id)

    evaluated = []
    for candidate in to_evaluate:
        filename = candidate["filename"]
        resume_data = parsed_resumes.get(filename, {})
        try:
            prompt = CANDIDATE_EVAL_PROMPT.format(
                preferences_json=json.dumps(prefs, indent=2),
                feedback_summary=feedback_summary,
                jd_text=_truncate(jd_text, 2000),
                filename=filename,
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
                resume_text_preview=_truncate(resume_data.get("raw_text", ""), 2000),
            )
            result = client.generate_json(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1)
            score = float(result.get("final_score", candidate.get("final_score", 0)))
            score = max(0.0, min(1.0, score))
            
            candidate.update({
                "llm_score": score,
                "llm_decision": result.get("decision", ""),
                "llm_reasoning": result.get("reasoning", []),
                "llm_strengths": result.get("strengths", []),
                "llm_weaknesses": result.get("weaknesses", []),
                "llm_risk_flags": result.get("risk_flags", []),
                "llm_interview_questions": result.get("interview_questions", []),
                "llm_evaluated": True
            })
        except Exception as e:
            logger.warning(f"LLM failed for {filename}: {e}")
            algo_result = _generate_algorithmic_evaluation(
                candidate=candidate,
                resume_data=resume_data,
                jd_text=jd_text,
                jd_data=jd_data,
            )
            candidate.update({
                "llm_score": algo_result.get("final_score", candidate.get("final_score", 0)),
                "llm_decision": algo_result.get("decision", ""),
                "llm_reasoning": algo_result.get("reasoning", []),
                "llm_strengths": algo_result.get("strengths", []),
                "llm_weaknesses": algo_result.get("weaknesses", []),
                "llm_risk_flags": algo_result.get("risk_flags", []),
                "llm_interview_questions": algo_result.get("interview_questions", []),
                "llm_evaluated": True
            })
        evaluated.append(candidate)

    evaluated.sort(key=lambda x: x.get("llm_score", 0), reverse=True)
    for c in remaining:
        filename = c["filename"]
        resume_data = parsed_resumes.get(filename, {})
        algo_result = _generate_algorithmic_evaluation(
            candidate=c,
            resume_data=resume_data,
            jd_text=jd_text,
            jd_data=jd_data,
        )
        c.update({
            "llm_score": algo_result.get("final_score", c.get("final_score", 0)),
            "llm_decision": algo_result.get("decision", ""),
            "llm_reasoning": algo_result.get("reasoning", []),
            "llm_strengths": algo_result.get("strengths", []),
            "llm_weaknesses": algo_result.get("weaknesses", []),
            "llm_risk_flags": algo_result.get("risk_flags", []),
            "llm_interview_questions": algo_result.get("interview_questions", []),
            "llm_evaluated": True
        })

    all_candidates = evaluated + remaining
    for i, c in enumerate(all_candidates, 1):
        c["rank"] = i
    return all_candidates
