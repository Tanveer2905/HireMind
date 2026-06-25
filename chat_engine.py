"""
chat_engine.py — Conversational Query Interface
Translates natural language queries into structured filters and searches
against parsed resume data. Uses LLM for query understanding when available,
falls back to rule-based parsing.
"""

import re
import logging
from typing import Any

from llm_client import LlamaClient
from utils import normalize_skill

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
QUERY_SYSTEM_PROMPT = """You are a search query parser for a hiring system. Convert natural language queries into structured JSON filters.

Available filter fields:
- skills_required: list of skill names the candidate must have
- skills_excluded: list of skill names the candidate must NOT have
- experience_min: minimum years of experience (number or null)
- experience_max: maximum years of experience (number or null)
- titles: list of job title keywords to match
- sort_by: one of "score", "experience", "skill_match", "recency"
- limit: max number of results to return (default 10)

RULES:
1. Output ONLY valid JSON, no other text.
2. Use null for fields that are not mentioned.
3. Skill names should be common canonical forms (e.g., "Python", "React", "Docker").
4. Parse experience ranges like "<3 years" as experience_max=3, ">5 years" as experience_min=5."""

QUERY_PARSE_PROMPT = """Convert this search query into structured filters:

Query: "{query}"

Output ONLY a JSON object:
{{
  "skills_required": [],
  "skills_excluded": [],
  "experience_min": null,
  "experience_max": null,
  "titles": [],
  "sort_by": "score",
  "limit": 10
}}"""

CHAT_SYSTEM_PROMPT = """You are an AI hiring copilot assistant. Answer questions about candidates, hiring, and the recruitment process. Be concise and helpful. When referring to candidate data, be specific about scores and skills."""

CHAT_RESPONSE_PROMPT = """Based on the following search results, answer the user's question.

User question: "{query}"

Search results ({count} candidates found):
{results_summary}

Provide a concise, helpful answer. Reference specific candidates by name when relevant."""


class ChatEngine:
    """
    Conversational query interface for the AI Hiring Copilot.

    Supports:
    - Natural language → filter → search (e.g., "Find Python devs with 3+ years")
    - Follow-up questions about candidates
    - Candidate comparisons
    - Interview question requests
    """

    def __init__(
        self,
        parsed_resumes: list[dict] | None = None,
        scored_results: list[dict] | None = None,
        jd_text: str = "",
        llm_client: LlamaClient | None = None,
    ):
        self.parsed_resumes = parsed_resumes or []
        self.scored_results = scored_results or []
        self.jd_text = jd_text
        self.llm_client = llm_client
        self._resume_lookup = {r["filename"]: r for r in self.parsed_resumes}

    def update_context(
        self,
        parsed_resumes: list[dict] | None = None,
        scored_results: list[dict] | None = None,
        jd_text: str | None = None,
    ) -> None:
        """Update the engine's context with new data."""
        if parsed_resumes is not None:
            self.parsed_resumes = parsed_resumes
            self._resume_lookup = {r["filename"]: r for r in parsed_resumes}
        if scored_results is not None:
            self.scored_results = scored_results
        if jd_text is not None:
            self.jd_text = jd_text

    def query(self, user_message: str) -> dict[str, Any]:
        """
        Process a natural language query.

        Args:
            user_message: The user's natural language query

        Returns:
            {
                "type": "filter" | "chat" | "error",
                "response": "human-readable response text",
                "candidates": [...],
                "filters_applied": {...} | None,
            }
        """
        message = user_message.strip()
        if not message:
            return {
                "type": "error",
                "response": "Please enter a query.",
                "candidates": [],
                "filters_applied": None,
            }

        # Detect query intent
        intent = self._detect_intent(message)

        if intent == "filter":
            return self._handle_filter_query(message)
        elif intent == "chat":
            return self._handle_chat_query(message)
        else:
            return self._handle_filter_query(message)

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------
    def _detect_intent(self, message: str) -> str:
        """Detect whether the message is a search/filter or a chat question."""
        msg_lower = message.lower()

        # Filter-like patterns
        filter_patterns = [
            r'\bfind\b', r'\bsearch\b', r'\bshow\b', r'\blist\b',
            r'\bwho\b.*\bhave?\b', r'\bwho\b.*\bwith\b',
            r'\bcandidates?\b.*\bwith\b', r'\bdevelopers?\b',
            r'\bengineers?\b', r'\bfilter\b', r'\bsort\b',
            r'\btop\b.*\d+', r'\bbest\b',
            r'\bexperience\b.*\byears?\b', r'\byears?\b.*\bexperience\b',
        ]

        for pattern in filter_patterns:
            if re.search(pattern, msg_lower):
                return "filter"

        # If no analysis data exists, treat as a general chat
        if not self.scored_results:
            return "chat"

        return "filter"

    # ------------------------------------------------------------------
    # Filter query handling
    # ------------------------------------------------------------------
    def _handle_filter_query(self, message: str) -> dict[str, Any]:
        """Parse a filter query and apply it against resume data."""
        filters = self._parse_filters(message)
        candidates = self._apply_filters(filters)

        # Build response text
        if candidates:
            response_parts = [f"Found **{len(candidates)}** matching candidate(s):"]
            for i, c in enumerate(candidates[:10], 1):
                name = c["filename"].replace(".pdf", "").replace("_", " ")
                score = c.get("final_score", 0)
                skills_str = ", ".join(c.get("matched_skills", [])[:5])
                exp = c.get("experience_years", 0)
                response_parts.append(
                    f"{i}. **{name}** — Score: {score:.3f} | "
                    f"Exp: {exp:.0f}yr | Skills: {skills_str or 'N/A'}"
                )
        else:
            response_parts = ["No candidates match your filters. Try broadening your search."]

        return {
            "type": "filter",
            "response": "\n".join(response_parts),
            "candidates": candidates[:10],
            "filters_applied": filters,
        }

    def _parse_filters(self, message: str) -> dict[str, Any]:
        """
        Parse natural language into structured filters.
        Uses LLM if available, falls back to rule-based parsing.
        """
        # Try LLM-based parsing
        if self.llm_client and self.llm_client.is_available():
            try:
                prompt = QUERY_PARSE_PROMPT.format(query=message)
                filters = self.llm_client.generate_json(
                    prompt=prompt,
                    system_prompt=QUERY_SYSTEM_PROMPT,
                    temperature=0.1,
                )
                logger.info(f"LLM parsed filters: {filters}")
                return self._normalize_filters(filters)
            except Exception as e:
                logger.warning(f"LLM filter parsing failed, using rules: {e}")

        # Rule-based fallback
        return self._rule_based_parse(message)

    def _rule_based_parse(self, message: str) -> dict[str, Any]:
        """Rule-based query parsing as fallback when LLM is unavailable."""
        msg_lower = message.lower()
        filters: dict[str, Any] = {
            "skills_required": [],
            "skills_excluded": [],
            "experience_min": None,
            "experience_max": None,
            "titles": [],
            "sort_by": "score",
            "limit": 10,
        }

        # Extract experience constraints
        exp_patterns = [
            (r'(?:less\s+than|under|<)\s*(\d+)\s*(?:years?|yrs?)', "max"),
            (r'(?:more\s+than|over|above|>)\s*(\d+)\s*(?:years?|yrs?)', "min"),
            (r'(\d+)\+\s*(?:years?|yrs?)', "min"),
            (r'(\d+)\s*(?:-|to)\s*(\d+)\s*(?:years?|yrs?)', "range"),
            (r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?experience', "exact"),
        ]

        for pattern, kind in exp_patterns:
            match = re.search(pattern, msg_lower)
            if match:
                if kind == "max":
                    filters["experience_max"] = int(match.group(1))
                elif kind == "min":
                    filters["experience_min"] = int(match.group(1))
                elif kind == "range":
                    filters["experience_min"] = int(match.group(1))
                    filters["experience_max"] = int(match.group(2))
                elif kind == "exact":
                    val = int(match.group(1))
                    filters["experience_min"] = max(0, val - 1)
                    filters["experience_max"] = val + 1
                break

        # Extract skill mentions by matching against known taxonomy
        from utils import _SKILL_ALIASES
        words_in_msg = set(re.findall(r'[a-zA-Z0-9#+.]+', msg_lower))

        # Also try multi-word matches
        for alias, canonical in _SKILL_ALIASES.items():
            if alias in msg_lower:
                if canonical not in filters["skills_required"]:
                    filters["skills_required"].append(canonical)

        # Extract "top N"
        top_match = re.search(r'top\s+(\d+)', msg_lower)
        if top_match:
            filters["limit"] = int(top_match.group(1))

        # Sort hints
        if "experience" in msg_lower:
            filters["sort_by"] = "experience"
        elif "skill" in msg_lower:
            filters["sort_by"] = "skill_match"
        elif "recent" in msg_lower:
            filters["sort_by"] = "recency"

        return filters

    def _normalize_filters(self, filters: dict) -> dict[str, Any]:
        """Normalize and validate filter values from LLM output."""
        normalized = {
            "skills_required": [],
            "skills_excluded": [],
            "experience_min": None,
            "experience_max": None,
            "titles": [],
            "sort_by": "score",
            "limit": 10,
        }

        # Skills
        for skill in filters.get("skills_required") or []:
            if isinstance(skill, str) and skill.strip():
                canonical = normalize_skill(skill.strip())
                normalized["skills_required"].append(canonical or skill.strip())

        for skill in filters.get("skills_excluded") or []:
            if isinstance(skill, str) and skill.strip():
                canonical = normalize_skill(skill.strip())
                normalized["skills_excluded"].append(canonical or skill.strip())

        # Experience
        for field in ["experience_min", "experience_max"]:
            val = filters.get(field)
            if val is not None:
                try:
                    normalized[field] = int(val)
                except (TypeError, ValueError):
                    pass

        # Titles
        for title in filters.get("titles") or []:
            if isinstance(title, str) and title.strip():
                normalized["titles"].append(title.strip().lower())

        # Sort
        valid_sorts = {"score", "experience", "skill_match", "recency"}
        sort_val = filters.get("sort_by", "score")
        normalized["sort_by"] = sort_val if sort_val in valid_sorts else "score"

        # Limit
        try:
            normalized["limit"] = max(1, min(50, int(filters.get("limit", 10))))
        except (TypeError, ValueError):
            normalized["limit"] = 10

        return normalized

    # ------------------------------------------------------------------
    # Filter application
    # ------------------------------------------------------------------
    def _apply_filters(self, filters: dict[str, Any]) -> list[dict]:
        """Apply structured filters to the scored results."""
        # Start with scored results if available, otherwise use parsed resumes
        if self.scored_results:
            candidates = list(self.scored_results)
        else:
            candidates = []
            for r in self.parsed_resumes:
                candidates.append({
                    "filename": r["filename"],
                    "skills": r.get("skills", []),
                    "experience_years": r.get("experience_years", 0),
                    "titles": r.get("titles", []),
                    "final_score": 0.5,
                    "matched_skills": r.get("skills", []),
                    "missing_skills": [],
                    "filtered": False,
                })

        filtered = []
        for c in candidates:
            if c.get("filtered"):
                continue

            # Skill requirements
            if filters["skills_required"]:
                resume_data = self._resume_lookup.get(c["filename"], {})
                resume_skills_lower = {
                    s.lower() for s in resume_data.get("skills", [])
                }
                resume_enriched_lower = {
                    s.lower() for s in resume_data.get("enriched_skills", [])
                }
                all_skills = resume_skills_lower | resume_enriched_lower

                required_lower = {s.lower() for s in filters["skills_required"]}
                if not required_lower.issubset(all_skills):
                    continue

            # Skill exclusions
            if filters["skills_excluded"]:
                resume_data = self._resume_lookup.get(c["filename"], {})
                resume_skills_lower = {
                    s.lower() for s in resume_data.get("skills", [])
                }
                excluded_lower = {s.lower() for s in filters["skills_excluded"]}
                if excluded_lower & resume_skills_lower:
                    continue

            # Experience constraints
            exp = c.get("experience_years", 0) or 0
            if filters["experience_min"] is not None and exp < filters["experience_min"]:
                continue
            if filters["experience_max"] is not None and exp > filters["experience_max"]:
                continue

            # Title matching
            if filters["titles"]:
                resume_data = self._resume_lookup.get(c["filename"], {})
                candidate_titles = [
                    t.lower() for t in resume_data.get("titles", [])
                ]
                title_match = any(
                    ft in " ".join(candidate_titles)
                    for ft in filters["titles"]
                )
                if not title_match:
                    continue

            filtered.append(c)

        # Sort
        sort_key = filters.get("sort_by", "score")
        sort_map = {
            "score": lambda x: x.get("final_score", 0),
            "experience": lambda x: x.get("experience_years", 0),
            "skill_match": lambda x: x.get("skill_match_pct", 0),
            "recency": lambda x: x.get("recency_score", 0),
        }
        key_fn = sort_map.get(sort_key, sort_map["score"])
        filtered.sort(key=key_fn, reverse=True)

        # Limit
        limit = filters.get("limit", 10)
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Chat (general Q&A) handling
    # ------------------------------------------------------------------
    def _handle_chat_query(self, message: str) -> dict[str, Any]:
        """Handle a general chat/Q&A query using the LLM."""
        if not self.llm_client or not self.llm_client.is_available():
            return {
                "type": "chat",
                "response": (
                    "LLM is not available for chat queries. "
                    "Try a search query like 'Find Python developers with 3+ years experience'."
                ),
                "candidates": [],
                "filters_applied": None,
            }

        # Build context from scored results
        results_summary = ""
        if self.scored_results:
            lines = []
            for r in self.scored_results[:10]:
                name = r["filename"].replace(".pdf", "").replace("_", " ")
                lines.append(
                    f"- {name}: score={r.get('final_score', 0):.3f}, "
                    f"skills_match={r.get('skill_match_pct', 0):.0f}%, "
                    f"exp={r.get('experience_years', 0):.0f}yr, "
                    f"matched=[{', '.join(r.get('matched_skills', [])[:5])}]"
                )
            results_summary = "\n".join(lines)
        else:
            results_summary = "No analysis results available yet."

        prompt = CHAT_RESPONSE_PROMPT.format(
            query=message,
            count=len(self.scored_results),
            results_summary=results_summary,
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=CHAT_SYSTEM_PROMPT,
                temperature=0.3,
            )
            return {
                "type": "chat",
                "response": response,
                "candidates": [],
                "filters_applied": None,
            }
        except Exception as e:
            return {
                "type": "error",
                "response": f"Chat query failed: {str(e)}",
                "candidates": [],
                "filters_applied": None,
            }
