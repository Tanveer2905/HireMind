import logging
from backend.memory_engine import get_preferences

logger = logging.getLogger(__name__)

def apply_personalization(user_id: str, candidates: list[dict]) -> list[dict]:
    """
    Adjust candidate scores purely based on the user's isolated preferences,
    without requiring ML training.
    """
    prefs = get_preferences(user_id)
    preferred_skills = set([s.lower() for s in prefs.get("preferred_skills", [])])
    rejected_patterns = set([s.lower() for s in prefs.get("rejected_patterns", [])])
    
    if not preferred_skills and not rejected_patterns:
        return candidates  # No preferences learned yet
        
    logger.info(f"Applying personalization for user {user_id}. Preferred: {len(preferred_skills)}, Rejected: {len(rejected_patterns)}")
    
    for c in candidates:
        candidate_skills = set([s.lower() for s in c.get("matched_skills", [])])
        
        boost = 0.0
        penalty = 0.0
        
        # Boost for preferred skills
        overlap = preferred_skills.intersection(candidate_skills)
        if overlap:
            boost += 0.02 * len(overlap)  # +2% per preferred skill
            
        # Penalize for rejected patterns
        bad_overlap = rejected_patterns.intersection(candidate_skills)
        if bad_overlap:
            penalty += 0.05 * len(bad_overlap)  # -5% per rejected pattern
            
        original_score = c.get("final_score", 0.0)
        new_score = original_score + boost - penalty
        
        # Clamp between 0 and 1
        c["final_score"] = max(0.0, min(1.0, new_score))
        c["personalization_applied"] = True
        
    # Re-sort candidates based on the new personalized scores
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Update ranks
    for idx, c in enumerate(candidates):
        c["rank"] = idx + 1
        
    return candidates
