import json
import os
from collections import Counter
from backend.user_context import get_feedback_path, get_preferences_path

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def record_feedback(user_id: str, candidate_id: str, decision: str, reason: str, candidate_skills: list):
    feedback_path = get_feedback_path(user_id)
    feedback_data = load_json(feedback_path, [])
    
    # Store feedback
    feedback_data.append({
        "candidate_id": candidate_id,
        "decision": decision,
        "reason": reason,
        "skills": candidate_skills
    })
    save_json(feedback_path, feedback_data)
    
    # Update preferences
    update_preferences(user_id, feedback_data)

def update_preferences(user_id: str, feedback_data: list):
    prefs_path = get_preferences_path(user_id)
    
    shortlisted_skills = []
    rejected_skills = []
    
    for item in feedback_data:
        if item["decision"] == "shortlisted":
            shortlisted_skills.extend(item.get("skills", []))
        elif item["decision"] == "rejected":
            rejected_skills.extend(item.get("skills", []))
            
    short_counts = Counter(shortlisted_skills)
    rej_counts = Counter(rejected_skills)
    
    # Simple rule: if it's shortlisted frequently, it's preferred
    preferred = [skill for skill, count in short_counts.items() if count >= 1 and short_counts[skill] > rej_counts.get(skill, 0)]
    
    # If it's rejected frequently (e.g., they reject overqualified or specific tech stacks)
    rejected = [skill for skill, count in rej_counts.items() if count >= 2 and rej_counts[skill] > short_counts.get(skill, 0)]
    
    preferences = {
        "preferred_skills": preferred,
        "rejected_patterns": rejected,
        "experience_range": "",
        "common_rejections": [f.get("reason", "") for f in feedback_data if f["decision"] == "rejected"][-5:]
    }
    save_json(prefs_path, preferences)

def get_preferences(user_id: str) -> dict:
    return load_json(get_preferences_path(user_id), {
        "preferred_skills": [],
        "rejected_patterns": [],
        "experience_range": "",
        "common_rejections": []
    })

def get_feedback_summary(user_id: str) -> str:
    feedback_data = load_json(get_feedback_path(user_id), [])
    if not feedback_data:
        return "No past decisions recorded."
    
    recent = feedback_data[-5:]
    summary = []
    for f in recent:
        summary.append(f"- {f['decision'].upper()} candidate '{f['candidate_id']}' because: {f['reason']}")
    return "\n".join(summary)
