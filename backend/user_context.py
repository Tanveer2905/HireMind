import os

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users")

def get_user_dir(user_id: str) -> str:
    user_dir = os.path.join(BASE_DATA_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_resumes_dir(user_id: str) -> str:
    path = os.path.join(get_user_dir(user_id), "resumes")
    os.makedirs(path, exist_ok=True)
    return path

def get_jd_dir(user_id: str) -> str:
    path = os.path.join(get_user_dir(user_id), "jd")
    os.makedirs(path, exist_ok=True)
    return path

def get_embeddings_dir(user_id: str) -> str:
    path = os.path.join(get_user_dir(user_id), "embeddings")
    os.makedirs(path, exist_ok=True)
    return path

def get_faiss_dir(user_id: str) -> str:
    path = os.path.join(get_user_dir(user_id), "faiss_index")
    os.makedirs(path, exist_ok=True)
    return path

def get_memory_path(user_id: str) -> str:
    return os.path.join(get_user_dir(user_id), "memory.json")

def get_preferences_path(user_id: str) -> str:
    return os.path.join(get_user_dir(user_id), "preferences.json")

def get_feedback_path(user_id: str) -> str:
    return os.path.join(get_user_dir(user_id), "feedback.json")
