import os
from backend.parser import batch_parse_resumes, parse_job_description
from backend.faiss_manager import FaissManager
from backend.scorer import rerank_candidates
from backend.personalization import apply_personalization

user_id = "a0654acd-5725-408b-bd64-d5211fd391d2"
jd_text = "We are looking for a Python developer."

print("Parsing resumes...")
parsed = batch_parse_resumes(user_id, use_cache=False)
print("Parsed count:", len(parsed))

if not parsed:
    print("No resumes found.")
    exit(1)

parsed_map = {p["filename"]: p for p in parsed}
file_hashes = {p["filename"]: p.get("file_hash", "") for p in parsed}

print("Parsing JD...")
jd_data = parse_job_description(jd_text)
must_haves = set()

print("Initializing FAISS...")
faiss_manager = FaissManager()
embeddings, filenames = faiss_manager.get_or_compute_embeddings(user_id, parsed, file_hashes)
print("Building index...")
faiss_manager.build_index(user_id, embeddings, filenames)

print("Searching...")
query_emb = faiss_manager.encode_query(jd_text)
faiss_results = faiss_manager.search(user_id, query_emb, top_k=len(filenames))

print("Reranking...")
scored = rerank_candidates(faiss_results, parsed_map, jd_data, must_haves, top_k=100)

print("Applying personalization...")
personalized = apply_personalization(user_id, scored)

print("Done! Success.")
