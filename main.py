"""
main.py — AI Recruiter CLI Entry Point
Orchestrates the full pipeline: ingest → parse → embed → index → score → rank → output
"""

import sys
import os
import io
import logging
import time
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    RESUMES_DIR, DATA_DIR, ensure_dirs,
    format_results_table, save_results_csv,
    file_hash, normalize_skill, extract_skills_from_text,
)
from parser import batch_parse_resumes, parse_job_description
from embeddings import EmbeddingEngine
from scorer import rerank_candidates

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CLI Input helpers
# ---------------------------------------------------------------------------
def get_multiline_input(prompt: str) -> str:
    """Get multi-line input from user. Empty line to finish."""
    print(prompt)
    print("(Enter an empty line when done)")
    print()
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                if lines:
                    break
                continue
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


def get_must_have_skills() -> set[str]:
    """Prompt user for must-have skills (hard filters)."""
    print()
    print("━" * 60)
    response = input(
        "Enter must-have skills (comma-separated), or press Enter to skip:\n> "
    ).strip()

    if not response:
        return set()

    skills = set()
    for item in response.split(","):
        item = item.strip()
        if not item:
            continue
        # Try to normalize
        canonical = normalize_skill(item)
        if canonical:
            skills.add(canonical)
        else:
            # Use as-is if not in taxonomy
            skills.add(item)

    if skills:
        print(f"  Must-have skills: {', '.join(sorted(skills))}")

    return skills


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    """Execute the full candidate ranking pipeline."""

    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🤖  AI RECRUITER — Candidate Ranking System  ".center(58) + "║")
    print("║" + "  Local-First • Offline • Free  ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # ------------------------------------------------------------------
    # Step 0: Check for resumes
    # ------------------------------------------------------------------
    ensure_dirs()

    pdf_files = list(RESUMES_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in: {RESUMES_DIR}")
        print(f"   Please place resume PDFs in the 'resumes' folder and try again.")
        print()
        sys.exit(1)

    print(f"📄 Found {len(pdf_files)} resume(s) in {RESUMES_DIR}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Get job description
    # ------------------------------------------------------------------
    print("━" * 60)
    jd_text = get_multiline_input("📋 Enter the Job Description:")

    if not jd_text.strip():
        print("❌ Job description cannot be empty.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Get must-have skills (hard filters)
    # ------------------------------------------------------------------
    must_have_skills = get_must_have_skills()

    # ------------------------------------------------------------------
    # Step 3: Parse job description
    # ------------------------------------------------------------------
    print()
    print("━" * 60)
    print("⚙️  Processing...")
    print()

    start_time = time.time()

    logger = logging.getLogger(__name__)
    logger.info("Parsing job description...")
    jd_data = parse_job_description(jd_text)

    print(f"  📋 JD Skills detected: {', '.join(jd_data['skills'][:15])}")
    if jd_data["required_experience"] > 0:
        print(f"  📅 Required experience: {jd_data['required_experience']:.0f} years")
    print()

    # ------------------------------------------------------------------
    # Step 4: Parse all resumes (with caching + multiprocessing)
    # ------------------------------------------------------------------
    logger.info("Parsing resumes...")
    print("  📄 Parsing resumes...")

    parsed_resumes = batch_parse_resumes(
        resume_dir=RESUMES_DIR,
        max_workers=min(4, os.cpu_count() or 1),
        use_cache=True,
    )

    if not parsed_resumes:
        print("❌ No resumes could be parsed. Check files and try again.")
        sys.exit(1)

    parse_time = time.time() - start_time
    print(f"  ✅ Parsed {len(parsed_resumes)} resumes ({parse_time:.1f}s)")
    print()

    # ------------------------------------------------------------------
    # Step 5: Generate embeddings & build FAISS index
    # ------------------------------------------------------------------
    logger.info("Generating embeddings...")
    print("  🧠 Generating embeddings...")

    embed_start = time.time()
    engine = EmbeddingEngine()

    # Compute file hashes for cache validation
    file_hashes = {}
    for pdf in pdf_files:
        file_hashes[pdf.name] = file_hash(pdf)

    # Get or compute embeddings (with caching)
    resume_embeddings, filenames = engine.get_or_compute_embeddings(
        parsed_resumes, file_hashes
    )

    # Build FAISS index
    engine.build_index(resume_embeddings, filenames)

    embed_time = time.time() - embed_start
    print(f"  ✅ Embeddings generated & indexed ({embed_time:.1f}s)")
    print()

    # ------------------------------------------------------------------
    # Step 6: Query with job description
    # ------------------------------------------------------------------
    logger.info("Searching for candidates...")
    print("  🔍 Searching for best candidates...")

    query_embedding = engine.encode_query(jd_text)

    # Get top 100 from FAISS (or all if fewer)
    top_k = min(100, len(parsed_resumes))
    faiss_results = engine.search(query_embedding, top_k=top_k)

    # ------------------------------------------------------------------
    # Step 7: Rerank using composite scoring
    # ------------------------------------------------------------------
    logger.info("Reranking candidates...")
    print("  🏆 Applying composite scoring & reranking...")

    # Build lookup dict for parsed resumes
    parsed_lookup = {r["filename"]: r for r in parsed_resumes}

    ranked_results = rerank_candidates(
        faiss_results=faiss_results,
        parsed_resumes=parsed_lookup,
        jd_data=jd_data,
        must_have_skills=must_have_skills if must_have_skills else None,
    )

    total_time = time.time() - start_time

    # ------------------------------------------------------------------
    # Step 8: Display results
    # ------------------------------------------------------------------
    print()
    table = format_results_table(ranked_results)
    print(table)

    # ------------------------------------------------------------------
    # Step 9: Export to CSV
    # ------------------------------------------------------------------
    print("━" * 60)
    participant_id = input("Enter your participant ID for the CSV filename (e.g. team_123) or press Enter for default: ").strip()
    if not participant_id:
        participant_id = "submission"
        
    csv_filename = f"{participant_id}.csv"
    if not csv_filename.endswith(".csv"):
        csv_filename += ".csv"
        
    csv_path = save_results_csv(ranked_results, path=DATA_DIR / csv_filename)
    print(f"  💾 Results exported to: {csv_path}")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Show detailed view for top 3
    # ------------------------------------------------------------------
    active_results = [r for r in ranked_results if not r.get("filtered")]
    if active_results:
        print("━" * 60)
        print("  📊 TOP CANDIDATES — Detailed Breakdown")
        print("━" * 60)
        for r in active_results[:3]:
            print()
            print(f"  #{r['rank']} {r['filename']}")
            print(f"     Final Score:     {r['final_score']:.4f}")
            print(f"     Semantic:        {r['semantic_score']:.4f} (weight: 35%)")
            print(f"     Skill Match:     {r['skill_score']:.4f} (weight: 25%)")
            print(f"     Experience:      {r['experience_score']:.4f} (weight: 20%)")
            print(f"     Recency:         {r['recency_score']:.4f} (weight: 10%)")
            print(f"     Keywords:        {r['keyword_score']:.4f} (weight: 10%)")
            print(f"     Experience Yrs:  {r['experience_years']}")
            if r.get('matched_skills'):
                print(f"     Matched Skills:  {', '.join(r['matched_skills'][:10])}")
            if r.get('missing_skills'):
                print(f"     Missing Skills:  {', '.join(r['missing_skills'][:5])}")
            print(f"     Assessment:      {r['explanation']}")
        print()
        print("━" * 60)

    print()
    print("  ✅ Done! Review the results above or check the CSV file.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    setup_logging(verbose=verbose)
    run_pipeline()
