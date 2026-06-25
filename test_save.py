import sys
from pathlib import Path
from utils import save_results_csv
from validate_submission import validate_submission

# Create 5 dummy results
results = []
for i in range(5):
    results.append({
        "filename": f"dummy_{i}.pdf",
        "rank": i+1,
        "final_score": 0.9 - (i * 0.1),
        "explanation": f"Candidate {i} is good"
    })

# Run save
path = Path("team_123.csv")
save_results_csv(results, path)

# Validate
errors = validate_submission(str(path))
if errors:
    print("ERRORS:", errors)
    sys.exit(1)
print("SUCCESS!")
