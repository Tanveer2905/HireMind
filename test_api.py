import urllib.request, json, time, sys, io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

time.sleep(3)  # Wait for server to start

data = json.dumps({
    'job_description': 'We are looking for a Senior Python Developer with 5+ years of experience in Python, Django or FastAPI, PostgreSQL, Docker, and AWS. Experience with machine learning libraries (TensorFlow, PyTorch) is a plus. Must have strong experience with REST API design and microservices architecture.',
    'must_have_skills': ['Python', 'Docker'],
    'use_llm_rerank': True
}).encode('utf-8')

req = urllib.request.Request('http://127.0.0.1:5000/api/analyze', data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=300) as response:
        resp = json.loads(response.read().decode('utf-8'))
        
        print("=" * 80)
        print(f"  LLM Used: {resp.get('llm_used')}")
        print(f"  Total Candidates: {resp.get('total_candidates')}")
        print("=" * 80)
        
        for r in resp.get('results', []):
            print()
            print(f"--- #{r.get('rank')} {r.get('filename')} ---")
            print(f"  LLM Evaluated: {r.get('llm_evaluated')}")
            print(f"  LLM Score:     {r.get('llm_score')}")
            print(f"  LLM Decision:  {r.get('llm_decision')}")
            print(f"  Final Score:   {r.get('final_score')}")
            
            reasoning = r.get('llm_reasoning', [])
            if reasoning:
                print(f"  Reasoning:")
                for reason in reasoning:
                    print(f"    - {reason}")
            
            strengths = r.get('llm_strengths', [])
            if strengths:
                print(f"  Strengths:")
                for s in strengths:
                    print(f"    + {s}")
            
            weaknesses = r.get('llm_weaknesses', [])
            if weaknesses:
                print(f"  Weaknesses:")
                for w in weaknesses:
                    print(f"    - {w}")
            
            risks = r.get('llm_risk_flags', [])
            if risks:
                print(f"  Red Flags:")
                for rf in risks:
                    print(f"    ⚠ {rf}")
            
            questions = r.get('llm_interview_questions', [])
            if questions:
                print(f"  Interview Questions:")
                for q in questions[:2]:
                    print(f"    ? {q}")
            
            print()
        
except Exception as e:
    print(f"Error: {e}")
