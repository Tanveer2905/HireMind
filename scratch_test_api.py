import urllib.request
import urllib.parse
import json
import os
import sys

def make_request(url, method="GET", data=None, headers=None):
    if headers is None:
        headers = {}
    
    req_data = None
    if data is not None:
        if isinstance(data, dict):
            req_data = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            req_data = data

    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res_content = response.read().decode("utf-8")
            return response.status, json.loads(res_content) if res_content else {}
    except urllib.error.HTTPError as e:
        err_content = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_content)
        except Exception:
            err_json = err_content
        print(f"HTTP Error {e.code} for {method} {url}: {err_json}")
        return e.code, err_json
    except Exception as e:
        print(f"Error connecting to {url}: {e}")
        return 500, str(e)

def run_tests():
    print("=== STARTING BACKEND INTEGRATION TESTS ===")
    base_url = "http://127.0.0.1:5000"
    
    email = "recruiter_test@example.com"
    password = "securepassword123"
    
    # 1. Register User
    print("\n1. Testing User Registration...")
    status, res = make_request(f"{base_url}/api/register", "POST", {"email": email, "password": password})
    print(f"Status: {status}, Response: {res}")
    
    # Allow registering again/already registered error
    if status == 400 and "already registered" in str(res):
        print("User already registered, proceeding to login.")
    elif status != 200:
        print("Registration failed!")
        return
        
    # 2. Login User
    print("\n2. Testing User Login...")
    status, res = make_request(f"{base_url}/api/login", "POST", {"email": email, "password": password})
    print(f"Status: {status}")
    if status != 200 or "access_token" not in res:
        print(f"Login failed! Response: {res}")
        return
        
    token = res["access_token"]
    auth_headers = {"Authorization": f"Bearer {token}"}
    print("Successfully obtained JWT token!")
    
    # 3. List resumes (should be empty initially for a new user, or we check)
    print("\n3. Testing list resumes (should be empty initially)...")
    status, res = make_request(f"{base_url}/api/resumes", "GET", headers=auth_headers)
    print(f"Status: {status}, Count: {res.get('count')}")
    
    # 4. Upload Resume
    print("\n4. Testing Upload Resumes...")
    # We will upload a dummy file or copy an existing resume PDF
    resume_source = "resumes/sarah_chen_ml_engineer.pdf"
    if not os.path.exists(resume_source):
        print("Warning: resumes/sarah_chen_ml_engineer.pdf not found, creating dummy content")
        with open("dummy.pdf", "wb") as f:
            f.write(b"%PDF-1.4 dummy pdf content for testing parsing...")
        resume_to_upload = "dummy.pdf"
    else:
        resume_to_upload = resume_source
        
    # Standard multipart form data manual formulation
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    with open(resume_to_upload, "rb") as f:
        file_content = f.read()
        
    filename = os.path.basename(resume_to_upload)
    
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode("utf-8") + file_content + f"\r\n--{boundary}--\r\n".encode("utf-8")
    
    upload_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    }
    
    status, res = make_request(f"{base_url}/api/upload", "POST", data=body, headers=upload_headers)
    print(f"Status: {status}, Uploaded: {res.get('uploaded')}, Errors: {res.get('errors')}")
    
    if os.path.exists("dummy.pdf"):
        os.remove("dummy.pdf")
        
    # Check resumes list again
    status, res = make_request(f"{base_url}/api/resumes", "GET", headers=auth_headers)
    print(f"Status: {status}, Count: {res.get('count')}, Resumes: {res.get('resumes')}")
    if res.get('count', 0) == 0:
        print("Upload failed to store file!")
        return
        
    # 5. Run Candidate Analysis
    print("\n5. Testing Candidate Analysis...")
    analysis_data = {
        "job_description": "We are looking for a Senior Machine Learning Engineer with experience in Python, AWS, Docker, Kubernetes.",
        "must_have_skills": ["Python"],
        "use_llm_rerank": False
    }
    status, res = make_request(f"{base_url}/api/analyze", "POST", data=analysis_data, headers=auth_headers)
    print(f"Status: {status}, Processing Time: {res.get('processing_time')}s")
    if status != 200 or "results" not in res:
        print(f"Analysis failed: {res}")
        return
        
    top_candidate = res["results"][0]
    print(f"Top Candidate: {top_candidate['filename']} with score {top_candidate['final_score']}")
    
    # 6. Test Chat
    print("\n6. Testing Copilot Chat query...")
    chat_data = {
        "message": "Who is our top candidate and what are their skills?"
    }
    status, res = make_request(f"{base_url}/api/chat", "POST", data=chat_data, headers=auth_headers)
    print(f"Status: {status}, Chat Type: {res.get('type')}")
    print(f"Chat Response:\n{res.get('response')}")
    
    # 7. Test Generate Interview Questions
    print("\n7. Testing Generate Interview Questions for top candidate...")
    candidate_filename = top_candidate["filename"]
    status, res = make_request(f"{base_url}/api/candidate/{urllib.parse.quote(candidate_filename)}/interview-questions", "POST", data={"count": 3}, headers=auth_headers)
    print(f"Status: {status}")
    if status == 200:
        print("Questions generated:")
        for idx, q in enumerate(res.get("questions", []), 1):
            print(f"  {idx}. {q.get('question')} (Purpose: {q.get('purpose')})")
            
    # 8. Test Decision Feedback
    print("\n8. Testing Decision Feedback...")
    feedback_data = {
        "filename": candidate_filename,
        "action": "shortlisted"
    }
    status, res = make_request(f"{base_url}/api/feedback", "POST", data=feedback_data, headers=auth_headers)
    print(f"Status: {status}, Recorded: {res.get('recorded')}, Total Feedback: {res.get('total_feedback')}")
    
    # 9. Test Feedback Stats
    print("\n9. Testing Feedback Stats...")
    status, res = make_request(f"{base_url}/api/feedback/stats", "GET", headers=auth_headers)
    print(f"Status: {status}, Total Stats: {res.get('total')}, Shortlisted: {res.get('shortlisted')}, Rejected: {res.get('rejected')}")
    
    # 10. Test Feedback Reset
    print("\n10. Testing Feedback Reset...")
    status, res = make_request(f"{base_url}/api/feedback/reset", "POST", headers=auth_headers)
    print(f"Status: {status}, Response: {res}")
    
    print("\n=== INTEGRATION TESTS COMPLETE! ===")

if __name__ == "__main__":
    run_tests()
