#!/usr/bin/env python3
"""
Test script for the feedback-based FitScore system
Demonstrates how feedback improves candidate evaluation over time
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
JOB_ID = "JD_001"
COMPANY_ID = "COMP_001"

def test_fitscore_calculation(resume_text, job_description, collateral=None):
    """Test FitScore calculation with optional feedback tracking"""
    url = f"{BASE_URL}/calculate-fitscore"
    
    payload = {
        "resume_text": resume_text,
        "job_description": job_description,
        "collateral": collateral,
        "job_id": JOB_ID,
        "company_id": COMPANY_ID,
        "use_gpt4": False  # Disable GPT-4 for testing
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calculating FitScore: {e}")
        return None

def submit_feedback(candidate_id, feedback_type, feedback_text, feedback_category, feedback_score):
    """Submit feedback for a candidate evaluation"""
    url = f"{BASE_URL}/feedback"
    
    payload = {
        "job_id": JOB_ID,
        "company_id": COMPANY_ID,
        "candidate_id": candidate_id,
        "feedback_type": feedback_type,
        "feedback_text": feedback_text,
        "feedback_category": feedback_category,
        "feedback_score": feedback_score
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error submitting feedback: {e}")
        return None

def get_feedback_summary():
    """Get summary of feedback and prompt versions"""
    url = f"{BASE_URL}/feedback/summary"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting feedback summary: {e}")
        return None

def get_prompt_versions():
    """Get all prompt versions"""
    url = f"{BASE_URL}/prompts/versions"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting prompt versions: {e}")
        return None

def main():
    """Main test function"""
    print("🚀 Testing Feedback-Based FitScore System")
    print("=" * 50)
    
    # Sample resume and job description
    sample_resume = """
    John Doe
    Software Engineer
    
    EDUCATION:
    University of California, Berkeley
    Bachelor of Science in Computer Science
    GPA: 3.6
    
    EXPERIENCE:
    Software Engineer
    TechCorp Inc.
    2021-2023 (2 years)
    - Developed web applications
    - Used Python, JavaScript, React
    - Worked on team projects
    
    Software Engineer Intern
    StartupXYZ
    2020-2021 (1 year)
    - Built basic features
    - Used Python, Django
    
    SKILLS:
    Python, JavaScript, React, Django, SQL, Git
    """
    
    sample_job_description = """
    Senior Software Engineer
    Tech Startup
    
    We are looking for a Senior Software Engineer to join our growing team.
    
    Requirements:
    - 3+ years of software engineering experience
    - Strong knowledge of Python, JavaScript, React
    - Experience with web development
    - Team collaboration skills
    
    Nice to have:
    - Startup experience
    - Full-stack development
    """
    
    # Test 1: Initial FitScore calculation
    print("\n📊 Test 1: Initial FitScore Calculation")
    print("-" * 40)
    
    result1 = test_fitscore_calculation(sample_resume, sample_job_description)
    if result1:
        print(f"Initial Score: {result1['total_score']:.2f}")
        print(f"Submittable: {result1['submittable']}")
        print(f"Prompt Version: {result1.get('prompt_version', 'Standard')}")
        print(f"Feedback Applied: {result1.get('feedback_applied', False)}")
    
    # Test 2: Submit negative feedback about skills
    print("\n💬 Test 2: Submitting Negative Feedback (Skills)")
    print("-" * 40)
    
    feedback1 = submit_feedback(
        candidate_id="CAND_001",
        feedback_type="negative",
        feedback_text="Candidate lacked required senior-level experience and technical depth",
        feedback_category="experience",
        feedback_score=-0.7
    )
    
    if feedback1:
        print(f"Feedback Status: {feedback1['message']}")
    
    # Test 3: Submit another negative feedback about skills
    print("\n💬 Test 3: Submitting Another Negative Feedback (Skills)")
    print("-" * 40)
    
    feedback2 = submit_feedback(
        candidate_id="CAND_002",
        feedback_type="negative",
        feedback_text="Skills assessment was too lenient, candidate missing critical capabilities",
        feedback_category="skills",
        feedback_score=-0.8
    )
    
    if feedback2:
        print(f"Feedback Status: {feedback2['message']}")
    
    # Wait a moment for feedback processing
    print("\n⏳ Waiting for feedback processing...")
    time.sleep(2)
    
    # Test 4: Check feedback summary
    print("\n📈 Test 4: Feedback Summary")
    print("-" * 40)
    
    summary = get_feedback_summary()
    if summary:
        print(f"Total Feedback: {summary['total_feedback']}")
        print(f"Total Prompts: {summary['total_prompts']}")
        print(f"Global Prompts: {summary['global_prompts']}")
        print(f"Local Prompts: {summary['local_prompts']}")
        print(f"Feedback by Category: {summary['feedback_by_category']}")
    
    # Test 5: Check prompt versions
    print("\n📋 Test 5: Prompt Versions")
    print("-" * 40)
    
    versions = get_prompt_versions()
    if versions:
        print(f"Total Prompt Versions: {len(versions['prompt_versions'])}")
        for version in versions['prompt_versions']:
            print(f"- {version['version_tag']} ({version['prompt_type']})")
            if version['feedback_pattern']:
                print(f"  Feedback: {version['feedback_pattern']}")
    
    # Test 6: Recalculate FitScore with feedback-applied prompt
    print("\n🔄 Test 6: FitScore with Feedback-Applied Prompt")
    print("-" * 40)
    
    result2 = test_fitscore_calculation(sample_resume, sample_job_description)
    if result2:
        print(f"Updated Score: {result2['total_score']:.2f}")
        print(f"Submittable: {result2['submittable']}")
        print(f"Prompt Version: {result2.get('prompt_version', 'Standard')}")
        print(f"Feedback Applied: {result2.get('feedback_applied', False)}")
        
        # Compare scores
        if result1 and result2:
            score_diff = result2['total_score'] - result1['total_score']
            print(f"Score Change: {score_diff:+.2f}")
    
    # Test 7: Submit positive feedback to balance
    print("\n💬 Test 7: Submitting Positive Feedback")
    print("-" * 40)
    
    feedback3 = submit_feedback(
        candidate_id="CAND_003",
        feedback_type="positive",
        feedback_text="Candidate evaluation was accurate and comprehensive",
        feedback_category="other",
        feedback_score=0.8
    )
    
    if feedback3:
        print(f"Feedback Status: {feedback3['message']}")
    
    print("\n✅ Testing Complete!")
    print("\nKey Features Demonstrated:")
    print("- Feedback collection and storage")
    print("- Automatic prompt versioning")
    print("- Local prompt updates based on feedback patterns")
    print("- Score adjustments based on feedback")
    print("- Version control and rollback capabilities")

if __name__ == "__main__":
    main() 