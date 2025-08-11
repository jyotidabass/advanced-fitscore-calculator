from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
import json
import os
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fitscore_calculator import FitScoreCalculator
from platform_integration_engine import PlatformIntegrationEngine
from reinforcement_feedback_agent import ReinforcementFeedbackAgent
from ai_submission_engine import AISubmissionEngine

# Initialize FastAPI app
app = FastAPI(
    title="Advanced FitScore Calculator & AI Hiring System",
    description="A comprehensive candidate evaluation system with advanced AI learning, reinforcement feedback, and autonomous hiring workflows",
    version="1.0.0"
)

# Initialize the FitScore calculator
calculator = FitScoreCalculator()

# Initialize the Platform Integration Engine
platform_engine = PlatformIntegrationEngine()

# Initialize the Reinforcement Learning Feedback Agent
reinforcement_agent = ReinforcementFeedbackAgent()

# Initialize the AI Submission Engine
ai_submission_engine = AISubmissionEngine(reinforcement_agent)

# Create templates directory and mount static files
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class FitScoreRequest(BaseModel):
    resume_text: str
    job_description: str
    collateral: Optional[str] = None
    openai_api_key: Optional[str] = None
    use_gpt4: bool = True
    job_id: Optional[str] = None
    company_id: Optional[str] = None

class FitScoreResponse(BaseModel):
    total_score: float
    education_score: float
    career_trajectory_score: float
    company_relevance_score: float
    tenure_stability_score: float
    most_important_skills_score: float
    bonus_signals_score: float
    red_flags_penalty: float
    submittable: bool
    recommendations: list
    details: dict
    timestamp: str
    prompt_version: Optional[str] = None
    feedback_applied: bool = False

class FeedbackRequest(BaseModel):
    job_id: str
    company_id: str
    candidate_id: str
    feedback_type: str  # "positive", "negative", "neutral"
    feedback_text: str
    feedback_category: str  # "skills", "experience", "culture_fit", "location", "other"
    feedback_score: float  # -1.0 to 1.0

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with form for testing FitScore calculator"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate-fitscore", response_model=FitScoreResponse)
async def calculate_fitscore(request: FitScoreRequest):
    """
    Calculate FitScore for a candidate based on resume and job description
    """
    try:
        # Set OpenAI API key if provided
        if request.openai_api_key:
            os.environ["OPENAI_API_KEY"] = request.openai_api_key
            # Reinitialize calculator with new API key
            global calculator
            calculator = FitScoreCalculator(openai_api_key=request.openai_api_key)
        
        # Calculate FitScore
        result = calculator.calculate_fitscore(
            resume_text=request.resume_text,
            job_description=request.job_description,
            collateral=request.collateral,
            use_gpt4=request.use_gpt4,
            job_id=request.job_id,
            company_id=request.company_id
        )
        
        # Convert to response model
        response = FitScoreResponse(
            total_score=result.total_score,
            education_score=result.education_score,
            career_trajectory_score=result.career_trajectory_score,
            company_relevance_score=result.company_relevance_score,
            tenure_stability_score=result.tenure_stability_score,
            most_important_skills_score=result.most_important_skills_score,
            bonus_signals_score=result.bonus_signals_score,
            red_flags_penalty=result.red_flags_penalty,
            submittable=result.total_score >= 8.2,
            recommendations=result.recommendations,
            details=result.details,
            timestamp=result.timestamp,
            prompt_version=result.prompt_version,
            feedback_applied=result.feedback_applied
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating FitScore: {str(e)}")

@app.post("/calculate-fitscore-form")
async def calculate_fitscore_form(
    resume_text: str = Form(...),
    job_description: str = Form(...),
    collateral: Optional[str] = Form(None),
    openai_api_key: Optional[str] = Form(None),
    use_gpt4: str = Form("on"),
    job_id: Optional[str] = Form(None),
    company_id: Optional[str] = Form(None)
):
    """
    Calculate FitScore using form data (for web interface)
    """
    try:
        # Set OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            global calculator
            calculator = FitScoreCalculator(openai_api_key=openai_api_key)
        
        # Convert checkbox value to boolean
        use_gpt4_bool = use_gpt4 == "on"
        
        # Calculate FitScore
        result = calculator.calculate_fitscore(
            resume_text=resume_text,
            job_description=job_description,
            collateral=collateral,
            use_gpt4=use_gpt4_bool,
            job_id=job_id,
            company_id=company_id
        )
        
        # Return results for template
        return {
            "success": True,
            "total_score": round(result.total_score, 2),
            "education_score": round(result.education_score, 2),
            "career_trajectory_score": round(result.career_trajectory_score, 2),
            "company_relevance_score": round(result.company_relevance_score, 2),
            "tenure_stability_score": round(result.tenure_stability_score, 2),
            "most_important_skills_score": round(result.most_important_skills_score, 2),
            "bonus_signals_score": round(result.bonus_signals_score, 2),
            "red_flags_penalty": round(result.red_flags_penalty, 2),
            "submittable": result.total_score >= 8.2,
            "recommendations": result.recommendations,
            "details": result.details,
            "timestamp": result.timestamp,
            "prompt_version": result.prompt_version,
            "feedback_applied": result.feedback_applied
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "FitScore Calculator API is running"}

@app.post("/feedback", response_model=FeedbackResponse)
async def add_feedback(request: FeedbackRequest):
    """
    Add feedback for a candidate evaluation to improve future scoring
    """
    try:
        success = calculator.add_feedback(
            job_id=request.job_id,
            company_id=request.company_id,
            candidate_id=request.candidate_id,
            feedback_type=request.feedback_type,
            feedback_text=request.feedback_text,
            feedback_category=request.feedback_category,
            feedback_score=request.feedback_score
        )
        
        if success:
            return FeedbackResponse(
                success=True,
                message="Feedback added successfully",
                feedback_id=request.candidate_id
            )
        else:
            return FeedbackResponse(
                success=False,
                message="Failed to add feedback"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding feedback: {str(e)}")

@app.get("/feedback/summary")
async def get_feedback_summary():
    """Get summary of feedback and prompt versions"""
    try:
        summary = calculator.get_feedback_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feedback summary: {str(e)}")

@app.get("/prompts/versions")
async def get_prompt_versions():
    """Get all prompt versions with metadata"""
    try:
        versions = calculator.get_prompt_versions()
        return {"prompt_versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting prompt versions: {str(e)}")

@app.post("/prompts/reset/{job_id}")
async def reset_to_global_prompt(job_id: str):
    """Reset a job to use the global prompt instead of local feedback-based prompt"""
    try:
        success = calculator.reset_to_global_prompt(job_id)
        if success:
            return {"success": True, "message": f"Job {job_id} reset to global prompt"}
        else:
            return {"success": False, "message": f"Failed to reset job {job_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting prompt: {str(e)}")

@app.post("/prompts/generate-local")
async def generate_local_prompt(
    job_description: str = Form(...),
    collateral: Optional[str] = Form(None),
    role_type: Optional[str] = Form(None),
    job_location: Optional[str] = Form(None)
):
    """Generate Smart Hiring Criteria (Local Prompt) for a specific role"""
    try:
        local_prompt = calculator.feedback_engine.generate_local_prompt(
            job_description=job_description,
            collateral=collateral,
            role_type=role_type,
            job_location=job_location
        )
        return {
            "success": True,
            "local_prompt": local_prompt,
            "message": "Local prompt generated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating local prompt: {str(e)}")

@app.post("/prompts/update-global")
async def update_global_prompt_with_trends(
    feedback_trends: List[Dict[str, Any]] = Body(...),
    current_version: str = Body(...)
):
    """Update Global Base Prompt based on industry-wide hiring feedback trends"""
    try:
        updated_prompt = calculator.feedback_engine.update_global_prompt_with_industry_trends(
            feedback_trends=feedback_trends,
            current_version=current_version
        )
        return {
            "success": True,
            "updated_prompt": updated_prompt,
            "message": "Global prompt updated with industry trends"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating global prompt: {str(e)}")

@app.post("/scoring/100-point-scale")
async def apply_100_point_scoring(scores: Dict[str, float] = Body(...)):
    """Apply 100-point scale scoring as per specification"""
    try:
        scoring_result = calculator.apply_100_point_scoring(scores)
        return {
            "success": True,
            "scoring_result": scoring_result,
            "message": "100-point scale scoring applied successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying 100-point scoring: {str(e)}")

# === Platform Integration Endpoints ===

@app.post("/platform/jobs")
async def create_or_update_job(job_data: Dict[str, Any] = Body(...)):
    """Job Creation / Modification with AI-Triggered Actions"""
    try:
        job_id = job_data.get("job_id", str(uuid.uuid4()))
        result = platform_engine.job_created_or_updated(job_id, job_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating/updating job: {str(e)}")

@app.post("/platform/candidates")
async def add_candidate_to_job(candidate_data: Dict[str, Any] = Body(...)):
    """Candidate Submission & AI Evaluation"""
    try:
        candidate_id = candidate_data.get("candidate_id", str(uuid.uuid4()))
        job_id = candidate_data.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required")
        
        result = platform_engine.candidate_added(candidate_id, job_id, candidate_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding candidate: {str(e)}")

@app.post("/platform/feedback")
async def log_client_feedback(feedback_data: Dict[str, Any] = Body(...)):
    """Client Feedback Logging (Internal Team)"""
    try:
        candidate_id = feedback_data.get("candidate_id")
        job_id = feedback_data.get("job_id")
        decision = feedback_data.get("decision")
        reason = feedback_data.get("reason")
        internal_team_member = feedback_data.get("internal_team_member", "unknown")
        
        if not all([candidate_id, job_id, decision]):
            raise HTTPException(status_code=400, detail="candidate_id, job_id, and decision are required")
        
        result = platform_engine.internal_team_logs_feedback(
            candidate_id=candidate_id,
            job_id=job_id,
            decision=decision,
            reason=reason,
            internal_team_member=internal_team_member
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error logging feedback: {str(e)}")

@app.get("/platform/jobs/{job_id}")
async def get_job_posting(job_id: str):
    """Get job posting with criteria"""
    try:
        job = platform_engine.get_job_posting(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"success": True, "job": job}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving job: {str(e)}")

@app.get("/platform/candidates/{candidate_id}")
async def get_candidate_profile(candidate_id: str):
    """Get candidate profile with evaluation"""
    try:
        candidate = platform_engine.get_candidate_profile(candidate_id)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return {"success": True, "candidate": candidate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving candidate: {str(e)}")

@app.get("/platform/workflows")
async def get_platform_workflows(job_id: Optional[str] = None):
    """Get platform workflows, optionally filtered by job ID"""
    try:
        workflows = platform_engine.get_platform_workflows(job_id)
        return {"success": True, "workflows": workflows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving workflows: {str(e)}")

@app.get("/platform/metrics")
async def get_operational_metrics(metric_type: Optional[str] = None):
    """Get operational metrics, optionally filtered by type"""
    try:
        metrics = platform_engine.get_operational_metrics(metric_type)
        return {"success": True, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

@app.get("/platform/data-logs")
async def get_platform_data_logs(job_id: Optional[str] = None):
    """Get platform data logs, optionally filtered by job ID"""
    try:
        logs = platform_engine.get_platform_data_logs(job_id)
        return {"success": True, "data_logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data logs: {str(e)}")

@app.get("/platform/guard-rails")
async def get_guard_rail_settings():
    """Get all guard-rail settings"""
    try:
        settings = platform_engine.get_guard_rail_settings()
        return {"success": True, "guard_rail_settings": settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving guard-rail settings: {str(e)}")

@app.put("/platform/guard-rails/{setting_name}")
async def update_guard_rail_setting(
    setting_name: str, 
    update_data: Dict[str, Any] = Body(...)
):
    """Update a guard-rail setting"""
    try:
        new_value = update_data.get("value")
        override_by = update_data.get("override_by")
        
        if new_value is None:
            raise HTTPException(status_code=400, detail="value is required")
        
        platform_engine.update_guard_rail_setting(setting_name, new_value, override_by)
        return {"success": True, "message": f"Guard-rail setting {setting_name} updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating guard-rail setting: {str(e)}")

# === Reinforcement Learning Feedback Endpoints ===

@app.post("/reinforcement/feedback-outcome")
async def process_feedback_outcome(feedback_data: Dict[str, Any] = Body(...)):
    """Process feedback outcome with reinforcement learning"""
    try:
        candidate_id = feedback_data.get("candidate_id")
        job_id = feedback_data.get("job_id")
        outcome = feedback_data.get("outcome")
        feedback_metadata = feedback_data.get("feedback_metadata", {})
        job_family = feedback_data.get("job_family", "software_engineer")
        
        if not all([candidate_id, job_id, outcome]):
            raise HTTPException(status_code=400, detail="candidate_id, job_id, and outcome are required")
        
        result = reinforcement_agent.process_feedback_outcome(
            candidate_id=candidate_id,
            job_id=job_id,
            outcome=outcome,
            feedback_metadata=feedback_metadata,
            job_family=job_family
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback outcome: {str(e)}")

@app.get("/reinforcement/weights/{job_family}")
async def get_dynamic_weights(job_family: str):
    """Get current dynamic weights for a job family"""
    try:
        weights = reinforcement_agent.get_dynamic_weights(job_family)
        if not weights:
            raise HTTPException(status_code=404, detail="Job family not found")
        return {"success": True, "weights": weights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving weights: {str(e)}")

@app.get("/reinforcement/learning-history/{job_family}")
async def get_learning_history(job_family: str):
    """Get learning history for a job family"""
    try:
        history = reinforcement_agent.get_learning_history(job_family)
        return {"success": True, "learning_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving learning history: {str(e)}")

@app.get("/reinforcement/success-rate/{job_family}")
async def get_success_rate(job_family: str):
    """Get current success rate for a job family"""
    try:
        success_rate = reinforcement_agent.get_success_rate(job_family)
        return {"success": True, "success_rate": success_rate, "job_family": job_family}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving success rate: {str(e)}")

@app.post("/reinforcement/reset-weights/{job_family}")
async def reset_weights_to_base(job_family: str):
    """Reset weights to base values for a job family"""
    try:
        success = reinforcement_agent.reset_weights_to_base(job_family)
        if success:
            return {"success": True, "message": f"Weights reset for {job_family}"}
        else:
            raise HTTPException(status_code=404, detail="Job family not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting weights: {str(e)}")

@app.get("/reinforcement/feedback-summary")
async def get_feedback_summary():
    """Get summary of all feedback outcomes"""
    try:
        summary = reinforcement_agent.get_feedback_summary()
        return {"success": True, "feedback_summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feedback summary: {str(e)}")

# === AI Submission Engine Endpoints ===

@app.post("/ai-submission/process")
async def process_candidate_submission(submission_data: Dict[str, Any] = Body(...)):
    """Process candidate submission through AI workflow"""
    try:
        candidate_id = submission_data.get("candidate_id")
        job_id = submission_data.get("job_id")
        recruiter_id = submission_data.get("recruiter_id")
        submission_notes = submission_data.get("submission_notes")
        
        if not all([candidate_id, job_id, recruiter_id]):
            raise HTTPException(status_code=400, detail="candidate_id, job_id, and recruiter_id are required")
        
        result = ai_submission_engine.process_candidate_submission(
            candidate_id=candidate_id,
            job_id=job_id,
            recruiter_id=recruiter_id,
            submission_notes=submission_notes
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing submission: {str(e)}")

@app.post("/ai-submission/approve/{workflow_id}")
async def approve_submission(
    workflow_id: str, 
    approval_data: Dict[str, Any] = Body(...)
):
    """Approve candidate submission by recruiter"""
    try:
        recruiter_id = approval_data.get("recruiter_id")
        notes = approval_data.get("notes")
        
        if not recruiter_id:
            raise HTTPException(status_code=400, detail="recruiter_id is required")
        
        result = ai_submission_engine.approve_submission(workflow_id, recruiter_id, notes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error approving submission: {str(e)}")

@app.post("/ai-submission/reject/{workflow_id}")
async def reject_submission(
    workflow_id: str, 
    rejection_data: Dict[str, Any] = Body(...)
):
    """Reject candidate submission by recruiter"""
    try:
        recruiter_id = rejection_data.get("recruiter_id")
        reason = rejection_data.get("reason")
        
        if not all([recruiter_id, reason]):
            raise HTTPException(status_code=400, detail="recruiter_id and reason are required")
        
        result = ai_submission_engine.reject_submission(workflow_id, recruiter_id, reason)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rejecting submission: {str(e)}")

@app.post("/ai-submission/client-feedback/{workflow_id}")
async def record_client_feedback(
    workflow_id: str, 
    feedback_data: Dict[str, Any] = Body(...)
):
    """Record client feedback and trigger learning"""
    try:
        outcome = feedback_data.get("outcome")
        feedback_metadata = feedback_data.get("feedback_metadata", {})
        
        if not outcome:
            raise HTTPException(status_code=400, detail="outcome is required")
        
        result = ai_submission_engine.record_client_feedback(workflow_id, outcome, feedback_metadata)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording client feedback: {str(e)}")

@app.get("/ai-submission/workflow/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    """Get current workflow status and details"""
    try:
        workflow = ai_submission_engine.get_workflow_status(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return {"success": True, "workflow": workflow}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving workflow: {str(e)}")

@app.get("/ai-submission/top-candidates/{job_id}")
async def get_top_candidates_for_job(job_id: str, limit: Optional[int] = 10):
    """Get top candidates for a specific job"""
    try:
        candidates = ai_submission_engine.get_top_candidates_for_job(job_id, limit)
        return {"success": True, "candidates": candidates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving candidates: {str(e)}")

@app.get("/api-docs")
async def api_docs():
    """API documentation endpoint"""
    return {
        "endpoints": {
            "POST /calculate-fitscore": "Calculate FitScore with JSON payload",
            "POST /calculate-fitscore-form": "Calculate FitScore with form data",
            "POST /feedback": "Add feedback for candidate evaluation",
            "GET /feedback/summary": "Get feedback and prompt summary",
            "GET /prompts/versions": "Get all prompt versions",
            "POST /prompts/reset/{job_id}": "Reset job to global prompt",
            "POST /prompts/generate-local": "Generate Smart Hiring Criteria (Local Prompt)",
            "POST /prompts/update-global": "Update Global Base Prompt with industry trends",
            "POST /scoring/100-point-scale": "Apply 100-point scale scoring",
            "POST /platform/jobs": "Job Creation/Modification with AI-Triggered Actions",
            "POST /platform/candidates": "Candidate Submission & AI Evaluation",
            "POST /platform/feedback": "Client Feedback Logging (Internal Team)",
            "GET /platform/jobs/{job_id}": "Get job posting with criteria",
            "GET /platform/candidates/{candidate_id}": "Get candidate profile with evaluation",
            "GET /platform/workflows": "Get platform workflows",
            "GET /platform/metrics": "Get operational metrics",
            "GET /platform/data-logs": "Get platform data logs",
            "GET /platform/guard-rails": "Get guard-rail settings",
            "PUT /platform/guard-rails/{setting_name}": "Update guard-rail setting",
            "POST /reinforcement/feedback-outcome": "Process feedback outcome with reinforcement learning",
            "GET /reinforcement/weights/{job_family}": "Get dynamic weights for job family",
            "GET /reinforcement/learning-history/{job_family}": "Get learning history for job family",
            "GET /reinforcement/success-rate/{job_family}": "Get success rate for job family",
            "POST /reinforcement/reset-weights/{job_family}": "Reset weights to base values",
            "GET /reinforcement/feedback-summary": "Get feedback summary",
            "POST /ai-submission/process": "Process candidate submission through AI workflow",
            "POST /ai-submission/approve/{workflow_id}": "Approve candidate submission",
            "POST /ai-submission/reject/{workflow_id}": "Reject candidate submission",
            "POST /ai-submission/client-feedback/{workflow_id}": "Record client feedback",
            "GET /ai-submission/workflow/{workflow_id}": "Get workflow status",
            "GET /ai-submission/top-candidates/{job_id}": "Get top candidates for job",
            "GET /": "Web interface for testing",
            "GET /health": "Health check",
            "GET /api-docs": "This documentation"
        },
        "example_request": {
            "resume_text": "Candidate resume text...",
            "job_description": "Job description text...",
            "collateral": "Additional context (optional)",
            "openai_api_key": "Your OpenAI API key (optional)",
            "use_gpt4": True,
            "job_id": "JD_001 (optional)",
            "company_id": "COMP_001 (optional)"
        },
        "feedback_example": {
            "job_id": "JD_001",
            "company_id": "COMP_001",
            "candidate_id": "CAND_001",
            "feedback_type": "negative",
            "feedback_text": "Candidate lacked required B2B experience",
            "feedback_category": "experience",
            "feedback_score": -0.5
        }
    }

# Vercel deployment requirement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000))) 
