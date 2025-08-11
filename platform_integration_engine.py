#!/usr/bin/env python3
"""
Platform Integration Engine for Comprehensive AI-Powered Recruitment Workflow
Implements the complete platform automation workflow as per specification
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

from platform_models import (
    JobPosting, CandidateProfile, ClientFeedback, HiringCriteria,
    ScoreCalibration, MarketTrends, PlatformWorkflow, OperationalMetrics,
    PlatformDataLog, AIPromptTemplate, GuardRailSettings,
    DecisionType, FeedbackCategory, CriteriaType, AI_ACTION_TYPES,
    DB_OPERATION_TYPES, WORKFLOW_STATUSES, DEFAULT_GUARD_RAILS
)

logger = logging.getLogger(__name__)

class PlatformIntegrationEngine:
    """
    Platform Integration Engine - Implements the complete AI-powered recruitment workflow
    
    Features:
    - Job Creation/Modification with Core + Adaptive Criteria
    - Candidate Submission & AI Evaluation
    - Client Feedback Logging & Analysis
    - Score Calibration for Outcome Discrepancies
    - Quarterly Market Trends Analysis
    - Operational Monitoring & Metrics
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the platform integration engine"""
        self.openai_api_key = openai_api_key
        
        # In-memory storage (replace with actual database in production)
        self.job_postings: Dict[str, JobPosting] = {}
        self.candidate_profiles: Dict[str, CandidateProfile] = {}
        self.client_feedback: Dict[str, ClientFeedback] = {}
        self.hiring_criteria: Dict[str, HiringCriteria] = {}
        self.score_calibrations: Dict[str, ScoreCalibration] = {}
        self.market_trends: Dict[str, MarketTrends] = {}
        self.platform_workflows: Dict[str, PlatformWorkflow] = {}
        self.operational_metrics: Dict[str, OperationalMetrics] = {}
        self.platform_data_logs: Dict[str, PlatformDataLog] = {}
        
        # AI Prompt Templates
        self.ai_prompt_templates: Dict[str, AIPromptTemplate] = {}
        self.guard_rail_settings: Dict[str, GuardRailSettings] = {}
        
        # Initialize default settings
        self._initialize_default_settings()
        self._initialize_ai_prompt_templates()
        
        logger.info("Platform Integration Engine initialized successfully")
    
    def _initialize_default_settings(self):
        """Initialize default guard-rail settings"""
        for setting_name, setting_value in DEFAULT_GUARD_RAILS.items():
            setting = GuardRailSettings(
                setting_id=f"guard_rail_{setting_name}",
                setting_name=setting_name,
                setting_value=setting_value,
                setting_type=self._get_setting_type(setting_value),
                description=f"Default setting for {setting_name}"
            )
            self.guard_rail_settings[setting.setting_id] = setting
    
    def _get_setting_type(self, value: Any) -> str:
        """Determine the type of a setting value"""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        else:
            return "string"
    
    def _initialize_ai_prompt_templates(self):
        """Initialize AI prompt templates as per specification"""
        
        # Client-Feedback Analysis Prompt
        client_feedback_template = AIPromptTemplate(
            template_id="client_feedback_analysis",
            template_type="client_feedback_analysis",
            template_content="""
Current Hiring Criteria:
{criteria}

New feedback logged by internal team:
"{reason}"

Safeguard: Only suggest incremental adjustments if ≥ 2–3 past cases match.
Avoid large or disruptive changes from isolated feedback.

Output:
- Incremental Criteria Adjustment (if any)
- Minor Prompt Update
- Alignment Explanation
            """,
            variables=["criteria", "reason"],
            safeguards={
                "pattern_validation_threshold": 2,
                "max_weight_change_per_cycle": 0.10,
                "human_oversight_required": True
            }
        )
        self.ai_prompt_templates[client_feedback_template.template_id] = client_feedback_template
        
        # Fit-Score Calibration Prompt
        score_calibration_template = AIPromptTemplate(
            template_id="fit_score_calibration",
            template_type="fit_score_calibration",
            template_content="""
Candidate Outcomes vs Scores:
{list_of_cases}

Current Criteria Weighting:
{weights}

Suggest slight (<10 %) weight tweaks to realign scores with decisions.

⚠️ Practical Safeguards Against Outlier Feedback:
- Pattern-validation threshold ≥ 2–3 similar feedback cases
- Max weight change per cycle 5–10 %
- Human oversight required for changes >5%
            """,
            variables=["list_of_cases", "weights"],
            safeguards={
                "max_weight_change_per_cycle": 0.10,
                "human_oversight_threshold": 0.05,
                "pattern_validation_threshold": 2
            }
        )
        self.ai_prompt_templates[score_calibration_template.template_id] = score_calibration_template
    
    # === Platform-Integrated AI Automation (as per specification) ===
    
    def job_created_or_updated(self, job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Job Creation / Modification
        
        Platform Action: Recruiter (only) creates or edits a job.
        AI-Triggered Actions:
        - Generate Core Smart Hiring Criteria
        - Quarterly, generate Adaptive Criteria from real-time market signals
        Result: Combined criteria stored in platform DB (labeled Core + Adaptive)
        """
        try:
            # Create workflow tracking
            workflow = self._create_workflow(job_id, "job_creation")
            
            # Fetch job description (simulate database operation)
            jd = self._fetch_job_description(job_id, job_data)
            self._log_db_operation(workflow, "FETCH_JOB_DESCRIPTION")
            
            # Generate Core Smart Hiring Criteria
            core = self._generate_core_criteria(jd)
            self._log_ai_action(workflow, "GENERATE_CORE_CRITERIA")
            
            # Check if quarterly update is needed
            quarterly_update_needed = self._check_quarterly_update_needed(job_id)
            
            if quarterly_update_needed:
                # Generate Adaptive Criteria from real-time market signals
                adap = self._generate_adaptive_criteria(jd, quarterly=True)
                self._log_ai_action(workflow, "GENERATE_ADAPTIVE_CRITERIA")
            else:
                adap = None
            
            # Store combined criteria in platform DB
            combined_criteria = self._combine_criteria(core, adap)
            self._store_combined_criteria(job_id, combined_criteria)
            self._log_db_operation(workflow, "STORE_COMBINED_CRITERIA")
            
            # Update workflow status
            self._complete_workflow(workflow)
            
            # Create or update job posting
            job_posting = JobPosting(
                job_id=job_id,
                company_id=job_data.get("company_id", "unknown"),
                title=job_data.get("title", ""),
                description=jd,
                requirements=job_data.get("requirements", []),
                location=job_data.get("location", ""),
                role_type=job_data.get("role_type", "unknown"),
                salary_range=job_data.get("salary_range"),
                core_criteria=core,
                adaptive_criteria=adap,
                combined_criteria=combined_criteria,
                last_quarterly_update=datetime.utcnow().isoformat() if quarterly_update_needed else None,
                market_trends_analyzed=quarterly_update_needed
            )
            self.job_postings[job_id] = job_posting
            
            return {
                "success": True,
                "job_id": job_id,
                "core_criteria_generated": True,
                "adaptive_criteria_generated": quarterly_update_needed,
                "combined_criteria_stored": True,
                "workflow_id": workflow.workflow_id
            }
            
        except Exception as e:
            logger.error(f"Error in job_created_or_updated: {e}")
            if 'workflow' in locals():
                self._fail_workflow(workflow, str(e))
            return {"success": False, "error": str(e)}
    
    def candidate_added(self, candidate_id: str, job_id: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Candidate Submission & Evaluation
        
        Platform Action: Recruiter attaches a candidate profile to the job.
        AI-Triggered Action: LLM evaluates the profile against stored criteria.
        Result: Fit-Score (1-10 scale) and evaluation summary saved to candidate record
        """
        try:
            # Create workflow tracking
            workflow = self._create_workflow(job_id, "candidate_evaluation")
            
            # Fetch candidate data (simulate database operation)
            profile = self._fetch_candidate_data(candidate_id, candidate_data)
            self._log_db_operation(workflow, "FETCH_CANDIDATE_DATA")
            
            # Fetch hiring criteria
            criteria = self._fetch_hiring_criteria(job_id)
            self._log_db_operation(workflow, "FETCH_HIRING_CRITERIA")
            
            # LLM evaluates the profile against stored criteria
            score, summary = self._evaluate_candidate_llm(profile, criteria)
            self._log_ai_action(workflow, "EVALUATE_CANDIDATE")
            
            # Store Fit-Score and evaluation summary in candidate record
            self._store_fit_score(candidate_id, job_id, score, summary)
            self._log_db_operation(workflow, "STORE_FIT_SCORE")
            
            # Create candidate profile
            candidate_profile = CandidateProfile(
                candidate_id=candidate_id,
                job_id=job_id,
                name=candidate_data.get("name", ""),
                email=candidate_data.get("email", ""),
                resume_text=candidate_data.get("resume_text", ""),
                resume_file_path=candidate_data.get("resume_file_path"),
                phone=candidate_data.get("phone"),
                location=candidate_data.get("location"),
                experience_years=candidate_data.get("experience_years"),
                current_company=candidate_data.get("current_company"),
                current_title=candidate_data.get("current_title"),
                fit_score=score,
                evaluation_summary=summary,
                evaluation_timestamp=datetime.utcnow().isoformat()
            )
            self.candidate_profiles[candidate_id] = candidate_profile
            
            # Update workflow status
            self._complete_workflow(workflow)
            
            return {
                "success": True,
                "candidate_id": candidate_id,
                "job_id": job_id,
                "fit_score": score,
                "evaluation_summary": summary,
                "stored_in_candidate_record": True,
                "workflow_id": workflow.workflow_id
            }
            
        except Exception as e:
            logger.error(f"Error in candidate_added: {e}")
            if 'workflow' in locals():
                self._fail_workflow(workflow, str(e))
            return {"success": False, "error": str(e)}
    
    def internal_team_logs_feedback(
        self, 
        candidate_id: str, 
        job_id: str, 
        decision: DecisionType, 
        reason: Optional[str] = None,
        internal_team_member: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Client Feedback Logging (Internal Team)
        
        Off-platform Action: Client e-mails or calls decision.
        Platform Action: Internal team logs Accept/Reject + reason.
        AI-Triggered Actions:
        1. Incremental feedback analysis (pattern-based).
        2. Score-vs-outcome discrepancy check ➔ on-the-spot calibration if required.
        """
        try:
            # Create workflow tracking
            workflow = self._create_workflow(job_id, "feedback_analysis")
            
            # Log feedback
            feedback_id = str(uuid.uuid4())
            client_feedback = ClientFeedback(
                feedback_id=feedback_id,
                candidate_id=candidate_id,
                job_id=job_id,
                company_id=self.job_postings.get(job_id, {}).get("company_id", "unknown"),
                decision=decision,
                feedback_reason=reason,
                feedback_category=self._categorize_feedback(reason),
                internal_team_member=internal_team_member
            )
            self.client_feedback[feedback_id] = client_feedback
            
            # Update candidate profile with client decision
            if candidate_id in self.candidate_profiles:
                self.candidate_profiles[candidate_id].client_decision = decision
                self.candidate_profiles[candidate_id].client_feedback = reason
                self.candidate_profiles[candidate_id].client_decision_timestamp = datetime.utcnow().isoformat()
            
            # Log feedback operation
            self._log_db_operation(workflow, "LOG_FEEDBACK")
            
            # Trigger AI analysis if reason provided
            if reason:
                # Check for similar feedback history
                history = self._fetch_similar_feedback(job_id, reason)
                self._log_db_operation(workflow, "FETCH_SIMILAR_FEEDBACK")
                
                # Check pattern validation threshold
                threshold = self.guard_rail_settings.get("guard_rail_pattern_validation_threshold").setting_value
                
                if len(history) >= threshold:
                    # Incremental feedback analysis
                    update = self._analyze_feedback_incremental_llm(reason, history)
                    self._log_ai_action(workflow, "ANALYZE_FEEDBACK")
                    
                    # Update hiring criteria
                    self._update_hiring_criteria(job_id, update)
                    self._log_db_operation(workflow, "UPDATE_HIRING_CRITERIA")
                    
                    # Mark feedback as analyzed
                    client_feedback.analyzed = True
                    client_feedback.analysis_timestamp = datetime.utcnow().isoformat()
                    client_feedback.pattern_matched = True
                    client_feedback.similar_feedback_count = len(history)
                    client_feedback.incremental_adjustment_applied = True
            
            # Check for score-vs-outcome discrepancies
            if self._detect_outcome_discrepancy(job_id):
                # Trigger immediate Fit-Score calibration
                recal = self._calibrate_scores_llm(job_id)
                self._log_ai_action(workflow, "CALIBRATE_SCORES")
                
                # Update hiring criteria with calibration
                self._update_hiring_criteria(job_id, recal)
                self._log_db_operation(workflow, "UPDATE_HIRING_CRITERIA")
            
            # Update workflow status
            self._complete_workflow(workflow)
            
            # Log platform data
            self._log_platform_data(job_id, candidate_id, decision, reason)
            
            return {
                "success": True,
                "feedback_id": feedback_id,
                "feedback_logged": True,
                "feedback_analyzed": reason is not None and len(history) >= threshold if 'history' in locals() else False,
                "score_calibration_triggered": self._detect_outcome_discrepancy(job_id),
                "workflow_id": workflow.workflow_id
            }
            
        except Exception as e:
            logger.error(f"Error in internal_team_logs_feedback: {e}")
            if 'workflow' in locals():
                self._fail_workflow(workflow, str(e))
            return {"success": False, "error": str(e)} 
    
    # === Helper Methods ===
    
    def _create_workflow(self, job_id: str, workflow_type: str) -> PlatformWorkflow:
        """Create a new platform workflow"""
        workflow = PlatformWorkflow(
            workflow_id=str(uuid.uuid4()),
            job_id=job_id,
            workflow_type=workflow_type,
            status=WORKFLOW_STATUSES["PENDING"]
        )
        self.platform_workflows[workflow.workflow_id] = workflow
        return workflow
    
    def _log_ai_action(self, workflow: PlatformWorkflow, action_type: str):
        """Log AI action in workflow"""
        workflow.ai_actions_triggered.append(action_type)
    
    def _log_db_operation(self, workflow: PlatformWorkflow, operation_type: str):
        """Log database operation in workflow"""
        workflow.database_operations.append(operation_type)
    
    def _complete_workflow(self, workflow: PlatformWorkflow):
        """Mark workflow as completed"""
        workflow.status = WORKFLOW_STATUSES["COMPLETED"]
        workflow.completed_timestamp = datetime.utcnow().isoformat()
    
    def _fail_workflow(self, workflow: PlatformWorkflow, error_message: str):
        """Mark workflow as failed"""
        workflow.status = WORKFLOW_STATUSES["FAILED"]
        workflow.error_messages.append(error_message)
        workflow.completed_timestamp = datetime.utcnow().isoformat()
    
    def _fetch_job_description(self, job_id: str, job_data: Dict[str, Any]) -> str:
        """Fetch job description (simulate database operation)"""
        return job_data.get("description", f"Job description for {job_id}")
    
    def _fetch_candidate_data(self, candidate_id: str, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch candidate data (simulate database operation)"""
        return candidate_data
    
    def _fetch_hiring_criteria(self, job_id: str) -> Dict[str, Any]:
        """Fetch hiring criteria for a job"""
        if job_id in self.job_postings:
            return self.job_postings[job_id].combined_criteria or {}
        return {}
    
    def _fetch_similar_feedback(self, job_id: str, reason: str) -> List[ClientFeedback]:
        """Fetch similar feedback for pattern analysis"""
        similar_feedback = []
        for feedback in self.client_feedback.values():
            if feedback.job_id == job_id and feedback.feedback_reason:
                # Simple similarity check (in production, use more sophisticated NLP)
                if any(keyword in feedback.feedback_reason.lower() for keyword in reason.lower().split()):
                    similar_feedback.append(feedback)
        return similar_feedback
    
    def _generate_core_criteria(self, job_description: str) -> Dict[str, Any]:
        """Generate core smart hiring criteria using AI"""
        # In production, this would call GPT-4/Claude API
        # For now, return a structured template
        return {
            "type": "core",
            "generated_at": datetime.utcnow().isoformat(),
            "job_description": job_description,
            "scoring_criteria": {
                "education": {"weight": 0.25, "criteria": "Degree requirements and institution quality"},
                "experience": {"weight": 0.30, "criteria": "Relevant work experience and progression"},
                "skills": {"weight": 0.25, "criteria": "Technical and soft skills"},
                "culture_fit": {"weight": 0.20, "criteria": "Company culture alignment"}
            }
        }
    
    def _generate_adaptive_criteria(self, job_description: str, quarterly: bool = False) -> Dict[str, Any]:
        """Generate adaptive criteria from market trends"""
        if not quarterly:
            return None
        
        # In production, this would analyze real-time market signals
        return {
            "type": "adaptive",
            "generated_at": datetime.utcnow().isoformat(),
            "quarterly_update": True,
            "market_signals": ["AI/ML demand increasing", "Remote work preference stable"],
            "emerging_skills": ["LLM deployment", "RAG systems", "Vector databases"],
            "declining_skills": ["Legacy frameworks", "On-premise systems"],
            "salary_trends": {"direction": "increasing", "percentage": 15}
        }
    
    def _combine_criteria(self, core: Dict[str, Any], adaptive: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine core and adaptive criteria"""
        combined = {
            "type": "combined",
            "core_criteria": core,
            "adaptive_criteria": adaptive,
            "combined_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        if adaptive:
            # Merge adaptive insights into core criteria
            if "emerging_skills" in adaptive:
                combined["emerging_skills"] = adaptive["emerging_skills"]
            if "market_signals" in adaptive:
                combined["market_signals"] = adaptive["market_signals"]
        
        return combined
    
    def _store_combined_criteria(self, job_id: str, criteria: Dict[str, Any]):
        """Store combined criteria in platform DB"""
        criteria_id = str(uuid.uuid4())
        hiring_criteria = HiringCriteria(
            criteria_id=criteria_id,
            job_id=job_id,
            criteria_type=CriteriaType.COMBINED,
            version="1.0",
            criteria_content=criteria,
            source="ai_generated",
            quarterly_update=criteria.get("adaptive_criteria") is not None
        )
        self.hiring_criteria[criteria_id] = hiring_criteria
    
    def _evaluate_candidate_llm(self, profile: Dict[str, Any], criteria: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate candidate using LLM against stored criteria"""
        # In production, this would call GPT-4/Claude API
        # For now, return a simulated evaluation
        
        # Simulate scoring based on profile data
        base_score = 7.0
        
        # Adjust based on experience
        if profile.get("experience_years", 0) >= 5:
            base_score += 1.0
        elif profile.get("experience_years", 0) >= 3:
            base_score += 0.5
        
        # Adjust based on skills match
        resume_text = profile.get("resume_text", "").lower()
        if "python" in resume_text or "javascript" in resume_text:
            base_score += 0.5
        
        # Ensure score is within 1-10 range
        score = max(1.0, min(10.0, base_score))
        
        summary = {
            "overall_score": score,
            "strengths": ["Relevant experience", "Technical skills"],
            "areas_for_improvement": ["Could benefit from more leadership experience"],
            "recommendation": "Strong candidate, recommend for next round"
        }
        
        return score, summary
    
    def _store_fit_score(self, candidate_id: str, job_id: str, score: float, summary: Dict[str, Any]):
        """Store Fit-Score and evaluation summary in candidate record"""
        # This is handled in the candidate profile creation
        pass
    
    def _categorize_feedback(self, reason: Optional[str]) -> Optional[FeedbackCategory]:
        """Categorize feedback based on reason"""
        if not reason:
            return None
        
        reason_lower = reason.lower()
        
        if any(word in reason_lower for word in ["skill", "technical", "technology"]):
            return FeedbackCategory.SKILLS
        elif any(word in reason_lower for word in ["experience", "background", "years"]):
            return FeedbackCategory.EXPERIENCE
        elif any(word in reason_lower for word in ["culture", "fit", "personality"]):
            return FeedbackCategory.CULTURE_FIT
        elif any(word in reason_lower for word in ["location", "remote", "onsite"]):
            return FeedbackCategory.LOCATION
        elif any(word in reason_lower for word in ["salary", "compensation", "pay"]):
            return FeedbackCategory.SALARY
        else:
            return FeedbackCategory.OTHER
    
    def _analyze_feedback_incremental_llm(self, reason: str, history: List[ClientFeedback]) -> Dict[str, Any]:
        """Analyze feedback incrementally using LLM"""
        # In production, this would call GPT-4/Claude API with the feedback analysis prompt
        # For now, return a structured analysis
        
        return {
            "feedback_analysis": {
                "reason": reason,
                "pattern_detected": True,
                "similar_cases": len(history),
                "recommended_adjustments": {
                    "skills_weight": 0.05,  # Increase skills weight by 5%
                    "experience_threshold": 0.02  # Slight adjustment to experience requirements
                }
            },
            "prompt_update": "Minor adjustment to skills weighting based on repeated feedback",
            "alignment_explanation": "Feedback indicates skills are more critical than initially weighted"
        }
    
    def _update_hiring_criteria(self, job_id: str, update: Dict[str, Any]):
        """Update hiring criteria based on feedback analysis"""
        # In production, this would update the database
        # For now, log the update
        logger.info(f"Updating hiring criteria for job {job_id}: {update}")
    
    def _detect_outcome_discrepancy(self, job_id: str) -> bool:
        """Detect score-vs-outcome discrepancies"""
        # Simple discrepancy detection logic
        # In production, this would use more sophisticated statistical analysis
        
        candidates_for_job = [c for c in self.candidate_profiles.values() if c.job_id == job_id]
        
        if len(candidates_for_job) < 3:
            return False  # Need more data for discrepancy detection
        
        # Check if high-scoring candidates are being rejected
        high_scoring_rejected = [
            c for c in candidates_for_job 
            if c.fit_score and c.fit_score >= 8.0 and c.client_decision == DecisionType.REJECT
        ]
        
        # Check if low-scoring candidates are being accepted
        low_scoring_accepted = [
            c for c in candidates_for_job 
            if c.fit_score and c.fit_score <= 6.0 and c.client_decision == DecisionType.ACCEPT
        ]
        
        # Trigger calibration if there are discrepancies
        return len(high_scoring_rejected) > 0 or len(low_scoring_accepted) > 0
    
    def _calibrate_scores_llm(self, job_id: str) -> Dict[str, Any]:
        """Calibrate scores using LLM for outcome discrepancies"""
        # In production, this would call GPT-4/Claude API with the calibration prompt
        # For now, return a structured calibration
        
        candidates_for_job = [c for c in self.candidate_profiles.values() if c.job_id == job_id]
        
        return {
            "calibration": {
                "trigger_reason": "Score-vs-outcome discrepancy detected",
                "candidate_outcomes": [
                    {
                        "candidate_id": c.candidate_id,
                        "fit_score": c.fit_score,
                        "decision": c.client_decision,
                        "feedback": c.client_feedback
                    }
                    for c in candidates_for_job if c.client_decision
                ],
                "recommended_adjustments": {
                    "skills_weight": -0.03,  # Decrease skills weight by 3%
                    "experience_weight": 0.02,  # Increase experience weight by 2%
                    "culture_fit_weight": 0.01  # Slight increase to culture fit
                },
                "explanation": "Adjusting weights to better align scores with client decisions"
            }
        }
    
    def _check_quarterly_update_needed(self, job_id: str) -> bool:
        """Check if quarterly update is needed"""
        if job_id in self.job_postings:
            last_update = self.job_postings[job_id].last_quarterly_update
            if last_update:
                last_update_date = datetime.fromisoformat(last_update)
                return datetime.utcnow() - last_update_date > timedelta(days=90)
        
        return True  # Default to needing update
    
    def _log_platform_data(self, job_id: str, candidate_id: str, decision: DecisionType, reason: Optional[str]):
        """Log platform data as per specification schema"""
        if candidate_id in self.candidate_profiles:
            candidate = self.candidate_profiles[candidate_id]
            
            data_log = PlatformDataLog(
                job_id=job_id,
                candidate_id=candidate_id,
                fit_score=candidate.fit_score or 0.0,
                client_decision=decision,
                feedback_reason=reason,
                evaluation_summary=candidate.evaluation_summary,
                criteria_version="1.0"  # In production, get actual version
            )
            
            log_id = str(uuid.uuid4())
            self.platform_data_logs[log_id] = data_log
    
    # === Public API Methods ===
    
    def get_job_posting(self, job_id: str) -> Optional[JobPosting]:
        """Get job posting by ID"""
        return self.job_postings.get(job_id)
    
    def get_candidate_profile(self, candidate_id: str) -> Optional[CandidateProfile]:
        """Get candidate profile by ID"""
        return self.candidate_profiles.get(candidate_id)
    
    def get_client_feedback(self, feedback_id: str) -> Optional[ClientFeedback]:
        """Get client feedback by ID"""
        return self.client_feedback.get(feedback_id)
    
    def get_platform_workflows(self, job_id: Optional[str] = None) -> List[PlatformWorkflow]:
        """Get platform workflows, optionally filtered by job ID"""
        if job_id:
            return [w for w in self.platform_workflows.values() if w.job_id == job_id]
        return list(self.platform_workflows.values())
    
    def get_operational_metrics(self, metric_type: Optional[str] = None) -> List[OperationalMetrics]:
        """Get operational metrics, optionally filtered by type"""
        if metric_type:
            return [m for m in self.operational_metrics.values() if m.metric_type == metric_type]
        return list(self.operational_metrics.values())
    
    def get_platform_data_logs(self, job_id: Optional[str] = None) -> List[PlatformDataLog]:
        """Get platform data logs, optionally filtered by job ID"""
        if job_id:
            return [l for l in self.platform_data_logs.values() if l.job_id == job_id]
        return list(self.platform_data_logs.values())
    
    def get_guard_rail_settings(self) -> Dict[str, GuardRailSettings]:
        """Get all guard-rail settings"""
        return self.guard_rail_settings.copy()
    
    def update_guard_rail_setting(self, setting_name: str, new_value: Any, override_by: Optional[str] = None):
        """Update a guard-rail setting"""
        for setting in self.guard_rail_settings.values():
            if setting.setting_name == setting_name:
                setting.setting_value = new_value
                setting.updated_at = datetime.utcnow().isoformat()
                if override_by:
                    setting.override_by = override_by
                    setting.override_timestamp = datetime.utcnow().isoformat()
                break 