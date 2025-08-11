#!/usr/bin/env python3
"""
AI Submission Engine with LangGraph-style Workflow
Implements the complete candidate submission pipeline with reinforcement learning
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid
from enum import Enum

from platform_models import (
    JobPosting, CandidateProfile, ClientFeedback, DecisionType,
    PlatformWorkflow, OperationalMetrics
)

from reinforcement_feedback_agent import ReinforcementFeedbackAgent

logger = logging.getLogger(__name__)

class SubmissionStatus(str, Enum):
    """Submission status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUBMITTED_TO_CLIENT = "submitted_to_client"
    CLIENT_REVIEWED = "client_reviewed"
    HIRED = "hired"
    REJECTED_BY_CLIENT = "rejected_by_client"

class WorkflowNode(str, Enum):
    """Workflow node types"""
    EVALUATE = "evaluate"
    SUBMIT = "submit"
    REJECT = "reject"
    CLIENT_REVIEW = "client_review"
    FEEDBACK_LEARNING = "feedback_learning"

@dataclass
class SubmissionWorkflow:
    """Submission workflow with LangGraph-style nodes"""
    workflow_id: str
    candidate_id: str
    job_id: str
    recruiter_id: str
    current_node: WorkflowNode
    status: SubmissionStatus
    fit_score: float
    submission_notes: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Workflow tracking
    node_history: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    feedback_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CandidateMatch:
    """Candidate match with scoring and metadata"""
    candidate_id: str
    job_id: str
    similarity_score: float  # Vector similarity (0-1)
    fit_score: float  # AI Fit Score (1-10)
    combined_score: float  # Weighted combination
    metadata: Dict[str, Any]
    match_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class AISubmissionEngine:
    """
    AI Submission Engine with LangGraph-style Workflow
    
    Features:
    - Vector-based candidate matching
    - AI Fit Score evaluation
    - Reinforcement learning feedback loop
    - Automated submission workflow
    - Recruiter approval system
    """
    
    def __init__(self, reinforcement_agent: Optional[ReinforcementFeedbackAgent] = None):
        """Initialize the AI submission engine"""
        self.reinforcement_agent = reinforcement_agent or ReinforcementFeedbackAgent()
        self.submission_workflows: Dict[str, SubmissionWorkflow] = {}
        self.candidate_matches: Dict[str, CandidateMatch] = {}
        
        # Workflow configuration
        self.workflow_config = {
            "min_fit_score": 9.0,  # Minimum score for submission
            "similarity_threshold": 0.8,  # Minimum vector similarity
            "auto_approval_threshold": 9.5,  # Auto-approve high scores
            "max_candidates_per_job": 10,  # Top N candidates to show
            "learning_enabled": True
        }
        
        logger.info("AI Submission Engine initialized successfully")
    
    def process_candidate_submission(
        self, 
        candidate_id: str, 
        job_id: str, 
        recruiter_id: str,
        submission_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process candidate submission through the complete workflow
        
        This implements the LangGraph-style workflow:
        1. Evaluate candidate against job
        2. Apply Fit Score evaluation
        3. Route to appropriate workflow node
        4. Handle recruiter approval
        5. Submit to client if approved
        """
        try:
            # Create submission workflow
            workflow_id = str(uuid.uuid4())
            
            # Get candidate and job data
            candidate = self._get_candidate_data(candidate_id)
            job = self._get_job_data(job_id)
            
            if not candidate or not job:
                return {"success": False, "error": "Candidate or job not found"}
            
            # Calculate similarity and Fit Score
            similarity_score = self._calculate_similarity(candidate, job)
            fit_score = self._calculate_fit_score(candidate, job)
            combined_score = self._calculate_combined_score(similarity_score, fit_score)
            
            # Determine initial workflow node
            current_node = self._determine_workflow_node(fit_score, similarity_score)
            
            # Create submission workflow
            workflow = SubmissionWorkflow(
                workflow_id=workflow_id,
                candidate_id=candidate_id,
                job_id=job_id,
                recruiter_id=recruiter_id,
                current_node=current_node,
                status=SubmissionStatus.PENDING,
                fit_score=fit_score,
                submission_notes=submission_notes
            )
            
            # Record initial evaluation
            workflow.node_history.append({
                "node": current_node,
                "timestamp": datetime.utcnow().isoformat(),
                "fit_score": fit_score,
                "similarity_score": similarity_score,
                "combined_score": combined_score
            })
            
            self.submission_workflows[workflow_id] = workflow
            
            # Store candidate match
            match_id = f"{candidate_id}_{job_id}"
            candidate_match = CandidateMatch(
                candidate_id=candidate_id,
                job_id=job_id,
                similarity_score=similarity_score,
                fit_score=fit_score,
                combined_score=combined_score,
                metadata={
                    "recruiter_id": recruiter_id,
                    "submission_notes": submission_notes,
                    "workflow_id": workflow_id
                }
            )
            self.candidate_matches[match_id] = candidate_match
            
            # Execute workflow node
            result = self._execute_workflow_node(workflow, current_node)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "current_node": current_node,
                "fit_score": fit_score,
                "similarity_score": similarity_score,
                "combined_score": combined_score,
                "status": workflow.status,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing candidate submission: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_workflow_node(self, workflow: SubmissionWorkflow, node: WorkflowNode) -> Dict[str, Any]:
        """Execute a specific workflow node"""
        if node == WorkflowNode.EVALUATE:
            return self._evaluate_candidate(workflow)
        elif node == WorkflowNode.SUBMIT:
            return self._submit_candidate(workflow)
        elif node == WorkflowNode.REJECT:
            return self._reject_candidate(workflow)
        elif node == WorkflowNode.CLIENT_REVIEW:
            return self._client_review(workflow)
        elif node == WorkflowNode.FEEDBACK_LEARNING:
            return self._feedback_learning(workflow)
        else:
            return {"error": f"Unknown workflow node: {node}"}
    
    def _evaluate_candidate(self, workflow: SubmissionWorkflow) -> Dict[str, Any]:
        """Evaluate candidate and determine next action"""
        if workflow.fit_score >= self.workflow_config["auto_approval_threshold"]:
            # Auto-approve high-scoring candidates
            workflow.current_node = WorkflowNode.SUBMIT
            workflow.status = SubmissionStatus.APPROVED
            workflow.node_history.append({
                "node": "evaluate",
                "action": "auto_approve",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": f"High fit score: {workflow.fit_score}"
            })
            return {"action": "auto_approve", "next_node": "submit"}
        
        elif workflow.fit_score >= self.workflow_config["min_fit_score"]:
            # Manual approval required
            workflow.current_node = WorkflowNode.SUBMIT
            workflow.status = SubmissionStatus.PENDING
            workflow.node_history.append({
                "node": "evaluate",
                "action": "manual_approval_required",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": f"Fit score {workflow.fit_score} requires manual approval"
            })
            return {"action": "manual_approval_required", "next_node": "submit"}
        
        else:
            # Reject low-scoring candidates
            workflow.current_node = WorkflowNode.REJECT
            workflow.status = SubmissionStatus.REJECTED
            workflow.node_history.append({
                "node": "evaluate",
                "action": "reject",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": f"Low fit score: {workflow.fit_score}"
            })
            return {"action": "reject", "next_node": "reject"}
    
    def _submit_candidate(self, workflow: SubmissionWorkflow) -> Dict[str, Any]:
        """Submit candidate to client"""
        if workflow.status == SubmissionStatus.PENDING:
            # Still waiting for manual approval
            return {"action": "waiting_for_approval", "status": "pending"}
        
        # Submit to client
        workflow.status = SubmissionStatus.SUBMITTED_TO_CLIENT
        workflow.current_node = WorkflowNode.CLIENT_REVIEW
        workflow.node_history.append({
            "node": "submit",
            "action": "submitted_to_client",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"action": "submitted_to_client", "next_node": "client_review"}
    
    def _reject_candidate(self, workflow: SubmissionWorkflow) -> Dict[str, Any]:
        """Reject candidate and apply learning"""
        workflow.status = SubmissionStatus.REJECTED
        workflow.node_history.append({
            "node": "reject",
            "action": "candidate_rejected",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": f"Fit score {workflow.fit_score} below threshold"
        })
        
        # Apply reinforcement learning if enabled
        if self.workflow_config["learning_enabled"]:
            self._apply_rejection_learning(workflow)
        
        return {"action": "candidate_rejected", "status": "rejected"}
    
    def _client_review(self, workflow: SubmissionWorkflow) -> Dict[str, Any]:
        """Handle client review process"""
        # This would typically be triggered by external client feedback
        # For now, we'll simulate the process
        workflow.node_history.append({
            "node": "client_review",
            "action": "awaiting_client_feedback",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"action": "awaiting_client_feedback", "status": "client_review"}
    
    def _feedback_learning(self, workflow: SubmissionWorkflow) -> Dict[str, Any]:
        """Apply feedback-based learning"""
        if not self.workflow_config["learning_enabled"]:
            return {"action": "learning_disabled"}
        
        # Apply reinforcement learning based on outcome
        outcome = workflow.feedback_metadata.get("outcome")
        if outcome:
            result = self.reinforcement_agent.process_feedback_outcome(
                candidate_id=workflow.candidate_id,
                job_id=workflow.job_id,
                outcome=outcome,
                feedback_metadata=workflow.feedback_metadata,
                job_family=workflow.feedback_metadata.get("job_family", "software_engineer")
            )
            
            workflow.node_history.append({
                "node": "feedback_learning",
                "action": "learning_applied",
                "timestamp": datetime.utcnow().isoformat(),
                "result": result
            })
            
            return {"action": "learning_applied", "result": result}
        
        return {"action": "no_outcome_data"}
    
    def approve_submission(self, workflow_id: str, recruiter_id: str, notes: Optional[str] = None) -> Dict[str, Any]:
        """Approve candidate submission by recruiter"""
        if workflow_id not in self.submission_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.submission_workflows[workflow_id]
        
        if workflow.recruiter_id != recruiter_id:
            return {"success": False, "error": "Unauthorized recruiter"}
        
        if workflow.status != SubmissionStatus.PENDING:
            return {"success": False, "error": "Workflow not in pending status"}
        
        # Approve submission
        workflow.status = SubmissionStatus.APPROVED
        workflow.submission_notes = notes
        workflow.updated_at = datetime.utcnow().isoformat()
        
        workflow.node_history.append({
            "node": "submit",
            "action": "manual_approval",
            "timestamp": datetime.utcnow().isoformat(),
            "recruiter_id": recruiter_id,
            "notes": notes
        })
        
        # Execute next workflow node
        result = self._execute_workflow_node(workflow, WorkflowNode.SUBMIT)
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": workflow.status,
            "result": result
        }
    
    def reject_submission(self, workflow_id: str, recruiter_id: str, reason: str) -> Dict[str, Any]:
        """Reject candidate submission by recruiter"""
        if workflow_id not in self.submission_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.submission_workflows[workflow_id]
        
        if workflow.recruiter_id != recruiter_id:
            return {"success": False, "error": "Unauthorized recruiter"}
        
        if workflow.status != SubmissionStatus.PENDING:
            return {"success": False, "error": "Workflow not in pending status"}
        
        # Reject submission
        workflow.status = SubmissionStatus.REJECTED
        workflow.updated_at = datetime.utcnow().isoformat()
        
        workflow.node_history.append({
            "node": "submit",
            "action": "manual_rejection",
            "timestamp": datetime.utcnow().isoformat(),
            "recruiter_id": recruiter_id,
            "reason": reason
        })
        
        # Execute rejection workflow
        result = self._execute_workflow_node(workflow, WorkflowNode.REJECT)
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": workflow.status,
            "result": result
        }
    
    def record_client_feedback(
        self, 
        workflow_id: str, 
        outcome: str, 
        feedback_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Record client feedback and trigger learning"""
        if workflow_id not in self.submission_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow = self.submission_workflows[workflow_id]
        
        # Update workflow with client feedback
        workflow.feedback_metadata = feedback_metadata
        workflow.updated_at = datetime.utcnow().isoformat()
        
        if outcome == "hired":
            workflow.status = SubmissionStatus.HIRED
        elif outcome == "rejected":
            workflow.status = SubmissionStatus.REJECTED_BY_CLIENT
        else:
            workflow.status = SubmissionStatus.CLIENT_REVIEWED
        
        workflow.node_history.append({
            "node": "client_review",
            "action": "client_feedback_received",
            "timestamp": datetime.utcnow().isoformat(),
            "outcome": outcome,
            "feedback_metadata": feedback_metadata
        })
        
        # Move to feedback learning node
        workflow.current_node = WorkflowNode.FEEDBACK_LEARNING
        result = self._execute_workflow_node(workflow, WorkflowNode.FEEDBACK_LEARNING)
        
        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": workflow.status,
            "result": result
        }
    
    def get_top_candidates_for_job(self, job_id: str, limit: Optional[int] = None) -> List[CandidateMatch]:
        """Get top candidates for a specific job"""
        limit = limit or self.workflow_config["max_candidates_per_job"]
        
        job_candidates = [
            match for match in self.candidate_matches.values() 
            if match.job_id == job_id
        ]
        
        # Sort by combined score (descending)
        sorted_candidates = sorted(
            job_candidates, 
            key=lambda x: x.combined_score, 
            reverse=True
        )
        
        return sorted_candidates[:limit]
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status and details"""
        if workflow_id not in self.submission_workflows:
            return None
        
        workflow = self.submission_workflows[workflow_id]
        
        return {
            "workflow_id": workflow.workflow_id,
            "candidate_id": workflow.candidate_id,
            "job_id": workflow.job_id,
            "recruiter_id": workflow.recruiter_id,
            "current_node": workflow.current_node,
            "status": workflow.status,
            "fit_score": workflow.fit_score,
            "submission_notes": workflow.submission_notes,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
            "node_history": workflow.node_history,
            "decisions": workflow.decisions
        }
    
    # Helper methods (simulated for now)
    def _get_candidate_data(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Get candidate data (simulated)"""
        # In production, this would fetch from database
        return {"id": candidate_id, "name": "Test Candidate", "skills": ["Python", "AI"]}
    
    def _get_job_data(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data (simulated)"""
        # In production, this would fetch from database
        return {"id": job_id, "title": "AI Engineer", "requirements": ["Python", "AI"]}
    
    def _calculate_similarity(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> float:
        """Calculate vector similarity between candidate and job (simulated)"""
        # In production, this would use pgvector or similar
        import random
        return random.uniform(0.7, 0.95)
    
    def _calculate_fit_score(self, candidate: Dict[str, Any], job: Dict[str, Any]) -> float:
        """Calculate AI Fit Score (simulated)"""
        # In production, this would use GPT-4/Claude
        import random
        return random.uniform(7.0, 9.8)
    
    def _calculate_combined_score(self, similarity: float, fit_score: float) -> float:
        """Calculate combined score from similarity and Fit Score"""
        # Weighted combination: 30% similarity + 70% Fit Score
        return (0.3 * similarity) + (0.7 * (fit_score / 10.0))
    
    def _determine_workflow_node(self, fit_score: float, similarity: float) -> WorkflowNode:
        """Determine the initial workflow node based on scores"""
        if fit_score >= self.workflow_config["min_fit_score"]:
            return WorkflowNode.EVALUATE
        else:
            return WorkflowNode.REJECT
    
    def _apply_rejection_learning(self, workflow: SubmissionWorkflow):
        """Apply learning from rejection"""
        if self.workflow_config["learning_enabled"]:
            # Record rejection for learning
            self.reinforcement_agent.process_feedback_outcome(
                candidate_id=workflow.candidate_id,
                job_id=workflow.job_id,
                outcome="rejected",
                feedback_metadata={
                    "fit_score": workflow.fit_score,
                    "reason": "Below threshold",
                    "workflow_id": workflow.workflow_id
                }
            ) 