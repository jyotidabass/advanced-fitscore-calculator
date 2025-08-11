#!/usr/bin/env python3
"""
Platform Integration Models for Comprehensive AI-Powered Recruitment Workflow
Database models, data structures, and platform integration components
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import json

class DecisionType(str, Enum):
    """Client decision types"""
    ACCEPT = "accept"
    REJECT = "reject"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"

class FeedbackCategory(str, Enum):
    """Feedback categories for analysis"""
    SKILLS = "skills"
    EXPERIENCE = "experience"
    CULTURE_FIT = "culture_fit"
    LOCATION = "location"
    SALARY = "salary"
    TIMING = "timing"
    OTHER = "other"

class CriteriaType(str, Enum):
    """Types of hiring criteria"""
    CORE = "core"
    ADAPTIVE = "adaptive"
    COMBINED = "combined"

@dataclass
class JobPosting:
    """Job posting model with platform integration"""
    job_id: str
    company_id: str
    title: str
    description: str
    requirements: List[str]
    location: str
    role_type: str  # remote, hybrid, onsite
    salary_range: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "active"
    recruiter_id: Optional[str] = None
    
    # Platform integration fields
    core_criteria: Optional[Dict[str, Any]] = None
    adaptive_criteria: Optional[Dict[str, Any]] = None
    combined_criteria: Optional[Dict[str, Any]] = None
    last_quarterly_update: Optional[str] = None
    market_trends_analyzed: bool = False

@dataclass
class CandidateProfile:
    """Candidate profile model with platform integration"""
    candidate_id: str
    job_id: str
    name: str
    email: str
    resume_text: str
    resume_file_path: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    experience_years: Optional[int] = None
    current_company: Optional[str] = None
    current_title: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    fit_score: Optional[float] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    evaluation_timestamp: Optional[str] = None
    shared_with_client: bool = False
    shared_timestamp: Optional[str] = None
    client_decision: Optional[DecisionType] = None
    client_feedback: Optional[str] = None
    client_decision_timestamp: Optional[str] = None

@dataclass
class ClientFeedback:
    """Client feedback logging model"""
    feedback_id: str
    candidate_id: str
    job_id: str
    company_id: str
    decision: DecisionType
    internal_team_member: str
    feedback_reason: Optional[str] = None
    feedback_category: Optional[FeedbackCategory] = None
    logged_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    analyzed: bool = False
    analysis_timestamp: Optional[str] = None
    pattern_matched: bool = False
    similar_feedback_count: int = 0
    incremental_adjustment_applied: bool = False

@dataclass
class HiringCriteria:
    """Hiring criteria model with versioning"""
    criteria_id: str
    job_id: str
    criteria_type: CriteriaType
    version: str
    criteria_content: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    source: str = "ai_generated"  # ai_generated, manual_override, feedback_adjusted
    feedback_patterns: List[str] = field(default_factory=list)
    market_trends_included: bool = False
    quarterly_update: bool = False

@dataclass
class ScoreCalibration:
    """Score calibration model for outcome discrepancies"""
    calibration_id: str
    job_id: str
    trigger_reason: str
    candidate_outcomes: List[Dict[str, Any]]
    original_weights: Dict[str, float]
    adjusted_weights: Dict[str, float]
    weight_changes: Dict[str, float]
    calibration_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    discrepancy_detected: bool = False
    calibration_applied: bool = False
    human_reviewed: bool = False
    reviewed_by: Optional[str] = None
    review_timestamp: Optional[str] = None

@dataclass
class MarketTrends:
    """Market trends analysis model"""
    trend_id: str
    job_id: str
    analysis_date: str
    market_signals: List[str]
    emerging_skills: List[str]
    declining_skills: List[str]
    industry_shifts: List[str]
    salary_trends: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    quarterly_analysis: bool = False
    applied_to_criteria: bool = False
    confidence_score: float = 0.0

@dataclass
class PlatformWorkflow:
    """Platform workflow tracking model"""
    workflow_id: str
    job_id: str
    workflow_type: str  # job_creation, candidate_evaluation, feedback_analysis, score_calibration
    status: str  # pending, in_progress, completed, failed
    trigger_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_timestamp: Optional[str] = None
    
    # Platform integration fields
    ai_actions_triggered: List[str] = field(default_factory=list)
    database_operations: List[str] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OperationalMetrics:
    """Operational metrics for monitoring"""
    metric_id: str
    metric_type: str  # performance, accuracy, feedback_analysis, score_calibration
    metric_name: str
    metric_value: Union[float, int, str]
    metric_unit: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    job_id: Optional[str] = None
    company_id: Optional[str] = None
    period: str = "daily"  # daily, weekly, monthly, quarterly
    trend_direction: Optional[str] = None  # increasing, decreasing, stable

# Platform Data-Logging Schema (as per specification)
@dataclass
class PlatformDataLog:
    """Platform data logging schema for feedback tracking"""
    job_id: str
    candidate_id: str
    fit_score: float
    client_decision: DecisionType
    feedback_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Additional platform fields
    evaluation_summary: Optional[Dict[str, Any]] = None
    criteria_version: Optional[str] = None
    score_calibration_applied: bool = False
    feedback_analyzed: bool = False

# AI Prompt Templates (as per specification)
@dataclass
class AIPromptTemplate:
    """AI prompt template for platform integration"""
    template_id: str
    template_type: str  # client_feedback_analysis, fit_score_calibration, market_trends
    template_content: str
    variables: List[str]
    safeguards: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    version: str = "1.0"
    active: bool = True
    usage_count: int = 0
    success_rate: float = 0.0

# Guard-rail Settings (as per specification)
@dataclass
class GuardRailSettings:
    """Guard-rail settings for platform safety"""
    setting_id: str
    setting_name: str
    setting_value: Union[int, float, str, bool]
    setting_type: str  # threshold, percentage, boolean, string
    description: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Platform integration fields
    active: bool = True
    override_allowed: bool = False
    override_by: Optional[str] = None
    override_timestamp: Optional[str] = None

# Default guard-rail settings
DEFAULT_GUARD_RAILS = {
    "pattern_validation_threshold": 2,  # ≥2-3 similar feedback cases
    "max_weight_change_per_cycle": 0.10,  # 5-10% max weight change
    "human_oversight_frequency": "monthly",  # Monthly manual review
    "quarterly_market_update": True,  # Quarterly adaptive criteria update
    "real_time_calibration": True,  # Real-time score calibration
    "feedback_analysis_delay": 0,  # Immediate analysis (0 seconds)
    "max_calibration_frequency": "daily"  # Max once per day per job
}

# Platform workflow statuses
WORKFLOW_STATUSES = {
    "PENDING": "pending",
    "IN_PROGRESS": "in_progress", 
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "HUMAN_REVIEW_REQUIRED": "human_review_required"
}

# AI action types
AI_ACTION_TYPES = {
    "GENERATE_CORE_CRITERIA": "generate_core_criteria",
    "GENERATE_ADAPTIVE_CRITERIA": "generate_adaptive_criteria",
    "EVALUATE_CANDIDATE": "evaluate_candidate",
    "ANALYZE_FEEDBACK": "analyze_feedback",
    "CALIBRATE_SCORES": "calibrate_scores",
    "ANALYZE_MARKET_TRENDS": "analyze_market_trends",
    "UPDATE_CRITERIA_WEIGHTS": "update_criteria_weights"
}

# Database operation types
DB_OPERATION_TYPES = {
    "STORE_COMBINED_CRITERIA": "store_combined_criteria",
    "STORE_FIT_SCORE": "store_fit_score",
    "LOG_FEEDBACK": "log_feedback",
    "UPDATE_HIRING_CRITERIA": "update_hiring_criteria",
    "FETCH_JOB_DESCRIPTION": "fetch_job_description",
    "FETCH_CANDIDATE_DATA": "fetch_candidate_data",
    "FETCH_HIRING_CRITERIA": "fetch_hiring_criteria",
    "FETCH_SIMILAR_FEEDBACK": "fetch_similar_feedback"
} 