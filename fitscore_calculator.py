import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import openai
import os
from dotenv import load_dotenv
import hashlib
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (optional)
try:
    load_dotenv()
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")

@dataclass
class FeedbackEntry:
    """Data class for storing feedback entries"""
    feedback_id: str
    job_id: str
    company_id: str
    candidate_id: str
    feedback_type: str  # "positive", "negative", "neutral"
    feedback_text: str
    feedback_category: str  # "skills", "experience", "culture_fit", "location", "other"
    feedback_score: float  # -1.0 to 1.0
    timestamp: str
    processed: bool = False

@dataclass
class PromptVersion:
    """Data class for storing prompt versions"""
    version_id: str
    prompt_type: str  # "global", "local"
    job_id: Optional[str] = None
    company_id: Optional[str] = None
    base_prompt_version: Optional[str] = None
    prompt_content: Dict[str, Any] = None
    version_tag: str = ""
    feedback_pattern: Optional[str] = None
    adjusted_fields: List[str] = None
    location_enforced: bool = False
    created_at: str = ""
    updated_at: str = ""

@dataclass
class FitScoreResult:
    """Data class to hold fitscore calculation results"""
    total_score: float
    education_score: float
    career_trajectory_score: float
    company_relevance_score: float
    tenure_stability_score: float
    most_important_skills_score: float
    bonus_signals_score: float
    red_flags_penalty: float
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: str
    prompt_version: Optional[str] = None
    feedback_applied: bool = False

class FeedbackEngine:
    """
    Adaptive Feedback Engine for updating prompts based on feedback patterns
    """
    
    def __init__(self):
        self.feedback_store: Dict[str, FeedbackEntry] = {}
        self.prompt_versions: Dict[str, PromptVersion] = {}
        self.feedback_patterns: Dict[str, List[str]] = {}
        
        # Initialize global base prompt
        self._initialize_global_prompt()
    
    def _initialize_global_prompt(self):
        """Initialize the global base prompt"""
        global_prompt = PromptVersion(
            version_id="global_v1.0",
            prompt_type="global",
            prompt_content=self._get_base_global_prompt(),
            version_tag="v1.0-global",
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        self.prompt_versions["global_v1.0"] = global_prompt
    
    def _get_base_global_prompt(self) -> Dict[str, Any]:
        """Get the base global prompt template - Smart Hiring Criteria Generator (Global Prompt)"""
        return {
            "version": "v1.0-global",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "description": "Global Smart Hiring Criteria Base - Standard rubric for all jobs (used as seed)",
            
            "education": {
                "tier1_weight": 0.25,
                "tier2_weight": 0.20,
                "specialty_bonus": 0.15,
                "graduate_degree_bonus": 0.10,
                "scoring_rules": {
                    "tier1_schools": "MIT, Stanford, Harvard, Berkeley, CMU, Caltech, Princeton, Yale, Columbia, UPenn, Cornell, University of Chicago, Northwestern, Johns Hopkins, Brown",
                    "tier2_schools": "UCLA, UCSD, USC, Michigan, Wisconsin, Washington, North Carolina, Virginia, NYU, Boston University, Rice, Vanderbilt, Emory, Georgetown, Notre Dame, Duke, Dartmouth, William & Mary, Boston College",
                    "specialty_programs": "CS Leaders, Engineering Leaders, Business MBA Leaders, Medical Leaders, Law Leaders"
                }
            },
            
            "career_trajectory": {
                "leadership_weight": 0.30,
                "progression_weight": 0.25,
                "scope_weight": 0.20,
                "ownership_weight": 0.15,
                "complexity_weight": 0.10,
                "scoring_rules": {
                    "leadership_indicators": ["manager", "director", "lead", "head", "chief", "vp", "cto", "ceo", "principal", "staff"],
                    "scope_indicators": ["team", "budget", "revenue", "strategy", "architect", "cross-functional", "stakeholder"],
                    "ownership_indicators": ["owned", "led", "managed", "responsible for", "delivered", "launched", "improved"],
                    "complexity_indicators": ["scalable", "distributed", "microservices", "architecture", "system design", "technical leadership"]
                }
            },
            
            "company_relevance": {
                "elite_company_bonus": 0.25,
                "industry_match_weight": 0.20,
                "role_type_weight": 0.15,
                "company_size_weight": 0.10,
                "scoring_rules": {
                    "elite_companies": {
                        "tech_startup": ["Stripe", "Scale AI", "Databricks", "Canva", "Airbnb", "Uber", "Palantir", "Snowflake", "MongoDB", "Twilio"],
                        "tech_enterprise": ["Google", "Meta", "Apple", "Amazon", "Microsoft", "Netflix", "Salesforce", "Oracle", "SAP", "Adobe"],
                        "big4_accounting": ["KPMG", "Deloitte", "EY", "PwC"],
                        "elite_law_firms": ["Cravath", "Skadden", "Sullivan & Cromwell", "Wachtell", "Davis Polk", "Simpson Thacher"],
                        "elite_healthcare": ["Mayo Clinic", "Cleveland Clinic", "Johns Hopkins", "Massachusetts General", "UCSF Medical Center"]
                    },
                    "company_types": ["startup", "enterprise", "law_firm", "accounting", "healthcare", "consulting", "financial", "academic", "government", "non_profit"]
                }
            },
            
            "tenure_stability": {
                "long_tenure_bonus": 0.30,
                "elite_company_tenure_bonus": 0.20,
                "internship_excellence_bonus": 0.15,
                "stability_pattern_weight": 0.35,
                "scoring_rules": {
                    "tenure_thresholds": {
                        "elite": 3.0,  # 3+ years average
                        "strong": 2.5,  # 2.5-3 years average
                        "good": 2.0,    # 2-2.5 years average
                        "reasonable": 1.5,  # 1.5-2 years average
                        "some_hopping": 1.0,  # 1-1.5 years average
                        "frequent_changes": 0.5,  # 0.5-1 year average
                        "very_short": 0.5  # less than 0.5 years average
                    }
                }
            },
            
            "most_important_skills": {
                "critical_skills_weight": 0.40,
                "technical_depth_weight": 0.25,
                "domain_expertise_weight": 0.20,
                "adaptability_weight": 0.15,
                "scoring_rules": {
                    "technical_skills": [
                        "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "php", "ruby", "swift", "kotlin", "scala",
                        "react", "vue", "angular", "node.js", "express", "django", "flask", "spring", "laravel", "asp.net",
                        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "github", "terraform", "ansible",
                        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "dynamodb", "cassandra",
                        "machine learning", "ai", "data science", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark", "hadoop"
                    ],
                    "enterprise_skills": [
                        "b2b", "enterprise", "saas", "enterprise software", "enterprise sales", "enterprise architecture",
                        "compliance", "security", "governance", "enterprise integration", "legacy systems", "mainframe",
                        "enterprise resource planning", "customer relationship management", "business intelligence", "data warehousing"
                    ],
                    "startup_skills": [
                        "mvp", "prototyping", "rapid development", "agile", "scrum", "lean methodology", "growth hacking",
                        "user acquisition", "product-market fit", "scaling", "fundraising", "venture capital", "angel investors"
                    ]
                }
            },
            
            "bonus_signals": {
                "exceptional_signals": 5.0,
                "strong_signals": 3.0,
                "some_signals": 1.0,
                "scoring_rules": {
                    "exceptional": ["patent", "published", "forbes", "founder", "board", "olympic", "military", "ted talk", "book", "award", "media coverage"],
                    "strong": ["open source", "speaking", "teaching", "certification", "hackathon", "leadership", "volunteer", "side project"],
                    "some": ["portfolio", "community", "course", "competition", "language"]
                }
            },
            
            "red_flags": {
                "major_penalty": -15.0,
                "moderate_penalty": -10.0,
                "minor_penalty": -5.0,
                "location_penalty": -15.0,
                "scoring_rules": {
                    "major": ["falsified", "plagiarized", "criminal", "ethical violation", "diploma mill", "unaccredited"],
                    "moderate": ["job hopping", "employment gap", "no progression", "short tenure", "concerning pattern"],
                    "minor": ["overqualified", "location mismatch", "missing certification"],
                    "location_constraints": {
                        "hybrid_onsite_radius": 20,  # km
                        "remote_allowed": True,
                        "strict_enforcement": True
                    }
                }
            },
            
            "scoring_scale": {
                "base_scale": 100,  # Score on 100-point scale
                "final_scale": 10,   # Then scale to 10
                "submittable_threshold": 8.2,  # Submittable (≥8.2) / Not Submittable (<8.2)
                "conversion_factor": 0.1  # 100 * 0.1 = 10
            }
        }
    
    def add_feedback(self, feedback: FeedbackEntry) -> bool:
        """Add new feedback entry"""
        try:
            self.feedback_store[feedback.feedback_id] = feedback
            logger.info(f"Added feedback: {feedback.feedback_id} for job: {feedback.job_id}")
            
            # Check if we should trigger prompt updates
            self._check_feedback_patterns(feedback.job_id, feedback.company_id)
            return True
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
    
    def _check_feedback_patterns(self, job_id: str, company_id: str):
        """Check if feedback patterns warrant prompt updates"""
        # Check local feedback patterns (≥2 for same JD)
        local_feedback = self._get_local_feedback(job_id)
        if len(local_feedback) >= 2:
            self._update_local_prompt(job_id, local_feedback)
        
        # Check global feedback patterns (≥3 across JDs)
        global_feedback = self._get_global_feedback()
        if len(global_feedback) >= 3:
            self._update_global_prompt(global_feedback)
    
    def _get_local_feedback(self, job_id: str) -> List[FeedbackEntry]:
        """Get feedback for a specific job"""
        return [f for f in self.feedback_store.values() if f.job_id == job_id]
    
    def _get_global_feedback(self) -> List[FeedbackEntry]:
        """Get feedback across all jobs"""
        return list(self.feedback_store.values())
    
    def _update_local_prompt(self, job_id: str, feedback_list: List[FeedbackEntry]):
        """Update local prompt based on job-specific feedback"""
        try:
            # Analyze feedback patterns
            patterns = self._analyze_feedback_patterns(feedback_list)
            
            if not patterns:
                return
            
            # Get current local prompt or create new one
            current_local = self._get_current_local_prompt(job_id)
            
            # Create updated prompt
            updated_prompt = self._apply_feedback_to_prompt(
                current_local.prompt_content if current_local else self._get_base_global_prompt(),
                patterns,
                "local"
            )
            
            # Create new version
            new_version = PromptVersion(
                version_id=f"local_{job_id}_{uuid.uuid4().hex[:8]}",
                prompt_type="local",
                job_id=job_id,
                base_prompt_version=current_local.version_tag if current_local else "global_v1.0",
                prompt_content=updated_prompt,
                version_tag=f"JD_{job_id}_v{self._get_next_local_version(job_id)}-local",
                feedback_pattern=patterns.get("summary", ""),
                adjusted_fields=patterns.get("adjusted_fields", []),
                location_enforced=patterns.get("location_enforced", False),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            
            self.prompt_versions[new_version.version_id] = new_version
            logger.info(f"Updated local prompt for job {job_id}: {new_version.version_tag}")
            
        except Exception as e:
            logger.error(f"Error updating local prompt: {e}")
    
    def _update_global_prompt(self, feedback_list: List[FeedbackEntry]):
        """Update global prompt based on cross-job feedback patterns"""
        try:
            # Analyze global feedback patterns
            patterns = self._analyze_feedback_patterns(feedback_list, is_global=True)
            
            if not patterns:
                return
            
            # Get current global prompt
            current_global = self._get_current_global_prompt()
            
            # Create updated prompt
            updated_prompt = self._apply_feedback_to_prompt(
                current_global.prompt_content,
                patterns,
                "global"
            )
            
            # Create new version
            new_version = PromptVersion(
                version_id=f"global_v{self._get_next_global_version()}",
                prompt_type="global",
                prompt_content=updated_prompt,
                version_tag=f"v{self._get_next_global_version()}-global",
                feedback_pattern=patterns.get("summary", ""),
                adjusted_fields=patterns.get("adjusted_fields", []),
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            
            self.prompt_versions[new_version.version_id] = new_version
            logger.info(f"Updated global prompt: {new_version.version_tag}")
            
        except Exception as e:
            logger.error(f"Error updating global prompt: {e}")
    
    def _analyze_feedback_patterns(self, feedback_list: List[FeedbackEntry], is_global: bool = False) -> Dict[str, Any]:
        """Analyze feedback to identify patterns"""
        if not feedback_list:
            return {}
        
        # Group feedback by category
        category_feedback = {}
        for feedback in feedback_list:
            category = feedback.feedback_category
            if category not in category_feedback:
                category_feedback[category] = []
            category_feedback[category].append(feedback)
        
        # Identify patterns
        patterns = {
            "summary": "",
            "adjusted_fields": [],
            "location_enforced": False
        }
        
        # Check for location-related feedback
        location_feedback = category_feedback.get("location", [])
        if len(location_feedback) >= 2:
            patterns["location_enforced"] = True
            patterns["adjusted_fields"].append("location_constraints")
            patterns["summary"] += "Location constraints enforced. "
        
        # Check for skills-related feedback
        skills_feedback = category_feedback.get("skills", [])
        if len(skills_feedback) >= 2:
            patterns["adjusted_fields"].append("most_important_skills")
            patterns["summary"] += "Skills requirements updated. "
        
        # Check for experience-related feedback
        experience_feedback = category_feedback.get("experience", [])
        if len(experience_feedback) >= 2:
            patterns["adjusted_fields"].append("career_trajectory")
            patterns["summary"] += "Experience requirements updated. "
        
        # Check for company relevance feedback
        company_feedback = category_feedback.get("culture_fit", [])
        if len(company_feedback) >= 2:
            patterns["adjusted_fields"].append("company_relevance")
            patterns["summary"] += "Company relevance criteria updated. "
        
        # Enhanced feedback analysis for B2B/enterprise patterns
        if is_global and len(feedback_list) >= 3:
            b2b_patterns = self._detect_b2b_enterprise_patterns(feedback_list)
            if b2b_patterns:
                patterns["b2b_enterprise_patterns"] = b2b_patterns
                patterns["adjusted_fields"].append("enterprise_experience_requirements")
                patterns["summary"] += "B2B/enterprise experience requirements updated. "
        
        return patterns
    
    def _detect_b2b_enterprise_patterns(self, feedback_list: List[FeedbackEntry]) -> Dict[str, Any]:
        """Detect B2B/enterprise experience patterns from feedback"""
        b2b_keywords = ["b2b", "enterprise", "saas", "enterprise software", "enterprise sales", "enterprise architecture"]
        enterprise_keywords = ["enterprise", "large company", "fortune 500", "corporate", "business to business"]
        consumer_keywords = ["consumer", "b2c", "retail", "e-commerce", "mobile app", "social media"]
        
        b2b_feedback = []
        enterprise_feedback = []
        consumer_feedback = []
        
        for feedback in feedback_list:
            feedback_lower = feedback.feedback_text.lower()
            
            # Check for B2B/enterprise feedback
            if any(keyword in feedback_lower for keyword in b2b_keywords + enterprise_keywords):
                if "no experience" in feedback_lower or "lacks" in feedback_lower or "missing" in feedback_lower:
                    b2b_feedback.append({
                        "type": "missing_b2b_experience",
                        "feedback": feedback.feedback_text,
                        "score": feedback.feedback_score
                    })
            
            # Check for consumer-only experience feedback
            if any(keyword in feedback_lower for keyword in consumer_keywords):
                if "only worked" in feedback_lower or "no enterprise" in feedback_lower:
                    consumer_feedback.append({
                        "type": "consumer_only_experience",
                        "feedback": feedback.feedback_text,
                        "score": feedback.feedback_score
                    })
        
        patterns = {}
        if len(b2b_feedback) >= 2:
            patterns["missing_b2b_experience"] = b2b_feedback
            patterns["recommendation"] = "Increase enterprise experience requirements"
        
        if len(consumer_feedback) >= 2:
            patterns["consumer_only_experience"] = consumer_feedback
            patterns["recommendation"] = "Add penalty for consumer-only experience in enterprise roles"
        
        return patterns
    
    def _apply_feedback_to_prompt(self, current_prompt: Dict[str, Any], patterns: Dict[str, Any], prompt_type: str) -> Dict[str, Any]:
        """
        Local Feedback-Based Prompt Adjuster
        
        GOAL: Adjust Smart Hiring Criteria for a specific JD or company based on repeated client feedback
        
        Only apply adjustment if ≥2 matching feedbacks
        Update Company Relevance, Most Important Skills, or Red Flags
        Do NOT affect other jobs or companies
        """
        updated_prompt = current_prompt.copy()
        
        # Apply location constraints if needed
        if patterns.get("location_enforced", False):
            if "location_constraints" not in updated_prompt:
                updated_prompt["location_constraints"] = {
                    "hybrid_onsite_radius": 20,  # km
                    "remote_allowed": False,
                    "strict_enforcement": True,
                    "location_penalty": -15.0,
                    "enforcement_reason": "Location constraints enforced based on feedback"
                }
        
        # Apply skills updates
        if "most_important_skills" in patterns.get("adjusted_fields", []):
            if "skills_adjustments" not in updated_prompt:
                updated_prompt["skills_adjustments"] = {
                    "feedback_based": True,
                    "updated_at": datetime.utcnow().isoformat(),
                    "adjustment_reason": patterns.get("summary", ""),
                    "adjustment_type": "skills_requirements_updated"
                }
        
        # Apply experience updates
        if "career_trajectory" in patterns.get("adjusted_fields", []):
            if "experience_adjustments" not in updated_prompt:
                updated_prompt["experience_adjustments"] = {
                    "feedback_based": True,
                    "updated_at": datetime.utcnow().isoformat(),
                    "adjustment_reason": patterns.get("summary", ""),
                    "adjustment_type": "experience_requirements_updated"
                }
        
        # Apply company relevance updates
        if "company_relevance" in patterns.get("adjusted_fields", []):
            if "company_adjustments" not in updated_prompt:
                updated_prompt["company_adjustments"] = {
                    "feedback_based": True,
                    "updated_at": datetime.utcnow().isoformat(),
                    "adjustment_reason": patterns.get("summary", ""),
                    "adjustment_type": "company_relevance_criteria_updated"
                }
        
        # Apply B2B/enterprise experience requirements
        if "enterprise_experience_requirements" in patterns.get("adjusted_fields", []):
            b2b_patterns = patterns.get("b2b_enterprise_patterns", {})
            if b2b_patterns:
                updated_prompt["enterprise_experience_adjustments"] = {
                    "feedback_based": True,
                    "updated_at": datetime.utcnow().isoformat(),
                    "adjustment_reason": "B2B/enterprise experience requirements updated based on feedback",
                    "b2b_patterns": b2b_patterns,
                    "adjustments": {
                        "enterprise_experience_bonus": 2.0,
                        "consumer_experience_penalty": -5.0,
                        "enterprise_skills_weight_increase": 0.05
                    }
                }
        
        # Add feedback summary and version metadata
        updated_prompt["feedback_metadata"] = {
            "last_updated": datetime.utcnow().isoformat(),
            "feedback_patterns": patterns,
            "adjustment_summary": patterns.get("summary", ""),
            "adjusted_fields": patterns.get("adjusted_fields", []),
            "prompt_type": prompt_type
        }
        
        return updated_prompt
    
    def update_global_prompt_with_industry_trends(self, feedback_trends: List[Dict[str, Any]], current_version: str) -> Dict[str, Any]:
        """
        Global Prompt Updater (Multi-JD Feedback)
        
        GOAL: Adjust the Global Base Prompt based on industry-wide hiring feedback
        
        Args:
            feedback_trends: List of feedback trends across multiple JDs
            current_version: Current prompt version (e.g., v2.3-global)
            
        Returns:
            Updated Global Prompt with new version and changelog
        """
        # Get current global prompt
        current_global = self._get_base_global_prompt()
        updated_prompt = current_global.copy()
        
        # Extract version number
        try:
            version_parts = current_version.split("v")[1].split("-")[0]
            major, minor = version_parts.split(".")
            new_minor = int(minor) + 1
            new_version = f"v{major}.{new_minor}-global"
        except:
            new_version = "v2.0-global"
        
        # Analyze industry trends
        industry_updates = {}
        
        for trend in feedback_trends:
            trend_text = trend.get("feedback_text", "").lower()
            trend_category = trend.get("feedback_category", "")
            
            # Check for LLM/AI trends
            if any(keyword in trend_text for keyword in ["llm", "large language model", "gpt", "claude", "ai model"]):
                if "llm_trends" not in industry_updates:
                    industry_updates["llm_trends"] = []
                industry_updates["llm_trends"].append({
                    "trend": "LLM deployment experience",
                    "feedback": trend.get("feedback_text", ""),
                    "date": datetime.utcnow().isoformat()
                })
            
            # Check for RAG/Vector DB trends
            if any(keyword in trend_text for keyword in ["rag", "retrieval augmented generation", "vector", "vector database", "embeddings"]):
                if "rag_trends" not in industry_updates:
                    industry_updates["rag_trends"] = []
                industry_updates["rag_trends"].append({
                    "trend": "RAG or vector DB background",
                    "feedback": trend.get("feedback_text", ""),
                    "date": datetime.utcnow().isoformat()
                })
            
            # Check for other emerging technology trends
            if any(keyword in trend_text for keyword in ["blockchain", "web3", "metaverse", "quantum", "edge computing"]):
                if "emerging_tech_trends" not in industry_updates:
                    industry_updates["emerging_tech_trends"] = []
                industry_updates["emerging_tech_trends"].append({
                    "trend": "Emerging technology experience",
                    "feedback": trend.get("feedback_text", ""),
                    "date": datetime.utcnow().isoformat()
                })
        
        # Apply industry updates to Most Important Skills section
        if industry_updates:
            if "industry_updates" not in updated_prompt:
                updated_prompt["industry_updates"] = {}
            
            updated_prompt["industry_updates"] = industry_updates
            
            # Update skills requirements based on trends
            if "llm_trends" in industry_updates:
                if "llm_skills" not in updated_prompt["most_important_skills"]["scoring_rules"]:
                    updated_prompt["most_important_skills"]["scoring_rules"]["llm_skills"] = [
                        "llm deployment", "prompt engineering", "fine-tuning", "ai model integration",
                        "openai api", "anthropic claude", "llama", "gpt-4", "claude-3"
                    ]
            
            if "rag_trends" in industry_updates:
                if "rag_skills" not in updated_prompt["most_important_skills"]["scoring_rules"]:
                    updated_prompt["most_important_skills"]["scoring_rules"]["rag_skills"] = [
                        "vector databases", "embeddings", "retrieval augmented generation", "semantic search",
                        "pinecone", "weaviate", "chroma", "qdrant", "faiss"
                    ]
        
        # Update version and metadata
        updated_prompt.update({
            "version": new_version,
            "updated_at": datetime.utcnow().isoformat(),
            "previous_version": current_version,
            "changelog": {
                "version": new_version,
                "updated_at": datetime.utcnow().isoformat(),
                "changes": industry_updates,
                "update_reason": "Industry-wide hiring feedback trends",
                "affected_sections": ["most_important_skills", "industry_updates"]
            }
        })
        
        return updated_prompt
    
    def _get_current_local_prompt(self, job_id: str) -> Optional[PromptVersion]:
        """Get current local prompt for a job"""
        for version in self.prompt_versions.values():
            if version.prompt_type == "local" and version.job_id == job_id:
                return version
        return None
    
    def _get_current_global_prompt(self) -> PromptVersion:
        """Get current global prompt"""
        for version in self.prompt_versions.values():
            if version.prompt_type == "global":
                return version
        return self.prompt_versions["global_v1.0"]
    
    def _get_next_local_version(self, job_id: str) -> int:
        """Get next local version number for a job"""
        versions = [v for v in self.prompt_versions.values() 
                   if v.prompt_type == "local" and v.job_id == job_id]
        return len(versions) + 1
    
    def _get_next_global_version(self) -> str:
        """Get next global version number"""
        versions = [v for v in self.prompt_versions.values() 
                   if v.prompt_type == "global"]
        return f"{len(versions) + 1}.0"
    
    def get_prompt_for_job(self, job_id: str, company_id: str) -> PromptVersion:
        """Get the appropriate prompt for a job (local if exists, otherwise global)"""
        local_prompt = self._get_current_local_prompt(job_id)
        if local_prompt:
            return local_prompt
        return self._get_current_global_prompt()
    
    def generate_local_prompt(self, job_description: str, collateral: str = None, role_type: str = None, job_location: str = None) -> Dict[str, Any]:
        """
        Smart Hiring Criteria Generator (Local Prompt)
        
        GOAL: Generate Smart Hiring Criteria to evaluate candidates for a specific role.
        
        Args:
            job_description: Job description text
            collateral: Optional additional information (hiring manager notes, success profile)
            role_type: Remote / Hybrid / On-Site
            job_location: e.g., San Francisco, CA
            
        Returns:
            Local prompt with tailored scoring rules and location constraints
        """
        # Get base global prompt
        base_prompt = self._get_base_global_prompt()
        
        # Create local prompt based on global
        local_prompt = base_prompt.copy()
        
        # Add job-specific information
        local_prompt.update({
            "job_description": job_description,
            "collateral": collateral,
            "role_type": role_type or self._detect_role_type(job_description),
            "job_location": job_location,
            "generated_at": datetime.utcnow().isoformat(),
            "prompt_type": "local"
        })
        
        # Apply location constraints based on role type
        if role_type and role_type.lower() in ["hybrid", "onsite", "on-site"]:
            local_prompt["location_constraints"] = {
                "hybrid_onsite_radius": 20,  # km
                "remote_allowed": False,
                "strict_enforcement": True,
                "location_penalty": -15.0,
                "enforcement_reason": f"Role is {role_type}, applying strict 20km radius enforcement"
            }
        elif role_type and role_type.lower() == "remote":
            local_prompt["location_constraints"] = {
                "hybrid_onsite_radius": None,
                "remote_allowed": True,
                "strict_enforcement": False,
                "location_penalty": 0.0,
                "enforcement_reason": "Remote role, no location restrictions"
            }
        
        # Tailor scoring rules based on collateral
        if collateral:
            local_prompt = self._tailor_scoring_rules(local_prompt, collateral)
        
        return local_prompt
    
    def _detect_role_type(self, job_description: str) -> str:
        """Detect role type from job description"""
        jd_lower = job_description.lower()
        
        if "remote" in jd_lower:
            return "remote"
        elif "hybrid" in jd_lower:
            return "hybrid"
        elif "onsite" in jd_lower or "on-site" in jd_lower:
            return "onsite"
        else:
            return "unknown"
    
    def _tailor_scoring_rules(self, prompt: Dict[str, Any], collateral: str) -> Dict[str, Any]:
        """Tailor scoring rules based on collateral information"""
        collateral_lower = collateral.lower()
        
        # Adjust weights based on collateral content
        if "startup" in collateral_lower or "early-stage" in collateral_lower:
            prompt["startup_adjustments"] = {
                "skills_weight_increase": 0.05,
                "experience_weight_increase": 0.03,
                "stability_weight_decrease": 0.05,
                "reasoning": "Startup environment values skills and experience over stability"
            }
        elif "enterprise" in collateral_lower or "large company" in collateral_lower:
            prompt["enterprise_adjustments"] = {
                "education_weight_increase": 0.05,
                "stability_weight_increase": 0.05,
                "skills_weight_decrease": 0.05,
                "reasoning": "Enterprise environment values education and stability"
            }
        elif "b2b" in collateral_lower or "enterprise sales" in collateral_lower:
            prompt["b2b_adjustments"] = {
                "company_relevance_weight_increase": 0.05,
                "enterprise_experience_bonus": 2.0,
                "consumer_experience_penalty": -5.0,
                "reasoning": "B2B roles require enterprise experience"
            }
        
        return prompt
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback and prompt versions"""
        return {
            "total_feedback": len(self.feedback_store),
            "total_prompts": len(self.prompt_versions),
            "global_prompts": len([v for v in self.prompt_versions.values() if v.prompt_type == "global"]),
            "local_prompts": len([v for v in self.prompt_versions.values() if v.prompt_type == "local"]),
            "feedback_by_category": self._get_feedback_by_category(),
            "recent_prompts": self._get_recent_prompts()
        }
    
    def _get_feedback_by_category(self) -> Dict[str, int]:
        """Get feedback count by category"""
        categories = {}
        for feedback in self.feedback_store.values():
            category = feedback.feedback_category
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _get_recent_prompts(self) -> List[Dict[str, Any]]:
        """Get recent prompt updates"""
        recent = []
        for version in sorted(self.prompt_versions.values(), 
                            key=lambda x: x.updated_at, reverse=True)[:5]:
            recent.append({
                "version_id": version.version_id,
                "type": version.prompt_type,
                "version_tag": version.version_tag,
                "updated_at": version.updated_at,
                "feedback_pattern": version.feedback_pattern
            })
        return recent

class FitScoreCalculator:
    """
    Comprehensive FitScore Calculator implementing the detailed evaluation system
    with dynamic weights, company/role-specific adjustments, and feedback-based learning.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the FitScore calculator"""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not provided. Some features may be limited.")
        
        # Initialize feedback engine
        self.feedback_engine = FeedbackEngine()
        
        # Default weights (can be adjusted per company/role)
        self.default_weights = {
            "education": 0.20,
            "career_trajectory": 0.20,
            "company_relevance": 0.15,
            "tenure_stability": 0.15,
            "most_important_skills": 0.20,
            "bonus_signals": 0.05,
            "red_flags": -0.15  # Penalty
        }
        
        # Comprehensive elite schools database with specialty recognition
        self.tier1_schools = {
            "US_TOP15": [
                "MIT", "Stanford", "Harvard", "Berkeley", "CMU", "Caltech", 
                "Princeton", "Yale", "Columbia", "UPenn", "Cornell", 
                "University of Chicago", "Northwestern", "Johns Hopkins", "Brown"
            ],
            "ENGINEERING_CS_ELITE": [
                "University of Waterloo", "Georgia Tech", "UIUC", "UT Austin", 
                "UW Seattle", "Purdue", "Virginia Tech"
            ],
            "INTERNATIONAL_ELITE": [
                "Oxford", "Cambridge", "ETH Zurich", "University of Toronto", 
                "IIT", "Tsinghua", "Peking University", "National University of Singapore",
                "University of Melbourne", "KAIST", "Technion"
            ],
            "BUSINESS_ELITE": [
                "Wharton", "Harvard Business", "Stanford GSB", "Kellogg", "Booth", "Sloan"
            ],
            "MEDICAL_ELITE": [
                "Harvard Medical", "Johns Hopkins", "UCSF", "Mayo Clinic", 
                "Stanford Medical", "Penn Medical"
            ],
            "LAW_ELITE": [
                "Harvard Law", "Yale Law", "Stanford Law", "Columbia Law", 
                "NYU Law", "Chicago Law"
            ]
        }
        
        self.tier2_schools = {
            "STRONG_UNIVERSITIES": [
                "UCLA", "UCSD", "USC", "Michigan", "Wisconsin", "Washington",
                "North Carolina", "Virginia", "NYU", "Boston University",
                "Rice", "Vanderbilt", "Emory", "Georgetown", "Notre Dame",
                "Duke", "Dartmouth", "William & Mary", "Boston College"
            ],
            "ENGINEERING_STRONG": [
                "Texas A&M", "Penn State", "Ohio State", "Arizona State",
                "UC Irvine", "UC Davis", "Rutgers", "Maryland",
                "UC Santa Barbara", "UC Santa Cruz", "Northeastern",
                "RIT", "WPI", "RPI", "Stevens Tech", "Colorado School of Mines"
            ],
            "INTERNATIONAL_STRONG": [
                "McGill", "UBC", "Queen's", "London School of Economics", 
                "Imperial College", "University of Sydney", "ANU",
                "University of Hong Kong", "HKUST", "Sciences Po", "Bocconi"
            ]
        }
        
        # Specialty program recognition
        self.specialty_programs = {
            "CS_LEADERS": ["MIT", "Stanford", "CMU", "Berkeley", "Waterloo", "UIUC", "Georgia Tech", "UT Austin", "UW Seattle"],
            "ENGINEERING_LEADERS": ["MIT", "Stanford", "Berkeley", "Caltech", "CMU", "Georgia Tech", "Purdue", "Michigan", "UIUC"],
            "BUSINESS_MBA_LEADERS": ["Wharton", "Harvard", "Stanford", "Kellogg", "Booth", "Sloan", "Columbia", "Tuck"],
            "MEDICAL_LEADERS": ["Harvard Medical", "Johns Hopkins", "UCSF", "Mayo Clinic", "Stanford Medical"],
            "LAW_LEADERS": ["Harvard Law", "Yale Law", "Stanford Law", "Columbia Law", "NYU Law", "Chicago Law"]
        }
        
        # Elite companies database
        self.elite_companies = {
            "TECH_STARTUP_ELITE": [
                "Stripe", "Scale AI", "Databricks", "Canva", "Airbnb", "Uber",
                "Palantir", "Snowflake", "MongoDB", "Twilio"
            ],
            "TECH_ENTERPRISE_ELITE": [
                "Google", "Meta", "Apple", "Amazon", "Microsoft", "Netflix",
                "Salesforce", "Oracle", "SAP", "Adobe"
            ],
            "BIG4_ACCOUNTING": [
                "KPMG", "Deloitte", "EY", "PwC"
            ],
            "ELITE_LAW_FIRMS": [
                "Cravath", "Skadden", "Sullivan & Cromwell", "Wachtell",
                "Davis Polk", "Simpson Thacher"
            ],
            "ELITE_HEALTHCARE": [
                "Mayo Clinic", "Cleveland Clinic", "Johns Hopkins",
                "Massachusetts General", "UCSF Medical Center"
            ]
        }

    def calculate_fitscore(
        self, 
        resume_text: str, 
        job_description: str, 
        collateral: Optional[str] = None,
        company_weights: Optional[Dict[str, float]] = None,
        use_gpt4: bool = True,
        job_id: Optional[str] = None,
        company_id: Optional[str] = None
    ) -> FitScoreResult:
        """
        Calculate comprehensive fitscore using resume and job description with optional GPT-4 enhancement
        and feedback-based prompt adaptation
        
        Args:
            resume_text: Candidate's resume text
            job_description: Job description text
            collateral: Optional additional information (company culture, specific requirements, etc.)
            company_weights: Optional custom weights for company/role
            use_gpt4: Whether to use GPT-4 for enhanced analysis (default: True)
            job_id: Optional job ID for feedback-based prompt selection
            company_id: Optional company ID for feedback-based prompt selection
            
        Returns:
            FitScoreResult with detailed scores and analysis
        """
        logger.info("Starting fitscore calculation with GPT-4 enhancement: %s", use_gpt4)
        
        # Get appropriate prompt based on feedback history
        prompt_version = None
        if job_id and company_id:
            prompt_version = self.feedback_engine.get_prompt_for_job(job_id, company_id)
            logger.info(f"Using prompt version: {prompt_version.version_tag}")
        
        # Step 1: Context Detection with GPT-4
        if use_gpt4 and self.client:
            context = self._detect_context_with_gpt4(job_description, resume_text)
            logger.info(f"Detected context: {context}")
        else:
            context = self._detect_context_fallback(job_description)
        
        # Step 2: Generate Smart Criteria with GPT-4
        if use_gpt4 and self.client:
            smart_criteria = self._generate_smart_criteria_with_gpt4(job_description, context)
            logger.info(f"Generated smart criteria: {smart_criteria}")
        else:
            smart_criteria = self._generate_smart_criteria_fallback(job_description, context)
        
        # Step 3: Dynamic Weight Adjustment with GPT-4 (considering feedback-based prompts)
        if use_gpt4 and self.client:
            weights = self._adjust_weights_dynamically_with_gpt4(context, smart_criteria)
            # Remove reasoning from weights dict for calculation
            weights.pop('reasoning', None)
        else:
            weights = company_weights or self.default_weights
            if collateral:
                weights = self._adjust_weights_for_collateral(weights, collateral)
            
            # Apply feedback-based prompt adjustments if available
            if prompt_version and prompt_version.prompt_content:
                weights = self._apply_prompt_adjustments(weights, prompt_version.prompt_content)
        
        # Step 4: Enhanced Skills Analysis with GPT-4
        if use_gpt4 and self.client:
            skills_analysis = self._extract_skills_with_gpt4(resume_text, job_description)
            logger.info(f"Enhanced skills analysis completed")
        else:
            skills_analysis = self._extract_skills_fallback(resume_text, job_description)
        
        # Step 5: Elite Evaluation against Smart Criteria with GPT-4
        if use_gpt4 and self.client:
            elite_evaluation = self._evaluate_against_smart_criteria_with_gpt4(resume_text, smart_criteria)
            logger.info(f"Elite evaluation completed")
        else:
            elite_evaluation = self._evaluate_against_smart_criteria_fallback(resume_text, smart_criteria)
        
        # Step 6: Traditional Component Evaluation (with GPT-4 enhancements where applicable)
        education_score, education_details = self._evaluate_education(resume_text, job_description)
        career_score, career_details = self._evaluate_career_trajectory(resume_text, job_description)
        company_score, company_details = self._evaluate_company_relevance(resume_text, job_description)
        tenure_score, tenure_details = self._evaluate_tenure_stability(resume_text, job_description)
        skills_score, skills_details = self._evaluate_most_important_skills(resume_text, job_description)
        bonus_score, bonus_details = self._evaluate_bonus_signals(resume_text, job_description)
        red_flags_penalty, red_flags_details = self._evaluate_red_flags(resume_text, job_description)
        
        # Apply location constraints if specified in prompt
        if prompt_version and prompt_version.location_enforced:
            red_flags_penalty, red_flags_details = self._apply_location_constraints(
                red_flags_penalty, red_flags_details, resume_text, job_description
            )
        
        # Step 7: Calculate weighted final score
        final_score = (
            education_score * weights["education"] +
            career_score * weights["career_trajectory"] +
            company_score * weights["company_relevance"] +
            tenure_score * weights["tenure_stability"] +
            skills_score * weights["most_important_skills"] +
            bonus_score * weights["bonus_signals"] +
            red_flags_penalty
        )
        
        # Step 8: Generate enhanced recommendations
        recommendations = self._generate_recommendations(
            final_score, education_score, career_score, company_score,
            tenure_score, skills_score, bonus_score, red_flags_penalty
        )
        
        # Step 9: Compile comprehensive results with GPT-4 insights
        details = {
            "education": education_details,
            "career_trajectory": career_details,
            "company_relevance": company_details,
            "tenure_stability": tenure_details,
            "most_important_skills": skills_details,
            "bonus_signals": bonus_details,
            "red_flags": red_flags_details,
            "weights_used": weights,
            "context_detection": context,
            "smart_criteria": smart_criteria,
            "skills_analysis": skills_analysis,
            "elite_evaluation": elite_evaluation,
            "gpt4_enhanced": use_gpt4 and self.client is not None,
            "prompt_version_used": prompt_version.version_tag if prompt_version else None
        }
        
        return FitScoreResult(
            total_score=final_score,
            education_score=education_score,
            career_trajectory_score=career_score,
            company_relevance_score=company_score,
            tenure_stability_score=tenure_score,
            most_important_skills_score=skills_score,
            bonus_signals_score=bonus_score,
            red_flags_penalty=red_flags_penalty,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat(),
            prompt_version=prompt_version.version_tag if prompt_version else None,
            feedback_applied=prompt_version is not None and prompt_version.prompt_type == "local"
        )

    def _evaluate_education(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Evaluate education based on tier system (20% weight)"""
        logger.info("Evaluating education")
        
        # Extract education information
        education_info = self._extract_education_info(resume_text)
        
        total_score = 0.0
        education_details = {
            "institutions": [],
            "total_score": 0.0,
            "tier_breakdown": {},
            "strengths": [],
            "concerns": []
        }
        
        for edu in education_info:
            institution_score = self._score_institution(edu["institution"], edu["degree_type"], edu["field"])
            total_score += institution_score
            
            education_details["institutions"].append({
                "institution": edu["institution"],
                "degree": edu["degree_type"],
                "field": edu["field"],
                "score": institution_score,
                "tier": self._get_institution_tier(edu["institution"])
            })
        
        # Calculate average score
        if education_info:
            avg_score = total_score / len(education_info)
        else:
            avg_score = 1.0  # Minimum score for no education
            education_details["concerns"].append("No education information found")
        
        # Apply graduate degree boost
        graduate_degrees = [edu for edu in education_info if "master" in edu["degree_type"].lower() or "phd" in edu["degree_type"].lower()]
        if graduate_degrees:
            avg_score = min(10.0, avg_score + 1.0)  # Boost for graduate degrees
            education_details["strengths"].append("Graduate degree(s) present")
        
        education_details["total_score"] = avg_score
        education_details["tier"] = self._get_institution_tier(education_info[0]["institution"]) if education_info else "No Education"
        
        return avg_score, education_details

    def _evaluate_career_trajectory(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Evaluate career trajectory and progression using detailed scoring (20% weight)"""
        logger.info("Evaluating career trajectory")
        
        # Extract work experience
        work_experience = self._extract_work_experience(resume_text)
        
        if not work_experience:
            return 1.0, {"error": "No work experience found", "score": 1.0}
        
        # Sort by date (most recent first)
        work_experience.sort(key=lambda x: x.get("end_date", "Present"), reverse=True)
        
        trajectory_details = {
            "positions": [],
            "progression_pattern": "",
            "leadership_roles": 0,
            "scope_increases": 0,
            "ownership_indicators": 0,
            "complexity_growth": 0,
            "score": 0.0,
            "progression_level": ""
        }
        
        # Analyze each position for progression indicators
        position_scores = []
        total_leadership = 0
        total_scope = 0
        total_ownership = 0
        total_complexity = 0
        
        for i, position in enumerate(work_experience):
            title = position["title"]
            description = position.get("description", "")
            
            # Base title score
            title_score = self._score_job_title(title)
            
            # Leadership indicators
            leadership_score = 0.0
            if any(word in title.lower() for word in ["manager", "director", "lead", "head", "chief", "vp", "cto", "ceo", "principal", "staff"]):
                leadership_score = 2.0
                total_leadership += 1
            
            # Scope and responsibility indicators
            scope_score = 0.0
            if any(word in description.lower() for word in ["team", "budget", "revenue", "strategy", "architect", "cross-functional", "stakeholder"]):
                scope_score = 1.5
                total_scope += 1
            
            # Ownership indicators
            ownership_score = 0.0
            if any(word in description.lower() for word in ["owned", "led", "managed", "responsible for", "delivered", "launched", "improved"]):
                ownership_score = 1.0
                total_ownership += 1
            
            # Complexity indicators
            complexity_score = 0.0
            if any(word in description.lower() for word in ["scalable", "distributed", "microservices", "architecture", "system design", "technical leadership"]):
                complexity_score = 1.0
                total_complexity += 1
            
            position_score = title_score + leadership_score + scope_score + ownership_score + complexity_score
            position_scores.append(position_score)
            
            trajectory_details["positions"].append({
                "title": title,
                "company": position["company"],
                "duration": position.get("duration", ""),
                "score": position_score,
                "leadership": leadership_score > 0,
                "scope": scope_score > 0,
                "ownership": ownership_score > 0,
                "complexity": complexity_score > 0
            })
        
        # Calculate progression metrics
        avg_position_score = sum(position_scores) / len(position_scores)
        
        # Analyze progression pattern
        if len(position_scores) >= 3:
            recent_scores = position_scores[:3]
            
            # Check for exceptional progression (9.5-10.0)
            if (recent_scores[0] >= 9.0 and recent_scores[1] >= 8.0 and 
                total_leadership >= 2 and total_ownership >= 2):
                progression_score = 9.5
                trajectory_details["progression_pattern"] = "Exceptional progression with leadership and ownership"
                trajectory_details["progression_level"] = "Exceptional (9.5-10.0)"
            
            # Check for clear upward progression (9.0-9.4)
            elif (recent_scores[0] > recent_scores[1] > recent_scores[2] and 
                  recent_scores[0] >= 7.5 and total_leadership >= 1):
                progression_score = 9.0
                trajectory_details["progression_pattern"] = "Clear upward progression with leadership"
                trajectory_details["progression_level"] = "Clear Upward (9.0-9.4)"
            
            # Check for strong progression (8.0-8.9)
            elif (recent_scores[0] > recent_scores[2] and 
                  recent_scores[0] >= 7.0 and total_scope >= 1):
                progression_score = 8.0
                trajectory_details["progression_pattern"] = "Strong progression with scope growth"
                trajectory_details["progression_level"] = "Strong (8.0-8.9)"
            
            # Check for good progression (7.0-7.9)
            elif recent_scores[0] >= 6.0 and total_ownership >= 1:
                progression_score = 7.0
                trajectory_details["progression_pattern"] = "Good progression with ownership"
                trajectory_details["progression_level"] = "Good (7.0-7.9)"
            
            # Check for steady progression (6.0-6.9)
            elif recent_scores[0] >= 5.0:
                progression_score = 6.0
                trajectory_details["progression_pattern"] = "Steady progression"
                trajectory_details["progression_level"] = "Steady (6.0-6.9)"
            
            # Limited progression (4.0-5.9)
            elif recent_scores[0] >= 4.0:
                progression_score = 4.0
                trajectory_details["progression_pattern"] = "Limited progression"
                trajectory_details["progression_level"] = "Limited (4.0-5.9)"
            
            # No progression (1.0-3.9)
            else:
                progression_score = 1.0
                trajectory_details["progression_pattern"] = "No clear progression"
                trajectory_details["progression_level"] = "No Progression (1.0-3.9)"
        
        else:
            # For candidates with fewer positions, base on current level
            if avg_position_score >= 8.0:
                progression_score = 8.0
            elif avg_position_score >= 6.0:
                progression_score = 6.0
            else:
                progression_score = 4.0
        
        # Update details
        trajectory_details["leadership_roles"] = total_leadership
        trajectory_details["scope_increases"] = total_scope
        trajectory_details["ownership_indicators"] = total_ownership
        trajectory_details["complexity_growth"] = total_complexity
        trajectory_details["score"] = progression_score
        
        return progression_score, trajectory_details

    def _evaluate_company_relevance(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Evaluate company relevance based on role type (15% weight)"""
        logger.info("Evaluating company relevance")
        
        work_experience = self._extract_work_experience(resume_text)
        role_type = self._detect_role_type(job_description)
        company_type = self._detect_company_type(job_description)
        
        company_details = {
            "role_type": role_type,
            "target_company_type": company_type,
            "companies": [],
            "relevance_score": 0.0
        }
        
        if not work_experience:
            return 1.0, {"error": "No work experience found", "score": 1.0}
        
        total_relevance = 0.0
        
        for position in work_experience:
            company_score = self._score_company_relevance(
                position["company"], 
                role_type, 
                company_type
            )
            total_relevance += company_score
            
            company_details["companies"].append({
                "company": position["company"],
                "role": position["title"],
                "relevance_score": company_score
            })
        
        avg_relevance = total_relevance / len(work_experience)
        company_details["relevance_score"] = avg_relevance
        
        return avg_relevance, company_details

    def _evaluate_tenure_stability(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Evaluate tenure and stability using detailed scoring rules (15% weight)"""
        logger.info("Evaluating tenure stability")
        
        work_experience = self._extract_work_experience(resume_text)
        
        if not work_experience:
            return 1.0, {"error": "No work experience found", "score": 1.0}
        
        tenure_details = {
            "positions": [],
            "average_tenure": 0.0,
            "tenure_pattern": "",
            "stability_score": 0.0,
            "excluded_positions": [],
            "internship_count": 0,
            "elite_company_tenure": 0.0,
            "tenure_level": ""
        }
        
        total_tenure = 0.0
        valid_positions = 0
        internship_count = 0
        elite_company_tenure = 0.0
        
        # Identify elite companies for tenure bonus
        elite_companies = []
        for category, companies in self.elite_companies.items():
            elite_companies.extend(companies)
        
        for position in work_experience:
            title = position["title"].lower()
            company = position["company"]
            
            # CRITICAL: Exclude internships, co-ops, and part-time student work
            if any(word in title for word in ["intern", "internship", "co-op", "coop", "part-time", "parttime"]):
                tenure_details["excluded_positions"].append({
                    "position": position["title"],
                    "company": position["company"],
                    "reason": "Internship/Co-op/Part-time (excluded from tenure calculation)"
                })
                internship_count += 1
                continue
            
            # Only count full-time positions post-graduation
            tenure_years = self._calculate_tenure_years(position.get("duration", ""))
            if tenure_years > 0:
                total_tenure += tenure_years
                valid_positions += 1
                
                # Check if it's an elite company for tenure bonus
                is_elite = any(elite_company.lower() in company.lower() for elite_company in elite_companies)
                if is_elite:
                    elite_company_tenure += tenure_years
                
                tenure_details["positions"].append({
                    "company": position["company"],
                    "title": position["title"],
                    "tenure_years": tenure_years,
                    "is_elite_company": is_elite
                })
        
        if valid_positions == 0:
            return 1.0, {"error": "No valid full-time positions found", "score": 1.0}
        
        avg_tenure = total_tenure / valid_positions
        tenure_details["average_tenure"] = avg_tenure
        tenure_details["internship_count"] = internship_count
        tenure_details["elite_company_tenure"] = elite_company_tenure
        
        # Apply detailed tenure scoring with decimals for precision
        if avg_tenure >= 3.0:
            stability_score = 9.5
            tenure_details["tenure_pattern"] = "Elite stability (3+ years average)"
            tenure_details["tenure_level"] = "Elite (9.5-10.0)"
        elif avg_tenure >= 2.5:
            stability_score = 8.5
            tenure_details["tenure_pattern"] = "Strong stability (2.5-3 years average)"
            tenure_details["tenure_level"] = "Strong (8.5-9.4)"
        elif avg_tenure >= 2.0:
            stability_score = 7.5
            tenure_details["tenure_pattern"] = "Good stability (2-2.5 years average)"
            tenure_details["tenure_level"] = "Good (7.5-8.4)"
        elif avg_tenure >= 1.5:
            stability_score = 6.5
            tenure_details["tenure_pattern"] = "Reasonable stability (1.5-2 years average)"
            tenure_details["tenure_level"] = "Reasonable (6.5-7.4)"
        elif avg_tenure >= 1.0:
            stability_score = 5.5
            tenure_details["tenure_pattern"] = "Some job hopping (1-1.5 years average)"
            tenure_details["tenure_level"] = "Some Hopping (5.5-6.4)"
        elif avg_tenure >= 0.5:
            stability_score = 4.0
            tenure_details["tenure_pattern"] = "Frequent job changes (0.5-1 year average)"
            tenure_details["tenure_level"] = "Frequent Changes (4.0-5.4)"
        else:
            stability_score = 1.0
            tenure_details["tenure_pattern"] = "Very short tenures (less than 0.5 years average)"
            tenure_details["tenure_level"] = "Very Short (1.0-3.9)"
        
        # Apply elite company tenure bonus
        if elite_company_tenure > 0:
            elite_bonus = min(0.5, elite_company_tenure * 0.1)  # Max 0.5 bonus
            stability_score = min(10.0, stability_score + elite_bonus)
            tenure_details["elite_tenure_bonus"] = elite_bonus
        
        # Apply internship excellence bonus (3-4 quality internships)
        if internship_count >= 3:
            internship_bonus = 0.3
            stability_score = min(10.0, stability_score + internship_bonus)
            tenure_details["internship_bonus"] = internship_bonus
        
        tenure_details["stability_score"] = stability_score
        return stability_score, tenure_details

    def _evaluate_most_important_skills(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Evaluate most important skills match (20% weight)"""
        logger.info("Evaluating skills match")
        
        # Extract required skills from job description
        required_skills = self._extract_required_skills(job_description)
        
        # Extract candidate skills from resume
        candidate_skills = self._extract_candidate_skills(resume_text)
        
        skills_details = {
            "required_skills": required_skills,
            "candidate_skills": candidate_skills,
            "matches": [],
            "missing": [],
            "match_percentage": 0.0,
            "score": 0.0
        }
        
        if not required_skills:
            return 5.0, {"error": "No required skills identified", "score": 5.0}
        
        matches = []
        missing = []
        
        for skill in required_skills:
            if self._skill_matches(skill, candidate_skills):
                matches.append(skill)
            else:
                missing.append(skill)
        
        match_percentage = len(matches) / len(required_skills) * 100
        
        # Score based on match percentage
        if match_percentage >= 90:
            skills_score = 9.0
        elif match_percentage >= 80:
            skills_score = 7.5
        elif match_percentage >= 70:
            skills_score = 6.0
        elif match_percentage >= 50:
            skills_score = 4.0
        else:
            skills_score = 1.0
        
        skills_details["matches"] = matches
        skills_details["missing"] = missing
        skills_details["match_percentage"] = match_percentage
        skills_details["score"] = skills_score
        
        return skills_score, skills_details

    def _evaluate_bonus_signals(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Evaluate bonus signals (5% weight)"""
        logger.info("Evaluating bonus signals")
        
        bonus_details = {
            "signals_found": [],
            "total_score": 0.0
        }
        
        bonus_score = 0.0
        
        # Check for exceptional signals (5 points)
        exceptional_signals = [
            "patent", "published", "forbes", "founder", "board", "olympic",
            "military", "ted talk", "book", "award", "media coverage"
        ]
        
        for signal in exceptional_signals:
            if signal in resume_text.lower():
                bonus_score += 5.0
                bonus_details["signals_found"].append(f"Exceptional: {signal}")
        
        # Check for strong signals (3-4 points)
        strong_signals = [
            "open source", "speaking", "teaching", "certification",
            "hackathon", "leadership", "volunteer", "side project"
        ]
        
        for signal in strong_signals:
            if signal in resume_text.lower():
                bonus_score += 3.0
                bonus_details["signals_found"].append(f"Strong: {signal}")
        
        # Check for some signals (1-2 points)
        some_signals = [
            "portfolio", "community", "course", "competition", "language"
        ]
        
        for signal in some_signals:
            if signal in resume_text.lower():
                bonus_score += 1.0
                bonus_details["signals_found"].append(f"Some: {signal}")
        
        # Cap bonus score at 5.0
        final_bonus_score = min(5.0, bonus_score)
        bonus_details["total_score"] = final_bonus_score
        
        return final_bonus_score, bonus_details

    def _evaluate_red_flags(self, resume_text: str, job_description: str) -> Tuple[float, Dict]:
        """Evaluate red flags (-15% penalty)"""
        logger.info("Evaluating red flags")
        
        red_flags_details = {
            "flags_found": [],
            "penalty": 0.0
        }
        
        penalty = 0.0
        
        # Check for major red flags (-15 points)
        major_flags = [
            "falsified", "plagiarized", "criminal", "ethical violation",
            "diploma mill", "unaccredited"
        ]
        
        for flag in major_flags:
            if flag in resume_text.lower():
                penalty -= 15.0
                red_flags_details["flags_found"].append(f"Major: {flag}")
        
        # Check for moderate red flags (-10 points)
        moderate_flags = [
            "job hopping", "employment gap", "no progression",
            "short tenure", "concerning pattern"
        ]
        
        for flag in moderate_flags:
            if flag in resume_text.lower():
                penalty -= 10.0
                red_flags_details["flags_found"].append(f"Moderate: {flag}")
        
        # Check for minor red flags (-5 points)
        minor_flags = [
            "overqualified", "location mismatch", "missing certification"
        ]
        
        for flag in minor_flags:
            if flag in resume_text.lower():
                penalty -= 5.0
                red_flags_details["flags_found"].append(f"Minor: {flag}")
        
        red_flags_details["penalty"] = penalty
        return penalty, red_flags_details

    def _generate_recommendations(
        self, 
        final_score: float,
        education_score: float,
        career_score: float,
        company_score: float,
        tenure_score: float,
        skills_score: float,
        bonus_score: float,
        red_flags_penalty: float
    ) -> List[str]:
        """Generate recommendations based on scores"""
        recommendations = []
        
        if final_score >= 8.2:
            recommendations.append("SUBMITTABLE CANDIDATE - Recommend to submit")
        else:
            recommendations.append("RECOMMENDED REJECT - Below elite hiring bar")
        
        if education_score < 6.0:
            recommendations.append("Education concerns - consider program strength and relevance")
        
        if career_score < 6.0:
            recommendations.append("Career trajectory concerns - limited progression visible")
        
        if company_score < 6.0:
            recommendations.append("Company relevance concerns - may not fit target environment")
        
        if tenure_score < 6.0:
            recommendations.append("Tenure stability concerns - frequent job changes")
        
        if skills_score < 6.0:
            recommendations.append("Skills gap - missing critical capabilities")
        
        if red_flags_penalty < -5.0:
            recommendations.append("Red flags detected - requires careful review")
        
        return recommendations

    # Helper methods for data extraction and scoring
    
    def _extract_education_info(self, resume_text: str) -> List[Dict]:
        """Extract education information from resume"""
        education_info = []
        
        # Simple regex patterns for education extraction
        education_patterns = [
            r"([A-Z][a-zA-Z\s&]+(?:University|College|Institute|School))",
            r"(Bachelor|Master|PhD|MBA|MS|BS|BA)\s+(?:of|in)?\s+([A-Za-z\s]+)",
        ]
        
        # Extract basic education info
        for pattern in education_patterns:
            matches = re.finditer(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                education_info.append({
                    "institution": match.group(1) if match.group(1) else "Unknown",
                    "degree_type": match.group(2) if len(match.groups()) > 1 else "Unknown",
                    "field": "General" if len(match.groups()) <= 1 else match.group(2)
                })
        
        return education_info

    def _extract_work_experience(self, resume_text: str) -> List[Dict]:
        """Extract work experience from resume with improved parsing"""
        work_experience = []
        
        # Enhanced patterns to extract job information
        # Look for patterns like "Senior Software Engineer\nGoogle Inc.\n2021-2024 (3 years)"
        lines = resume_text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for job titles
            if any(title in line.lower() for title in ['engineer', 'manager', 'director', 'analyst', 'developer', 'consultant', 'lead', 'senior', 'principal', 'staff']):
                title = line
                company = "Unknown Company"
                duration = "Unknown"
                description = ""
                
                # Look for company name in next few lines
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not any(skip in next_line.lower() for skip in ['experience', 'education', 'skills', 'bonus']):
                        # Check if it looks like a company name
                        if any(company_indicator in next_line.lower() for company_indicator in ['inc', 'corp', 'llc', 'ltd', 'company', 'google', 'microsoft', 'amazon', 'apple', 'meta', 'ibm', 'oracle']):
                            company = next_line
                            break
                
                # Look for duration in the same area
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if re.search(r'\d{4}-\d{4}|\(\d+\s+years?\)|\(\d+\s+months?\)', next_line):
                        duration = next_line
                        break
                
                # Collect description from subsequent lines
                desc_lines = []
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith('-') or next_line.startswith('•'):
                        desc_lines.append(next_line)
                    elif next_line and not any(skip in next_line.lower() for skip in ['experience', 'education', 'skills', 'bonus']):
                        break
                description = ' '.join(desc_lines)
                
                work_experience.append({
                    "title": title,
                    "company": company,
                    "duration": duration,
                    "description": description
                })
        
        return work_experience

    def _score_institution(self, institution: str, degree_type: str, field: str) -> float:
        """Score an educational institution based on comprehensive tier classification"""
        institution_lower = institution.lower()
        
        # Check Tier 1 schools with specialty recognition
        for category, schools in self.tier1_schools.items():
            for school in schools:
                if school.lower() in institution_lower:
                    base_score = 9.5
                    
                    # Apply specialty program bonuses
                    if field and self._is_specialty_match(field, category):
                        base_score = min(10.0, base_score + 0.5)
                    
                    # Graduate degree boost
                    if degree_type and ("master" in degree_type.lower() or "phd" in degree_type.lower()):
                        base_score = min(10.0, base_score + 0.3)
                    
                    return base_score
        
        # Check Tier 2 schools
        for category, schools in self.tier2_schools.items():
            for school in schools:
                if school.lower() in institution_lower:
                    base_score = 7.5
                    
                    # Apply specialty program bonuses
                    if field and self._is_specialty_match(field, category):
                        base_score = min(8.5, base_score + 0.5)
                    
                    # Graduate degree boost
                    if degree_type and ("master" in degree_type.lower() or "phd" in degree_type.lower()):
                        base_score = min(8.5, base_score + 0.3)
                    
                    return base_score
        
        # Check specialty programs for non-tier schools
        for specialty, schools in self.specialty_programs.items():
            for school in schools:
                if school.lower() in institution_lower and field and self._is_specialty_match(field, specialty):
                    return 8.0
        
        # Default scoring for other institutions
        if "university" in institution_lower or "college" in institution_lower:
            base_score = 5.0
            # Graduate degree boost for any institution
            if degree_type and ("master" in degree_type.lower() or "phd" in degree_type.lower()):
                base_score = min(6.5, base_score + 0.5)
            return base_score
        else:
            return 3.0
    
    def _is_specialty_match(self, field: str, category: str) -> bool:
        """Check if field of study matches specialty category"""
        field_lower = field.lower()
        
        if "CS" in category or "COMPUTER" in category:
            return any(term in field_lower for term in ["computer", "software", "cs", "computing"])
        elif "ENGINEERING" in category:
            return any(term in field_lower for term in ["engineering", "mechanical", "electrical", "civil", "chemical"])
        elif "BUSINESS" in category or "MBA" in category:
            return any(term in field_lower for term in ["business", "mba", "management", "finance", "economics"])
        elif "MEDICAL" in category:
            return any(term in field_lower for term in ["medical", "medicine", "health", "nursing", "pharmacy"])
        elif "LAW" in category:
            return any(term in field_lower for term in ["law", "legal", "juris", "jd"])
        
        return False

    def _get_institution_tier(self, institution: str) -> str:
        """Get institution tier classification"""
        institution_lower = institution.lower()
        
        # Check Tier 1 schools
        for category, schools in self.tier1_schools.items():
            for school in schools:
                if school.lower() in institution_lower:
                    return f"Tier 1 - {category.replace('_', ' ').title()}"
        
        # Check Tier 2 schools
        for category, schools in self.tier2_schools.items():
            for school in schools:
                if school.lower() in institution_lower:
                    return f"Tier 2 - {category.replace('_', ' ').title()}"
        
        # Check specialty programs
        for specialty, schools in self.specialty_programs.items():
            for school in schools:
                if school.lower() in institution_lower:
                    return f"Specialty - {specialty.replace('_', ' ').title()}"
        
        return "Tier 3"

    def _score_job_title(self, title: str) -> float:
        """Score job title based on seniority"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ["ceo", "cto", "cfo", "vp", "director"]):
            return 9.0
        elif any(word in title_lower for word in ["senior", "lead", "principal"]):
            return 7.0
        elif any(word in title_lower for word in ["manager", "supervisor"]):
            return 6.0
        elif any(word in title_lower for word in ["engineer", "analyst", "developer"]):
            return 5.0
        else:
            return 3.0

    def _detect_role_type(self, job_description: str) -> str:
        """Detect role type from job description"""
        jd_lower = job_description.lower()
        
        if any(word in jd_lower for word in ["software", "engineer", "developer", "programmer"]):
            return "technical"
        elif any(word in jd_lower for word in ["manager", "director", "lead"]):
            return "management"
        elif any(word in jd_lower for word in ["sales", "account", "business"]):
            return "sales"
        elif any(word in jd_lower for word in ["legal", "attorney", "law"]):
            return "legal"
        elif any(word in jd_lower for word in ["accounting", "cpa", "audit"]):
            return "accounting"
        elif any(word in jd_lower for word in ["healthcare", "medical", "nurse"]):
            return "healthcare"
        else:
            return "general"

    def _detect_company_type(self, job_description: str) -> str:
        """Detect company type from job description"""
        jd_lower = job_description.lower()
        
        if any(word in jd_lower for word in ["startup", "seed", "series", "early-stage"]):
            return "startup"
        elif any(word in jd_lower for word in ["enterprise", "fortune", "large company"]):
            return "enterprise"
        elif any(word in jd_lower for word in ["law firm", "llp", "amlaw"]):
            return "law_firm"
        elif any(word in jd_lower for word in ["accounting", "cpa"]):
            return "accounting"
        elif any(word in jd_lower for word in ["healthcare", "hospital"]):
            return "healthcare"
        else:
            return "general"

    def _score_company_relevance(self, company: str, role_type: str, company_type: str) -> float:
        """Score company relevance based on role and company type"""
        company_lower = company.lower()
        
        if role_type == "technical":
            if company_type == "startup":
                for elite_startup in self.elite_companies["TECH_STARTUP_ELITE"]:
                    if elite_startup.lower() in company_lower:
                        return 9.0
            elif company_type == "enterprise":
                for elite_enterprise in self.elite_companies["TECH_ENTERPRISE_ELITE"]:
                    if elite_enterprise.lower() in company_lower:
                        return 9.0
        
        elif role_type == "accounting":
            for big4 in self.elite_companies["BIG4_ACCOUNTING"]:
                if big4.lower() in company_lower:
                    return 9.0
        
        elif role_type == "legal":
            for elite_law in self.elite_companies["ELITE_LAW_FIRMS"]:
                if elite_law.lower() in company_lower:
                    return 9.0
        
        elif role_type == "healthcare":
            for elite_healthcare in self.elite_companies["ELITE_HEALTHCARE"]:
                if elite_healthcare.lower() in company_lower:
                    return 9.0
        
        # Default score
        return 5.0

    def _calculate_tenure_years(self, duration: str) -> float:
        """Calculate tenure in years from duration string with improved parsing"""
        if not duration or duration == "Unknown":
            return 0.0
        
        duration_lower = duration.lower()
        
        # Pattern 1: "2021-2024 (3 years)"
        years_match = re.search(r'\((\d+(?:\.\d+)?)\s+years?\)', duration_lower)
        if years_match:
            return float(years_match.group(1))
        
        # Pattern 2: "2021-2024" - calculate years
        year_range_match = re.search(r'(\d{4})-(\d{4})', duration)
        if year_range_match:
            start_year = int(year_range_match.group(1))
            end_year = int(year_range_match.group(2))
            return end_year - start_year
        
        # Pattern 3: "3 years" or "2.5 years"
        years_only_match = re.search(r'(\d+(?:\.\d+)?)\s+years?', duration_lower)
        if years_only_match:
            return float(years_only_match.group(1))
        
        # Pattern 4: "6 months" - convert to years
        months_match = re.search(r'(\d+)\s+months?', duration_lower)
        if months_match:
            months = int(months_match.group(1))
            return months / 12.0
        
        # Pattern 5: Just a number (assume years)
        number_match = re.search(r'^(\d+(?:\.\d+)?)$', duration.strip())
        if number_match:
            return float(number_match.group(1))
        
        return 1.0  # Default to 1 year if can't parse

    def _extract_required_skills(self, job_description: str) -> List[str]:
        """Extract required skills from job description"""
        skills = []
        
        # Comprehensive technical skills database
        tech_skills = [
            # Programming Languages
            "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "php", "ruby", "swift", "kotlin", "scala",
            
            # Web Technologies
            "react", "vue", "angular", "node.js", "express", "django", "flask", "spring", "laravel", "asp.net",
            
            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "github", "terraform", "ansible",
            
            # Databases
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "dynamodb", "cassandra",
            
            # Data & AI
            "machine learning", "ai", "data science", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark", "hadoop",
            
            # Mobile & Desktop
            "ios", "android", "react native", "flutter", "xamarin", "electron",
            
            # Other Technologies
            "graphql", "rest api", "microservices", "serverless", "blockchain", "cybersecurity", "devops", "sre"
        ]
        
        jd_lower = job_description.lower()
        for skill in tech_skills:
            if skill in jd_lower:
                skills.append(skill)
        
        return skills

    def _extract_candidate_skills(self, resume_text: str) -> List[str]:
        """Extract candidate skills from resume"""
        skills = []
        
        # Comprehensive technical skills database (same as required skills)
        tech_skills = [
            # Programming Languages
            "python", "java", "javascript", "typescript", "go", "rust", "c++", "c#", "php", "ruby", "swift", "kotlin", "scala",
            
            # Web Technologies
            "react", "vue", "angular", "node.js", "express", "django", "flask", "spring", "laravel", "asp.net",
            
            # Cloud & DevOps
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab", "github", "terraform", "ansible",
            
            # Databases
            "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "dynamodb", "cassandra",
            
            # Data & AI
            "machine learning", "ai", "data science", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "spark", "hadoop",
            
            # Mobile & Desktop
            "ios", "android", "react native", "flutter", "xamarin", "electron",
            
            # Other Technologies
            "graphql", "rest api", "microservices", "serverless", "blockchain", "cybersecurity", "devops", "sre"
        ]
        
        resume_lower = resume_text.lower()
        for skill in tech_skills:
            if skill in resume_lower:
                skills.append(skill)
        
        return skills

    def _skill_matches(self, required_skill: str, candidate_skills: List[str]) -> bool:
        """Check if candidate has required skill"""
        return required_skill.lower() in [skill.lower() for skill in candidate_skills]

    def _adjust_weights_for_collateral(self, weights: Dict[str, float], collateral: str) -> Dict[str, float]:
        """Adjust weights based on collateral information"""
        adjusted_weights = weights.copy()
        collateral_lower = collateral.lower()
        
        # Adjust weights based on collateral content
        if "startup" in collateral_lower or "early-stage" in collateral_lower:
            # Startups may value skills and company relevance more
            adjusted_weights["most_important_skills"] += 0.05
            adjusted_weights["company_relevance"] += 0.05
            adjusted_weights["education"] -= 0.05
            adjusted_weights["tenure_stability"] -= 0.05
        
        elif "enterprise" in collateral_lower or "large company" in collateral_lower:
            # Enterprises may value education and stability more
            adjusted_weights["education"] += 0.05
            adjusted_weights["tenure_stability"] += 0.05
            adjusted_weights["most_important_skills"] -= 0.05
            adjusted_weights["company_relevance"] -= 0.05
        
        elif "leadership" in collateral_lower or "management" in collateral_lower:
            # Leadership roles may value career trajectory more
            adjusted_weights["career_trajectory"] += 0.05
            adjusted_weights["bonus_signals"] += 0.02
            adjusted_weights["most_important_skills"] -= 0.07
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight != 1.0:
            for key in adjusted_weights:
                if key != "red_flags":  # Don't normalize penalty
                    adjusted_weights[key] = adjusted_weights[key] / total_weight
        
        return adjusted_weights

    def to_dict(self, result: FitScoreResult) -> Dict:
        """Convert FitScoreResult to dictionary"""
        return {
            "total_score": result.total_score,
            "education_score": result.education_score,
            "career_trajectory_score": result.career_trajectory_score,
            "company_relevance_score": result.company_relevance_score,
            "tenure_stability_score": result.tenure_stability_score,
            "most_important_skills_score": result.most_important_skills_score,
            "bonus_signals_score": result.bonus_signals_score,
            "red_flags_penalty": result.red_flags_penalty,
            "details": result.details,
            "recommendations": result.recommendations,
            "timestamp": result.timestamp,
            "submittable": result.total_score >= 8.2
        }

    def to_json(self, result: FitScoreResult) -> str:
        """Convert FitScoreResult to JSON string"""
        return json.dumps(self.to_dict(result), indent=2)

    def _detect_context_with_gpt4(self, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Use GPT-4 to intelligently detect context, industry, role type, and company type
        """
        if not self.client:
            logger.warning("OpenAI client not available, using fallback context detection")
            return self._detect_context_fallback(job_description)
        
        try:
            prompt = f"""
            Analyze the following job description and resume to detect:
            1. Industry (tech, healthcare, law, finance, etc.)
            2. Company type (startup, enterprise, law firm, accounting, healthcare, etc.)
            3. Role type (technical, management, sales, legal, accounting, healthcare, etc.)
            4. Role level (entry, mid, senior, executive)
            5. Key requirements and preferences
            
            Job Description:
            {job_description}
            
            Resume:
            {resume_text}
            
            Return a JSON object with:
            {{
                "industry": "detected industry",
                "company_type": "startup|enterprise|law_firm|accounting|healthcare|consulting|financial|academic|government|non_profit",
                "role_type": "technical|management|sales|legal|accounting|healthcare|consulting|financial|academic|government|non_profit",
                "role_level": "entry|mid|senior|executive",
                "key_requirements": ["requirement1", "requirement2"],
                "preferences": ["preference1", "preference2"],
                "company_size": "small|medium|large",
                "growth_stage": "seed|series_a|series_b|series_c|established|public"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            context_data = json.loads(response.choices[0].message.content)
            logger.info(f"GPT-4 detected context: {context_data}")
            return context_data
            
        except Exception as e:
            logger.error(f"GPT-4 context detection failed: {e}")
            return self._detect_context_fallback(job_description)

    def _generate_smart_criteria_with_gpt4(self, job_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use GPT-4 to generate elite hiring criteria based on job description and context
        """
        if not self.client:
            logger.warning("OpenAI client not available, using fallback criteria")
            return self._generate_smart_criteria_fallback(job_description, context)
        
        try:
            prompt = f"""
            Based on the job description and context, generate elite hiring criteria for top 1-2% performers.
            
            Context: {json.dumps(context, indent=2)}
            
            Job Description:
            {job_description}
            
            Generate elite criteria in JSON format:
            {{
                "mission_critical_skills": [
                    {{
                        "skill": "skill name",
                        "description": "what they must be able to do",
                        "importance": "critical|high|medium"
                    }}
                ],
                "elite_company_benchmarks": [
                    "company1",
                    "company2"
                ],
                "expected_outcomes": [
                    "outcome1",
                    "outcome2"
                ],
                "domain_mastery_requirements": [
                    "requirement1",
                    "requirement2"
                ],
                "leadership_indicators": [
                    "indicator1",
                    "indicator2"
                ],
                "technical_complexity": "low|medium|high",
                "scale_requirements": "small|medium|large",
                "industry_specific_requirements": [
                    "requirement1",
                    "requirement2"
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            criteria = json.loads(response.choices[0].message.content)
            logger.info(f"GPT-4 generated smart criteria: {criteria}")
            return criteria
            
        except Exception as e:
            logger.error(f"GPT-4 criteria generation failed: {e}")
            return self._generate_smart_criteria_fallback(job_description, context)

    def _extract_skills_with_gpt4(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Use GPT-4 to extract and match skills more intelligently
        """
        if not self.client:
            logger.warning("OpenAI client not available, using fallback skills extraction")
            return self._extract_skills_fallback(resume_text, job_description)
        
        try:
            prompt = f"""
            Extract and analyze skills from the resume and job description.
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            
            Return JSON with:
            {{
                "candidate_skills": [
                    {{
                        "skill": "skill name",
                        "evidence": "where/how it's mentioned",
                        "proficiency": "basic|intermediate|advanced|expert",
                        "years_experience": "estimated years"
                    }}
                ],
                "required_skills": [
                    {{
                        "skill": "skill name",
                        "importance": "required|preferred|nice_to_have",
                        "description": "what they need to do with it"
                    }}
                ],
                "skill_matches": [
                    {{
                        "skill": "skill name",
                        "match_quality": "exact|partial|inferred|missing",
                        "candidate_evidence": "how candidate demonstrates it",
                        "requirement_level": "what job requires"
                    }}
                ],
                "missing_critical_skills": ["skill1", "skill2"],
                "inferred_skills": [
                    {{
                        "skill": "skill name",
                        "reasoning": "why we can infer this",
                        "confidence": "high|medium|low"
                    }}
                ]
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            skills_analysis = json.loads(response.choices[0].message.content)
            logger.info(f"GPT-4 skills analysis completed")
            return skills_analysis
            
        except Exception as e:
            logger.error(f"GPT-4 skills extraction failed: {e}")
            return self._extract_skills_fallback(resume_text, job_description)

    def _evaluate_against_smart_criteria_with_gpt4(self, resume_text: str, smart_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use GPT-4 to evaluate candidate against elite smart criteria
        """
        if not self.client:
            logger.warning("OpenAI client not available, using fallback evaluation")
            return self._evaluate_against_smart_criteria_fallback(resume_text, smart_criteria)
        
        try:
            prompt = f"""
            Evaluate the candidate against elite hiring criteria.
            
            Resume:
            {resume_text}
            
            Elite Criteria:
            {json.dumps(smart_criteria, indent=2)}
            
            Return evaluation in JSON:
            {{
                "mission_critical_skills_score": {{
                    "score": 0-10,
                    "matches": ["skill1", "skill2"],
                    "gaps": ["skill1", "skill2"],
                    "reasoning": "detailed explanation"
                }},
                "elite_company_benchmark_score": {{
                    "score": 0-10,
                    "company_matches": ["company1", "company2"],
                    "reasoning": "explanation"
                }},
                "expected_outcomes_score": {{
                    "score": 0-10,
                    "outcomes_demonstrated": ["outcome1", "outcome2"],
                    "missing_outcomes": ["outcome1", "outcome2"],
                    "reasoning": "explanation"
                }},
                "domain_mastery_score": {{
                    "score": 0-10,
                    "mastery_areas": ["area1", "area2"],
                    "gaps": ["area1", "area2"],
                    "reasoning": "explanation"
                }},
                "leadership_score": {{
                    "score": 0-10,
                    "leadership_evidence": ["evidence1", "evidence2"],
                    "reasoning": "explanation"
                }},
                "overall_elite_score": {{
                    "score": 0-10,
                    "strengths": ["strength1", "strength2"],
                    "concerns": ["concern1", "concern2"],
                    "recommendation": "submit|reject|consider"
                }}
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            logger.info(f"GPT-4 elite evaluation completed")
            return evaluation
            
        except Exception as e:
            logger.error(f"GPT-4 elite evaluation failed: {e}")
            return self._evaluate_against_smart_criteria_fallback(resume_text, smart_criteria)

    def _adjust_weights_dynamically_with_gpt4(self, context: Dict[str, Any], smart_criteria: Dict[str, Any]) -> Dict[str, float]:
        """
        Use GPT-4 to dynamically adjust weights based on context and smart criteria
        """
        if not self.client:
            logger.warning("OpenAI client not available, using fallback weight adjustment")
            return self._adjust_weights_for_collateral(self.default_weights, "")
        
        try:
            prompt = f"""
            Based on the context and smart criteria, adjust the scoring weights for the FitScore evaluation.
            
            Context: {json.dumps(context, indent=2)}
            Smart Criteria: {json.dumps(smart_criteria, indent=2)}
            
            Current default weights:
            - Education: 20%
            - Career Trajectory: 20%
            - Company Relevance: 15%
            - Tenure Stability: 15%
            - Most Important Skills: 20%
            - Bonus Signals: 5%
            - Red Flags: -15% (penalty)
            
            Adjust weights based on:
            1. Industry requirements
            2. Company type (startup vs enterprise)
            3. Role level and complexity
            4. Growth stage and company size
            5. Specific role requirements
            
            Return adjusted weights in JSON:
            {{
                "education": 0.XX,
                "career_trajectory": 0.XX,
                "company_relevance": 0.XX,
                "tenure_stability": 0.XX,
                "most_important_skills": 0.XX,
                "bonus_signals": 0.XX,
                "red_flags": -0.XX,
                "reasoning": "explanation of adjustments"
            }}
            
            Ensure weights sum to 1.0 (excluding red_flags penalty).
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            adjusted_weights = json.loads(response.choices[0].message.content)
            logger.info(f"GPT-4 adjusted weights: {adjusted_weights}")
            return adjusted_weights
            
        except Exception as e:
            logger.error(f"GPT-4 weight adjustment failed: {e}")
            return self._adjust_weights_for_collateral(self.default_weights, "")

    # Fallback methods for when GPT-4 is not available
    def _detect_context_fallback(self, job_description: str) -> Dict[str, Any]:
        """Fallback context detection using pattern matching"""
        jd_lower = job_description.lower()
        
        # Detect industry
        industry = "general"
        if any(word in jd_lower for word in ["software", "engineer", "developer", "tech"]):
            industry = "tech"
        elif any(word in jd_lower for word in ["healthcare", "medical", "hospital", "clinic"]):
            industry = "healthcare"
        elif any(word in jd_lower for word in ["law", "legal", "attorney", "lawyer"]):
            industry = "law"
        elif any(word in jd_lower for word in ["accounting", "cpa", "audit", "finance"]):
            industry = "finance"
        
        # Detect company type
        company_type = "general"
        if any(word in jd_lower for word in ["startup", "seed", "series", "early-stage"]):
            company_type = "startup"
        elif any(word in jd_lower for word in ["enterprise", "fortune", "large company"]):
            company_type = "enterprise"
        elif any(word in jd_lower for word in ["law firm", "llp", "amlaw"]):
            company_type = "law_firm"
        elif any(word in jd_lower for word in ["accounting firm", "big 4"]):
            company_type = "accounting"
        elif any(word in jd_lower for word in ["hospital", "healthcare system"]):
            company_type = "healthcare"
        
        return {
            "industry": industry,
            "company_type": company_type,
            "role_type": self._detect_role_type(job_description),
            "role_level": "mid",  # Default
            "key_requirements": [],
            "preferences": [],
            "company_size": "medium",
            "growth_stage": "established"
        }

    def _generate_smart_criteria_fallback(self, job_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback smart criteria generation"""
        return {
            "mission_critical_skills": [],
            "elite_company_benchmarks": [],
            "expected_outcomes": [],
            "domain_mastery_requirements": [],
            "leadership_indicators": [],
            "technical_complexity": "medium",
            "scale_requirements": "medium",
            "industry_specific_requirements": []
        }

    def _extract_skills_fallback(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Fallback skills extraction using existing methods"""
        required_skills = self._extract_required_skills(job_description)
        candidate_skills = self._extract_candidate_skills(resume_text)
        
        return {
            "candidate_skills": [{"skill": skill, "evidence": "resume", "proficiency": "unknown", "years_experience": "unknown"} for skill in candidate_skills],
            "required_skills": [{"skill": skill, "importance": "required", "description": ""} for skill in required_skills],
            "skill_matches": [],
            "missing_critical_skills": [],
            "inferred_skills": []
        }

    def _evaluate_against_smart_criteria_fallback(self, resume_text: str, smart_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback evaluation using existing scoring methods"""
        return {
            "mission_critical_skills_score": {"score": 5.0, "matches": [], "gaps": [], "reasoning": "Fallback evaluation"},
            "elite_company_benchmark_score": {"score": 5.0, "company_matches": [], "reasoning": "Fallback evaluation"},
            "expected_outcomes_score": {"score": 5.0, "outcomes_demonstrated": [], "missing_outcomes": [], "reasoning": "Fallback evaluation"},
            "domain_mastery_score": {"score": 5.0, "mastery_areas": [], "gaps": [], "reasoning": "Fallback evaluation"},
            "leadership_score": {"score": 5.0, "leadership_evidence": [], "reasoning": "Fallback evaluation"},
            "overall_elite_score": {"score": 5.0, "strengths": [], "concerns": [], "recommendation": "consider"}
        }

    # Feedback-based system methods
    
    def add_feedback(
        self,
        job_id: str,
        company_id: str,
        candidate_id: str,
        feedback_type: str,
        feedback_text: str,
        feedback_category: str,
        feedback_score: float
    ) -> bool:
        """
        Add feedback for a candidate evaluation to improve future scoring
        
        Args:
            job_id: Unique identifier for the job
            company_id: Unique identifier for the company
            candidate_id: Unique identifier for the candidate
            feedback_type: "positive", "negative", or "neutral"
            feedback_text: Detailed feedback text
            feedback_category: "skills", "experience", "culture_fit", "location", "other"
            feedback_score: Score from -1.0 to 1.0
            
        Returns:
            bool: True if feedback was added successfully
        """
        try:
            feedback = FeedbackEntry(
                feedback_id=str(uuid.uuid4()),
                job_id=job_id,
                company_id=company_id,
                candidate_id=candidate_id,
                feedback_type=feedback_type,
                feedback_text=feedback_text,
                feedback_category=feedback_category,
                feedback_score=feedback_score,
                timestamp=datetime.utcnow().isoformat()
            )
            
            success = self.feedback_engine.add_feedback(feedback)
            if success:
                logger.info(f"Feedback added successfully for job {job_id}, candidate {candidate_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of feedback and prompt versions"""
        return self.feedback_engine.get_feedback_summary()
    
    def get_prompt_versions(self) -> List[Dict[str, Any]]:
        """Get all prompt versions with metadata"""
        versions = []
        for version in self.feedback_engine.prompt_versions.values():
            versions.append({
                "version_id": version.version_id,
                "prompt_type": version.prompt_type,
                "job_id": version.job_id,
                "company_id": version.company_id,
                "version_tag": version.version_tag,
                "feedback_pattern": version.feedback_pattern,
                "adjusted_fields": version.adjusted_fields,
                "location_enforced": version.location_enforced,
                "created_at": version.created_at,
                "updated_at": version.updated_at
            })
        return versions
    
    def reset_to_global_prompt(self, job_id: str) -> bool:
        """Reset a job to use the global prompt instead of local feedback-based prompt"""
        try:
            # Remove local prompts for this job
            local_versions = [v for v in self.feedback_engine.prompt_versions.values() 
                            if v.prompt_type == "local" and v.job_id == job_id]
            
            for version in local_versions:
                del self.feedback_engine.prompt_versions[version.version_id]
            
            logger.info(f"Reset job {job_id} to global prompt")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting to global prompt: {e}")
            return False
    
    def _apply_prompt_adjustments(self, weights: Dict[str, float], prompt_content: Dict[str, Any]) -> Dict[str, float]:
        """Apply feedback-based prompt adjustments to weights"""
        adjusted_weights = weights.copy()
        
        # Apply skills adjustments
        if "skills_adjustments" in prompt_content:
            skills_adj = prompt_content["skills_adjustments"]
            if skills_adj.get("feedback_based", False):
                # Increase skills weight based on feedback
                adjusted_weights["most_important_skills"] = min(0.30, adjusted_weights["most_important_skills"] + 0.05)
                # Decrease other weights proportionally
                adjusted_weights["education"] = max(0.15, adjusted_weights["education"] - 0.025)
                adjusted_weights["career_trajectory"] = max(0.15, adjusted_weights["career_trajectory"] - 0.025)
        
        # Apply experience adjustments
        if "experience_adjustments" in prompt_content:
            exp_adj = prompt_content["experience_adjustments"]
            if exp_adj.get("feedback_based", False):
                # Increase career trajectory weight based on feedback
                adjusted_weights["career_trajectory"] = min(0.30, adjusted_weights["career_trajectory"] + 0.05)
                # Decrease other weights proportionally
                adjusted_weights["education"] = max(0.15, adjusted_weights["education"] - 0.025)
                adjusted_weights["most_important_skills"] = max(0.15, adjusted_weights["most_important_skills"] - 0.025)
        
        # Apply company relevance adjustments
        if "company_adjustments" in prompt_content:
            comp_adj = prompt_content["company_adjustments"]
            if comp_adj.get("feedback_based", False):
                # Increase company relevance weight based on feedback
                adjusted_weights["company_relevance"] = min(0.25, adjusted_weights["company_relevance"] + 0.05)
                # Decrease other weights proportionally
                adjusted_weights["tenure_stability"] = max(0.10, adjusted_weights["tenure_stability"] - 0.025)
                adjusted_weights["bonus_signals"] = max(0.03, adjusted_weights["bonus_signals"] - 0.025)
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(adjusted_weights.values())
        if total_weight != 1.0:
            for key in adjusted_weights:
                if key != "red_flags":  # Don't normalize penalty
                    adjusted_weights[key] = adjusted_weights[key] / total_weight
        
        return adjusted_weights
    
    def _apply_location_constraints(
        self, 
        red_flags_penalty: float, 
        red_flags_details: Dict[str, Any], 
        resume_text: str, 
        job_description: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Enhanced location constraints with strict 20km enforcement for Hybrid/On-Site roles
        
        Applies Red Flag (–15 raw score penalty) if candidate is outside 20km radius
        """
        jd_lower = job_description.lower()
        is_remote = "remote" in jd_lower
        is_hybrid = "hybrid" in jd_lower
        is_onsite = "onsite" in jd_lower or "on-site" in jd_lower
        
        # Enhanced location detection with more comprehensive city database
        if (is_hybrid or is_onsite) and not is_remote:
            # Major cities that are typically outside 20km radius
            major_cities = {
                "new york": {"state": "NY", "typical_distance": "50+ km"},
                "california": {"state": "CA", "typical_distance": "100+ km"},
                "texas": {"state": "TX", "typical_distance": "100+ km"},
                "florida": {"state": "FL", "typical_distance": "100+ km"},
                "washington": {"state": "WA", "typical_distance": "50+ km"},
                "illinois": {"state": "IL", "typical_distance": "50+ km"},
                "massachusetts": {"state": "MA", "typical_distance": "30+ km"},
                "colorado": {"state": "CO", "typical_distance": "100+ km"},
                "georgia": {"state": "GA", "typical_distance": "100+ km"},
                "north carolina": {"state": "NC", "typical_distance": "100+ km"},
                "virginia": {"state": "VA", "typical_distance": "100+ km"},
                "pennsylvania": {"state": "PA", "typical_distance": "100+ km"},
                "ohio": {"state": "OH", "typical_distance": "100+ km"},
                "michigan": {"state": "MI", "typical_distance": "100+ km"},
                "wisconsin": {"state": "WI", "typical_distance": "100+ km"}
            }
            
            resume_lower = resume_text.lower()
            location_penalties_applied = []
            
            for city, info in major_cities.items():
                if city in resume_lower:
                    # Apply strict 20km enforcement penalty
                    location_penalty = -15.0
                    red_flags_penalty += location_penalty
                    
                    location_penalties_applied.append({
                        "location": f"{city.title()}, {info['state']}",
                        "penalty": location_penalty,
                        "distance": info['typical_distance'],
                        "reason": f"Location outside 20km radius for {is_hybrid and 'hybrid' or 'onsite'} role",
                        "enforcement": "Strict 20km radius enforcement applied"
                    })
                    
                    red_flags_details["flags_found"].append(f"Location: {city.title()} (outside 20km radius)")
            
            # Add location penalties to red flags details
            if location_penalties_applied:
                if "location_penalties" not in red_flags_details:
                    red_flags_details["location_penalties"] = []
                red_flags_details["location_penalties"].extend(location_penalties_applied)
                
                # Add enforcement summary
                red_flags_details["location_enforcement"] = {
                    "enforced": True,
                    "radius_km": 20,
                    "penalties_applied": len(location_penalties_applied),
                    "total_penalty": sum(p["penalty"] for p in location_penalties_applied),
                    "enforcement_reason": f"Hybrid/On-Site role requires strict 20km radius enforcement"
                }
        
        return red_flags_penalty, red_flags_details
    
    def convert_to_100_point_scale(self, score_10: float) -> float:
        """
        Convert 10-point scale score to 100-point scale as per specification
        
        GOAL: Score on 100-point scale, then scale to 10
        """
        return score_10 * 10.0
    
    def convert_to_10_point_scale(self, score_100: float) -> float:
        """
        Convert 100-point scale score to 10-point scale as per specification
        
        GOAL: Score on 100-point scale, then scale to 10
        """
        return score_100 * 0.1
    
    def apply_100_point_scoring(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply 100-point scale scoring as per specification
        
        Output: Score Breakdown by Category, Final Raw Score + Scaled Score
        """
        # Convert all scores to 100-point scale
        scores_100 = {}
        for category, score in scores.items():
            scores_100[f"{category}_100"] = self.convert_to_100_point_scale(score)
        
        # Calculate total raw score on 100-point scale
        total_raw_100 = sum(scores_100.values())
        
        # Scale back to 10-point scale
        total_scaled_10 = self.convert_to_10_point_scale(total_raw_100)
        
        return {
            "scores_100_point": scores_100,
            "total_raw_100": total_raw_100,
            "total_scaled_10": total_scaled_10,
            "conversion_applied": True,
            "scoring_method": "100-point scale with 10-point scaling as per specification"
        }


# Example usage and testing
def main():
    """Example usage of the FitScore calculator"""
    
    # Initialize calculator
    calculator = FitScoreCalculator()
    
    # Sample resume and job description
    sample_resume = """
    John Doe
    Software Engineer
    
    EDUCATION:
    Massachusetts Institute of Technology
    Bachelor of Science in Computer Science
    GPA: 3.8
    
    EXPERIENCE:
    Senior Software Engineer
    Google Inc.
    2020-2023 (3 years)
    - Led team of 10 engineers
    - Built scalable microservices
    - Used Python, React, AWS, Docker
    
    Software Engineer
    Microsoft Corporation
    2018-2020 (2 years)
    - Developed web applications
    - Used JavaScript, Node.js, SQL
    
    SKILLS:
    Python, JavaScript, React, Node.js, AWS, Docker, Kubernetes, SQL, MongoDB
    
    BONUS:
    - Open source contributor
    - Published technical articles
    - Speaking at conferences
    """
    
    sample_job_description = """
    Senior Software Engineer
    Tech Startup
    
    We are looking for a Senior Software Engineer to join our growing team.
    
    Requirements:
    - 5+ years of software engineering experience
    - Strong knowledge of Python, JavaScript, React
    - Experience with AWS, Docker, Kubernetes
    - Experience with microservices architecture
    - Leadership experience preferred
    
    Nice to have:
    - Machine learning experience
    - Open source contributions
    - Startup experience
    """
    
    # Calculate fitscore
    result = calculator.calculate_fitscore(
        resume_text=sample_resume,
        job_description=sample_job_description,
        collateral="Startup environment with fast-paced culture and emphasis on technical skills."
    )
    
    # Print results
    print("=== FITSCORE CALCULATION RESULTS ===")
    print(f"Total Score: {result.total_score:.2f}")
    print(f"Submittable: {result.total_score >= 8.2}")
    print(f"\nCategory Scores:")
    print(f"Education: {result.education_score:.2f}")
    print(f"Career Trajectory: {result.career_trajectory_score:.2f}")
    print(f"Company Relevance: {result.company_relevance_score:.2f}")
    print(f"Tenure Stability: {result.tenure_stability_score:.2f}")
    print(f"Most Important Skills: {result.most_important_skills_score:.2f}")
    print(f"Bonus Signals: {result.bonus_signals_score:.2f}")
    print(f"Red Flags Penalty: {result.red_flags_penalty:.2f}")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"- {rec}")
    
    print(f"\nDetailed Results (JSON):")
    print(calculator.to_json(result))


def test_fitscore():
    """Quick test function to verify the calculator works"""
    calculator = FitScoreCalculator()
    
    # Simple test case
    resume = "Software Engineer at Google with Python, React, AWS experience. MIT graduate."
    jd = "Looking for Python developer with React and AWS experience."
    
    result = calculator.calculate_fitscore(resume, jd)
    
    print(f"Test Result: {result.total_score:.2f}")
    print(f"Submittable: {result.total_score >= 8.2}")
    return result


if __name__ == "__main__":
    main() 
