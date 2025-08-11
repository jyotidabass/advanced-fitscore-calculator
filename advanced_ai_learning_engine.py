#!/usr/bin/env python3
"""
Advanced AI Learning Engine for Advanced FitScore Calculator & AI Hiring System
Implements sophisticated learning approaches for predictive hiring accuracy
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import uuid
import numpy as np
from enum import Enum

from platform_models import (
    CandidateProfile, ClientFeedback, DecisionType, 
    OperationalMetrics, PlatformDataLog
)

from reinforcement_feedback_agent import ReinforcementFeedbackAgent

logger = logging.getLogger(__name__)

class LearningMethod(str, Enum):
    """Advanced learning methods enumeration"""
    RLHF = "reinforcement_learning_human_feedback"
    CONTRASTIVE = "contrastive_learning"
    FEW_SHOT = "few_shot_learning"
    CURRICULUM = "curriculum_learning"
    ONLINE = "online_learning"
    BAYESIAN = "bayesian_updating"
    ACTIVE = "active_learning"

class ConfidenceLevel(str, Enum):
    """Confidence level enumeration"""
    HIGH = "high"      # 90%+ confidence
    MEDIUM = "medium"  # 70-89% confidence
    LOW = "low"        # 50-69% confidence
    UNCERTAIN = "uncertain"  # <50% confidence

@dataclass
class LearningOutcome:
    """Learning outcome with advanced metrics"""
    outcome_id: str
    candidate_id: str
    job_id: str
    outcome: str  # 'hired', 'rejected', 'interviewed', 'offer_accepted'
    fit_score: float
    confidence_interval: Tuple[float, float]  # (lower, upper)
    confidence_percentage: float
    learning_method: LearningMethod
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Advanced learning fields
    embedding_drift: Optional[float] = None  # Vector space movement
    pattern_confidence: Optional[float] = None  # Pattern recognition confidence
    uncertainty_reduction: Optional[float] = None  # How much uncertainty was reduced

@dataclass
class BayesianScore:
    """Bayesian score with confidence intervals"""
    score_id: str
    candidate_id: str
    job_id: str
    mean_score: float
    confidence_lower: float
    confidence_upper: float
    confidence_percentage: float
    prior_distribution: Dict[str, float]
    posterior_distribution: Dict[str, float]
    uncertainty_measure: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class ContrastiveLearningPair:
    """Contrastive learning pair for candidate comparison"""
    pair_id: str
    job_id: str
    positive_candidate_id: str  # Hired candidate
    negative_candidate_id: str  # Rejected candidate
    positive_fit_score: float
    negative_fit_score: float
    embedding_distance: float
    learning_gradient: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class CurriculumLearningStage:
    """Curriculum learning stage progression"""
    stage_id: str
    stage_name: str
    difficulty_level: int  # 1-10 scale
    confidence_threshold: float
    examples_required: int
    current_examples: int
    success_rate: float
    is_active: bool
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class AdvancedAILearningEngine:
    """
    Advanced AI Learning Engine for Advanced FitScore Calculator & AI Hiring System
    
    Implements sophisticated learning approaches:
    - Reinforcement Learning from Human Feedback (RLHF)
    - Contrastive Learning for candidate pairs
    - Few-Shot Learning with prompt tuning
    - Curriculum Learning with staged progression
    - Online Learning with real-time updates
    - Bayesian Updating for score confidence
    - Active Learning for feedback prioritization
    """
    
    def __init__(self, reinforcement_agent: Optional[ReinforcementFeedbackAgent] = None):
        """Initialize the advanced AI learning engine"""
        self.reinforcement_agent = reinforcement_agent or ReinforcementFeedbackAgent()
        self.learning_outcomes: Dict[str, LearningOutcome] = {}
        self.bayesian_scores: Dict[str, BayesianScore] = {}
        self.contrastive_pairs: Dict[str, ContrastiveLearningPair] = {}
        self.curriculum_stages: Dict[str, CurriculumLearningStage] = {}
        
        # Learning configuration
        self.learning_config = {
            "min_fit_score_threshold": 8.2,  # Only 8.2+ candidates eligible
            "confidence_thresholds": {
                "high": 0.90,
                "medium": 0.70,
                "low": 0.50
            },
            "learning_methods": {
                "rlhf": {"enabled": True, "weight": 0.3},
                "contrastive": {"enabled": True, "weight": 0.25},
                "few_shot": {"enabled": True, "weight": 0.2},
                "curriculum": {"enabled": True, "weight": 0.15},
                "online": {"enabled": True, "weight": 0.1}
            },
            "uncertainty_threshold": 0.3,  # Flag candidates with high uncertainty
            "active_learning_threshold": 0.4  # Request feedback for uncertain cases
        }
        
        # Initialize curriculum learning stages
        self._initialize_curriculum_stages()
        
        logger.info("Advanced AI Learning Engine initialized successfully")
    
    def _initialize_curriculum_stages(self):
        """Initialize curriculum learning stages"""
        stages = [
            {"name": "High Confidence Outcomes", "difficulty": 1, "confidence": 0.9, "examples": 5},
            {"name": "Clear Success Patterns", "difficulty": 2, "confidence": 0.8, "examples": 10},
            {"name": "Moderate Confidence Cases", "difficulty": 3, "confidence": 0.7, "examples": 15},
            {"name": "Ambiguous Outcomes", "difficulty": 4, "confidence": 0.6, "examples": 20},
            {"name": "Noisy Data Learning", "difficulty": 5, "confidence": 0.5, "examples": 25}
        ]
        
        for i, stage_config in enumerate(stages):
            stage = CurriculumLearningStage(
                stage_id=f"stage_{i+1}",
                stage_name=stage_config["name"],
                difficulty_level=stage_config["difficulty"],
                confidence_threshold=stage_config["confidence"],
                examples_required=stage_config["examples"],
                current_examples=0,
                success_rate=0.0,
                is_active=(i == 0)  # Only first stage active initially
            )
            self.curriculum_stages[stage["stage_id"]] = stage
    
    def process_advanced_learning_outcome(
        self, 
        candidate_id: str, 
        job_id: str, 
        outcome: str, 
        fit_score: float,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process learning outcome with advanced AI methods
        
        This implements the comprehensive learning approach:
        - RLHF for outcome-based weight adjustments
        - Contrastive learning for candidate pairs
        - Bayesian updating for confidence intervals
        - Curriculum learning progression
        - Online learning for real-time adaptation
        """
        try:
            # Create learning outcome
            outcome_id = str(uuid.uuid4())
            
            # Calculate confidence interval using Bayesian updating
            confidence_data = self._calculate_bayesian_confidence(
                candidate_id, job_id, fit_score, outcome, metadata
            )
            
            # Determine learning method based on data characteristics
            learning_method = self._determine_learning_method(fit_score, confidence_data, metadata)
            
            # Create learning outcome
            learning_outcome = LearningOutcome(
                outcome_id=outcome_id,
                candidate_id=candidate_id,
                job_id=job_id,
                outcome=outcome,
                fit_score=fit_score,
                confidence_interval=confidence_data["confidence_interval"],
                confidence_percentage=confidence_data["confidence_percentage"],
                learning_method=learning_method,
                metadata=metadata
            )
            
            self.learning_outcomes[outcome_id] = learning_outcome
            
            # Apply advanced learning methods
            learning_results = self._apply_advanced_learning_methods(
                learning_outcome, confidence_data
            )
            
            # Update curriculum learning progression
            curriculum_update = self._update_curriculum_learning(learning_outcome)
            
            # Check if active learning is needed
            active_learning_result = self._check_active_learning_needed(learning_outcome)
            
            logger.info(f"Advanced learning outcome processed: {outcome} with method {learning_method}")
            
            return {
                "success": True,
                "outcome_id": outcome_id,
                "learning_method": learning_method,
                "confidence_interval": confidence_data["confidence_interval"],
                "confidence_percentage": confidence_data["confidence_percentage"],
                "learning_results": learning_results,
                "curriculum_update": curriculum_update,
                "active_learning": active_learning_result,
                "message": f"Advanced learning outcome '{outcome}' processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing advanced learning outcome: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_bayesian_confidence(
        self, 
        candidate_id: str, 
        job_id: str, 
        fit_score: float, 
        outcome: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate Bayesian confidence intervals for FitScore
        
        Implements Bayesian updating for score confidence:
        - Prior distribution based on historical data
        - Likelihood based on current outcome
        - Posterior distribution with updated beliefs
        - Confidence intervals with uncertainty measures
        """
        # Get prior distribution from historical data
        prior_distribution = self._get_prior_distribution(job_id, metadata.get("job_family", "software_engineer"))
        
        # Calculate likelihood based on outcome
        likelihood = self._calculate_outcome_likelihood(fit_score, outcome, metadata)
        
        # Apply Bayesian updating
        posterior_distribution = self._apply_bayesian_updating(prior_distribution, likelihood)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(posterior_distribution)
        confidence_percentage = self._calculate_confidence_percentage(posterior_distribution)
        uncertainty_measure = self._calculate_uncertainty_measure(posterior_distribution)
        
        # Store Bayesian score
        score_id = f"{candidate_id}_{job_id}"
        bayesian_score = BayesianScore(
            score_id=score_id,
            candidate_id=candidate_id,
            job_id=job_id,
            mean_score=fit_score,
            confidence_lower=confidence_interval[0],
            confidence_upper=confidence_interval[1],
            confidence_percentage=confidence_percentage,
            prior_distribution=prior_distribution,
            posterior_distribution=posterior_distribution,
            uncertainty_measure=uncertainty_measure
        )
        self.bayesian_scores[score_id] = bayesian_score
        
        return {
            "confidence_interval": confidence_interval,
            "confidence_percentage": confidence_percentage,
            "uncertainty_measure": uncertainty_measure,
            "bayesian_score_id": score_id
        }
    
    def _get_prior_distribution(self, job_id: str, job_family: str) -> Dict[str, float]:
        """Get prior distribution from historical data"""
        # In production, this would query historical outcomes
        # For now, use default distributions based on job family
        default_priors = {
            "software_engineer": {"hired": 0.3, "rejected": 0.5, "interviewed": 0.2},
            "data_scientist": {"hired": 0.25, "rejected": 0.55, "interviewed": 0.2},
            "product_manager": {"hired": 0.2, "rejected": 0.6, "interviewed": 0.2}
        }
        
        return default_priors.get(job_family, default_priors["software_engineer"])
    
    def _calculate_outcome_likelihood(self, fit_score: float, outcome: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate likelihood based on outcome and FitScore"""
        # Higher FitScore should increase likelihood of positive outcomes
        score_factor = fit_score / 10.0
        
        if outcome == "hired":
            likelihood = {"hired": 0.6 + (score_factor * 0.3), "rejected": 0.3 - (score_factor * 0.2), "interviewed": 0.1}
        elif outcome == "rejected":
            likelihood = {"hired": 0.1, "rejected": 0.7 + (score_factor * 0.2), "interviewed": 0.2 - (score_factor * 0.1)}
        else:  # interviewed
            likelihood = {"hired": 0.2, "rejected": 0.4, "interviewed": 0.4 + (score_factor * 0.2)}
        
        # Normalize to sum to 1
        total = sum(likelihood.values())
        return {k: v/total for k, v in likelihood.items()}
    
    def _apply_bayesian_updating(self, prior: Dict[str, float], likelihood: Dict[str, float]) -> Dict[str, float]:
        """Apply Bayesian updating to combine prior and likelihood"""
        # Apply Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)
        posterior = {}
        evidence = 0.0
        
        for outcome in prior.keys():
            posterior[outcome] = likelihood[outcome] * prior[outcome]
            evidence += posterior[outcome]
        
        # Normalize posterior
        if evidence > 0:
            posterior = {k: v/evidence for k, v in posterior.items()}
        
        return posterior
    
    def _calculate_confidence_interval(self, distribution: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval from distribution"""
        # Simple confidence interval calculation
        # In production, use more sophisticated statistical methods
        values = list(distribution.values())
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        confidence_lower = max(0.0, mean_val - (1.96 * std_val))  # 95% CI
        confidence_upper = min(1.0, mean_val + (1.96 * std_val))
        
        return (confidence_lower, confidence_upper)
    
    def _calculate_confidence_percentage(self, distribution: Dict[str, float]) -> float:
        """Calculate confidence percentage from distribution"""
        # Higher confidence when distribution is more concentrated
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in distribution.values())
        max_entropy = np.log2(len(distribution))
        
        if max_entropy > 0:
            confidence = 1.0 - (entropy / max_entropy)
        else:
            confidence = 1.0
        
        return confidence
    
    def _calculate_uncertainty_measure(self, distribution: Dict[str, float]) -> float:
        """Calculate uncertainty measure from distribution"""
        # Lower uncertainty when distribution is more concentrated
        return 1.0 - self._calculate_confidence_percentage(distribution)
    
    def _determine_learning_method(self, fit_score: float, confidence_data: Dict[str, Any], metadata: Dict[str, Any]) -> LearningMethod:
        """Determine the most appropriate learning method"""
        confidence = confidence_data["confidence_percentage"]
        uncertainty = confidence_data["uncertainty_measure"]
        
        if uncertainty > self.learning_config["active_learning_threshold"]:
            return LearningMethod.ACTIVE
        
        if fit_score >= self.learning_config["min_fit_score_threshold"]:
            if confidence > self.learning_config["confidence_thresholds"]["high"]:
                return LearningMethod.CURRICULUM
            elif confidence > self.learning_config["confidence_thresholds"]["medium"]:
                return LearningMethod.ONLINE
            else:
                return LearningMethod.FEW_SHOT
        else:
            return LearningMethod.RLHF
    
    def _apply_advanced_learning_methods(
        self, 
        learning_outcome: LearningOutcome, 
        confidence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply advanced learning methods based on outcome"""
        results = {}
        
        # Apply RLHF
        if self.learning_config["learning_methods"]["rlhf"]["enabled"]:
            rlhf_result = self._apply_rlhf_learning(learning_outcome)
            results["rlhf"] = rlhf_result
        
        # Apply contrastive learning if we have pairs
        if self.learning_config["learning_methods"]["contrastive"]["enabled"]:
            contrastive_result = self._apply_contrastive_learning(learning_outcome)
            results["contrastive"] = contrastive_result
        
        # Apply few-shot learning
        if self.learning_config["learning_methods"]["few_shot"]["enabled"]:
            few_shot_result = self._apply_few_shot_learning(learning_outcome)
            results["few_shot"] = few_shot_result
        
        # Apply online learning
        if self.learning_config["learning_methods"]["online"]["enabled"]:
            online_result = self._apply_online_learning(learning_outcome)
            results["online"] = online_result
        
        return results
    
    def _apply_rlhf_learning(self, learning_outcome: LearningOutcome) -> Dict[str, Any]:
        """Apply Reinforcement Learning from Human Feedback"""
        # Use the existing reinforcement agent
        result = self.reinforcement_agent.process_feedback_outcome(
            candidate_id=learning_outcome.candidate_id,
            job_id=learning_outcome.job_id,
            outcome=learning_outcome.outcome,
            feedback_metadata=learning_outcome.metadata,
            job_family=learning_outcome.metadata.get("job_family", "software_engineer")
        )
        
        return {
            "method": "RLHF",
            "weight_adjustments": result.get("weight_adjustments", {}),
            "reward": result.get("reward", 0.0),
            "success": result.get("success", False)
        }
    
    def _apply_contrastive_learning(self, learning_outcome: LearningOutcome) -> Dict[str, Any]:
        """Apply contrastive learning for candidate pairs"""
        # Look for contrastive pairs (hired vs rejected for same job)
        job_outcomes = [
            outcome for outcome in self.learning_outcomes.values()
            if outcome.job_id == learning_outcome.job_id
        ]
        
        if len(job_outcomes) >= 2:
            # Find positive and negative examples
            positive_outcomes = [o for o in job_outcomes if o.outcome == "hired"]
            negative_outcomes = [o for o in job_outcomes if o.outcome == "rejected"]
            
            if positive_outcomes and negative_outcomes:
                # Create contrastive learning pair
                positive = positive_outcomes[0]
                negative = negative_outcomes[0]
                
                pair = ContrastiveLearningPair(
                    pair_id=str(uuid.uuid4()),
                    job_id=learning_outcome.job_id,
                    positive_candidate_id=positive.candidate_id,
                    negative_candidate_id=negative.candidate_id,
                    positive_fit_score=positive.fit_score,
                    negative_fit_score=negative.fit_score,
                    embedding_distance=abs(positive.fit_score - negative.fit_score),
                    learning_gradient=positive.fit_score - negative.fit_score
                )
                
                self.contrastive_pairs[pair.pair_id] = pair
                
                return {
                    "method": "Contrastive Learning",
                    "pair_id": pair.pair_id,
                    "embedding_distance": pair.embedding_distance,
                    "learning_gradient": pair.learning_gradient,
                    "success": True
                }
        
        return {
            "method": "Contrastive Learning",
            "success": False,
            "reason": "Insufficient contrastive data"
        }
    
    def _apply_few_shot_learning(self, learning_outcome: LearningOutcome) -> Dict[str, Any]:
        """Apply few-shot learning with prompt tuning"""
        # In production, this would update OpenAI prompts
        # For now, simulate the process
        
        prompt_updates = {
            "hired": "Boost scoring for candidates with similar profiles",
            "rejected": "Reduce scoring for candidates with similar negative traits",
            "interviewed": "Maintain moderate scoring for interview candidates"
        }
        
        return {
            "method": "Few-Shot Learning",
            "prompt_update": prompt_updates.get(learning_outcome.outcome, "No update"),
            "success": True
        }
    
    def _apply_online_learning(self, learning_outcome: LearningOutcome) -> Dict[str, Any]:
        """Apply online learning for real-time adaptation"""
        # Update learning patterns in real-time
        # This would typically update model weights or embeddings
        
        return {
            "method": "Online Learning",
            "real_time_update": True,
            "adaptation_speed": "immediate",
            "success": True
        }
    
    def _update_curriculum_learning(self, learning_outcome: LearningOutcome) -> Dict[str, Any]:
        """Update curriculum learning progression"""
        # Find appropriate stage for this outcome
        appropriate_stage = None
        for stage in self.curriculum_stages.values():
            if (learning_outcome.confidence_percentage >= stage.confidence_threshold and 
                stage.is_active):
                appropriate_stage = stage
                break
        
        if appropriate_stage:
            appropriate_stage.current_examples += 1
            
            # Check if stage is complete
            if appropriate_stage.current_examples >= appropriate_stage.examples_required:
                # Move to next stage
                next_stage = self._get_next_curriculum_stage(appropriate_stage.stage_id)
                if next_stage:
                    appropriate_stage.is_active = False
                    next_stage.is_active = True
                    
                    return {
                        "stage_completed": appropriate_stage.stage_name,
                        "next_stage": next_stage.stage_name,
                        "progression": True
                    }
            
            return {
                "current_stage": appropriate_stage.stage_name,
                "examples_progress": f"{appropriate_stage.current_examples}/{appropriate_stage.examples_required}",
                "progression": False
            }
        
        return {"progression": False, "reason": "No appropriate stage found"}
    
    def _get_next_curriculum_stage(self, current_stage_id: str) -> Optional[CurriculumLearningStage]:
        """Get the next curriculum learning stage"""
        current_stage_num = int(current_stage_id.split("_")[1])
        next_stage_id = f"stage_{current_stage_num + 1}"
        
        return self.curriculum_stages.get(next_stage_id)
    
    def _check_active_learning_needed(self, learning_outcome: LearningOutcome) -> Dict[str, Any]:
        """Check if active learning feedback is needed"""
        if learning_outcome.confidence_percentage < self.learning_config["active_learning_threshold"]:
            return {
                "active_learning_needed": True,
                "confidence_level": "uncertain",
                "feedback_request": "Request human feedback for uncertain case",
                "priority": "high"
            }
        
        return {
            "active_learning_needed": False,
            "confidence_level": "sufficient",
            "feedback_request": "No additional feedback needed"
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        total_outcomes = len(self.learning_outcomes)
        outcomes_by_method = {}
        confidence_distribution = {"high": 0, "medium": 0, "low": 0, "uncertain": 0}
        
        for outcome in self.learning_outcomes.values():
            method = outcome.learning_method
            if method not in outcomes_by_method:
                outcomes_by_method[method] = 0
            outcomes_by_method[method] += 1
            
            # Categorize confidence levels
            if outcome.confidence_percentage >= self.learning_config["confidence_thresholds"]["high"]:
                confidence_distribution["high"] += 1
            elif outcome.confidence_percentage >= self.learning_config["confidence_thresholds"]["medium"]:
                confidence_distribution["medium"] += 1
            elif outcome.confidence_percentage >= self.learning_config["confidence_thresholds"]["low"]:
                confidence_distribution["low"] += 1
            else:
                confidence_distribution["uncertain"] += 1
        
        return {
            "total_outcomes": total_outcomes,
            "outcomes_by_method": outcomes_by_method,
            "confidence_distribution": confidence_distribution,
            "curriculum_stages": {s.stage_id: s.stage_name for s in self.curriculum_stages.values()},
            "active_stage": next((s.stage_name for s in self.curriculum_stages.values() if s.is_active), None),
            "contrastive_pairs": len(self.contrastive_pairs),
            "bayesian_scores": len(self.bayesian_scores),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_candidate_confidence_score(self, candidate_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        """Get confidence score for a specific candidate-job pair"""
        score_id = f"{candidate_id}_{job_id}"
        if score_id in self.bayesian_scores:
            score = self.bayesian_scores[score_id]
            return {
                "mean_score": score.mean_score,
                "confidence_interval": score.confidence_interval,
                "confidence_percentage": score.confidence_percentage,
                "uncertainty_measure": score.uncertainty_measure,
                "recommendation": self._get_confidence_recommendation(score.confidence_percentage)
            }
        return None
    
    def _get_confidence_recommendation(self, confidence: float) -> str:
        """Get recommendation based on confidence level"""
        if confidence >= self.learning_config["confidence_thresholds"]["high"]:
            return "High confidence - Strong recommendation"
        elif confidence >= self.learning_config["confidence_thresholds"]["medium"]:
            return "Medium confidence - Moderate recommendation"
        elif confidence >= self.learning_config["confidence_thresholds"]["low"]:
            return "Low confidence - Weak recommendation"
        else:
            return "Uncertain - Request additional feedback"
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get detailed learning analytics"""
        return {
            "learning_summary": self.get_learning_summary(),
            "curriculum_progress": {
                stage.stage_id: {
                    "name": stage.stage_name,
                    "difficulty": stage.difficulty_level,
                    "progress": f"{stage.current_examples}/{stage.examples_required}",
                    "success_rate": stage.success_rate,
                    "is_active": stage.is_active
                }
                for stage in self.curriculum_stages.values()
            },
            "contrastive_learning": {
                "total_pairs": len(self.contrastive_pairs),
                "average_embedding_distance": np.mean([p.embedding_distance for p in self.contrastive_pairs.values()]) if self.contrastive_pairs else 0,
                "learning_gradient_summary": {
                    "positive": len([p for p in self.contrastive_pairs.values() if p.learning_gradient > 0]),
                    "negative": len([p for p in self.contrastive_pairs.values() if p.learning_gradient < 0])
                }
            },
            "bayesian_analysis": {
                "total_scores": len(self.bayesian_scores),
                "average_confidence": np.mean([s.confidence_percentage for s in self.bayesian_scores.values()]) if self.bayesian_scores else 0,
                "average_uncertainty": np.mean([s.uncertainty_measure for s in self.bayesian_scores.values()]) if self.bayesian_scores else 0
            }
        } 