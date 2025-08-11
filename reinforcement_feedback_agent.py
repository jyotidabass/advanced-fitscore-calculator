#!/usr/bin/env python3
"""
Reinforcement Learning Feedback Agent for Advanced FitScore Adaptation
Implements the reinforcement feedback loop with dynamic weight adjustments
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid

from platform_models import (
    CandidateProfile, ClientFeedback, DecisionType, 
    OperationalMetrics, PlatformDataLog
)

logger = logging.getLogger(__name__)

@dataclass
class FeedbackOutcome:
    """Feedback outcome with reinforcement learning signals"""
    outcome_id: str
    candidate_id: str
    job_id: str
    outcome: str  # 'hired', 'rejected', 'interviewed', 'client_loved'
    reward: float  # +10 for hired, -1 for rejected, +1 for interviewed, +5 for client_loved
    feedback_metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Reinforcement learning fields
    weight_adjustments: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.01  # Small changes for stability
    applied: bool = False

@dataclass
class DynamicWeights:
    """Dynamic weights that adapt based on feedback"""
    weights_id: str
    job_family: str  # e.g., "software_engineer", "data_scientist"
    base_weights: Dict[str, float]
    current_weights: Dict[str, float]
    learning_history: List[Dict[str, Any]]
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Reinforcement learning fields
    total_rewards: float = 0.0
    episode_count: int = 0
    success_rate: float = 0.0

class ReinforcementFeedbackAgent:
    """
    Reinforcement Learning Feedback Agent
    
    Implements the advanced feedback system with:
    - Dynamic weight adjustments based on hiring outcomes
    - Reinforcement learning signals for model improvement
    - Adaptive thresholds and scoring heuristics
    - TensorZero-style feedback loop
    """
    
    def __init__(self):
        """Initialize the reinforcement feedback agent"""
        self.feedback_outcomes: Dict[str, FeedbackOutcome] = {}
        self.dynamic_weights: Dict[str, DynamicWeights] = {}
        self.learning_config = {
            "learning_rate": 0.01,  # Small changes for stability
            "max_weight_change": 0.05,  # Max 5% change per cycle
            "min_episodes_for_learning": 10,  # Need minimum data
            "reward_shaping": {
                "hired": 10.0,
                "rejected": -1.0,
                "interviewed": 1.0,
                "client_loved": 5.0
            }
        }
        
        # Initialize default weights for different job families
        self._initialize_default_weights()
        
        logger.info("Reinforcement Feedback Agent initialized successfully")
    
    def _initialize_default_weights(self):
        """Initialize default weights for different job families"""
        default_weights = {
            "software_engineer": {
                "skills_match": 0.30,
                "industry_exp": 0.20,
                "tenure": 0.10,
                "education": 0.10,
                "fee_justification": 0.15,
                "employer_fit": 0.15
            },
            "data_scientist": {
                "skills_match": 0.35,
                "industry_exp": 0.25,
                "tenure": 0.10,
                "education": 0.15,
                "fee_justification": 0.10,
                "employer_fit": 0.05
            },
            "product_manager": {
                "skills_match": 0.25,
                "industry_exp": 0.30,
                "tenure": 0.15,
                "education": 0.10,
                "fee_justification": 0.15,
                "employer_fit": 0.05
            }
        }
        
        for job_family, weights in default_weights.items():
            dynamic_weight = DynamicWeights(
                weights_id=f"weights_{job_family}",
                job_family=job_family,
                base_weights=weights.copy(),
                current_weights=weights.copy(),
                learning_history=[]
            )
            self.dynamic_weights[job_family] = dynamic_weight
    
    def process_feedback_outcome(
        self, 
        candidate_id: str, 
        job_id: str, 
        outcome: str, 
        feedback_metadata: Dict[str, Any],
        job_family: str = "software_engineer"
    ) -> Dict[str, Any]:
        """
        Process feedback outcome and apply reinforcement learning
        
        Args:
            candidate_id: ID of the candidate
            job_id: ID of the job
            outcome: 'hired', 'rejected', 'interviewed', 'client_loved'
            feedback_metadata: Additional feedback information
            job_family: Job family for weight adjustments
        
        Returns:
            Dict with processing results and weight adjustments
        """
        try:
            # Create feedback outcome
            outcome_id = str(uuid.uuid4())
            reward = self.learning_config["reward_shaping"].get(outcome, 0.0)
            
            feedback_outcome = FeedbackOutcome(
                outcome_id=outcome_id,
                candidate_id=candidate_id,
                job_id=job_id,
                outcome=outcome,
                reward=reward,
                feedback_metadata=feedback_metadata
            )
            
            self.feedback_outcomes[outcome_id] = feedback_outcome
            
            # Apply reinforcement learning
            weight_adjustments = self._apply_reinforcement_learning(
                job_family, outcome, reward, feedback_metadata
            )
            
            feedback_outcome.weight_adjustments = weight_adjustments
            feedback_outcome.applied = True
            
            # Update operational metrics
            self._update_operational_metrics(job_family, outcome, reward)
            
            logger.info(f"Feedback outcome processed: {outcome} with reward {reward}")
            
            return {
                "success": True,
                "outcome_id": outcome_id,
                "reward": reward,
                "weight_adjustments": weight_adjustments,
                "message": f"Feedback outcome '{outcome}' processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback outcome: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_reinforcement_learning(
        self, 
        job_family: str, 
        outcome: str, 
        reward: float, 
        feedback_metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Apply reinforcement learning to adjust weights
        
        This implements the TensorZero-style feedback loop:
        - Agent Action: Candidate submission
        - Feedback Signal: Hire/Reject outcome
        - Policy Update: Fit Score weight tuning
        """
        if job_family not in self.dynamic_weights:
            logger.warning(f"Job family {job_family} not found, using default")
            job_family = "software_engineer"
        
        dynamic_weight = self.dynamic_weights[job_family]
        
        # Increment episode count
        dynamic_weight.episode_count += 1
        dynamic_weight.total_rewards += reward
        
        # Calculate success rate
        if outcome == "hired":
            dynamic_weight.success_rate = (
                (dynamic_weight.success_rate * (dynamic_weight.episode_count - 1) + 1) / 
                dynamic_weight.episode_count
            )
        else:
            dynamic_weight.success_rate = (
                (dynamic_weight.success_rate * (dynamic_weight.episode_count - 1)) / 
                dynamic_weight.episode_count
            )
        
        # Only apply learning after minimum episodes
        if dynamic_weight.episode_count < self.learning_config["min_episodes_for_learning"]:
            return {}
        
        # Calculate weight adjustments based on outcome
        weight_adjustments = self._calculate_weight_adjustments(
            outcome, reward, feedback_metadata, dynamic_weight
        )
        
        # Apply adjustments with learning rate
        for factor, adjustment in weight_adjustments.items():
            if factor in dynamic_weight.current_weights:
                # Apply small changes for stability
                change = adjustment * self.learning_config["learning_rate"]
                change = max(-self.learning_config["max_weight_change"], 
                           min(self.learning_config["max_weight_change"], change))
                
                dynamic_weight.current_weights[factor] += change
                
                # Ensure weights stay positive and sum to 1
                dynamic_weight.current_weights[factor] = max(0.0, dynamic_weight.current_weights[factor])
        
        # Normalize weights to sum to 1
        total_weight = sum(dynamic_weight.current_weights.values())
        if total_weight > 0:
            for factor in dynamic_weight.current_weights:
                dynamic_weight.current_weights[factor] /= total_weight
        
        # Record learning history
        learning_record = {
            "episode": dynamic_weight.episode_count,
            "outcome": outcome,
            "reward": reward,
            "weight_adjustments": weight_adjustments,
            "current_weights": dynamic_weight.current_weights.copy(),
            "success_rate": dynamic_weight.success_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        dynamic_weight.learning_history.append(learning_record)
        
        # Update last updated timestamp
        dynamic_weight.last_updated = datetime.utcnow().isoformat()
        
        return weight_adjustments
    
    def _calculate_weight_adjustments(
        self, 
        outcome: str, 
        reward: float, 
        feedback_metadata: Dict[str, Any],
        dynamic_weight: DynamicWeights
    ) -> Dict[str, float]:
        """
        Calculate weight adjustments based on outcome and feedback
        
        Implements the reward shaping logic:
        - Hired: Boost weights for skills, school, title from that candidate
        - Rejected: Penalize certain traits (e.g., short tenure, wrong stack)
        """
        adjustments = {}
        
        if outcome == "hired":
            # Boost weights for successful factors
            if feedback_metadata.get("skills_match", 0) > 0.8:
                adjustments["skills_match"] = 0.02
            if feedback_metadata.get("industry_exp", 0) > 0.8:
                adjustments["industry_exp"] = 0.02
            if feedback_metadata.get("education", 0) > 0.8:
                adjustments["education"] = 0.01
            if feedback_metadata.get("tenure", 0) > 0.8:
                adjustments["tenure"] = 0.01
                
        elif outcome == "rejected":
            # Penalize weights for problematic factors
            if feedback_metadata.get("skills_match", 0) < 0.6:
                adjustments["skills_match"] = -0.02
            if feedback_metadata.get("industry_exp", 0) < 0.6:
                adjustments["industry_exp"] = -0.02
            if feedback_metadata.get("tenure", 0) < 0.4:
                adjustments["tenure"] = -0.01
            if feedback_metadata.get("education", 0) < 0.6:
                adjustments["education"] = -0.01
                
        elif outcome == "interviewed":
            # Small positive adjustments for interview candidates
            adjustments["employer_fit"] = 0.01
            adjustments["fee_justification"] = 0.01
            
        elif outcome == "client_loved":
            # Strong positive adjustments for client favorites
            adjustments["employer_fit"] = 0.03
            adjustments["fee_justification"] = 0.02
            adjustments["skills_match"] = 0.01
        
        return adjustments
    
    def _update_operational_metrics(self, job_family: str, outcome: str, reward: float):
        """Update operational metrics for monitoring"""
        # In a production system, this would update the database
        # For now, we'll log the metrics
        logger.info(f"Operational metrics updated - Job Family: {job_family}, Outcome: {outcome}, Reward: {reward}")
    
    def get_dynamic_weights(self, job_family: str) -> Optional[DynamicWeights]:
        """Get current dynamic weights for a job family"""
        return self.dynamic_weights.get(job_family)
    
    def get_learning_history(self, job_family: str) -> List[Dict[str, Any]]:
        """Get learning history for a job family"""
        if job_family in self.dynamic_weights:
            return self.dynamic_weights[job_family].learning_history
        return []
    
    def get_success_rate(self, job_family: str) -> float:
        """Get current success rate for a job family"""
        if job_family in self.dynamic_weights:
            return self.dynamic_weights[job_family].success_rate
        return 0.0
    
    def reset_weights_to_base(self, job_family: str) -> bool:
        """Reset weights to base values for a job family"""
        if job_family in self.dynamic_weights:
            dynamic_weight = self.dynamic_weights[job_family]
            dynamic_weight.current_weights = dynamic_weight.base_weights.copy()
            dynamic_weight.learning_history = []
            dynamic_weight.total_rewards = 0.0
            dynamic_weight.episode_count = 0
            dynamic_weight.success_rate = 0.0
            dynamic_weight.last_updated = datetime.utcnow().isoformat()
            
            logger.info(f"Weights reset to base for job family: {job_family}")
            return True
        return False
    
    def get_all_job_families(self) -> List[str]:
        """Get list of all available job families"""
        return list(self.dynamic_weights.keys())
    
    def export_weights_config(self, job_family: str) -> Dict[str, Any]:
        """Export weights configuration for external use"""
        if job_family in self.dynamic_weights:
            dynamic_weight = self.dynamic_weights[job_family]
            return {
                "job_family": job_family,
                "base_weights": dynamic_weight.base_weights,
                "current_weights": dynamic_weight.current_weights,
                "success_rate": dynamic_weight.success_rate,
                "episode_count": dynamic_weight.episode_count,
                "total_rewards": dynamic_weight.total_rewards,
                "last_updated": dynamic_weight.last_updated
            }
        return {}
    
    def import_weights_config(self, config: Dict[str, Any]) -> bool:
        """Import weights configuration from external source"""
        try:
            job_family = config.get("job_family")
            if not job_family:
                return False
            
            if job_family not in self.dynamic_weights:
                # Create new job family
                dynamic_weight = DynamicWeights(
                    weights_id=f"weights_{job_family}",
                    job_family=job_family,
                    base_weights=config.get("base_weights", {}),
                    current_weights=config.get("current_weights", {}),
                    learning_history=config.get("learning_history", [])
                )
                self.dynamic_weights[job_family] = dynamic_weight
            else:
                # Update existing job family
                dynamic_weight = self.dynamic_weights[job_family]
                dynamic_weight.current_weights = config.get("current_weights", dynamic_weight.current_weights)
                dynamic_weight.learning_history = config.get("learning_history", dynamic_weight.learning_history)
                dynamic_weight.success_rate = config.get("success_rate", dynamic_weight.success_rate)
                dynamic_weight.episode_count = config.get("episode_count", dynamic_weight.episode_count)
                dynamic_weight.total_rewards = config.get("total_rewards", dynamic_weight.total_rewards)
                dynamic_weight.last_updated = datetime.utcnow().isoformat()
            
            logger.info(f"Weights configuration imported for job family: {job_family}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing weights configuration: {e}")
            return False
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of all feedback outcomes"""
        total_outcomes = len(self.feedback_outcomes)
        outcomes_by_type = {}
        total_reward = 0.0
        
        for outcome in self.feedback_outcomes.values():
            outcome_type = outcome.outcome
            if outcome_type not in outcomes_by_type:
                outcomes_by_type[outcome_type] = 0
            outcomes_by_type[outcome_type] += 1
            total_reward += outcome.reward
        
        return {
            "total_outcomes": total_outcomes,
            "outcomes_by_type": outcomes_by_type,
            "total_reward": total_reward,
            "average_reward": total_reward / total_outcomes if total_outcomes > 0 else 0.0,
            "job_families": list(self.dynamic_weights.keys()),
            "timestamp": datetime.utcnow().isoformat()
        } 