"""
Lifespan Manager - Tracks agent's life cycle and manages developmental trajectories
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class LifespanConfig:
    """CONFIGURATION FOR LIFESPAN MANAGER"""
    max_ticks: int = 1000  # MAXIMUM NUMBER OF LIFE TICKS
    early_life_threshold: float = 0.2  # FRACTION OF LIFE CONSIDERED "EARLY"
    mid_life_threshold: float = 0.6  # FRACTION OF LIFE CONSIDERED "MID"
    plasticity_decay_rate: float = 0.995  # RATE OF PLASTICITY DECAY PER TICK
    memory_dominance_growth: float = 1.005  # RATE OF MEMORY DOMINANCE GROWTH
    identity_stability_growth: float = 1.002  # RATE OF IDENTITY STABILITY GROWTH

class LifespanManager:
    """
    TRACKS AGENT'S LIFE CYCLE AND MANAGES DEVELOPMENTAL TRAJECTORIES
    
    EARLY LIFE = HIGH PLASTICITY, LATE LIFE = RECALL DOMINANCE
    ANALOGY: DEVELOPMENTAL TRAJECTORIES, AGING
    """
    
    def __init__(self, config: LifespanConfig = None):
        self.config = config or LifespanConfig()
        
        # LIFE CYCLE STATE
        self.current_tick = 0
        self.life_stage = "infancy"
        self.is_alive = True
        
        # DEVELOPMENTAL METRICS
        self.plasticity = 1.0  # ABILITY TO LEARN AND CHANGE
        self.memory_dominance = 0.1  # RELIANCE ON MEMORY VS NEW LEARNING
        self.identity_stability = 0.1  # STABILITY OF IDENTITY OVER TIME
        
        # LIFE HISTORY
        self.life_history = []
        self.milestones = []
        self.death_cause = None
        
        # DEVELOPMENTAL PHASES
        self.developmental_phases = self._initialize_phases()
        
        logger.info(f"INITIALIZED LIFESPAN MANAGER WITH {self.config.max_ticks} MAX TICKS")
    
    def tick(self) -> Dict[str, Any]:
        """
        ADVANCE ONE TICK IN THE LIFE CYCLE
        
        RETURNS CURRENT LIFE STATE AND DEVELOPMENTAL METRICS
        """
        if not self.is_alive:
            logger.warning("AGENT IS DEAD, CANNOT TICK")
            return self.get_life_state()
        
        # ADVANCE TICK COUNTER
        self.current_tick += 1
        
        # UPDATE DEVELOPMENTAL METRICS
        self._update_developmental_metrics()
        
        # CHECK FOR LIFE STAGE TRANSITIONS
        self._check_life_stage_transitions()
        
        # RECORD LIFE HISTORY
        self._record_life_history()
        
        # CHECK FOR DEATH
        if self._should_die():
            self._die()
        
        # LOG PROGRESS
        if self.current_tick % 100 == 0:
            logger.info(f"LIFE TICK {self.current_tick}/{self.config.max_ticks} - "
                       f"STAGE: {self.life_stage}, PLASTICITY: {self.plasticity:.3f}")
        
        return self.get_life_state()
    
    def _update_developmental_metrics(self) -> None:
        """UPDATE DEVELOPMENTAL METRICS BASED ON LIFE STAGE"""
        # PLASTICITY DECAYS OVER TIME
        self.plasticity *= self.config.plasticity_decay_rate
        
        # MEMORY DOMINANCE GROWS OVER TIME
        self.memory_dominance *= self.config.memory_dominance_growth
        
        # IDENTITY STABILITY GROWS OVER TIME
        self.identity_stability *= self.config.identity_stability_growth
        
        # CLAMP VALUES TO VALID RANGES
        self.plasticity = max(0.0, min(1.0, self.plasticity))
        self.memory_dominance = max(0.0, min(1.0, self.memory_dominance))
        self.identity_stability = max(0.0, min(1.0, self.identity_stability))
    
    def _check_life_stage_transitions(self) -> None:
        """CHECK AND EXECUTE LIFE STAGE TRANSITIONS"""
        life_progress = self.current_tick / self.config.max_ticks
        
        old_stage = self.life_stage
        
        # DETERMINE LIFE STAGE BASED ON PROGRESS
        if life_progress < self.config.early_life_threshold:
            new_stage = "infancy"
        elif life_progress < self.config.mid_life_threshold:
            new_stage = "adolescence"
        else:
            new_stage = "maturity"
        
        # RECORD STAGE TRANSITION
        if new_stage != old_stage:
            self.life_stage = new_stage
            self._record_milestone(f"transitioned_to_{new_stage}", self.current_tick)
            logger.info(f"LIFE STAGE TRANSITION: {old_stage} -> {new_stage}")
    
    def _record_life_history(self) -> None:
        """RECORD CURRENT LIFE STATE IN HISTORY"""
        life_record = {
            "tick": self.current_tick,
            "life_stage": self.life_stage,
            "plasticity": self.plasticity,
            "memory_dominance": self.memory_dominance,
            "identity_stability": self.identity_stability,
            "timestamp": datetime.now()
        }
        
        self.life_history.append(life_record)
    
    def _should_die(self) -> bool:
        """DETERMINE IF AGENT SHOULD DIE"""
        # NATURAL DEATH FROM OLD AGE
        if self.current_tick >= self.config.max_ticks:
            return True
        
        # DEATH FROM EXTREME INSTABILITY
        if self.identity_stability < 0.01:
            return True
        
        # DEATH FROM COMPLETE MEMORY LOSS
        if self.memory_dominance < 0.01:
            return True
        
        return False
    
    def _die(self) -> None:
        """EXECUTE AGENT DEATH"""
        self.is_alive = False
        
        # DETERMINE CAUSE OF DEATH
        if self.current_tick >= self.config.max_ticks:
            self.death_cause = "natural_death_old_age"
        elif self.identity_stability < 0.01:
            self.death_cause = "death_from_identity_instability"
        elif self.memory_dominance < 0.01:
            self.death_cause = "death_from_memory_loss"
        else:
            self.death_cause = "unknown_death"
        
        # RECORD DEATH MILESTONE
        self._record_milestone("death", self.current_tick, {"cause": self.death_cause})
        
        logger.info(f"AGENT DIED AT TICK {self.current_tick} - CAUSE: {self.death_cause}")
    
    def _record_milestone(self, milestone_type: str, tick: int, 
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """RECORD A LIFE MILESTONE"""
        milestone = {
            "type": milestone_type,
            "tick": tick,
            "life_progress": tick / self.config.max_ticks,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.milestones.append(milestone)
    
    def get_life_state(self) -> Dict[str, Any]:
        """RETURN CURRENT LIFE STATE"""
        return {
            "current_tick": self.current_tick,
            "max_ticks": self.config.max_ticks,
            "life_progress": self.current_tick / self.config.max_ticks,
            "life_stage": self.life_stage,
            "is_alive": self.is_alive,
            "plasticity": self.plasticity,
            "memory_dominance": self.memory_dominance,
            "identity_stability": self.identity_stability,
            "death_cause": self.death_cause
        }
    
    def get_developmental_phase(self) -> Dict[str, Any]:
        """RETURN CURRENT DEVELOPMENTAL PHASE CHARACTERISTICS"""
        phase = self.developmental_phases.get(self.life_stage, {})
        
        # ADD CURRENT METRICS
        phase.update({
            "current_plasticity": self.plasticity,
            "current_memory_dominance": self.memory_dominance,
            "current_identity_stability": self.identity_stability
        })
        
        return phase
    
    def _initialize_phases(self) -> Dict[str, Dict[str, Any]]:
        """INITIALIZE DEVELOPMENTAL PHASES"""
        return {
            "infancy": {
                "description": "High plasticity, rapid learning, minimal memory reliance",
                "characteristics": ["exploration", "adaptation", "growth"],
                "optimal_activities": ["learning", "experimentation", "identity_formation"]
            },
            "adolescence": {
                "description": "Balanced plasticity and memory, identity development",
                "characteristics": ["integration", "self_discovery", "pattern_recognition"],
                "optimal_activities": ["reflection", "memory_integration", "identity_refinement"]
            },
            "maturity": {
                "description": "Lower plasticity, high memory dominance, stable identity",
                "characteristics": ["wisdom", "stability", "experience_integration"],
                "optimal_activities": ["memory_consolidation", "narrative_generation", "legacy_building"]
            }
        }
    
    def get_life_summary(self) -> Dict[str, Any]:
        """RETURN COMPREHENSIVE LIFE SUMMARY"""
        if not self.life_history:
            return {"status": "no_life_history"}
        
        # CALCULATE LIFE STATISTICS
        life_stats = {
            "total_ticks": self.current_tick,
            "life_stages": {},
            "milestones": len(self.milestones),
            "final_state": self.get_life_state()
        }
        
        # ANALYZE LIFE STAGES
        for record in self.life_history:
            stage = record["life_stage"]
            if stage not in life_stats["life_stages"]:
                life_stats["life_stages"][stage] = {
                    "ticks": 0,
                    "avg_plasticity": 0.0,
                    "avg_memory_dominance": 0.0,
                    "avg_identity_stability": 0.0
                }
            
            stage_stats = life_stats["life_stages"][stage]
            stage_stats["ticks"] += 1
            stage_stats["avg_plasticity"] += record["plasticity"]
            stage_stats["avg_memory_dominance"] += record["memory_dominance"]
            stage_stats["avg_identity_stability"] += record["identity_stability"]
        
        # CALCULATE AVERAGES
        for stage_stats in life_stats["life_stages"].values():
            ticks = stage_stats["ticks"]
            stage_stats["avg_plasticity"] /= ticks
            stage_stats["avg_memory_dominance"] /= ticks
            stage_stats["avg_identity_stability"] /= ticks
        
        return life_stats
    
    def export_life_data(self, filepath: str) -> None:
        """EXPORT COMPLETE LIFE DATA TO JSON FILE"""
        export_data = {
            "life_history": self.life_history,
            "milestones": self.milestones,
            "life_summary": self.get_life_summary(),
            "config": self.config,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"EXPORTED LIFE DATA TO {filepath}")
    
    def get_developmental_trajectory(self) -> Dict[str, np.ndarray]:
        """RETURN DEVELOPMENTAL TRAJECTORIES FOR VISUALIZATION"""
        if not self.life_history:
            return {}
        
        # EXTRACT TRAJECTORIES
        ticks = [record["tick"] for record in self.life_history]
        plasticity = [record["plasticity"] for record in self.life_history]
        memory_dominance = [record["memory_dominance"] for record in self.life_history]
        identity_stability = [record["identity_stability"] for record in self.life_history]
        
        return {
            "ticks": np.array(ticks),
            "plasticity": np.array(plasticity),
            "memory_dominance": np.array(memory_dominance),
            "identity_stability": np.array(identity_stability)
        }
    
    def reset_life(self) -> None:
        """RESET AGENT TO BEGINNING OF LIFE"""
        self.current_tick = 0
        self.life_stage = "infancy"
        self.is_alive = True
        self.plasticity = 1.0
        self.memory_dominance = 0.1
        self.identity_stability = 0.1
        self.life_history = []
        self.milestones = []
        self.death_cause = None
        
        logger.info("RESET AGENT TO BEGINNING OF LIFE")
    
    def extend_life(self, additional_ticks: int) -> None:
        """EXTEND AGENT'S LIFESPAN BY ADDITIONAL TICKS"""
        if not self.is_alive:
            logger.warning("CANNOT EXTEND LIFE OF DEAD AGENT")
            return
        
        self.config.max_ticks += additional_ticks
        logger.info(f"EXTENDED LIFESPAN BY {additional_ticks} TICKS TO {self.config.max_ticks}")
    
    def get_optimal_activities(self) -> List[str]:
        """RETURN OPTIMAL ACTIVITIES FOR CURRENT LIFE STAGE"""
        phase = self.developmental_phases.get(self.life_stage, {})
        return phase.get("optimal_activities", [])
    
    def get_life_expectancy(self) -> int:
        """RETURN ESTIMATED TICKS REMAINING IN LIFE"""
        if not self.is_alive:
            return 0
        
        return self.config.max_ticks - self.current_tick
