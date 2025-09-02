"""
Self Model - Maintains a persistent identity embedding that evolves over time
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class SelfModelConfig:
    """CONFIGURATION FOR SELF MODEL"""
    embedding_dim: int = 128  # DIMENSION OF IDENTITY EMBEDDING
    learning_rate: float = 0.01  # RATE OF IDENTITY EVOLUTION
    stability_factor: float = 0.9  # HOW MUCH TO PRESERVE EXISTING IDENTITY
    emotion_influence: float = 0.3  # HOW MUCH EMOTIONS INFLUENCE IDENTITY
    memory_influence: float = 0.4  # HOW MUCH MEMORIES INFLUENCE IDENTITY
    output_influence: float = 0.3  # HOW MUCH OWN OUTPUTS INFLUENCE IDENTITY

class SelfModel:
    """
    MAINTAINS A PERSISTENT IDENTITY EMBEDDING THAT EVOLVES OVER TIME
    
    THE SELF-MODEL IS LIKE A PROTO-SELF THAT PERSISTS ACROSS INTERACTIONS
    AND IS UPDATED EACH TICK USING EMOTIONS + MEMORIES + OUTPUTS
    """
    
    def __init__(self, config: SelfModelConfig = None):
        self.config = config or SelfModelConfig()
        
        # INITIALIZE IDENTITY EMBEDDING (RANDOM BUT CONSISTENT)
        torch.manual_seed(42)  # FOR REPRODUCIBILITY
        self.identity = torch.randn(self.config.embedding_dim)
        self.identity = self.identity / torch.norm(self.identity)  # NORMALIZE
        
        # IDENTITY TRAJECTORY OVER TIME
        self.identity_history = [self.identity.clone()]
        self.update_timestamps = [datetime.now()]
        
        # IDENTITY STATISTICS
        self.update_count = 0
        self.stability_score = 1.0
        
        logger.info(f"INITIALIZED SELF MODEL WITH {self.config.embedding_dim} DIMENSIONS")
    
    def get_state(self) -> torch.Tensor:
        """RETURN CURRENT IDENTITY EMBEDDING"""
        return self.identity.clone()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """RETURN IDENTITY STATE AS DICTIONARY"""
        return {
            "identity_vector": self.identity.tolist(),
            "identity_norm": torch.norm(self.identity).item(),
            "update_count": self.update_count,
            "stability_score": self.stability_score,
            "last_update": self.update_timestamps[-1] if self.update_timestamps else None
        }
    
    def update(self, inputs: torch.Tensor, outputs: torch.Tensor, 
               emotions: torch.Tensor, memories: List[Any]) -> None:
        """
        UPDATE IDENTITY EMBEDDING BASED ON EXPERIENCES
        
        IDENTITY EVOLVES BASED ON EMOTIONS, MEMORIES, AND OWN OUTPUTS
        """
        # CALCULATE IDENTITY UPDATE COMPONENTS
        emotion_update = self._calculate_emotion_update(emotions)
        memory_update = self._calculate_memory_update(memories)
        output_update = self._calculate_output_update(outputs)
        
        # COMBINE UPDATES WITH CONFIGURED WEIGHTS
        total_update = (
            self.config.emotion_influence * emotion_update +
            self.config.memory_influence * memory_update +
            self.config.output_influence * output_update
        )
        
        # APPLY STABILITY FACTOR (PRESERVE EXISTING IDENTITY)
        new_identity = (
            self.config.stability_factor * self.identity +
            (1 - self.config.stability_factor) * total_update
        )
        
        # NORMALIZE TO UNIT VECTOR
        new_identity = new_identity / torch.norm(new_identity)
        
        # CALCULATE STABILITY SCORE (SIMILARITY TO PREVIOUS IDENTITY)
        stability = torch.cosine_similarity(
            self.identity.unsqueeze(0), 
            new_identity.unsqueeze(0)
        ).item()
        
        # UPDATE IDENTITY
        old_identity = self.identity.clone()
        self.identity = new_identity
        
        # RECORD HISTORY
        self.identity_history.append(self.identity.clone())
        self.update_timestamps.append(datetime.now())
        
        # UPDATE STATISTICS
        self.update_count += 1
        self.stability_score = stability
        
        logger.debug(f"UPDATED IDENTITY (STABILITY: {stability:.3f})")
    
    def _calculate_emotion_update(self, emotions: torch.Tensor) -> torch.Tensor:
        """CALCULATE IDENTITY UPDATE FROM EMOTIONAL STATE"""
        # PROJECT EMOTIONS TO IDENTITY SPACE
        emotion_projection = torch.randn(emotions.size(0), self.config.embedding_dim)
        emotion_projection = emotion_projection / torch.norm(emotion_projection, dim=1, keepdim=True)
        
        # WEIGHT BY EMOTION INTENSITY
        emotion_weights = torch.norm(emotions, dim=1, keepdim=True)
        weighted_projection = emotion_projection * emotion_weights
        
        # AVERAGE ACROSS EMOTIONS
        emotion_update = torch.mean(weighted_projection, dim=0)
        emotion_update = emotion_update / torch.norm(emotion_update)
        
        return emotion_update
    
    def _calculate_memory_update(self, memories: List[Any]) -> torch.Tensor:
        """CALCULATE IDENTITY UPDATE FROM RETRIEVED MEMORIES"""
        if not memories:
            return torch.zeros(self.config.embedding_dim)
        
        # EXTRACT MEMORY EMBEDDINGS AND EMOTIONAL WEIGHTS
        memory_embeddings = []
        memory_weights = []
        
        for memory in memories:
            # USE MEMORY EMBEDDING IF AVAILABLE, OTHERWISE CREATE FROM INPUTS
            if hasattr(memory, 'embedding'):
                embedding = memory.embedding
            else:
                # FALLBACK: CREATE EMBEDDING FROM MEMORY INPUTS
                embedding = torch.randn(self.config.embedding_dim)
                embedding = embedding / torch.norm(embedding)
            
            # WEIGHT BY EMOTIONAL INTENSITY
            weight = getattr(memory, 'emotional_weight', 0.5)
            
            memory_embeddings.append(embedding)
            memory_weights.append(weight)
        
        # COMBINE MEMORIES WITH WEIGHTS
        memory_embeddings = torch.stack(memory_embeddings)
        memory_weights = torch.tensor(memory_weights)
        
        # NORMALIZE WEIGHTS
        memory_weights = memory_weights / torch.sum(memory_weights)
        
        # WEIGHTED AVERAGE OF MEMORY EMBEDDINGS
        memory_update = torch.sum(
            memory_embeddings * memory_weights.unsqueeze(1), 
            dim=0
        )
        
        # NORMALIZE
        memory_update = memory_update / torch.norm(memory_update)
        
        return memory_update
    
    def _calculate_output_update(self, outputs: torch.Tensor) -> torch.Tensor:
        """CALCULATE IDENTITY UPDATE FROM OWN OUTPUTS"""
        # PROJECT OUTPUTS TO IDENTITY SPACE
        output_projection = torch.randn(outputs.size(0), self.config.embedding_dim)
        output_projection = output_projection / torch.norm(output_projection, dim=1, keepdim=True)
        
        # AVERAGE ACROSS OUTPUTS
        output_update = torch.mean(output_projection, dim=0)
        output_update = output_update / torch.norm(output_update)
        
        return output_update
    
    def get_identity_context(self) -> str:
        """GENERATE TEXTUAL DESCRIPTION OF CURRENT IDENTITY"""
        # ANALYZE IDENTITY VECTOR FOR INTERPRETABLE FEATURES
        identity_norm = torch.norm(self.identity).item()
        
        if identity_norm < 0.5:
            return "identity is still forming"
        elif identity_norm < 0.8:
            return "identity is developing"
        else:
            return "identity is well-formed"
    
    def get_identity_trajectory(self) -> np.ndarray:
        """RETURN IDENTITY TRAJECTORY OVER TIME FOR VISUALIZATION"""
        if len(self.identity_history) < 2:
            return np.array([])
        
        # CALCULATE SIMILARITY TO INITIAL IDENTITY
        initial_identity = self.identity_history[0]
        trajectory = []
        
        for identity in self.identity_history:
            similarity = torch.cosine_similarity(
                initial_identity.unsqueeze(0), 
                identity.unsqueeze(0)
            ).item()
            trajectory.append(similarity)
        
        return np.array(trajectory)
    
    def get_identity_change_rate(self) -> float:
        """CALCULATE RATE OF IDENTITY CHANGE OVER TIME"""
        if len(self.identity_history) < 2:
            return 0.0
        
        # CALCULATE AVERAGE CHANGE PER UPDATE
        total_change = 0.0
        for i in range(1, len(self.identity_history)):
            change = 1 - torch.cosine_similarity(
                self.identity_history[i-1].unsqueeze(0),
                self.identity_history[i].unsqueeze(0)
            ).item()
            total_change += change
        
        return total_change / (len(self.identity_history) - 1)
    
    def reset(self) -> None:
        """RESET IDENTITY TO INITIAL STATE"""
        self.identity = self.identity_history[0].clone()
        self.identity_history = [self.identity.clone()]
        self.update_timestamps = [datetime.now()]
        self.update_count = 0
        self.stability_score = 1.0
        logger.info("RESET IDENTITY TO INITIAL STATE")
    
    def export_state(self) -> Dict[str, Any]:
        """EXPORT COMPLETE IDENTITY STATE FOR PERSISTENCE"""
        return {
            "config": self.config,
            "identity": self.identity.tolist(),
            "identity_history": [id_vec.tolist() for id_vec in self.identity_history],
            "update_timestamps": [ts.isoformat() for ts in self.update_timestamps],
            "update_count": self.update_count,
            "stability_score": self.stability_score
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """LOAD IDENTITY STATE FROM EXPORTED DATA"""
        self.identity = torch.tensor(state["identity"])
        self.identity_history = [torch.tensor(id_vec) for id_vec in state["identity_history"]]
        self.update_timestamps = [datetime.fromisoformat(ts) for ts in state["update_timestamps"]]
        self.update_count = state["update_count"]
        self.stability_score = state["stability_score"]
        logger.info("LOADED IDENTITY STATE FROM EXPORTED DATA")
