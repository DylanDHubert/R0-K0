"""
Emotion Engine - Manages emotional state and applies emotional bias to model outputs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import csv
import os
from datetime import datetime
import json

# LANGCHAIN INTEGRATION
try:
    from overlay.utils.langchain import LangChainConfig
except ImportError:
    # FALLBACK IF LANGCHAIN UTILITIES NOT AVAILABLE
    @dataclass
    class LangChainConfig:
        provider: str = "openai"
        model_name: str = "gpt-3.5-turbo"
        api_key: Optional[str] = None
        temperature: float = 0.1
        max_retries: int = 3
        rate_limit_delay: float = 0.1

logger = logging.getLogger(__name__)

class EmotionInference:
    """BASE CLASS FOR EMOTION INFERENCE - TAKES TEXT, OUTPUTS N-DIM VECTOR"""
    
    def __init__(self, config: 'EmotionConfig'):
        self.config = config
    
    def infer(self, text: str) -> torch.Tensor:
        """INFER EMOTION VECTOR FROM TEXT - MUST BE IMPLEMENTED BY SUBCLASSES"""
        raise NotImplementedError
    
    def get_cache_path(self, run_dir: str) -> str:
        """GET PATH FOR EMOTION CACHE FILE IN RUN DIRECTORY"""
        return os.path.join(run_dir, "emotions.csv")
    
    def save_to_cache(self, text: str, emotion_vector: torch.Tensor, run_dir: str, 
                      api_model: str = "unknown") -> None:
        """SAVE EMOTION INFERENCE TO CSV CACHE"""
        cache_path = self.get_cache_path(run_dir)
        
        # CREATE RUN DIRECTORY IF IT DOESN'T EXIST
        os.makedirs(run_dir, exist_ok=True)
        
        # CHECK IF CSV EXISTS AND CREATE HEADERS
        file_exists = os.path.exists(cache_path)
        
        with open(cache_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'api_model', 'text'] + [f'dim_{i}' for i in range(self.config.dimension)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # PREPARE ROW DATA
            row_data = {
                'timestamp': datetime.now().isoformat(),
                'api_model': api_model,
                'text': text
            }
            
            # ADD EMOTION DIMENSIONS
            for i, value in enumerate(emotion_vector.tolist()):
                row_data[f'dim_{i}'] = value
            
            writer.writerow(row_data)
        
        logger.debug(f"SAVED EMOTION INFERENCE TO CACHE: {cache_path}")

class APIBasedInference(EmotionInference):
    """API-BASED EMOTION INFERENCE USING EXTERNAL LLM SERVICE"""
    
    def __init__(self, config: 'EmotionConfig', langchain_config=None, model_name: str = "unknown"):
        super().__init__(config)
        self.langchain_config = langchain_config
        self.model_name = model_name
        self.run_dir = None
        self.llm = None
        self.prompt_template = None
        
        # INITIALIZE LANGCHAIN COMPONENTS IF CONFIG PROVIDED
        if langchain_config:
            self._initialize_langchain()
    
    def _initialize_langchain(self):
        """INITIALIZE LANGCHAIN LLM AND PROMPT TEMPLATE"""
        try:
            from overlay.utils.langchain import create_llm, create_emotion_prompt
            
            self.llm = create_llm(self.langchain_config)
            
            # GET EMOTION LABELS FROM CONFIG OR USE DEFAULT ONES
            emotion_labels = self.config.labels
            if not emotion_labels:
                # USE DEFAULT MEANINGFUL LABELS IF NONE PROVIDED
                default_labels = ["happy", "sad", "scared"]
                if self.config.dimension <= len(default_labels):
                    emotion_labels = default_labels[:self.config.dimension]
                else:
                    emotion_labels = default_labels + [f"emotion_{i}" for i in range(len(default_labels), self.config.dimension)]
            
            self.prompt_template = create_emotion_prompt(
                self.config.dimension, 
                emotion_labels
            )
            
            if self.llm:
                logger.info(f"INITIALIZED LANGCHAIN LLM: {type(self.llm).__name__}")
            else:
                logger.warning("FAILED TO CREATE LANGCHAIN LLM")
                
        except ImportError:
            logger.warning("LANGCHAIN UTILITIES NOT AVAILABLE")
        except Exception as e:
            logger.error(f"FAILED TO INITIALIZE LANGCHAIN: {e}")
    
    def set_run_directory(self, run_dir: str):
        """SET THE RUN DIRECTORY FOR CACHING"""
        self.run_dir = run_dir
    
    def infer(self, text: str) -> torch.Tensor:
        """INFER EMOTION VECTOR FROM TEXT USING API"""
        if not self.llm or not self.prompt_template:
            logger.warning("LANGCHAIN NOT INITIALIZED - RETURNING RANDOM EMOTION VECTOR")
            return torch.randn(self.config.dimension) * 0.1
        
        try:
            # CALL LANGCHAIN LLM
            from overlay.utils.langchain import call_llm_with_prompt
            
            response = call_llm_with_prompt(self.llm, self.prompt_template, text)
            
            # PARSE RESPONSE INTO EMOTION VECTOR
            from overlay.utils.langchain import EmotionOutputParser
            parser = EmotionOutputParser(self.config.dimension)
            emotion_vector = parser.parse(response)
            
            # CONVERT TO TENSOR
            emotion_vector = torch.tensor(emotion_vector, dtype=torch.float32)
            
            # SAVE TO CACHE IF RUN DIRECTORY IS SET
            if self.run_dir:
                self.save_to_cache(text, emotion_vector, self.run_dir, self.model_name)
            
            return emotion_vector
            
        except Exception as e:
            logger.error(f"API INFERENCE FAILED: {e}")
            # FALLBACK TO RANDOM VECTOR
            return torch.randn(self.config.dimension) * 0.1

@dataclass
class EmotionConfig:
    """CONFIGURATION FOR EMOTION ENGINE"""
    dimension: int = 3  # CONFIGURABLE EMOTIONAL DIMENSIONS
    decay_rate: float = 0.95  # EMOTIONAL DECAY PER TICK
    max_intensity: float = 1.0  # MAXIMUM EMOTIONAL INTENSITY
    bias_strength: float = 0.3  # HOW STRONGLY EMOTIONS BIAS OUTPUTS
    update_sensitivity: float = 0.1  # HOW SENSITIVE EMOTIONS ARE TO INPUTS
    labels: Optional[List[str]] = None  # CUSTOM LABELS FOR EMOTION DIMENSIONS

class EmotionEngine:
    """
    MANAGES EMOTIONAL STATE VECTOR AND APPLIES EMOTIONAL BIAS TO MODEL OUTPUTS
    
    EMOTIONS ARE MODELED AS A VECTOR OF STATE NODES (HAPPY, SAD, SCARED, ETC.)
    THAT BIAS EMBEDDINGS, LOGITS, AND MEMORY GATES.
    
    DESIGN: 3D EMOTION SPACE PROJECTED ONTO INPUT EMBEDDINGS
    """
    
    def __init__(self, config: EmotionConfig = None):
        self.config = config or EmotionConfig()
        
        # INITIALIZE EMOTIONAL STATE VECTOR
        self.emotions = torch.zeros(self.config.dimension)
        
        # EMOTION DIMENSION LABELS (N-DIMENSIONAL) - USE CONFIG LABELS OR DEFAULT MEANINGFUL ONES
        if self.config.labels:
            self.emotion_labels = self.config.labels
        else:
            # DEFAULT MEANINGFUL EMOTION LABELS FOR 3 DIMENSIONS
            default_labels = ["happy", "sad", "scared"]
            if self.config.dimension <= len(default_labels):
                self.emotion_labels = default_labels[:self.config.dimension]
            else:
                # EXTEND WITH GENERIC LABELS IF MORE THAN 3 DIMENSIONS
                self.emotion_labels = default_labels + [f"emotion_{i}" for i in range(len(default_labels), self.config.dimension)]
        
        # EMOTIONAL MEMORY - TRACKS INTENSITY OVER TIME
        self.emotion_history = []
        
        # EMOTION INFERENCE COMPONENT
        self.emotion_inference = None
        
        logger.info(f"INITIALIZED EMOTION ENGINE WITH {self.config.dimension} DIMENSIONS: {', '.join(self.emotion_labels)}")
    
    def get_state(self) -> torch.Tensor:
        """RETURN CURRENT EMOTIONAL STATE VECTOR"""
        return self.emotions.clone()
    
    def get_state_dict(self) -> Dict[str, float]:
        """RETURN EMOTIONAL STATE AS LABELED DICTIONARY"""
        return dict(zip(self.emotion_labels, self.emotions.tolist()))
    
    def get_state_probs(self) -> torch.Tensor:
        """RETURN CURRENT EMOTIONAL STATE AS PROBABILITIES"""
        return torch.sigmoid(self.emotions)
    
    def get_state_probs_dict(self) -> Dict[str, float]:
        """RETURN EMOTIONAL STATE AS LABELED PROBABILITY DICTIONARY"""
        probs = self.get_state_probs()
        return dict(zip(self.emotion_labels, probs.tolist()))
    
    def apply_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        APPLY EMOTIONAL BIAS TO MODEL LOGITS
        
        EMOTIONS BIAS THE PROBABILITY DISTRIBUTION OF OUTPUT TOKENS
        """
        if logits.dim() == 1:
            # SINGLE TOKEN LOGITS
            bias = self.emotions.mean() * self.config.bias_strength
            biased_logits = logits + bias
        else:
            # BATCH OF LOGITS
            bias = self.emotions.mean() * self.config.bias_strength
            biased_logits = logits + bias
            
        return biased_logits
    
    def apply_to_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        APPLY EMOTIONAL BIAS TO EMBEDDINGS
        
        DESIGN: PROJECT 3D EMOTION SPACE ONTO INPUT EMBEDDINGS
        EMOTIONS MODIFY THE SEMANTIC SPACE OF INPUT REPRESENTATIONS
        """
        # PROJECT EMOTIONS TO EMBEDDING SPACE
        # SIMPLE LINEAR PROJECTION: EMOTIONS INFLUENCE FIRST DIMENSIONS
        bias_dim = min(self.config.dimension, embeddings.size(-1))
        
        # CREATE EMOTIONAL BIAS VECTOR
        emotion_bias = self.emotions[:bias_dim].unsqueeze(0).expand(embeddings.size(0), -1)
        
        # APPLY BIAS TO EMBEDDINGS
        embeddings[..., :bias_dim] += emotion_bias * self.config.bias_strength
        
        return embeddings
    
    def update(self, inputs: torch.Tensor, outputs: torch.Tensor, 
               text_output: Optional[str] = None, emotion_gradient: Optional[torch.Tensor] = None) -> None:
        """
        UPDATE EMOTIONAL STATE BASED ON INPUTS, OUTPUTS, OR DIRECT EMOTION GRADIENT
        
        PRIORITY: emotion_gradient > text_output inference > fallback random
        """
        if emotion_gradient is not None:
            # DIRECT EMOTION GRADIENT PROVIDED
            if emotion_gradient.size(0) != self.config.dimension:
                logger.error(f"EMOTION GRADIENT WRONG DIMENSIONS: {emotion_gradient.size(0)} != {self.config.dimension}")
                emotion_gradient = torch.randn(self.config.dimension) * 0.1
            
            # UPDATE EMOTIONS WITH PROVIDED GRADIENT
            self.emotions += emotion_gradient * self.config.update_sensitivity
            
        elif text_output and self.emotion_inference:
            # USE EMOTION INFERENCE COMPONENT
            inferred_gradient = self.emotion_inference.infer(text_output)
            
            # VALIDATE EMOTION DIMENSIONS
            if inferred_gradient.size(0) != self.config.dimension:
                logger.error(f"EMOTION INFERENCE RETURNED WRONG DIMENSIONS: {inferred_gradient.size(0)} != {self.config.dimension}")
                inferred_gradient = torch.randn(self.config.dimension) * 0.1
            
            # UPDATE EMOTIONS WITH INFERRED VALUES
            self.emotions += inferred_gradient * self.config.update_sensitivity
            
        else:
            # FALLBACK: SIMPLIFIED RANDOM UPDATES
            # INFER EMOTION FROM INPUTS/OUTPUTS (SIMPLIFIED)
            input_impact = torch.randn(self.config.dimension) * 0.1 * self.config.update_sensitivity
            output_impact = torch.randn(self.config.dimension) * 0.1 * self.config.update_sensitivity
            
            # COMBINE IMPACTS
            total_impact = input_impact + output_impact
            
            # UPDATE EMOTIONS
            self.emotions += total_impact
        
        # APPLY DECAY
        self.emotions *= self.config.decay_rate
        
        # CLAMP TO VALID RANGE
        self.emotions = torch.clamp(self.emotions, -self.config.max_intensity, self.config.max_intensity)
        
        # RECORD EMOTIONAL STATE (RAW VALUES, NOT PROBABILITIES)
        self.emotion_history.append(self.emotions.clone())
        
        logger.debug(f"UPDATED EMOTIONS: {self.get_state_dict()}")
    
    def get_emotional_context(self) -> str:
        """GENERATE TEXTUAL DESCRIPTION OF CURRENT EMOTIONAL STATE"""
        dominant_emotions = []
        for i, intensity in enumerate(self.emotions):
            if abs(intensity) > 0.3:
                emotion = self.emotion_labels[i]
                if intensity > 0:
                    dominant_emotions.append(f"feeling {emotion}")
                else:
                    dominant_emotions.append(f"feeling low {emotion}")
        
        if not dominant_emotions:
            return "feeling neutral"
        
        return f"currently {' and '.join(dominant_emotions)}"
    
    def reset(self) -> None:
        """RESET EMOTIONAL STATE TO NEUTRAL"""
        self.emotions = torch.zeros(self.config.dimension)
        self.emotion_history = []
        logger.info("RESET EMOTIONAL STATE TO NEUTRAL")
    
    def get_emotion_trajectory(self) -> np.ndarray:
        """RETURN EMOTIONAL TRAJECTORY OVER TIME FOR VISUALIZATION"""
        if not self.emotion_history:
            return np.array([])
        return torch.stack(self.emotion_history).numpy()
    
    def set_emotion_inference(self, inference_component: EmotionInference):
        """SET EMOTION INFERENCE COMPONENT"""
        self.emotion_inference = inference_component
        
        # ENSURE THE INFERENCE COMPONENT HAS ACCESS TO THE EMOTION LABELS
        if hasattr(inference_component, 'config') and hasattr(self, 'emotion_labels'):
            # UPDATE THE INFERENCE COMPONENT'S CONFIG WITH OUR LABELS IF IT DOESN'T HAVE THEM
            if not inference_component.config.labels:
                inference_component.config.labels = self.emotion_labels.copy()
                logger.info(f"UPDATED INFERENCE COMPONENT WITH EMOTION LABELS: {', '.join(self.emotion_labels)}")
        
        logger.info(f"SET EMOTION INFERENCE COMPONENT: {type(inference_component).__name__}")
    
    def set_run_directory(self, run_dir: str):
        """SET RUN DIRECTORY FOR EMOTION INFERENCE CACHING"""
        if self.emotion_inference and hasattr(self.emotion_inference, 'set_run_directory'):
            self.emotion_inference.set_run_directory(run_dir)
            logger.info(f"SET RUN DIRECTORY FOR EMOTION CACHING: {run_dir}")
    
    def infer_emotion(self, text: str) -> torch.Tensor:
        """INFER EMOTION FROM TEXT INPUT USING EMOTION INFERENCE COMPONENT"""
        if self.emotion_inference:
            try:
                emotion_vector = self.emotion_inference.infer(text)
                logger.debug(f"INFERRED EMOTION VECTOR: {emotion_vector}")
                return emotion_vector
            except Exception as e:
                logger.error(f"EMOTION INFERENCE FAILED: {e}")
                # FALLBACK TO RANDOM EMOTION
                return torch.randn(self.config.dimension) * 0.1
        else:
            logger.warning("NO EMOTION INFERENCE COMPONENT SET - RETURNING RANDOM EMOTION")
            return torch.randn(self.config.dimension) * 0.1
