"""
Base Model Adapter - Unified interface for different LLM backends
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseModelAdapter(ABC):
    """
    ABSTRACT BASE CLASS FOR LLM ADAPTERS
    
    PROVIDES UNIFIED INTERFACE FOR DIFFERENT LLM BACKENDS (GPT-2, LLaMA, MISTRAL, ETC.)
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        logger.info(f"INITIALIZING BASE MODEL ADAPTER FOR {model_name}")
    
    @abstractmethod
    def load_model(self) -> None:
        """LOAD THE UNDERLYING MODEL AND TOKENIZER"""
        pass
    
    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """ENCODE TEXT TO EMBEDDINGS"""
        pass
    
    @abstractmethod
    def decode(self, embeddings: torch.Tensor) -> str:
        """DECODE EMBEDDINGS BACK TO TEXT"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 100, 
                temperature: float = 1.0) -> str:
        """GENERATE TEXT FROM PROMPT"""
        pass
    
    def forward(self, inputs: torch.Tensor, memories: List[Any] = None,
                emotions: torch.Tensor = None, identity: torch.Tensor = None) -> torch.Tensor:
        """
        FORWARD PASS THROUGH THE MODEL WITH EMOTIONAL AND MEMORY CONTEXT
        
        THIS IS THE MAIN INTERFACE FOR THE AGENT LOOP
        """
        # CONVERT INPUTS TO TEXT IF NEEDED
        if isinstance(inputs, torch.Tensor):
            # SIMPLIFIED: CONVERT TENSOR TO TEXT PROMPT
            prompt = self._tensor_to_prompt(inputs)
        else:
            prompt = str(inputs)
        
        # ENHANCE PROMPT WITH MEMORIES
        if memories:
            prompt = self._enhance_with_memories(prompt, memories)
        
        # ENHANCE PROMPT WITH EMOTIONAL CONTEXT
        if emotions is not None:
            prompt = self._enhance_with_emotions(prompt, emotions)
        
        # ENHANCE PROMPT WITH IDENTITY
        if identity is not None:
            prompt = self._enhance_with_identity(prompt, identity)
        
        # GENERATE RESPONSE
        response = self.generate(prompt, max_length=100, temperature=0.8)
        
        # CONVERT RESPONSE TO TENSOR
        response_tensor = self.encode(response)
        
        return response_tensor
    
    def _tensor_to_prompt(self, tensor: torch.Tensor) -> str:
        """CONVERT TENSOR INPUT TO TEXT PROMPT"""
        # SIMPLIFIED CONVERSION - IN PRACTICE, YOU'D USE A PROPER DECODER
        if tensor.dim() == 1:
            # SINGLE VECTOR
            prompt = f"Processing input vector of dimension {tensor.size(0)}"
        else:
            # BATCH OF VECTORS
            prompt = f"Processing {tensor.size(0)} input vectors of dimension {tensor.size(1)}"
        
        return prompt
    
    def _enhance_with_memories(self, prompt: str, memories: List[Any]) -> str:
        """ENHANCE PROMPT WITH MEMORY CONTEXT"""
        if not memories:
            return prompt
        
        memory_context = "Based on my memories: "
        memory_descriptions = []
        
        for i, memory in enumerate(memories[:3]):  # LIMIT TO 3 MEMORIES
            if hasattr(memory, 'emotional_weight'):
                weight = memory.emotional_weight
                memory_descriptions.append(f"memory_{i} (emotional_weight: {weight:.2f})")
            else:
                memory_descriptions.append(f"memory_{i}")
        
        memory_context += ", ".join(memory_descriptions)
        
        enhanced_prompt = f"{prompt}\n\n{memory_context}"
        return enhanced_prompt
    
    def _enhance_with_emotions(self, prompt: str, emotions: torch.Tensor) -> str:
        """ENHANCE PROMPT WITH EMOTIONAL CONTEXT"""
        if emotions is None:
            return prompt
        
        # CONVERT EMOTIONS TO TEXTUAL DESCRIPTION
        emotion_labels = ["joy", "fear", "curiosity", "sadness", "anger", "surprise"]
        emotion_descriptions = []
        
        for i, intensity in enumerate(emotions):
            if i < len(emotion_labels):
                emotion = emotion_labels[i]
                if abs(intensity) > 0.3:
                    if intensity > 0:
                        emotion_descriptions.append(f"feeling {emotion}")
                    else:
                        emotion_descriptions.append(f"feeling low {emotion}")
        
        if emotion_descriptions:
            emotional_context = f"I am {' and '.join(emotion_descriptions)}."
            enhanced_prompt = f"{prompt}\n\n{emotional_context}"
            return enhanced_prompt
        
        return prompt
    
    def _enhance_with_identity(self, prompt: str, identity: torch.Tensor) -> str:
        """ENHANCE PROMPT WITH IDENTITY CONTEXT"""
        if identity is None:
            return prompt
        
        # SIMPLIFIED IDENTITY CONTEXT
        identity_norm = torch.norm(identity).item()
        
        if identity_norm > 0.8:
            identity_context = "I have a strong sense of self."
        elif identity_norm > 0.5:
            identity_context = "I am developing my sense of self."
        else:
            identity_context = "I am still forming my sense of self."
        
        enhanced_prompt = f"{prompt}\n\n{identity_context}"
        return enhanced_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """RETURN INFORMATION ABOUT THE LOADED MODEL"""
        if self.model is None:
            return {"status": "model_not_loaded"}
        
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "model_type": type(self.model).__name__
        }
        
        # ADD MODEL-SPECIFIC INFORMATION
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'vocab_size'):
                info["vocab_size"] = config.vocab_size
            if hasattr(config, 'hidden_size'):
                info["hidden_size"] = config.hidden_size
            if hasattr(config, 'num_layers'):
                info["num_layers"] = config.num_layers
        
        return info
    
    def is_loaded(self) -> bool:
        """CHECK IF MODEL IS LOADED"""
        return self.model is not None
    
    def to_device(self, device: str) -> None:
        """MOVE MODEL TO SPECIFIED DEVICE"""
        if self.model is not None:
            self.model = self.model.to(device)
            self.device = device
            logger.info(f"MOVED MODEL TO DEVICE: {device}")
    
    def unload(self) -> None:
        """UNLOAD MODEL TO FREE MEMORY"""
        if self.model is not None:
            del self.model
            self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            logger.info("UNLOADED MODEL")
    
    def __del__(self):
        """CLEANUP WHEN ADAPTER IS DESTROYED"""
        self.unload()
