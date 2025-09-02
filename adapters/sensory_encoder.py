"""
Sensory Encoder Adapter - Unified interface for different sensory encoders
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class SensoryEncoderAdapter(ABC):
    """
    ABSTRACT BASE CLASS FOR SENSORY ENCODERS
    
    PROVIDES UNIFIED INTERFACE FOR DIFFERENT SENSORY MODALITIES (VISION, AUDIO, TEXT)
    """
    
    def __init__(self, encoder_name: str, device: str = "cpu"):
        self.encoder_name = encoder_name
        self.device = device
        self.encoder = None
        self.feature_dim = None
        
        logger.info(f"INITIALIZING SENSORY ENCODER ADAPTER FOR {encoder_name}")
    
    @abstractmethod
    def load_encoder(self) -> None:
        """LOAD THE UNDERLYING ENCODER MODEL"""
        pass
    
    @abstractmethod
    def encode(self, inputs: Any) -> torch.Tensor:
        """ENCODE SENSORY INPUTS TO EMBEDDINGS"""
        pass
    
    @abstractmethod
    def get_feature_dimension(self) -> int:
        """RETURN THE DIMENSION OF ENCODED FEATURES"""
        pass
    
    def process_inputs(self, inputs: Any) -> torch.Tensor:
        """
        PROCESS SENSORY INPUTS THROUGH THE ENCODER
        
        THIS IS THE MAIN INTERFACE FOR THE AGENT LOOP
        """
        # VALIDATE INPUTS
        if not self._validate_inputs(inputs):
            raise ValueError(f"INVALID INPUT TYPE FOR {self.encoder_name}")
        
        # ENCODE INPUTS
        embeddings = self.encode(inputs)
        
        # VALIDATE OUTPUTS
        if not self._validate_embeddings(embeddings):
            raise ValueError(f"INVALID EMBEDDING OUTPUT FROM {self.encoder_name}")
        
        return embeddings
    
    def _validate_inputs(self, inputs: Any) -> bool:
        """VALIDATE THAT INPUTS ARE COMPATIBLE WITH THIS ENCODER"""
        # BASE IMPLEMENTATION - SUBCLASSES SHOULD OVERRIDE
        return inputs is not None
    
    def _validate_embeddings(self, embeddings: torch.Tensor) -> bool:
        """VALIDATE THAT EMBEDDINGS HAVE EXPECTED SHAPE"""
        if not isinstance(embeddings, torch.Tensor):
            return False
        
        if embeddings.dim() < 1:
            return False
        
        if self.feature_dim and embeddings.size(-1) != self.feature_dim:
            return False
        
        return True
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """RETURN INFORMATION ABOUT THE LOADED ENCODER"""
        if self.encoder is None:
            return {"status": "encoder_not_loaded"}
        
        info = {
            "encoder_name": self.encoder_name,
            "device": self.device,
            "encoder_type": type(self.encoder).__name__,
            "feature_dimension": self.get_feature_dimension()
        }
        
        return info
    
    def is_loaded(self) -> bool:
        """CHECK IF ENCODER IS LOADED"""
        return self.encoder is not None
    
    def to_device(self, device: str) -> None:
        """MOVE ENCODER TO SPECIFIED DEVICE"""
        if self.encoder is not None:
            self.encoder = self.encoder.to(device)
            self.device = device
            logger.info(f"MOVED ENCODER TO DEVICE: {device}")
    
    def unload(self) -> None:
        """UNLOAD ENCODER TO FREE MEMORY"""
        if self.encoder is not None:
            del self.encoder
            self.encoder = None
            logger.info("UNLOADED ENCODER")
    
    def __del__(self):
        """CLEANUP WHEN ADAPTER IS DESTROYED"""
        self.unload()

class TextEncoderAdapter(SensoryEncoderAdapter):
    """TEXT ENCODER ADAPTER FOR SENTENCE TRANSFORMERS"""
    
    def __init__(self, encoder_name: str, device: str = "cpu"):
        super().__init__(encoder_name, device)
        self.max_length = 512
    
    def load_encoder(self) -> None:
        """LOAD SENTENCE TRANSFORMER ENCODER"""
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.encoder_name, device=self.device)
            self.feature_dim = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"LOADED TEXT ENCODER: {self.encoder_name}")
        except ImportError:
            logger.error("SENTENCE_TRANSFORMERS NOT INSTALLED")
            raise
    
    def encode(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        """ENCODE TEXT INPUTS TO EMBEDDINGS"""
        if not self.is_loaded():
            self.load_encoder()
        
        # HANDLE SINGLE STRING OR LIST OF STRINGS
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # ENCODE USING SENTENCE TRANSFORMER
        embeddings = self.encoder.encode(inputs, convert_to_tensor=True)
        
        return embeddings
    
    def get_feature_dimension(self) -> int:
        """RETURN THE DIMENSION OF ENCODED FEATURES"""
        if self.feature_dim is None and self.is_loaded():
            self.feature_dim = self.encoder.get_sentence_embedding_dimension()
        return self.feature_dim or 768  # DEFAULT DIMENSION
    
    def _validate_inputs(self, inputs: Any) -> bool:
        """VALIDATE TEXT INPUTS"""
        if isinstance(inputs, str):
            return len(inputs.strip()) > 0
        elif isinstance(inputs, list):
            return all(isinstance(item, str) and len(item.strip()) > 0 for item in inputs)
        return False

class VisionEncoderAdapter(SensoryEncoderAdapter):
    """VISION ENCODER ADAPTER FOR CLIP"""
    
    def __init__(self, encoder_name: str, device: str = "cpu"):
        super().__init__(encoder_name, device)
        self.image_size = 224
    
    def load_encoder(self) -> None:
        """LOAD CLIP VISION ENCODER"""
        try:
            import clip
            self.encoder, _ = clip.load(self.encoder_name, device=self.device)
            self.feature_dim = 512  # CLIP FEATURE DIMENSION
            logger.info(f"LOADED VISION ENCODER: {self.encoder_name}")
        except ImportError:
            logger.error("CLIP NOT INSTALLED")
            raise
    
    def encode(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """ENCODE VISION INPUTS TO EMBEDDINGS"""
        if not self.is_loaded():
            self.load_encoder()
        
        # HANDLE SINGLE TENSOR OR LIST OF TENSORS
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)  # ADD BATCH DIMENSION
            inputs = [inputs]
        
        # PROCESS EACH IMAGE
        embeddings = []
        for image in inputs:
            # PREPROCESS IMAGE
            processed_image = self._preprocess_image(image)
            
            # ENCODE USING CLIP
            with torch.no_grad():
                image_features = self.encoder.encode_image(processed_image)
                embeddings.append(image_features)
        
        # STACK EMBEDDINGS
        return torch.stack(embeddings)
    
    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """PREPROCESS IMAGE FOR CLIP ENCODER"""
        # SIMPLIFIED PREPROCESSING - IN PRACTICE, YOU'D USE CLIP'S TRANSFORMS
        if image.dim() == 3:
            # CHW FORMAT
            if image.size(0) == 3:  # RGB
                pass
            elif image.size(-1) == 3:  # HWC FORMAT
                image = image.permute(2, 0, 1)
            else:
                raise ValueError("UNSUPPORTED IMAGE FORMAT")
        
        # NORMALIZE TO [0, 1] RANGE
        if image.max() > 1.0:
            image = image / 255.0
        
        # RESIZE TO CLIP INPUT SIZE (SIMPLIFIED)
        # IN PRACTICE, YOU'D USE PROPER RESIZING AND NORMALIZATION
        
        return image
    
    def get_feature_dimension(self) -> int:
        """RETURN THE DIMENSION OF ENCODED FEATURES"""
        return self.feature_dim or 512
    
    def _validate_inputs(self, inputs: Any) -> bool:
        """VALIDATE VISION INPUTS"""
        if isinstance(inputs, torch.Tensor):
            return inputs.dim() >= 2
        elif isinstance(inputs, list):
            return all(isinstance(item, torch.Tensor) and item.dim() >= 2 for item in inputs)
        return False

class AudioEncoderAdapter(SensoryEncoderAdapter):
    """AUDIO ENCODER ADAPTER FOR WHISPER"""
    
    def __init__(self, encoder_name: str, device: str = "cpu"):
        super().__init__(encoder_name, device)
        self.sample_rate = 16000
    
    def load_encoder(self) -> None:
        """LOAD WHISPER AUDIO ENCODER"""
        try:
            import whisper
            self.encoder = whisper.load_model(self.encoder_name, device=self.device)
            self.feature_dim = 1024  # WHISPER FEATURE DIMENSION
            logger.info(f"LOADED AUDIO ENCODER: {self.encoder_name}")
        except ImportError:
            logger.error("WHISPER NOT INSTALLED")
            raise
    
    def encode(self, inputs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """ENCODE AUDIO INPUTS TO EMBEDDINGS"""
        if not self.is_loaded():
            self.load_encoder()
        
        # HANDLE SINGLE TENSOR OR LIST OF TENSORS
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)  # ADD BATCH DIMENSION
            inputs = [inputs]
        
        # PROCESS EACH AUDIO SAMPLE
        embeddings = []
        for audio in inputs:
            # PREPROCESS AUDIO
            processed_audio = self._preprocess_audio(audio)
            
            # ENCODE USING WHISPER
            with torch.no_grad():
                # EXTRACT AUDIO FEATURES
                audio_features = self.encoder.encoder(processed_audio)
                
                # POOL FEATURES (SIMPLIFIED)
                audio_embedding = torch.mean(audio_features, dim=1)
                embeddings.append(audio_embedding)
        
        # STACK EMBEDDINGS
        return torch.stack(embeddings)
    
    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """PREPROCESS AUDIO FOR WHISPER ENCODER"""
        # SIMPLIFIED PREPROCESSING - IN PRACTICE, YOU'D USE WHISPER'S TRANSFORMS
        
        # ENSURE CORRECT SHAPE
        if audio.dim() == 1:
            # MONO AUDIO
            pass
        elif audio.dim() == 2:
            # STEREO AUDIO - CONVERT TO MONO
            audio = torch.mean(audio, dim=0)
        else:
            raise ValueError("UNSUPPORTED AUDIO FORMAT")
        
        # NORMALIZE AUDIO
        if audio.max() > 1.0:
            audio = audio / audio.max()
        
        # RESAMPLE TO WHISPER SAMPLE RATE (SIMPLIFIED)
        # IN PRACTICE, YOU'D USE PROPER RESAMPLING
        
        return audio
    
    def get_feature_dimension(self) -> int:
        """RETURN THE DIMENSION OF ENCODED FEATURES"""
        return self.feature_dim or 1024
    
    def _validate_inputs(self, inputs: Any) -> bool:
        """VALIDATE AUDIO INPUTS"""
        if isinstance(inputs, torch.Tensor):
            return inputs.dim() >= 1
        elif isinstance(inputs, list):
            return all(isinstance(item, torch.Tensor) and item.dim() >= 1 for item in inputs)
        return False
