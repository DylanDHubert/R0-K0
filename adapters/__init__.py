"""
Adapters - Interface classes for different LLMs, sensory encoders, and memory backends
"""

from .base_model import BaseModelAdapter
from .sensory_encoder import SensoryEncoderAdapter
from .memory_store import MemoryStoreAdapter

__all__ = [
    "BaseModelAdapter",
    "SensoryEncoderAdapter", 
    "MemoryStoreAdapter"
]
