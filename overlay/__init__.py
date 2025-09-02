"""
Emotion-Weighted Memory LLM - A Plug-and-Play Cognitive Overlay for Pretrained Models
"""

from .core import AgentLoop
from .emotion import EmotionEngine
from .memory import EpisodicMemory
from .self_model import SelfModel
from .sleep import SleepCycle
from .narrative import NarrativeEngine
from .lifespan import LifespanManager

__version__ = "0.1.0"
__all__ = [
    "AgentLoop",
    "EmotionEngine", 
    "EpisodicMemory",
    "SelfModel",
    "SleepCycle",
    "NarrativeEngine",
    "LifespanManager"
]
