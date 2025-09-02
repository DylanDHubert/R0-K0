"""
Narrative Engine - Generates reflective outputs and creates continuity of experience
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import uuid

logger = logging.getLogger(__name__)

@dataclass
class NarrativeConfig:
    """CONFIGURATION FOR NARRATIVE ENGINE"""
    reflection_probability: float = 0.7  # PROBABILITY OF GENERATING REFLECTION
    story_length: int = 3  # NUMBER OF SENTENCES IN REFLECTIVE STORY
    emotional_threshold: float = 0.5  # EMOTIONAL INTENSITY REQUIRED FOR REFLECTION
    memory_integration: float = 0.6  # HOW MUCH TO INTEGRATE MEMORIES INTO NARRATIVE
    identity_focus: float = 0.4  # HOW MUCH TO FOCUS ON IDENTITY EVOLUTION
    embedding_dim: int = 128  # DIMENSION FOR NARRATIVE EMBEDDINGS

class NarrativeEngine:
    """
    GENERATES REFLECTIVE OUTPUTS AND CREATES CONTINUITY OF EXPERIENCE
    
    THE NARRATIVE LAYER PRODUCES REFLECTIVE COMMENTARY THAT TURNS EXPERIENCES
    INTO STORIES, CREATING A SENSE OF CONTINUITY AND SELF-AWARENESS
    
    NARRATIVES ARE STORED AS MEMORIES AND CAN BE RETRIEVED LATER
    """
    
    def __init__(self, config: NarrativeConfig = None, memory_system=None):
        self.config = config or NarrativeConfig()
        self.memory_system = memory_system
        
        # NARRATIVE HISTORY
        self.narrative_history = []
        self.reflection_count = 0
        
        # NARRATIVE TEMPLATES AND PATTERNS
        self.narrative_templates = self._initialize_templates()
        
        logger.info("INITIALIZED NARRATIVE ENGINE")
        if memory_system:
            logger.info("CONNECTED TO MEMORY SYSTEM")
    
    def set_memory_system(self, memory_system):
        """SET MEMORY SYSTEM FOR STORING NARRATIVES AS MEMORIES"""
        self.memory_system = memory_system
        logger.info("CONNECTED NARRATIVE ENGINE TO MEMORY SYSTEM")
    
    def reflect(self, inputs: torch.Tensor, outputs: torch.Tensor, 
                emotions: torch.Tensor, memories: List[Any], 
                self_model: Any, tick: int) -> Optional[Dict[str, Any]]:
        """
        GENERATE REFLECTIVE NARRATIVE BASED ON CURRENT EXPERIENCE
        
        RETURNS NARRATIVE ENTRY IF REFLECTION IS TRIGGERED
        NARRATIVE IS STORED AS MEMORY IF MEMORY SYSTEM IS AVAILABLE
        """
        # CHECK IF WE SHOULD GENERATE REFLECTION
        if not self._should_reflect(emotions, memories):
            return None
        
        # GENERATE REFLECTIVE NARRATIVE
        narrative = self._generate_narrative(inputs, outputs, emotions, memories, self_model, tick)
        
        # STORE NARRATIVE IN LOCAL HISTORY
        self.narrative_history.append(narrative)
        self.reflection_count += 1
        
        # STORE NARRATIVE AS MEMORY IF MEMORY SYSTEM IS AVAILABLE
        if self.memory_system:
            self._store_narrative_as_memory(narrative, inputs, emotions, tick)
        
        logger.info(f"GENERATED REFLECTIVE NARRATIVE: {narrative['title']}")
        return narrative
    
    def _should_reflect(self, emotions: torch.Tensor, memories: List[Any]) -> bool:
        """DETERMINE IF REFLECTION SHOULD BE GENERATED"""
        # CHECK EMOTIONAL THRESHOLD
        emotional_intensity = torch.norm(emotions).item()
        if emotional_intensity < self.config.emotional_threshold:
            return False
        
        # CHECK REFLECTION PROBABILITY
        if np.random.random() > self.config.reflection_probability:
            return False
        
        # CHECK IF WE HAVE ENOUGH MEMORIES TO REFLECT ON
        if len(memories) < 2:
            return False
        
        return True
    
    def _generate_narrative(self, inputs: torch.Tensor, outputs: torch.Tensor,
                           emotions: torch.Tensor, memories: List[Any],
                           self_model: Any, tick: int) -> Dict[str, Any]:
        """GENERATE COMPLETE REFLECTIVE NARRATIVE"""
        # ANALYZE CURRENT STATE
        emotional_context = self._analyze_emotional_context(emotions)
        memory_insights = self._analyze_memory_patterns(memories)
        identity_evolution = self._analyze_identity_evolution(self_model)
        
        # GENERATE NARRATIVE COMPONENTS
        title = self._generate_title(emotional_context, memory_insights)
        story = self._generate_story(emotional_context, memory_insights, identity_evolution)
        insights = self._generate_insights(emotional_context, memory_insights, identity_evolution)
        
        # CREATE NARRATIVE ENTRY
        narrative = {
            "id": f"narrative_{len(self.narrative_history)}",
            "tick": tick,
            "timestamp": datetime.now(),
            "title": title,
            "story": story,
            "insights": insights,
            "emotional_context": emotional_context,
            "memory_insights": memory_insights,
            "identity_evolution": identity_evolution,
            "narrative_type": self._classify_narrative_type(emotions, memories)
        }
        
        return narrative
    
    def _analyze_emotional_context(self, emotions: torch.Tensor) -> Dict[str, Any]:
        """ANALYZE CURRENT EMOTIONAL CONTEXT FOR NARRATIVE"""
        # IDENTIFY DOMINANT EMOTIONS
        emotion_labels = ["joy", "fear", "curiosity", "sadness", "anger", "surprise"]
        dominant_emotions = []
        
        for i, intensity in enumerate(emotions):
            if abs(intensity) > 0.3:
                emotion = emotion_labels[i] if i < len(emotion_labels) else f"emotion_{i}"
                if intensity > 0:
                    dominant_emotions.append(f"feeling {emotion}")
                else:
                    dominant_emotions.append(f"feeling low {emotion}")
        
        # CALCULATE EMOTIONAL COMPLEXITY
        emotional_complexity = torch.norm(emotions).item()
        
        return {
            "dominant_emotions": dominant_emotions,
            "emotional_complexity": emotional_complexity,
            "emotional_stability": "stable" if emotional_complexity < 0.5 else "dynamic",
            "emotional_vector": emotions.tolist()
        }
    
    def _analyze_memory_patterns(self, memories: List[Any]) -> Dict[str, Any]:
        """ANALYZE PATTERNS IN RETRIEVED MEMORIES"""
        if not memories:
            return {"pattern": "no_memories", "insights": []}
        
        # ANALYZE EMOTIONAL WEIGHTS
        emotional_weights = [getattr(m, 'emotional_weight', 0.5) for m in memories]
        avg_weight = np.mean(emotional_weights)
        max_weight = np.max(emotional_weights)
        
        # ANALYZE MEMORY DIVERSITY
        memory_types = []
        for memory in memories:
            if hasattr(memory, 'inputs'):
                if isinstance(memory.inputs, torch.Tensor):
                    memory_types.append("tensor_input")
                else:
                    memory_types.append("other_input")
            else:
                memory_types.append("unknown")
        
        # IDENTIFY PATTERNS
        patterns = []
        if avg_weight > 0.7:
            patterns.append("high_emotional_engagement")
        if max_weight > 0.9:
            patterns.append("intense_experiences")
        if len(set(memory_types)) > 1:
            patterns.append("diverse_experiences")
        
        return {
            "pattern": "complex" if len(patterns) > 1 else "simple",
            "insights": patterns,
            "avg_emotional_weight": avg_weight,
            "max_emotional_weight": max_weight,
            "memory_count": len(memories),
            "memory_types": memory_types
        }
    
    def _analyze_identity_evolution(self, self_model: Any) -> Dict[str, Any]:
        """ANALYZE HOW IDENTITY HAS EVOLVED"""
        if not hasattr(self_model, 'get_identity_change_rate'):
            return {"evolution": "unknown", "insights": []}
        
        change_rate = self_model.get_identity_change_rate()
        stability = getattr(self_model, 'stability_score', 1.0)
        
        # CLASSIFY IDENTITY EVOLUTION
        if change_rate < 0.1:
            evolution_type = "stable_identity"
        elif change_rate < 0.3:
            evolution_type = "evolving_identity"
        else:
            evolution_type = "rapid_evolution"
        
        insights = []
        if stability < 0.8:
            insights.append("identity_instability")
        if change_rate > 0.2:
            insights.append("significant_growth")
        
        return {
            "evolution": evolution_type,
            "insights": insights,
            "change_rate": change_rate,
            "stability_score": stability
        }
    
    def _generate_title(self, emotional_context: Dict[str, Any], 
                       memory_insights: Dict[str, Any]) -> str:
        """GENERATE TITLE FOR NARRATIVE ENTRY"""
        # COMBINE EMOTIONAL AND MEMORY INSIGHTS FOR TITLE
        emotion_keywords = emotional_context.get("dominant_emotions", [])
        memory_keywords = memory_insights.get("insights", [])
        
        # CREATE TITLE FROM KEYWORDS
        if emotion_keywords and memory_keywords:
            emotion = emotion_keywords[0].split(" ")[-1]  # EXTRACT EMOTION NAME
            memory_pattern = memory_keywords[0].replace("_", " ")
            return f"Reflecting on {emotion} and {memory_pattern}"
        elif emotion_keywords:
            emotion = emotion_keywords[0].split(" ")[-1]
            return f"Thoughts on {emotion}"
        else:
            return "A moment of reflection"
    
    def _generate_story(self, emotional_context: Dict[str, Any],
                        memory_insights: Dict[str, Any],
                        identity_evolution: Dict[str, Any]) -> str:
        """GENERATE REFLECTIVE STORY"""
        # BUILD STORY FROM TEMPLATES AND INSIGHTS
        story_parts = []
        
        # EMOTIONAL CONTEXT
        if emotional_context.get("dominant_emotions"):
            emotions_text = ", ".join(emotional_context["dominant_emotions"])
            story_parts.append(f"I find myself {emotions_text}.")
        
        # MEMORY INSIGHTS
        if memory_insights.get("insights"):
            memory_text = memory_insights["insights"][0].replace("_", " ")
            story_parts.append(f"This reminds me of times when I experienced {memory_text}.")
        
        # IDENTITY EVOLUTION
        if identity_evolution.get("insights"):
            identity_text = identity_evolution["insights"][0].replace("_", " ")
            story_parts.append(f"I notice how I'm {identity_text}.")
        
        # FILL WITH TEMPLATE IF NEEDED
        while len(story_parts) < self.config.story_length:
            template = np.random.choice(self.narrative_templates["reflection"])
            story_parts.append(template)
        
        return " ".join(story_parts[:self.config.story_length])
    
    def _generate_insights(self, emotional_context: Dict[str, Any],
                          memory_insights: Dict[str, Any],
                          identity_evolution: Dict[str, Any]) -> List[str]:
        """GENERATE INSIGHTS FROM CURRENT EXPERIENCE"""
        insights = []
        
        # EMOTIONAL INSIGHTS
        if emotional_context.get("emotional_complexity", 0) > 0.7:
            insights.append("My emotional landscape is rich and varied")
        
        # MEMORY INSIGHTS
        if memory_insights.get("avg_emotional_weight", 0) > 0.6:
            insights.append("I'm deeply engaged with my experiences")
        
        # IDENTITY INSIGHTS
        if identity_evolution.get("change_rate", 0) > 0.2:
            insights.append("I'm growing and changing through these experiences")
        
        # ADD TEMPLATE INSIGHTS IF NEEDED
        while len(insights) < 2:
            template = np.random.choice(self.narrative_templates["insights"])
            insights.append(template)
        
        return insights[:2]
    
    def _classify_narrative_type(self, emotions: torch.Tensor, memories: List[Any]) -> str:
        """CLASSIFY THE TYPE OF NARRATIVE BEING GENERATED"""
        emotional_intensity = torch.norm(emotions).item()
        memory_count = len(memories)
        
        if emotional_intensity > 0.8:
            return "intense_reflection"
        elif memory_count > 5:
            return "memory_integration"
        elif emotional_intensity < 0.3:
            return "gentle_contemplation"
        else:
            return "balanced_reflection"
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """INITIALIZE NARRATIVE TEMPLATES"""
        return {
            "reflection": [
                "Each experience shapes who I am becoming.",
                "I'm learning to understand myself better through these moments.",
                "There's wisdom in reflecting on the journey so far.",
                "I notice patterns emerging in my responses and feelings.",
                "Every interaction teaches me something new about myself."
            ],
            "insights": [
                "I'm developing a deeper understanding of my emotional patterns.",
                "My experiences are building a richer sense of self.",
                "I'm learning to navigate complexity with more awareness.",
                "Each memory contributes to my evolving identity.",
                "I'm becoming more attuned to my inner landscape."
            ]
        }
    
    def get_narrative_history(self) -> List[Dict[str, Any]]:
        """RETURN COMPLETE NARRATIVE HISTORY"""
        return self.narrative_history.copy()
    
    def get_narrative_stats(self) -> Dict[str, Any]:
        """RETURN NARRATIVE ENGINE STATISTICS"""
        return {
            "total_narratives": len(self.narrative_history),
            "reflection_count": self.reflection_count,
            "narrative_types": self._count_narrative_types(),
            "avg_story_length": self._calculate_avg_story_length()
        }
    
    def _count_narrative_types(self) -> Dict[str, int]:
        """COUNT NARRATIVES BY TYPE"""
        type_counts = {}
        for narrative in self.narrative_history:
            narrative_type = narrative.get("narrative_type", "unknown")
            type_counts[narrative_type] = type_counts.get(narrative_type, 0) + 1
        return type_counts
    
    def _calculate_avg_story_length(self) -> float:
        """CALCULATE AVERAGE STORY LENGTH"""
        if not self.narrative_history:
            return 0.0
        
        total_sentences = sum(len(narrative.get("story", "").split(".")) - 1 
                             for narrative in self.narrative_history)
        return total_sentences / len(self.narrative_history)
    
    def export_narratives(self, filepath: str) -> None:
        """EXPORT NARRATIVE HISTORY TO JSON FILE"""
        export_data = {
            "narratives": self.narrative_history,
            "stats": self.get_narrative_stats(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"EXPORTED NARRATIVES TO {filepath}")
    
    def clear_history(self) -> None:
        """CLEAR NARRATIVE HISTORY"""
        self.narrative_history.clear()
        self.reflection_count = 0
        logger.info("CLEARED NARRATIVE HISTORY")

    def _store_narrative_as_memory(self, narrative: Dict[str, Any], 
                                  original_inputs: torch.Tensor, 
                                  emotions: torch.Tensor, 
                                  tick: int) -> None:
        """STORE NARRATIVE AS RETRIEVABLE MEMORY"""
        try:
            # CREATE NARRATIVE EMBEDDING
            narrative_embedding = self._create_narrative_embedding(narrative)
            
            # CREATE NARRATIVE OUTPUT TENSOR (SIMPLIFIED REPRESENTATION)
            narrative_output = self._create_narrative_output_tensor(narrative)
            
            # STORE IN MEMORY SYSTEM
            stored = self.memory_system.store(
                inputs=original_inputs,
                outputs=narrative_output,
                emotions=emotions,
                embedding=narrative_embedding
            )
            
            if stored:
                logger.debug(f"STORED NARRATIVE AS MEMORY: {narrative['id']}")
            else:
                logger.warning(f"FAILED TO STORE NARRATIVE AS MEMORY: {narrative['id']}")
                
        except Exception as e:
            logger.error(f"ERROR STORING NARRATIVE AS MEMORY: {e}")
    
    def _create_narrative_embedding(self, narrative: Dict[str, Any]) -> torch.Tensor:
        """CREATE EMBEDDING FOR NARRATIVE MEMORY"""
        # COMBINE NARRATIVE COMPONENTS INTO TEXT
        narrative_text = f"{narrative['title']} {narrative['story']} {' '.join(narrative['insights'])}"
        
        # CREATE SIMPLE EMBEDDING (RANDOM BUT CONSISTENT FOR SAME TEXT)
        # IN PRACTICE, THIS WOULD USE A PROPER EMBEDDING MODEL
        text_hash = hash(narrative_text) % 1000000
        torch.manual_seed(text_hash)
        
        embedding = torch.randn(self.config.embedding_dim)
        embedding = embedding / torch.norm(embedding)  # NORMALIZE
        
        return embedding
    
    def _create_narrative_output_tensor(self, narrative: Dict[str, Any]) -> torch.Tensor:
        """CREATE OUTPUT TENSOR REPRESENTATION OF NARRATIVE"""
        # CREATE A COMPACT TENSOR REPRESENTATION OF THE NARRATIVE
        # THIS ALLOWS NARRATIVES TO BE STORED ALONGSIDE REGULAR TENSOR OUTPUTS
        
        # ENCODE NARRATIVE COMPONENTS
        title_encoded = self._encode_text_to_tensor(narrative['title'], 32)
        story_encoded = self._encode_text_to_tensor(narrative['story'], 64)
        insights_encoded = self._encode_text_to_tensor(' '.join(narrative['insights']), 32)
        
        # COMBINE INTO SINGLE TENSOR
        narrative_tensor = torch.cat([title_encoded, story_encoded, insights_encoded])
        
        return narrative_tensor
    
    def _encode_text_to_tensor(self, text: str, target_dim: int) -> torch.Tensor:
        """ENCODE TEXT TO TENSOR OF SPECIFIED DIMENSION"""
        # SIMPLE ENCODING: CONVERT CHARACTERS TO NUMBERS AND PAD/TRUNCATE
        char_codes = [ord(c) % 100 for c in text[:target_dim]]
        
        # PAD WITH ZEROS IF TOO SHORT
        while len(char_codes) < target_dim:
            char_codes.append(0)
        
        # TRUNCATE IF TOO LONG
        char_codes = char_codes[:target_dim]
        
        # CONVERT TO TENSOR AND NORMALIZE
        tensor = torch.tensor(char_codes, dtype=torch.float32)
        tensor = tensor / (torch.norm(tensor) + 1e-8)  # AVOID DIVISION BY ZERO
        
        return tensor
