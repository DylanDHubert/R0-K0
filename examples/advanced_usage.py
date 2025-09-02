"""
Advanced Usage Example - Demonstrates sophisticated features of the emotion-weighted memory LLM system
"""

import torch
import numpy as np
import logging
from overlay import (
    AgentLoop, EmotionEngine, EpisodicMemory, SelfModel, 
    LifespanManager, SleepCycle, NarrativeEngine
)
from adapters import QdrantMemoryStore, TextEncoderAdapter
from overlay.emotion import EmotionConfig
from overlay.memory import MemoryConfig
from overlay.self_model import SelfModelConfig
from overlay.sleep import SleepConfig
from overlay.narrative import NarrativeConfig
from overlay.lifespan import LifespanConfig

# SETUP LOGGING
logging.basicConfig(level=logging.INFO)

def create_custom_configurations():
    """CREATE CUSTOM CONFIGURATIONS FOR ADVANCED FEATURES"""
    print("⚙️ CREATING CUSTOM CONFIGURATIONS...")
    
    # EMOTION ENGINE WITH CUSTOM CONFIG
    emotion_config = EmotionConfig(
        dimension=8,  # EXTENDED EMOTIONAL DIMENSIONS
        decay_rate=0.98,  # SLOWER EMOTIONAL DECAY
        bias_strength=0.4,  # STRONGER EMOTIONAL BIAS
        update_sensitivity=0.15  # HIGHER EMOTIONAL SENSITIVITY
    )
    
    # MEMORY SYSTEM WITH CUSTOM CONFIG
    memory_config = MemoryConfig(
        max_memories=2000,  # LARGER MEMORY CAPACITY
        storage_threshold=0.05,  # LOWER STORAGE THRESHOLD
        retrieval_k=8,  # RETRIEVE MORE MEMORIES
        emotional_resonance_weight=0.8  # HEAVY EMOTIONAL RESONANCE
    )
    
    # SELF MODEL WITH CUSTOM CONFIG
    self_model_config = SelfModelConfig(
        embedding_dim=256,  # LARGER IDENTITY EMBEDDING
        learning_rate=0.02,  # FASTER IDENTITY EVOLUTION
        stability_factor=0.85,  # LESS STABLE IDENTITY
        emotion_influence=0.4,  # MORE EMOTIONAL INFLUENCE
        memory_influence=0.5,  # MORE MEMORY INFLUENCE
        output_influence=0.1  # LESS OUTPUT INFLUENCE
    )
    
    # SLEEP CYCLE WITH CUSTOM CONFIG
    sleep_config = SleepConfig(
        sleep_interval=8,  # MORE FREQUENT SLEEP
        consolidation_strength=0.3,  # STRONGER CONSOLIDATION
        forgetting_rate=0.15,  # MORE AGGRESSIVE FORGETTING
        dream_probability=0.5,  # HIGHER DREAM PROBABILITY
        max_dreams_per_cycle=5  # MORE DREAMS PER CYCLE
    )
    
    # NARRATIVE ENGINE WITH CUSTOM CONFIG
    narrative_config = NarrativeConfig(
        reflection_probability=0.8,  # HIGHER REFLECTION PROBABILITY
        story_length=5,  # LONGER REFLECTIVE STORIES
        emotional_threshold=0.3,  # LOWER EMOTIONAL THRESHOLD
        memory_integration=0.7,  # MORE MEMORY INTEGRATION
        identity_focus=0.5  # MORE IDENTITY FOCUS
    )
    
    # LIFESPAN MANAGER WITH CUSTOM CONFIG
    lifespan_config = LifespanConfig(
        max_ticks=2000,  # LONGER LIFE
        early_life_threshold=0.15,  # SHORTER INFANCY
        mid_life_threshold=0.7,  # LONGER ADOLESCENCE
        plasticity_decay_rate=0.998,  # SLOWER PLASTICITY DECAY
        memory_dominance_growth=1.008,  # FASTER MEMORY DOMINANCE GROWTH
        identity_stability_growth=1.003  # FASTER IDENTITY STABILITY GROWTH
    )
    
    return {
        "emotion": emotion_config,
        "memory": memory_config,
        "self_model": self_model_config,
        "sleep": sleep_config,
        "narrative": narrative_config,
        "lifespan": lifespan_config
    }

def create_advanced_llm_adapter():
    """CREATE ADVANCED LLM ADAPTER WITH EMOTIONAL CONTEXT INTEGRATION"""
    print("🤖 CREATING ADVANCED LLM ADAPTER...")
    
    class AdvancedLLMAdapter:
        """ADVANCED LLM ADAPTER WITH EMOTIONAL AND MEMORY CONTEXT"""
        
        def __init__(self):
            self.model_name = "advanced_emotion_aware_model"
            self.context_window = []
            self.max_context_length = 1000
        
        def forward(self, inputs, memories=None, emotions=None, identity=None):
            """ADVANCED FORWARD PASS WITH CONTEXT INTEGRATION"""
            
            # CREATE RICH CONTEXT
            context = self._build_context(inputs, memories, emotions, identity)
            
            # SIMULATE ADVANCED PROCESSING
            batch_size = inputs.size(0)
            output_dim = 128  # LARGER OUTPUT DIMENSION
            
            # BASE OUTPUTS
            outputs = torch.randn(batch_size, output_dim)
            
            # APPLY EMOTIONAL BIAS
            if emotions is not None:
                emotional_bias = self._calculate_emotional_bias(emotions, output_dim)
                outputs += emotional_bias * 0.3
            
            # APPLY MEMORY INFLUENCE
            if memories:
                memory_influence = self._calculate_memory_influence(memories, output_dim)
                outputs += memory_influence * 0.2
            
            # APPLY IDENTITY INFLUENCE
            if identity is not None:
                identity_influence = self._calculate_identity_influence(identity, output_dim)
                outputs += identity_influence * 0.15
            
            # APPLY INPUT INFLUENCE
            input_influence = inputs[:, :output_dim] * 0.1
            outputs += input_influence
            
            # ADD CONTEXTUAL VARIANCE
            context_variance = torch.randn_like(outputs) * 0.05
            outputs += context_variance
            
            return outputs
        
        def _build_context(self, inputs, memories, emotions, identity):
            """BUILD RICH CONTEXT FROM ALL COMPONENTS"""
            context = {
                "input_shape": inputs.shape,
                "memory_count": len(memories) if memories else 0,
                "emotional_state": emotions.tolist() if emotions is not None else None,
                "identity_present": identity is not None
            }
            
            # ADD TO CONTEXT WINDOW
            self.context_window.append(context)
            if len(self.context_window) > self.max_context_length:
                self.context_window.pop(0)
            
            return context
        
        def _calculate_emotional_bias(self, emotions, output_dim):
            """CALCULATE EMOTIONAL BIAS FOR OUTPUTS"""
            # PROJECT EMOTIONS TO OUTPUT SPACE
            emotion_projection = torch.randn(emotions.size(0), output_dim)
            emotion_projection = emotion_projection / torch.norm(emotion_projection, dim=1, keepdim=True)
            
            # WEIGHT BY EMOTION INTENSITY
            emotion_weights = torch.norm(emotions, dim=1, keepdim=True)
            emotional_bias = emotion_projection * emotion_weights
            
            return torch.mean(emotional_bias, dim=0)
        
        def _calculate_memory_influence(self, memories, output_dim):
            """CALCULATE MEMORY INFLUENCE FOR OUTPUTS"""
            if not memories:
                return torch.zeros(output_dim)
            
            # EXTRACT MEMORY EMBEDDINGS
            memory_embeddings = []
            memory_weights = []
            
            for memory in memories:
                if hasattr(memory, 'embedding'):
                    embedding = memory.embedding
                else:
                    embedding = torch.randn(output_dim)
                
                weight = getattr(memory, 'emotional_weight', 0.5)
                
                memory_embeddings.append(embedding)
                memory_weights.append(weight)
            
            # COMBINE MEMORIES
            memory_embeddings = torch.stack(memory_embeddings)
            memory_weights = torch.tensor(memory_weights)
            
            # NORMALIZE WEIGHTS
            memory_weights = memory_weights / torch.sum(memory_weights)
            
            # WEIGHTED AVERAGE
            memory_influence = torch.sum(
                memory_embeddings * memory_weights.unsqueeze(1), 
                dim=0
            )
            
            return memory_influence
        
        def _calculate_identity_influence(self, identity, output_dim):
            """CALCULATE IDENTITY INFLUENCE FOR OUTPUTS"""
            # PROJECT IDENTITY TO OUTPUT SPACE
            identity_projection = torch.randn(identity.size(0), output_dim)
            identity_projection = identity_projection / torch.norm(identity_projection, dim=1, keepdim=True)
            
            # AVERAGE ACROSS IDENTITIES
            identity_influence = torch.mean(identity_projection, dim=0)
            
            return identity_influence
        
        def get_context_summary(self):
            """RETURN SUMMARY OF CONTEXT WINDOW"""
            return {
                "context_length": len(self.context_window),
                "recent_contexts": self.context_window[-5:] if self.context_window else []
            }
    
    return AdvancedLLMAdapter()

def create_advanced_sensory_encoders():
    """CREATE ADVANCED SENSORY ENCODERS WITH MULTIMODAL INTEGRATION"""
    print("👁️ CREATING ADVANCED SENSORY ENCODERS...")
    
    class AdvancedTextEncoder:
        """ADVANCED TEXT ENCODER WITH EMOTIONAL CONTEXT"""
        
        def __init__(self):
            self.name = "advanced_text_encoder"
            self.feature_dim = 256
        
        def process_inputs(self, inputs):
            """PROCESS TEXT INPUTS WITH ADVANCED FEATURES"""
            if isinstance(inputs, str):
                # ANALYZE TEXT EMOTIONAL CONTENT
                emotional_content = self._analyze_emotional_content(inputs)
                
                # CREATE ENHANCED EMBEDDING
                base_embedding = torch.randn(1, self.feature_dim)
                emotional_enhancement = torch.tensor(emotional_content).unsqueeze(0) * 0.2
                
                enhanced_embedding = base_embedding + emotional_enhancement
                return enhanced_embedding
            else:
                return torch.randn(1, self.feature_dim)
        
        def _analyze_emotional_content(self, text):
            """ANALYZE EMOTIONAL CONTENT OF TEXT"""
            # SIMPLIFIED EMOTIONAL ANALYSIS
            emotional_scores = torch.zeros(8)  # 8 EMOTIONAL DIMENSIONS
            
            text_lower = text.lower()
            
            # JOY
            if any(word in text_lower for word in ["happy", "joy", "excited", "wonderful"]):
                emotional_scores[0] = 0.8
            
            # FEAR
            if any(word in text_lower for word in ["scared", "afraid", "terrified", "worried"]):
                emotional_scores[1] = 0.8
            
            # CURIOSITY
            if any(word in text_lower for word in ["curious", "wonder", "question", "explore"]):
                emotional_scores[2] = 0.8
            
            # SADNESS
            if any(word in text_lower for word in ["sad", "depressed", "melancholy", "grief"]):
                emotional_scores[3] = 0.8
            
            # ANGER
            if any(word in text_lower for word in ["angry", "furious", "rage", "irritated"]):
                emotional_scores[4] = 0.8
            
            # SURPRISE
            if any(word in text_lower for word in ["surprised", "amazed", "shocked", "astonished"]):
                emotional_scores[5] = 0.8
            
            # LOVE
            if any(word in text_lower for word in ["love", "affection", "tender", "caring"]):
                emotional_scores[6] = 0.8
            
            # CONFUSION
            if any(word in text_lower for word in ["confused", "puzzled", "uncertain", "unclear"]):
                emotional_scores[7] = 0.8
            
            return emotional_scores
    
    class AdvancedVisionEncoder:
        """ADVANCED VISION ENCODER WITH ATTENTION MECHANISMS"""
        
        def __init__(self):
            self.name = "advanced_vision_encoder"
            self.feature_dim = 512
        
        def process_inputs(self, inputs):
            """PROCESS VISION INPUTS WITH ATTENTION"""
            if isinstance(inputs, torch.Tensor):
                # APPLY ATTENTION MECHANISM
                attention_weights = self._calculate_attention_weights(inputs)
                
                # CREATE ATTENTION-ENHANCED EMBEDDING
                base_embedding = torch.randn(1, self.feature_dim)
                attention_enhancement = attention_weights[:self.feature_dim] * 0.3
                
                enhanced_embedding = base_embedding + attention_enhancement
                return enhanced_embedding
            else:
                return torch.randn(1, self.feature_dim)
        
        def _calculate_attention_weights(self, inputs):
            """CALCULATE ATTENTION WEIGHTS FOR VISION INPUTS"""
            # SIMPLIFIED ATTENTION MECHANISM
            if inputs.dim() >= 2:
                # CALCULATE SPATIAL ATTENTION
                spatial_attention = torch.mean(inputs, dim=0)
                attention_weights = torch.softmax(spatial_attention, dim=0)
                return attention_weights
            else:
                return torch.ones(inputs.size(0))
    
    class AdvancedAudioEncoder:
        """ADVANCED AUDIO ENCODER WITH TEMPORAL MODELING"""
        
        def __init__(self):
            self.name = "advanced_audio_encoder"
            self.feature_dim = 1024
        
        def process_inputs(self, inputs):
            """PROCESS AUDIO INPUTS WITH TEMPORAL MODELING"""
            if isinstance(inputs, torch.Tensor):
                # APPLY TEMPORAL MODELING
                temporal_features = self._extract_temporal_features(inputs)
                
                # CREATE TEMPORAL-ENHANCED EMBEDDING
                base_embedding = torch.randn(1, self.feature_dim)
                temporal_enhancement = temporal_features[:self.feature_dim] * 0.25
                
                enhanced_embedding = base_embedding + temporal_enhancement
                return enhanced_embedding
            else:
                return torch.randn(1, self.feature_dim)
        
        def _extract_temporal_features(self, inputs):
            """EXTRACT TEMPORAL FEATURES FROM AUDIO"""
            # SIMPLIFIED TEMPORAL FEATURE EXTRACTION
            if inputs.dim() >= 1:
                # CALCULATE TEMPORAL STATISTICS
                temporal_mean = torch.mean(inputs, dim=0)
                temporal_std = torch.std(inputs, dim=0)
                temporal_features = torch.cat([temporal_mean, temporal_std])
                return temporal_features
            else:
                return torch.zeros(self.feature_dim)
    
    return [
        AdvancedTextEncoder(),
        AdvancedVisionEncoder(),
        AdvancedAudioEncoder()
    ]

def run_advanced_example():
    """RUN THE ADVANCED USAGE EXAMPLE"""
    print("🚀 STARTING ADVANCED EMOTION-WEIGHTED MEMORY LLM EXAMPLE")
    print("=" * 70)
    
    # STEP 1: CREATE CUSTOM CONFIGURATIONS
    configs = create_custom_configurations()
    
    # STEP 2: CREATE ADVANCED COMPONENTS
    print("\n2️⃣ INITIALIZING ADVANCED COMPONENTS...")
    
    # EMOTION ENGINE WITH CUSTOM CONFIG
    emotion_engine = EmotionEngine(config=configs["emotion"])
    
    # MEMORY SYSTEM WITH CUSTOM CONFIG
    memory_system = EpisodicMemory(config=configs["memory"])
    
    # SELF MODEL WITH CUSTOM CONFIG
    self_model = SelfModel(config=configs["self_model"])
    
    # SLEEP CYCLE WITH CUSTOM CONFIG
    sleep_cycle = SleepCycle(config=configs["sleep"])
    
    # NARRATIVE ENGINE WITH CUSTOM CONFIG
    narrative_engine = NarrativeEngine(config=configs["narrative"])
    
    # LIFESPAN MANAGER WITH CUSTOM CONFIG
    lifespan_manager = LifespanManager(config=configs["lifespan"])
    
    # STEP 3: CREATE ADVANCED ADAPTERS
    print("\n3️⃣ CREATING ADVANCED ADAPTERS...")
    llm_adapter = create_advanced_llm_adapter()
    sensory_encoders = create_advanced_sensory_encoders()
    
    # STEP 4: CREATE AGENT LOOP
    print("\n4️⃣ CREATING ADVANCED AGENT LOOP...")
    agent = AgentLoop(
        llm_adapter=llm_adapter,
        sensory_encoders=sensory_encoders,
        memory_system=memory_system,
        emotion_engine=emotion_engine,
        self_model=self_model,
        lifespan_manager=lifespan_manager,
        config=AgentLoop.AgentConfig(
            max_ticks=100,  # LONGER LIFE FOR ADVANCED FEATURES
            tick_delay=0.05,  # FASTER TICKS
            log_level="INFO",
            enable_sleep=True,
            enable_narrative=True
        )
    )
    
    # STEP 5: RUN THE AGENT
    print("\n5️⃣ RUNNING ADVANCED AGENT LIFE CYCLE...")
    print("AGENT WILL LIVE FOR 100 TICKS WITH ADVANCED FEATURES...")
    print("-" * 50)
    
    try:
        final_summary = agent.run(max_ticks=100)
        
        # STEP 6: DISPLAY ADVANCED RESULTS
        print("\n6️⃣ ADVANCED AGENT LIFE COMPLETED!")
        print("=" * 70)
        
        # BASIC LIFE SUMMARY
        print(f"📊 ADVANCED LIFE SUMMARY:")
        print(f"   • Total Ticks: {final_summary['total_ticks']}")
        print(f"   • Total Time: {final_summary['total_time']:.2f}s")
        print(f"   • Life Stage: {final_summary['lifespan_summary']['final_state']['life_stage']}")
        
        # ADVANCED EMOTIONAL ANALYSIS
        print(f"\n😊 ADVANCED EMOTIONAL ANALYSIS:")
        emotions = final_summary['final_emotions']
        emotion_labels = ["joy", "fear", "curiosity", "sadness", "anger", "surprise", "love", "confusion"]
        
        for i, (emotion, intensity) in enumerate(emotions.items()):
            if i < len(emotion_labels):
                label = emotion_labels[i]
                if abs(intensity) > 0.1:
                    print(f"   • {label}: {intensity:.3f}")
        
        # ADVANCED IDENTITY ANALYSIS
        print(f"\n🆔 ADVANCED IDENTITY ANALYSIS:")
        identity = final_summary['final_identity']
        print(f"   • Identity Norm: {identity['identity_norm']:.3f}")
        print(f"   • Update Count: {identity['update_count']}")
        print(f"   • Stability Score: {identity['stability_score']:.3f}")
        print(f"   • Identity Context: {self_model.get_identity_context()}")
        
        # ADVANCED MEMORY ANALYSIS
        print(f"\n🧠 ADVANCED MEMORY ANALYSIS:")
        memory_stats = final_summary['memory_stats']
        print(f"   • Total Memories: {memory_stats['total_memories']}")
        print(f"   • Storage Count: {memory_stats['storage_count']}")
        print(f"   • Retrieval Count: {memory_stats['retrieval_count']}")
        print(f"   • Avg Emotional Weight: {memory_stats['avg_emotional_weight']:.3f}")
        
        # ADVANCED SLEEP ANALYSIS
        if 'sleep_stats' in final_summary:
            sleep_stats = final_summary['sleep_stats']
            print(f"\n😴 ADVANCED SLEEP ANALYSIS:")
            print(f"   • Sleep Cycles: {sleep_stats['sleep_cycles']}")
            print(f"   • Dreams Generated: {sleep_stats['dreams_generated']}")
            print(f"   • Memories Consolidated: {sleep_stats['memories_consolidated']}")
            print(f"   • Memories Forgotten: {sleep_stats['memories_forgotten']}")
            
            # SHOW DREAM MEMORIES
            dream_memories = sleep_cycle.get_dream_memories()
            if dream_memories:
                print(f"   • Dream Types: {[dream['dream_type'] for dream in dream_memories]}")
        
        # ADVANCED NARRATIVE ANALYSIS
        if 'narrative_stats' in final_summary:
            narrative_stats = final_summary['narrative_stats']
            print(f"\n📖 ADVANCED NARRATIVE ANALYSIS:")
            print(f"   • Total Narratives: {narrative_stats['total_narratives']}")
            print(f"   • Reflection Count: {narrative_stats['reflection_count']}")
            print(f"   • Narrative Types: {narrative_stats['narrative_types']}")
            print(f"   • Avg Story Length: {narrative_stats['avg_story_length']:.1f}")
        
        # ADVANCED LIFESPAN ANALYSIS
        print(f"\n⏰ ADVANCED LIFESPAN ANALYSIS:")
        lifespan_trajectory = lifespan_manager.get_developmental_trajectory()
        if lifespan_trajectory:
            print(f"   • Final Plasticity: {lifespan_trajectory['plasticity'][-1]:.3f}")
            print(f"   • Final Memory Dominance: {lifespan_trajectory['memory_dominance'][-1]:.3f}")
            print(f"   • Final Identity Stability: {lifespan_trajectory['identity_stability'][-1]:.3f}")
            
            # CALCULATE DEVELOPMENTAL RATES
            plasticity_rate = (lifespan_trajectory['plasticity'][0] - lifespan_trajectory['plasticity'][-1]) / len(lifespan_trajectory['plasticity'])
            memory_rate = (lifespan_trajectory['memory_dominance'][-1] - lifespan_trajectory['memory_dominance'][0]) / len(lifespan_trajectory['memory_dominance'])
            
            print(f"   • Plasticity Decay Rate: {plasticity_rate:.4f} per tick")
            print(f"   • Memory Dominance Growth Rate: {memory_rate:.4f} per tick")
        
        # ADVANCED COMPONENT ANALYSIS
        print(f"\n🔧 ADVANCED COMPONENT ANALYSIS:")
        
        # LLM CONTEXT ANALYSIS
        llm_context = llm_adapter.get_context_summary()
        print(f"   • LLM Context Length: {llm_context['context_length']}")
        
        # SENSORY ENCODER ANALYSIS
        for encoder in sensory_encoders:
            print(f"   • {encoder.name}: {encoder.feature_dim} dimensions")
        
        print("\n" + "=" * 70)
        print("🎉 ADVANCED EXAMPLE COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\n⏹️ ADVANCED AGENT LIFE INTERRUPTED BY USER")
    except Exception as e:
        print(f"\n❌ ERROR DURING ADVANCED AGENT LIFE: {e}")
        raise

if __name__ == "__main__":
    run_advanced_example()
