"""
Basic Usage Example - Demonstrates the emotion-weighted memory LLM system
"""

import torch
import logging
from overlay import AgentLoop, EmotionEngine, EpisodicMemory, SelfModel, LifespanManager
from adapters import FAISSMemoryStore

# SETUP LOGGING
logging.basicConfig(level=logging.INFO)

def main():
    """MAIN EXAMPLE FUNCTION"""
    print("🚀 STARTING EMOTION-WEIGHTED MEMORY LLM EXAMPLE")
    print("=" * 60)
    
    # STEP 1: CREATE MEMORY STORE
    print("\n1️⃣ CREATING MEMORY STORE...")
    memory_store = FAISSMemoryStore(config={"dimension": 128})
    memory_system = EpisodicMemory()
    
    # STEP 2: CREATE CORE COMPONENTS
    print("\n2️⃣ INITIALIZING CORE COMPONENTS...")
    emotion_engine = EmotionEngine()
    self_model = SelfModel()
    lifespan_manager = LifespanManager()
    
    # STEP 3: CREATE DUMMY LLM ADAPTER
    print("\n3️⃣ CREATING DUMMY LLM ADAPTER...")
    class DummyLLMAdapter:
        """DUMMY LLM ADAPTER FOR DEMONSTRATION"""
        
        def __init__(self):
            self.model_name = "dummy_model"
        
        def forward(self, inputs, memories=None, emotions=None, identity=None):
            """DUMMY FORWARD PASS"""
            # CREATE SIMULATED OUTPUTS
            batch_size = inputs.size(0)
            output_dim = 64
            outputs = torch.randn(batch_size, output_dim)
            
            # APPLY SOME VARIATION BASED ON INPUTS
            outputs += inputs[:, :output_dim] * 0.1
            
            return outputs
    
    llm_adapter = DummyLLMAdapter()
    
    # STEP 4: CREATE SENSORY ENCODERS
    print("\n4️⃣ CREATING SENSORY ENCODERS...")
    class DummySensoryEncoder:
        """DUMMY SENSORY ENCODER FOR DEMONSTRATION"""
        
        def __init__(self, name):
            self.name = name
        
        def process_inputs(self, inputs):
            """DUMMY INPUT PROCESSING"""
            # CONVERT INPUTS TO EMBEDDINGS
            if isinstance(inputs, str):
                # TEXT INPUT
                return torch.randn(1, 128)
            elif isinstance(inputs, torch.Tensor):
                # ALREADY TENSOR
                return inputs
            else:
                # OTHER INPUTS
                return torch.randn(1, 128)
    
    sensory_encoders = [
        DummySensoryEncoder("text_encoder"),
        DummySensoryEncoder("vision_encoder"),
        DummySensoryEncoder("audio_encoder")
    ]
    
    # STEP 5: CREATE AGENT LOOP
    print("\n5️⃣ CREATING AGENT LOOP...")
    agent = AgentLoop(
        llm_adapter=llm_adapter,
        sensory_encoders=sensory_encoders,
        memory_system=memory_system,
        emotion_engine=emotion_engine,
        self_model=self_model,
        lifespan_manager=lifespan_manager,
        config=AgentLoop.AgentConfig(
            max_ticks=50,  # SHORT LIFE FOR DEMONSTRATION
            tick_delay=0.1,  # FAST TICKS
            log_level="INFO"
        )
    )
    
    # STEP 6: RUN THE AGENT
    print("\n6️⃣ RUNNING AGENT LIFE CYCLE...")
    print("AGENT WILL LIVE FOR 50 TICKS...")
    print("-" * 40)
    
    try:
        final_summary = agent.run(max_ticks=50)
        
        # STEP 7: DISPLAY RESULTS
        print("\n7️⃣ AGENT LIFE COMPLETED!")
        print("=" * 60)
        
        print(f"📊 LIFE SUMMARY:")
        print(f"   • Total Ticks: {final_summary['total_ticks']}")
        print(f"   • Total Time: {final_summary['total_time']:.2f}s")
        print(f"   • Life Stage: {final_summary['lifespan_summary']['final_state']['life_stage']}")
        
        print(f"\n🧠 MEMORY STATISTICS:")
        memory_stats = final_summary['memory_stats']
        print(f"   • Total Memories: {memory_stats['total_memories']}")
        print(f"   • Storage Count: {memory_stats['storage_count']}")
        print(f"   • Retrieval Count: {memory_stats['retrieval_count']}")
        
        print(f"\n😊 FINAL EMOTIONAL STATE:")
        emotions = final_summary['final_emotions']
        for emotion, intensity in emotions.items():
            if abs(intensity) > 0.1:
                print(f"   • {emotion}: {intensity:.3f}")
        
        print(f"\n🆔 IDENTITY EVOLUTION:")
        identity = final_summary['final_identity']
        print(f"   • Identity Norm: {identity['identity_norm']:.3f}")
        print(f"   • Update Count: {identity['update_count']}")
        print(f"   • Stability Score: {identity['stability_score']:.3f}")
        
        # STEP 8: SHOW LIFE TRAJECTORIES
        print(f"\n📈 LIFE TRAJECTORIES:")
        lifespan_trajectory = lifespan_manager.get_developmental_trajectory()
        if lifespan_trajectory:
            print(f"   • Plasticity: {lifespan_trajectory['plasticity'][-1]:.3f} (started at 1.0)")
            print(f"   • Memory Dominance: {lifespan_trajectory['memory_dominance'][-1]:.3f} (started at 0.1)")
            print(f"   • Identity Stability: {lifespan_trajectory['identity_stability'][-1]:.3f} (started at 0.1)")
        
        # STEP 9: SHOW SLEEP AND NARRATIVE STATS
        if 'sleep_stats' in final_summary:
            sleep_stats = final_summary['sleep_stats']
            print(f"\n😴 SLEEP STATISTICS:")
            print(f"   • Sleep Cycles: {sleep_stats['sleep_cycles']}")
            print(f"   • Dreams Generated: {sleep_stats['dreams_generated']}")
            print(f"   • Memories Consolidated: {sleep_stats['memories_consolidated']}")
        
        if 'narrative_stats' in final_summary:
            narrative_stats = final_summary['narrative_stats']
            print(f"\n📖 NARRATIVE STATISTICS:")
            print(f"   • Total Narratives: {narrative_stats['total_narratives']}")
            print(f"   • Reflection Count: {narrative_stats['reflection_count']}")
        
        print("\n" + "=" * 60)
        print("🎉 EXAMPLE COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\n⏹️ AGENT LIFE INTERRUPTED BY USER")
    except Exception as e:
        print(f"\n❌ ERROR DURING AGENT LIFE: {e}")
        raise

def demonstrate_components():
    """DEMONSTRATE INDIVIDUAL COMPONENTS"""
    print("\n🔧 COMPONENT DEMONSTRATION")
    print("=" * 40)
    
    # EMOTION ENGINE DEMO
    print("\n😊 EMOTION ENGINE:")
    emotions = EmotionEngine()
    print(f"   Initial state: {emotions.get_state_dict()}")
    
    # SIMULATE EMOTIONAL UPDATE
    dummy_inputs = torch.randn(1, 10)
    dummy_outputs = torch.randn(1, 10)
    emotions.update(dummy_inputs, dummy_outputs)
    print(f"   After update: {emotions.get_state_dict()}")
    
    # SELF MODEL DEMO
    print("\n🆔 SELF MODEL:")
    self_model = SelfModel()
    print(f"   Initial identity: {self_model.get_identity_context()}")
    
    # SIMULATE IDENTITY UPDATE
    self_model.update(dummy_inputs, dummy_outputs, emotions.get_state(), [])
    print(f"   After update: {self_model.get_identity_context()}")
    
    # MEMORY SYSTEM DEMO
    print("\n🧠 MEMORY SYSTEM:")
    memory = EpisodicMemory()
    
    # STORE SOME MEMORIES
    for i in range(3):
        inputs = torch.randn(1, 128)
        outputs = torch.randn(1, 64)
        emotions_state = torch.randn(6) * 0.5  # MODERATE EMOTIONAL INTENSITY
        embedding = torch.randn(1, 128)
        
        stored = memory.store(inputs, outputs, emotions_state, embedding)
        print(f"   Memory {i+1} stored: {stored}")
    
    # RETRIEVE MEMORIES
    query = torch.randn(1, 128)
    retrieved = memory.retrieve(query, emotions.get_state())
    print(f"   Retrieved {len(retrieved)} memories")
    
    # LIFESPAN MANAGER DEMO
    print("\n⏰ LIFESPAN MANAGER:")
    lifespan = LifespanManager()
    print(f"   Initial stage: {lifespan.get_life_state()['life_stage']}")
    
    # ADVANCE A FEW TICKS
    for i in range(5):
        lifespan.tick()
        state = lifespan.get_life_state()
        print(f"   Tick {i+1}: {state['life_stage']} (plasticity: {state['plasticity']:.3f})")

if __name__ == "__main__":
    # RUN COMPONENT DEMONSTRATION
    demonstrate_components()
    
    # RUN MAIN EXAMPLE
    main()
