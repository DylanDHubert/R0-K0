"""
Command Line Interface for Emotion-Weighted Memory LLM
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from . import AgentLoop, EmotionEngine, EpisodicMemory, SelfModel, LifespanManager
from .core import AgentConfig
from adapters import FAISSMemoryStore

def setup_logging(level: str = "INFO") -> None:
    """SETUP LOGGING CONFIGURATION"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_dummy_llm_adapter():
    """CREATE DUMMY LLM ADAPTER FOR CLI"""
    class DummyLLMAdapter:
        def __init__(self):
            self.model_name = "dummy_model"
        
        def forward(self, inputs, memories=None, emotions=None, identity=None):
            import torch
            batch_size = inputs.size(0)
            output_dim = 64
            outputs = torch.randn(batch_size, output_dim)
            outputs += inputs[:, :output_dim] * 0.1
            return outputs
    
    return DummyLLMAdapter()

def create_dummy_sensory_encoders():
    """CREATE DUMMY SENSORY ENCODERS FOR CLI"""
    class DummySensoryEncoder:
        def __init__(self, name):
            self.name = name
        
        def process_inputs(self, inputs):
            import torch
            if isinstance(inputs, str):
                return torch.randn(1, 128)
            elif isinstance(inputs, torch.Tensor):
                return inputs
            else:
                return torch.randn(1, 128)
    
    return [
        DummySensoryEncoder("text_encoder"),
        DummySensoryEncoder("vision_encoder"),
        DummySensoryEncoder("audio_encoder")
    ]

def run_agent(args) -> None:
    """RUN THE AGENT WITH SPECIFIED PARAMETERS"""
    print("🚀 INITIALIZING EMOTION-WEIGHTED MEMORY LLM AGENT")
    print("=" * 60)
    
    # CREATE COMPONENTS
    print("CREATING COMPONENTS...")
    
    # MEMORY STORE
    memory_store = FAISSMemoryStore(config={"dimension": 128})
    memory_system = EpisodicMemory()
    
    # CORE COMPONENTS
    emotion_engine = EmotionEngine()
    self_model = SelfModel()
    lifespan_manager = LifespanManager()
    
    # ADAPTERS
    llm_adapter = create_dummy_llm_adapter()
    sensory_encoders = create_dummy_sensory_encoders()
    
    # AGENT CONFIG
    agent_config = AgentConfig(
        max_ticks=args.ticks,
        tick_delay=args.delay,
        log_level=args.log_level,
        enable_sleep=args.enable_sleep,
        enable_narrative=args.enable_narrative
    )
    
    # CREATE AGENT
    agent = AgentLoop(
        llm_adapter=llm_adapter,
        sensory_encoders=sensory_encoders,
        memory_system=memory_system,
        emotion_engine=emotion_engine,
        self_model=self_model,
        lifespan_manager=lifespan_manager,
        config=agent_config
    )
    
    # RUN AGENT
    print(f"RUNNING AGENT FOR {args.ticks} TICKS...")
    print("-" * 40)
    
    try:
        final_summary = agent.run(max_ticks=args.ticks)
        
        # DISPLAY RESULTS
        print("\n📊 AGENT LIFE COMPLETED!")
        print("=" * 60)
        
        print(f"LIFE SUMMARY:")
        print(f"  • Total Ticks: {final_summary['total_ticks']}")
        print(f"  • Total Time: {final_summary['total_time']:.2f}s")
        print(f"  • Life Stage: {final_summary['lifespan_summary']['final_state']['life_stage']}")
        
        print(f"\nMEMORY STATISTICS:")
        memory_stats = final_summary['memory_stats']
        print(f"  • Total Memories: {memory_stats['total_memories']}")
        print(f"  • Storage Count: {memory_stats['storage_count']}")
        print(f"  • Retrieval Count: {memory_stats['retrieval_count']}")
        
        print(f"\nEMOTIONAL STATE:")
        emotions = final_summary['final_emotions']
        for emotion, intensity in emotions.items():
            if abs(intensity) > 0.1:
                print(f"  • {emotion}: {intensity:.3f}")
        
        print(f"\nIDENTITY:")
        identity = final_summary['final_identity']
        print(f"  • Identity Norm: {identity['identity_norm']:.3f}")
        print(f"  • Update Count: {identity['update_count']}")
        print(f"  • Stability Score: {identity['stability_score']:.3f}")
        
        # EXPORT DATA IF REQUESTED
        if args.export:
            export_path = Path(args.export)
            agent.export_life_data(str(export_path))
            print(f"\n💾 LIFE DATA EXPORTED TO: {export_path}")
        
        print("\n🎉 AGENT LIFE COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("\n⏹️ AGENT LIFE INTERRUPTED BY USER")
    except Exception as e:
        print(f"\n❌ ERROR DURING AGENT LIFE: {e}")
        if args.verbose:
            raise
        sys.exit(1)

def demonstrate_components(args) -> None:
    """DEMONSTRATE INDIVIDUAL COMPONENTS"""
    print("🔧 COMPONENT DEMONSTRATION")
    print("=" * 40)
    
    # EMOTION ENGINE
    print("\n😊 EMOTION ENGINE:")
    emotions = EmotionEngine()
    print(f"  Initial state: {emotions.get_state_dict()}")
    
    import torch
    dummy_inputs = torch.randn(1, 10)
    dummy_outputs = torch.randn(1, 10)
    emotions.update(dummy_inputs, dummy_outputs)
    print(f"  After update: {emotions.get_state_dict()}")
    
    # SELF MODEL
    print("\n🆔 SELF MODEL:")
    self_model = SelfModel()
    print(f"  Initial identity: {self_model.get_identity_context()}")
    
    self_model.update(dummy_inputs, dummy_outputs, emotions.get_state(), [])
    print(f"  After update: {self_model.get_identity_context()}")
    
    # MEMORY SYSTEM
    print("\n🧠 MEMORY SYSTEM:")
    memory = EpisodicMemory()
    
    for i in range(3):
        inputs = torch.randn(1, 128)
        outputs = torch.randn(1, 64)
        emotions_state = torch.randn(6) * 0.5
        embedding = torch.randn(1, 128)
        
        stored = memory.store(inputs, outputs, emotions_state, embedding)
        print(f"  Memory {i+1} stored: {stored}")
    
    query = torch.randn(1, 128)
    retrieved = memory.retrieve(query, emotions.get_state())
    print(f"  Retrieved {len(retrieved)} memories")
    
    # LIFESPAN MANAGER
    print("\n⏰ LIFESPAN MANAGER:")
    lifespan = LifespanManager()
    print(f"  Initial stage: {lifespan.get_life_state()['life_stage']}")
    
    for i in range(5):
        lifespan.tick()
        state = lifespan.get_life_state()
        print(f"  Tick {i+1}: {state['life_stage']} (plasticity: {state['plasticity']:.3f})")
    
    print("\n✅ COMPONENT DEMONSTRATION COMPLETED!")

def main():
    """MAIN CLI ENTRY POINT"""
    parser = argparse.ArgumentParser(
        description="Emotion-Weighted Memory LLM - A Plug-and-Play Cognitive Overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # RUN BASIC AGENT FOR 100 TICKS
  emotion-llm run --ticks 100
  
  # RUN AGENT WITH SLEEP AND NARRATIVE
  emotion-llm run --ticks 200 --enable-sleep --enable-narrative
  
  # RUN AGENT WITH FAST TICKS AND EXPORT DATA
  emotion-llm run --ticks 50 --delay 0.05 --export life_data.json
  
  # DEMONSTRATE INDIVIDUAL COMPONENTS
  emotion-llm demo
  
  # RUN WITH VERBOSE LOGGING
  emotion-llm run --ticks 100 --log-level DEBUG --verbose
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # RUN COMMAND
    run_parser = subparsers.add_parser("run", help="Run the agent")
    run_parser.add_argument(
        "--ticks", "-t", type=int, default=100,
        help="Number of life ticks (default: 100)"
    )
    run_parser.add_argument(
        "--delay", "-d", type=float, default=0.1,
        help="Delay between ticks in seconds (default: 0.1)"
    )
    run_parser.add_argument(
        "--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level (default: INFO)"
    )
    run_parser.add_argument(
        "--enable-sleep", action="store_true",
        help="Enable sleep cycles"
    )
    run_parser.add_argument(
        "--enable-narrative", action="store_true",
        help="Enable narrative generation"
    )
    run_parser.add_argument(
        "--export", "-e", type=str,
        help="Export life data to JSON file"
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose error reporting"
    )
    
    # DEMO COMMAND
    demo_parser = subparsers.add_parser("demo", help="Demonstrate individual components")
    demo_parser.add_argument(
        "--log-level", "-l", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level (default: INFO)"
    )
    
    # VERSION COMMAND
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    # PARSE ARGUMENTS
    args = parser.parse_args()
    
    # HANDLE VERSION COMMAND
    if args.command == "version":
        from . import __version__
        print(f"Emotion-Weighted Memory LLM v{__version__}")
        return
    
    # HANDLE NO COMMAND
    if not args.command:
        parser.print_help()
        return
    
    # SETUP LOGGING
    setup_logging(args.log_level)
    
    # EXECUTE COMMAND
    if args.command == "run":
        run_agent(args)
    elif args.command == "demo":
        demonstrate_components(args)
    else:
        print(f"UNKNOWN COMMAND: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
