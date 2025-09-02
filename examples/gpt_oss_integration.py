"""
EXAMPLE: GPT-OSS-20B INTEGRATION WITH ATTENTION-BASED MEMORY INFLUENCE
"""

import logging
import torch
from overlay.core import AgentLoop
from overlay.emotion import EmotionConfig
from overlay.memory import MemoryConfig
from overlay.lifespan import LifespanConfig
from overlay.self_model import SelfModelConfig
from overlay.narrative import NarrativeConfig
from overlay.sleep import SleepConfig
from overlay.utils.langchain import LangChainConfig

# SETUP LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """DEMONSTRATE GPT-OSS-20B AGENT WITH MEMORY INFLUENCE"""
    
    # CONFIGURE COMPONENTS
    emotion_config = EmotionConfig(
        dimension=6,
        labels=["joy", "sadness", "anger", "fear", "surprise", "disgust"]
    )
    
    memory_config = MemoryConfig(
        vector_store_type="chroma",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim=384,
        similarity_threshold=0.3
    )
    
    lifespan_config = LifespanConfig(
        max_ticks=1000,
        life_stages=["infancy", "adolescence", "maturity"]
    )
    
    self_model_config = SelfModelConfig(
        embedding_dim=128,
        stability_factor=0.95
    )
    
    narrative_config = NarrativeConfig(
        reflection_threshold=0.7,
        reflection_probability=0.3,
        memory_threshold=5
    )
    
    sleep_config = SleepConfig(
        sleep_interval=50,
        dream_probability=0.8
    )
    
    # GPT-OSS-20B CONFIGURATION
    llm_config = LangChainConfig(
        provider="huggingface",
        model_name="openai/gpt-oss-20b",
        reasoning_level="medium",  # low, medium, high
        max_tokens=256,
        temperature=0.7
    )
    
    # CREATE AGENT
    agent = AgentLoop(
        run_id="gpt_oss_demo",
        emotion_config=emotion_config,
        memory_config=memory_config,
        lifespan_config=lifespan_config,
        self_model_config=self_model_config,
        narrative_config=narrative_config,
        sleep_config=sleep_config,
        llm_config=llm_config
    )
    
    logger.info("AGENT CREATED - STARTING INTERACTION")
    
    # INTERACT WITH AGENT
    test_inputs = [
        "Hello! How are you feeling today?",
        "Tell me about your memories so far.",
        "What's your current identity like?",
        "Can you reflect on our conversation?",
        "What would you like to dream about?"
    ]
    
    for i, input_text in enumerate(test_inputs):
        logger.info(f"\n--- INTERACTION {i+1} ---")
        logger.info(f"INPUT: {input_text}")
        
        # PROCESS INPUT
        response = agent.process_input(input_text)
        logger.info(f"RESPONSE: {response}")
        
        # GET STATUS
        status = agent.get_status()
        logger.info(f"TICK: {status['tick']}")
        logger.info(f"EMOTIONS: {status['emotions']}")
        logger.info(f"LIFE STAGE: {status['life_stage']}")
        logger.info(f"MEMORIES: {status['memory_count']}")
        
        if not status['alive']:
            logger.warning("AGENT DIED - STOPPING")
            break
    
    logger.info("\n--- FINAL STATUS ---")
    final_status = agent.get_status()
    for key, value in final_status.items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()
