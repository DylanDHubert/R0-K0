#!/usr/bin/env python3
"""
COLAB FULL DEBUG TEST - ISOLATED COMPONENT TESTING WITH STEP-BY-STEP DEBUG

This file is designed to be copy-pasted into Colab cells for comprehensive testing.
Each section can be run in separate cells to isolate and debug components.

SETUP INSTRUCTIONS:
1. Copy this file to Colab
2. Install dependencies: !pip install torch transformers sentence-transformers langchain openai
3. Set OpenAI API key: import os; os.environ["OPENAI_API_KEY"] = "your-key-here"
4. Run each section in separate cells
"""

import os
import sys
import torch
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# SETUP COMPREHENSIVE LOGGING
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('colab_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# FORCE DEBUG LEVEL FOR ALL COMPONENTS
logging.getLogger('overlay').setLevel(logging.DEBUG)
logging.getLogger('overlay.emotion').setLevel(logging.DEBUG)
logging.getLogger('overlay.memory').setLevel(logging.DEBUG)
logging.getLogger('overlay.core').setLevel(logging.DEBUG)

print("🔧 LOGGING SETUP COMPLETE - DEBUG LEVEL ENABLED")

# ============================================================================
# SECTION 1: DEPENDENCY INSTALLATION AND SETUP
# ============================================================================

def install_dependencies():
    """INSTALL ALL REQUIRED DEPENDENCIES FOR COLAB"""
    print("📦 INSTALLING DEPENDENCIES...")
    
    # INSTALL PACKAGES
    os.system("pip install torch transformers sentence-transformers langchain openai chromadb")
    
    # VERIFY INSTALLATIONS
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
    
    try:
        import sentence_transformers
        print(f"✅ Sentence-Transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ Sentence-Transformers: {e}")
    
    try:
        import langchain
        print(f"✅ LangChain: {langchain.__version__}")
    except ImportError as e:
        print(f"❌ LangChain: {e}")
    
    try:
        import openai
        print(f"✅ OpenAI: {openai.__version__}")
    except ImportError as e:
        print(f"❌ OpenAI: {e}")
    
    print("📦 DEPENDENCY INSTALLATION COMPLETE")

# ============================================================================
# SECTION 2: OPENAI API SETUP
# ============================================================================

def setup_openai_api():
    """SETUP OPENAI API FOR EMOTION INFERENCE"""
    print("🔑 SETTING UP OPENAI API...")
    
    # CHECK FOR API KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY NOT FOUND IN ENVIRONMENT")
        print("💡 SET YOUR API KEY: os.environ['OPENAI_API_KEY'] = 'your-key-here'")
        return False
    
    print(f"✅ OPENAI API KEY FOUND: {api_key[:8]}...")
    
    # TEST API CONNECTION
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print(f"✅ OPENAI API TEST SUCCESSFUL: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ OPENAI API TEST FAILED: {e}")
        return False

# ============================================================================
# SECTION 3: COMPONENT ISOLATION TESTS
# ============================================================================

def test_emotion_engine_isolation():
    """TEST EMOTION ENGINE IN ISOLATION"""
    print("\n🧠 TESTING EMOTION ENGINE ISOLATION")
    print("=" * 50)
    
    try:
        # IMPORT COMPONENTS
        from overlay.emotion import EmotionEngine, EmotionConfig, APIBasedInference
        from overlay.utils.langchain import LangChainConfig
        
        print("✅ IMPORTS SUCCESSFUL")
        
        # CREATE CONFIG
        emotion_config = EmotionConfig(
            dimension=6,
            labels=["joy", "sadness", "anger", "fear", "surprise", "disgust"],
            decay_rate=0.95,
            bias_strength=0.3
        )
        print(f"✅ EMOTION CONFIG CREATED: {emotion_config}")
        
        # CREATE EMOTION ENGINE
        emotion_engine = EmotionEngine(emotion_config)
        print(f"✅ EMOTION ENGINE CREATED: {emotion_engine}")
        
        # TEST INITIAL STATE
        initial_state = emotion_engine.get_state()
        print(f"✅ INITIAL EMOTION STATE: {initial_state}")
        print(f"✅ EMOTION SHAPE: {initial_state.shape}")
        
        # TEST PROBABILITY CONVERSION
        probs = emotion_engine.get_state_probs()
        print(f"✅ EMOTION PROBABILITIES: {probs}")
        print(f"✅ PROB SHAPE: {probs.shape}")
        
        # TEST EMOTION UPDATE
        test_gradient = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
        emotion_engine.update(inputs=torch.tensor([0]), outputs=torch.tensor([0]), emotion_gradient=test_gradient)
        updated_state = emotion_engine.get_state()
        print(f"✅ UPDATED EMOTION STATE: {updated_state}")
        
        # TEST EMOTION INFERENCE COMPONENT
        langchain_config = LangChainConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        emotion_inference = APIBasedInference(
            config=emotion_config,
            langchain_config=langchain_config,
            model_name="gpt-3.5-turbo"
        )
        print(f"✅ EMOTION INFERENCE COMPONENT CREATED: {emotion_inference}")
        
        # SET THE INFERENCE COMPONENT
        emotion_engine.set_emotion_inference(emotion_inference)
        print("✅ EMOTION INFERENCE COMPONENT SET")
        
        # TEST EMOTION INFERENCE
        test_text = "I am feeling very happy and excited today!"
        inferred_emotions = emotion_engine.infer_emotion(test_text)
        print(f"✅ INFERRED EMOTIONS: {inferred_emotions}")
        print(f"✅ INFERRED SHAPE: {inferred_emotions.shape}")
        
        print("🎉 EMOTION ENGINE ISOLATION TEST PASSED!")
        return emotion_engine
        
    except Exception as e:
        print(f"❌ EMOTION ENGINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_system_isolation():
    """TEST MEMORY SYSTEM IN ISOLATION"""
    print("\n🗄️ TESTING MEMORY SYSTEM ISOLATION")
    print("=" * 50)
    
    try:
        # IMPORT COMPONENTS
        from overlay.memory import EpisodicMemory, MemoryConfig, MemoryEntry
        from overlay.utils.langchain import LangChainConfig
        
        print("✅ IMPORTS SUCCESSFUL")
        
        # CREATE CONFIG
        memory_config = MemoryConfig(
            max_memories=100,
            storage_threshold=0.1,
            embedding_dim=384,
            vector_store_type="chroma",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        print(f"✅ MEMORY CONFIG CREATED: {memory_config}")
        
        # CREATE MEMORY SYSTEM
        memory_system = EpisodicMemory(memory_config, run_dir="./test_memories")
        print(f"✅ MEMORY SYSTEM CREATED: {memory_system}")
        
        # TEST MEMORY STORAGE
        test_inputs = torch.randn(128)
        test_outputs = torch.randn(128)
        test_emotions = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.2, -0.3])
        test_embedding = torch.randn(384)
        test_attention = torch.randn(64, 64)  # SIMPLE ATTENTION PATTERN
        
        # CREATE MEMORY ENTRY
        memory_entry = MemoryEntry(
            id="test_memory_001",
            timestamp=datetime.now(),
            inputs=test_inputs,
            outputs=test_outputs,
            emotions=test_emotions,
            embedding=test_embedding,
            emotional_weight=0.8,
            attention_weights=test_attention
        )
        print(f"✅ MEMORY ENTRY CREATED: {memory_entry}")
        print(f"✅ ATTENTION WEIGHTS SHAPE: {memory_entry.attention_weights.shape}")
        
        # TEST TENSOR SAVING
        tensor_file = memory_entry.save_tensors("./test_memories")
        print(f"✅ TENSORS SAVED TO: {tensor_file}")
        
        # TEST TENSOR LOADING
        loaded_tensors = MemoryEntry.load_tensors("./test_memories", "test_memory_001.pt")
        print(f"✅ TENSORS LOADED: {list(loaded_tensors.keys())}")
        print(f"✅ ATTENTION WEIGHTS LOADED: {loaded_tensors['attention_weights'].shape}")
        
        # TEST MEMORY STORAGE
        stored = memory_system.store(
            inputs=test_inputs,
            outputs=test_outputs,
            emotions=test_emotions,
            embedding=test_embedding,
            attention_weights=test_attention
        )
        print(f"✅ MEMORY STORED: {stored}")
        
        # TEST MEMORY RETRIEVAL
        retrieved_memories = memory_system.retrieve(
            query_embedding=test_embedding,
            emotions=test_emotions,
            k=5
        )
        print(f"✅ MEMORIES RETRIEVED: {len(retrieved_memories)}")
        
        print("🎉 MEMORY SYSTEM ISOLATION TEST PASSED!")
        return memory_system
        
    except Exception as e:
        print(f"❌ MEMORY SYSTEM TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_sentence_transformer_isolation():
    """TEST SENTENCE TRANSFORMER IN ISOLATION"""
    print("\n🔤 TESTING SENTENCE TRANSFORMER ISOLATION")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("✅ IMPORTS SUCCESSFUL")
        
        # LOAD MODEL
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ MODEL LOADED: {model}")
        
        # TEST EMBEDDING GENERATION
        test_texts = [
            "Hello, how are you feeling?",
            "I am very happy and excited!",
            "This is a test of the embedding system."
        ]
        
        embeddings = model.encode(test_texts, convert_to_tensor=True)
        print(f"✅ EMBEDDINGS GENERATED: {embeddings.shape}")
        print(f"✅ EMBEDDING DIMENSION: {embeddings.shape[-1]}")
        
        # TEST SIMILARITY
        similarity_matrix = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        print(f"✅ SIMILARITY MATRIX: {similarity_matrix}")
        
        print("🎉 SENTENCE TRANSFORMER ISOLATION TEST PASSED!")
        return model
        
    except Exception as e:
        print(f"❌ SENTENCE TRANSFORMER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# SECTION 4: INTEGRATION TESTS
# ============================================================================

def test_emotion_memory_integration():
    """TEST EMOTION AND MEMORY INTEGRATION"""
    print("\n🔗 TESTING EMOTION-MEMORY INTEGRATION")
    print("=" * 50)
    
    try:
        # GET ISOLATED COMPONENTS
        emotion_engine = test_emotion_engine_isolation()
        memory_system = test_memory_system_isolation()
        
        if not emotion_engine or not memory_system:
            print("❌ REQUIRED COMPONENTS NOT AVAILABLE")
            return False
        
        print("✅ BOTH COMPONENTS AVAILABLE")
        
        # TEST INTEGRATED FLOW
        test_input = "I am feeling very sad and lonely today."
        
        # 1. EMOTION INFERENCE
        print("\n1️⃣ EMOTION INFERENCE STEP")
        emotion_gradient = emotion_engine.infer_emotion(test_input)
        print(f"   INFERRED EMOTIONS: {emotion_gradient}")
        
        # 2. EMOTION UPDATE
        print("\n2️⃣ EMOTION UPDATE STEP")
        emotion_engine.update(inputs=torch.tensor([0]), outputs=torch.tensor([0]), emotion_gradient=emotion_gradient)
        current_emotions = emotion_engine.get_state()
        print(f"   CURRENT EMOTIONS: {current_emotions}")
        
        # 3. CREATE EMBEDDING
        print("\n3️⃣ EMBEDDING CREATION STEP")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        input_embedding = sentence_model.encode(test_input, convert_to_tensor=True)
        print(f"   INPUT EMBEDDING SHAPE: {input_embedding.shape}")
        
        # 4. MEMORY STORAGE
        print("\n4️⃣ MEMORY STORAGE STEP")
        stored = memory_system.store(
            inputs=torch.tensor([ord(c) for c in test_input[:128]]),
            outputs=torch.tensor([0] * 128),
            emotions=current_emotions,
            embedding=input_embedding,
            attention_weights=torch.randn(64, 64)  # DUMMY ATTENTION
        )
        print(f"   MEMORY STORED: {stored}")
        
        # 5. MEMORY RETRIEVAL
        print("\n5️⃣ MEMORY RETRIEVAL STEP")
        retrieved = memory_system.retrieve(
            query_embedding=input_embedding,
            emotions=current_emotions,
            k=5
        )
        print(f"   MEMORIES RETRIEVED: {len(retrieved)}")
        
        print("🎉 EMOTION-MEMORY INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ EMOTION-MEMORY INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# SECTION 5: FULL AGENT TEST
# ============================================================================

def test_full_agent():
    """TEST FULL AGENT WITH ALL COMPONENTS"""
    print("\n🤖 TESTING FULL AGENT")
    print("=" * 50)
    
    try:
        # IMPORT AGENT
        from overlay.core import AgentLoop
        from overlay.utils.langchain import LangChainConfig
        
        print("✅ IMPORTS SUCCESSFUL")
        
        # CREATE LLM CONFIG (USING OPENAI FOR NOW)
        llm_config = LangChainConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            reasoning_level="medium",
            max_tokens=128,
            temperature=0.7
        )
        print(f"✅ LLM CONFIG CREATED: {llm_config}")
        
        # CREATE AGENT
        agent = AgentLoop(
            run_id="colab_test_agent",
            llm_config=llm_config
        )
        print(f"✅ AGENT CREATED: {agent}")
        
        # GET INITIAL STATUS
        initial_status = agent.get_status()
        print(f"✅ INITIAL STATUS: {initial_status}")
        
        print("🎉 FULL AGENT TEST PASSED!")
        return agent
        
    except Exception as e:
        print(f"❌ FULL AGENT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# SECTION 6: 2-TICK ITERATION TEST
# ============================================================================

def test_2_tick_iteration(agent):
    """TEST 2-TICK ITERATION WITH FULL DEBUG"""
    print("\n🔄 TESTING 2-TICK ITERATION")
    print("=" * 50)
    
    if not agent:
        print("❌ AGENT NOT AVAILABLE")
        return False
    
    try:
        # TICK 1: SEED INPUT
        print("\n🔄 TICK 1: SEED INPUT")
        print("-" * 30)
        
        seed_input = "Hello! I'm starting my journey. How are you feeling today?"
        print(f"   SEED INPUT: {seed_input}")
        
        # PROCESS TICK 1
        print("\n   PROCESSING TICK 1...")
        response_1 = agent.process_input(seed_input)
        print(f"   RESPONSE 1: {response_1}")
        
        # GET STATUS AFTER TICK 1
        status_1 = agent.get_status()
        print(f"   STATUS AFTER TICK 1: {status_1}")
        
        # TICK 2: RESPONSE AS INPUT
        print("\n🔄 TICK 2: RESPONSE AS INPUT")
        print("-" * 30)
        
        # USE RESPONSE 1 AS INPUT FOR TICK 2
        tick_2_input = response_1
        print(f"   TICK 2 INPUT: {tick_2_input}")
        
        # PROCESS TICK 2
        print("\n   PROCESSING TICK 2...")
        response_2 = agent.process_input(tick_2_input)
        print(f"   RESPONSE 2: {response_2}")
        
        # GET STATUS AFTER TICK 2
        status_2 = agent.get_status()
        print(f"   STATUS AFTER TICK 2: {status_2}")
        
        # COMPARE STATES
        print("\n📊 STATE COMPARISON")
        print("-" * 30)
        print(f"   TICK 1 EMOTIONS: {status_1['emotions']}")
        print(f"   TICK 2 EMOTIONS: {status_2['emotions']}")
        print(f"   TICK 1 MEMORIES: {status_1['memory_count']}")
        print(f"   TICK 2 MEMORIES: {status_2['memory_count']}")
        print(f"   ATTENTION EVOLUTION: {status_2['attention_evolution']}")
        
        print("🎉 2-TICK ITERATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ 2-TICK ITERATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# SECTION 7: MAIN TEST RUNNER
# ============================================================================

def run_full_test_suite():
    """RUN THE COMPLETE TEST SUITE"""
    print("🚀 STARTING FULL TEST SUITE")
    print("=" * 60)
    
    # 1. DEPENDENCIES
    print("\n1️⃣ INSTALLING DEPENDENCIES")
    install_dependencies()
    
    # 2. OPENAI SETUP
    print("\n2️⃣ SETTING UP OPENAI API")
    if not setup_openai_api():
        print("❌ OPENAI SETUP FAILED - SKIPPING API TESTS")
        return False
    
    # 3. COMPONENT ISOLATION TESTS
    print("\n3️⃣ RUNNING COMPONENT ISOLATION TESTS")
    emotion_engine = test_emotion_engine_isolation()
    memory_system = test_memory_system_isolation()
    sentence_model = test_sentence_transformer_isolation()
    
    # 4. INTEGRATION TESTS
    print("\n4️⃣ RUNNING INTEGRATION TESTS")
    test_emotion_memory_integration()
    
    # 5. FULL AGENT TEST
    print("\n5️⃣ RUNNING FULL AGENT TEST")
    agent = test_full_agent()
    
    # 6. 2-TICK ITERATION
    if agent:
        print("\n6️⃣ RUNNING 2-TICK ITERATION TEST")
        test_2_tick_iteration(agent)
    
    print("\n🎉 FULL TEST SUITE COMPLETE!")
    return True

# ============================================================================
# SECTION 8: INDIVIDUAL TEST RUNNERS
# ============================================================================

if __name__ == "__main__":
    print("🔧 COLAB FULL DEBUG TEST FILE")
    print("💡 COPY SECTIONS INTO SEPARATE COLAB CELLS")
    print("💡 OR RUN: run_full_test_suite()")
    
    # UNCOMMENT TO RUN FULL SUITE
    # run_full_test_suite()
