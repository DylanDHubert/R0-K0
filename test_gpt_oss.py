#!/usr/bin/env python3
"""
SIMPLE TEST FOR GPT-OSS-20B INTEGRATION
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from overlay.core import AgentLoop
from overlay.utils.langchain import LangChainConfig

def test_basic_integration():
    """TEST BASIC AGENT CREATION AND CONFIGURATION"""
    print("🧪 TESTING GPT-OSS-20B INTEGRATION")
    
    try:
        # CREATE MINIMAL CONFIG
        llm_config = LangChainConfig(
            provider="huggingface",
            model_name="openai/gpt-oss-20b",
            reasoning_level="low",  # FASTEST FOR TESTING
            max_tokens=64,
            temperature=0.7
        )
        
        print("✅ LangChainConfig created successfully")
        
        # CREATE AGENT
        agent = AgentLoop(
            run_id="test_run",
            llm_config=llm_config
        )
        
        print("✅ AgentLoop created successfully")
        print(f"📁 Run directory: {agent.run_dir}")
        print(f"🧠 LLM initialized: {agent.llm is not None}")
        
        # TEST STATUS
        status = agent.get_status()
        print(f"📊 Initial status: {status}")
        
        print("\n🎉 BASIC INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_influence():
    """TEST MEMORY INFLUENCE SYSTEM"""
    print("\n🧪 TESTING MEMORY INFLUENCE SYSTEM")
    
    try:
        # CREATE AGENT WITH MEMORY
        llm_config = LangChainConfig(
            provider="huggingface",
            model_name="openai/gpt-oss-20b",
            reasoning_level="low",
            max_tokens=64
        )
        
        agent = AgentLoop(llm_config=llm_config)
        
        # TEST MEMORY RETRIEVAL
        memories = agent._retrieve_relevant_memories("test input", agent.emotion_engine.get_state())
        print(f"✅ Memory retrieval: {len(memories)} memories found")
        
        # TEST IDENTITY CONVERSION
        identity = agent.self_model.get_identity_embedding()
        identity_text = agent._tensor_to_identity_description(identity)
        print(f"✅ Identity conversion: {identity_text}")
        
        print("🎉 MEMORY INFLUENCE TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ MEMORY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 STARTING GPT-OSS-20B INTEGRATION TESTS\n")
    
    test1_passed = test_basic_integration()
    test2_passed = test_memory_influence()
    
    print(f"\n📋 TEST SUMMARY:")
    print(f"   Basic Integration: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Memory Influence: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! GPT-OSS-20B INTEGRATION READY!")
    else:
        print("\n⚠️  SOME TESTS FAILED - CHECK ERRORS ABOVE")
        sys.exit(1)
