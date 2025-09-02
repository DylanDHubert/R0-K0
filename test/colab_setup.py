#!/usr/bin/env python3
"""
COLAB SETUP SCRIPT - INITIAL ENVIRONMENT SETUP

Run this first in Colab to set up the environment.
"""

import os
import sys

def setup_colab_environment():
    """SETUP COLAB ENVIRONMENT FOR TESTING"""
    print("🔧 SETTING UP COLAB ENVIRONMENT")
    
    # 1. INSTALL DEPENDENCIES
    print("\n1️⃣ INSTALLING DEPENDENCIES...")
    os.system("pip install torch transformers sentence-transformers langchain openai chromadb")
    
    # 2. CLONE REPOSITORY (IF NOT ALREADY DONE)
    print("\n2️⃣ CHECKING REPOSITORY...")
    if not os.path.exists("R0-K0"):
        print("   CLONING R0-K0 REPOSITORY...")
        os.system("git clone https://github.com/yourusername/R0-K0.git")
        os.chdir("R0-K0")
    else:
        print("   REPOSITORY ALREADY EXISTS")
        if os.path.exists("R0-K0"):
            os.chdir("R0-K0")
    
    # 3. ADD TO PYTHON PATH
    print("\n3️⃣ SETTING PYTHON PATH...")
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"   ADDED {current_dir} TO PYTHON PATH")
    
    # 4. VERIFY IMPORTS
    print("\n4️⃣ VERIFYING IMPORTS...")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"   ❌ PyTorch: {e}")
    
    try:
        import transformers
        print(f"   ✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"   ❌ Transformers: {e}")
    
    try:
        import sentence_transformers
        print(f"   ✅ Sentence-Transformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"   ❌ Sentence-Transformers: {e}")
    
    try:
        import langchain
        print(f"   ✅ LangChain: {langchain.__version__}")
    except ImportError as e:
        print(f"   ❌ LangChain: {e}")
    
    try:
        import openai
        print(f"   ✅ OpenAI: {openai.__version__}")
    except ImportError as e:
        print(f"   ❌ OpenAI: {e}")
    
    # 5. SETUP OPENAI API KEY
    print("\n5️⃣ OPENAI API KEY SETUP...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ❌ OPENAI_API_KEY NOT SET")
        print("   💡 SET YOUR API KEY: os.environ['OPENAI_API_KEY'] = 'your-key-here'")
        print("   💡 OR USE: getpass() to input securely")
    else:
        print(f"   ✅ OPENAI API KEY FOUND: {api_key[:8]}...")
    
    # 6. TEST BASIC IMPORTS
    print("\n6️⃣ TESTING BASIC IMPORTS...")
    try:
        # TEST OVERLAY IMPORTS
        from overlay.emotion import EmotionEngine, EmotionConfig
        print("   ✅ Overlay emotion imports successful")
        
        from overlay.memory import EpisodicMemory, MemoryConfig
        print("   ✅ Overlay memory imports successful")
        
        from overlay.core import AgentLoop
        print("   ✅ Overlay core imports successful")
        
    except ImportError as e:
        print(f"   ❌ Overlay imports failed: {e}")
        print("   💡 Make sure you're in the R0-K0 directory")
    
    print("\n🎉 COLAB ENVIRONMENT SETUP COMPLETE!")
    print("💡 Next: Set your OpenAI API key and run the test suite")

def get_openai_key():
    """SECURELY GET OPENAI API KEY FROM USER"""
    try:
        from getpass import getpass
        api_key = getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
        print("✅ API key set successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to set API key: {e}")
        return False

if __name__ == "__main__":
    setup_colab_environment()
    
    # ASK FOR API KEY IF NOT SET
    if not os.getenv("OPENAI_API_KEY"):
        print("\n🔑 SETTING UP OPENAI API KEY...")
        get_openai_key()
