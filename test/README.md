# 🧪 Test Directory - Colab Testing Suite

This directory contains comprehensive testing files for the R0-K0 cognitive overlay system, designed to be run in Google Colab.

## 📁 Files

- **`colab_setup.py`** - Initial environment setup and dependency installation
- **`colab_full_debug_test.py`** - Complete test suite with isolated component testing

## 🚀 Quick Start in Colab

### 1. **Environment Setup**
```python
# Copy and paste this into your first Colab cell
!git clone https://github.com/yourusername/R0-K0.git
%cd R0-K0
exec(open('test/colab_setup.py').read())
setup_colab_environment()
```

### 2. **Set OpenAI API Key**
```python
# Set your OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 3. **Run Individual Tests**
```python
# Copy sections from colab_full_debug_test.py into separate cells

# Test emotion engine isolation
exec(open('test/colab_full_debug_test.py').read())
test_emotion_engine_isolation()

# Test memory system isolation
test_memory_system_isolation()

# Test sentence transformer
test_sentence_transformer_isolation()
```

### 4. **Run Integration Tests**
```python
# Test emotion-memory integration
test_emotion_memory_integration()

# Test full agent
agent = test_full_agent()

# Test 2-tick iteration
test_2_tick_iteration(agent)
```

### 5. **Run Complete Test Suite**
```python
# Run everything at once
run_full_test_suite()
```

## 🔍 What Each Test Does

### **Component Isolation Tests**
- **Emotion Engine**: Tests emotion inference, state management, and OpenAI integration
- **Memory System**: Tests memory storage, retrieval, and attention weight handling
- **Sentence Transformer**: Tests embedding generation and similarity calculations

### **Integration Tests**
- **Emotion-Memory**: Tests how emotions and memories work together
- **Full Agent**: Tests the complete agent with all components
- **2-Tick Iteration**: Tests the feedback loop where each output becomes the next input

## 📊 Debug Output

Each test provides comprehensive debug output including:
- ✅ Success confirmations
- ❌ Error details with full tracebacks
- 🔍 Component state information
- 📈 Performance metrics
- 🧠 Attention evolution tracking

## 🚨 Troubleshooting

### **Common Issues**
1. **Import Errors**: Make sure you're in the R0-K0 directory
2. **API Key Issues**: Verify your OpenAI API key is set correctly
3. **Memory Issues**: Colab has limited RAM - some models may not fit
4. **Dependency Issues**: Run the setup script first

### **Debug Mode**
All tests run with DEBUG level logging. Check the console output for detailed information about what's happening at each step.

## 🎯 Expected Results

### **Successful Run Should Show**
- ✅ All component imports working
- ✅ OpenAI API connection successful
- ✅ Emotion inference working
- ✅ Memory storage and retrieval working
- ✅ Attention weights being captured and modified
- ✅ 2-tick iteration completing successfully

### **Key Metrics to Watch**
- **Emotion Evolution**: How emotions change between ticks
- **Memory Growth**: Number of memories stored
- **Attention Changes**: How attention patterns evolve
- **Response Coherence**: Whether responses build on previous interactions

## 💡 Tips for Colab

1. **Use GPU Runtime**: Enable GPU for better performance
2. **Monitor Memory**: Watch Colab's memory usage
3. **Save Outputs**: Copy important results to a separate document
4. **Run Incrementally**: Test components one at a time to isolate issues

## 🔧 Customization

You can modify the test parameters in `colab_full_debug_test.py`:
- Change emotion dimensions
- Adjust memory thresholds
- Modify attention influence factors
- Customize test inputs

**Happy Testing! 🎉**

