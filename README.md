# R0-K0: Cognitive Overlay for Pretrained Models

**EMOTION-WEIGHTED MEMORY SYSTEM WITH GPT-OSS-20B INTEGRATION**

A cognitive overlay system that adds emotional intelligence, episodic memory, and identity formation to pretrained language models through attention-based memory influence.

## 🚀 Key Features

- **GPT-OSS-20B Integration**: Full integration with OpenAI's open-weight reasoning model
- **Attention-Based Memory Influence**: Memories directly modify LLM attention weights
- **6D Emotional Engine**: Configurable emotional dimensions with sigmoid probabilities
- **Episodic Memory System**: LangChain-powered vector similarity search with persistent storage
- **Identity Formation**: Persistent self-model that evolves through experiences
- **Narrative Reflection**: Automatic story generation and memory integration
- **Sleep Cycles**: Memory consolidation and dream generation
- **Lifespan Management**: Developmental stages from infancy to maturity

## 🧠 How It Works

### Memory Flow Architecture

```
Input Text → Emotion Inference → Memory Retrieval → Identity Update → LLM Processing
                                    ↓
                              Memory Influence
                                    ↓
                            Modified Attention Weights
                                    ↓
                              GPT-OSS-20B Generation
                                    ↓
                              Response + Memory Storage
```

**Each tick generates a complete response with memory-influenced attention:**

1. **Input Processing**: Text input triggers emotion inference
2. **Memory Retrieval**: Relevant memories found via semantic similarity
3. **Identity Integration**: Self-model updates based on current context
4. **Attention Modification**: Memories influence final attention layer
5. **Full Response Generation**: GPT-OSS-20B generates complete response
6. **Memory Storage**: Interaction stored as retrievable memory

### GPT-OSS-20B Internal Reasoning

The system leverages GPT-OSS-20B's **"Full chain-of-thought"** capabilities:

- **Configurable reasoning levels**: Low, Medium, High
- **Visible reasoning process**: Model shows thinking steps
- **Harmony response format**: Required for proper functioning
- **Agentic capabilities**: Function calling, structured outputs

## 🏗️ Architecture

### Core Components

- **`EmotionEngine`**: N-dimensional emotional state with sigmoid probabilities
- **`EpisodicMemory`**: Hybrid storage (LangChain vectors + .pt tensor files)
- **`SelfModel`**: Persistent identity embedding with stability tracking
- **`NarrativeEngine`**: Reflective story generation and memory integration
- **`SleepCycle`**: Memory consolidation and dream fusion
- **`LifespanManager`**: Life stage progression and developmental metrics

### Memory System

- **Vector Store**: Chroma/FAISS for semantic search
- **Tensor Storage**: .pt files for persistent tensor data
- **Hybrid Approach**: Best of both worlds - fast search + persistent storage

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from overlay.core import AgentLoop
from overlay.utils.langchain import LangChainConfig

# CONFIGURE GPT-OSS-20B
llm_config = LangChainConfig(
    provider="huggingface",
    model_name="openai/gpt-oss-20b",
    reasoning_level="medium",
    max_tokens=256
)

# CREATE AGENT
agent = AgentLoop(llm_config=llm_config)

# INTERACT
response = agent.process_input("Hello! How are you feeling?")
print(response)
```

### Run Example

```bash
python examples/gpt_oss_integration.py
```

## 🔧 Configuration

### Emotion Engine
```python
EmotionConfig(
    dimension=6,  # CONFIGURABLE DIMENSIONS
    labels=["joy", "sadness", "anger", "fear", "surprise", "disgust"]
)
```

### Memory System
```python
MemoryConfig(
    vector_store_type="chroma",  # OR "faiss"
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold=0.3
)
```

### GPT-OSS-20B
```python
LangChainConfig(
    reasoning_level="high",  # low, medium, high
    max_tokens=512,
    temperature=0.7
)
```

## 📁 Project Structure

```
R0-K0/
├── overlay/
│   ├── core.py              # MAIN AGENT LOOP WITH GPT-OSS-20B
│   ├── emotion.py           # EMOTIONAL ENGINE
│   ├── memory.py            # EPISODIC MEMORY SYSTEM
│   ├── self_model.py        # IDENTITY FORMATION
│   ├── narrative.py         # REFLECTIVE STORY GENERATION
│   ├── sleep.py             # MEMORY CONSOLIDATION
│   ├── lifespan.py          # LIFE CYCLE MANAGEMENT
│   └── utils/
│       └── langchain.py     # LLM INTEGRATION UTILITIES
├── examples/
│   └── gpt_oss_integration.py  # GPT-OSS-20B DEMO
└── requirements.txt
```

## 🎯 Use Cases

- **Conversational AI**: Emotional intelligence and memory
- **Personal Assistants**: Persistent identity and learning
- **Creative Writing**: Narrative generation and reflection
- **Research**: Long-term memory and reasoning
- **Education**: Adaptive learning with emotional context

## 🔬 Technical Details

### Memory Influence Mechanism

```python
# MEMORIES MODIFY ATTENTION WEIGHTS
final_attention = model.get_final_attention_layer()
modified_attention = final_attention * (1 + memory_influence * 0.3)

# GENERATE WITH MODIFIED ATTENTION
response = model.generate(
    prompt, 
    attention_weights=modified_attention
)
```

### Emotional Processing

```python
# SIGMOID PROBABILITIES FOR REALISTIC EMOTION LEVELS
emotion_probs = torch.sigmoid(emotion_vector)
# NOT softmax - allows multiple emotions simultaneously
```

### Identity Evolution

```python
# IDENTITY UPDATES BASED ON EXPERIENCES
new_identity = current_identity * stability_factor + 
               influence * (1 - stability_factor)
```

## 🚧 Future Enhancements

- **Attention Visualization**: See how memories influence generation
- **Dream Analysis**: Interpret dream fusion patterns
- **Emotional Contagion**: Emotions spread between interactions
- **Multi-Modal Integration**: Images, audio, and text
- **Collaborative Memory**: Shared memories between agents

## 📚 Dependencies

- **PyTorch**: Tensor operations and neural networks
- **Transformers**: GPT-OSS-20B integration
- **LangChain**: LLM orchestration and memory
- **ChromaDB/FAISS**: Vector similarity search
- **Sentence-Transformers**: Text embeddings

## 🤝 Contributing

This is an experimental system for exploring cognitive architectures in AI. Contributions welcome!

## 📄 License

Apache 2.0 - Build freely for experimentation and commercial use.
