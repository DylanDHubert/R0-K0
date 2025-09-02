"""
Episodic Memory - Stores and retrieves experiences with emotional weighting using LangChain
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import uuid
import json
import os

# LANGCHAIN INTEGRATION
try:
    from langchain.memory import VectorStoreRetrieverMemory
    from langchain.vectorstores import Chroma, FAISS
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LANGCHAIN NOT AVAILABLE - USING FALLBACK MEMORY")

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """
    SINGLE MEMORY ENTRY WITH EMOTIONAL WEIGHTING AND ATTENTION WEIGHTS
    
    DIMENSIONS:
    - inputs: torch.Tensor (variable shape) - sensory/input data
    - outputs: torch.Tensor (variable shape) - model/action outputs  
    - emotions: torch.Tensor (N-dim) - emotional state vector
    - embedding: torch.Tensor (embedding_dim) - semantic representation
    - emotional_weight: float (0.0 to 1.0) - memory strength
    - attention_weights: Optional[torch.Tensor] - captured attention patterns
    - metadata: Dict - additional context (timestamp, access stats, etc.)
    """
    id: str
    timestamp: datetime
    inputs: torch.Tensor
    outputs: torch.Tensor
    emotions: torch.Tensor
    embedding: torch.Tensor
    emotional_weight: float
    attention_weights: Optional[torch.Tensor] = None
    access_count: int = 0
    last_accessed: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_document(self) -> 'Document':
        """CONVERT TO LANGCHAIN DOCUMENT FOR VECTOR STORE"""
        if not LANGCHAIN_AVAILABLE:
            return None
            
        # CREATE TEXT REPRESENTATION FOR VECTOR STORE
        content = f"Input: {self.inputs.shape}, Output: {self.outputs.shape}, Emotions: {self.emotions.tolist()}"
        
        # STORE TENSOR FILE PATHS AND BASIC METADATA
        metadata = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "emotional_weight": self.emotional_weight,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "emotions": self.emotions.tolist(),
            "input_shape": list(self.inputs.shape),
            "output_shape": list(self.outputs.shape),
            "embedding_shape": list(self.embedding.shape),
            "tensor_file": f"{self.id}.pt"  # TORCH FILE PATH
        }
        
        # ADD CUSTOM METADATA
        metadata.update(self.metadata)
        
        return Document(page_content=content, metadata=metadata)
    
    def save_tensors(self, run_dir: str) -> str:
        """SAVE TENSORS TO TORCH FILE AND RETURN FILE PATH"""
        memory_dir = os.path.join(run_dir, "memories")
        os.makedirs(memory_dir, exist_ok=True)
        
        file_path = os.path.join(memory_dir, f"{self.id}.pt")
        
        # SAVE ALL TENSORS TO SINGLE FILE INCLUDING ATTENTION WEIGHTS
        save_dict = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "emotions": self.emotions,
            "embedding": self.embedding
        }
        
        # ADD ATTENTION WEIGHTS IF AVAILABLE
        if self.attention_weights is not None:
            save_dict["attention_weights"] = self.attention_weights
        
        torch.save(save_dict, file_path)
        
        return file_path
    
    @classmethod
    def load_tensors(cls, run_dir: str, tensor_file: str) -> Dict[str, torch.Tensor]:
        """LOAD TENSORS FROM TORCH FILE INCLUDING ATTENTION WEIGHTS"""
        file_path = os.path.join(run_dir, "memories", tensor_file)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TENSOR FILE NOT FOUND: {file_path}")
        
        tensors = torch.load(file_path)
        
        # ENSURE ATTENTION WEIGHTS ARE INCLUDED (EVEN IF NONE)
        if "attention_weights" not in tensors:
            tensors["attention_weights"] = None
            
        return tensors
    
    @classmethod
    def from_document(cls, doc: 'Document', run_dir: str) -> 'MemoryEntry':
        """RECONSTRUCT MEMORY ENTRY FROM LANGCHAIN DOCUMENT AND TORCH FILE"""
        metadata = doc.metadata
        
        # LOAD TENSORS FROM FILE
        tensor_file = metadata.get("tensor_file")
        if not tensor_file:
            raise ValueError("NO TENSOR FILE SPECIFIED IN DOCUMENT")
        
        tensors = cls.load_tensors(run_dir, tensor_file)
        
        return cls(
            id=metadata["id"],
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            inputs=tensors["inputs"],
            outputs=tensors["outputs"],
            emotions=tensors["emotions"],
            embedding=tensors["embedding"],
            emotional_weight=metadata["emotional_weight"],
            attention_weights=tensors.get("attention_weights"),
            access_count=metadata.get("access_count", 0),
            last_accessed=datetime.fromisoformat(metadata["last_accessed"]) if metadata.get("last_accessed") else None,
            metadata={k: v for k, v in metadata.items() if k not in ["id", "timestamp", "emotional_weight", "access_count", "last_accessed", "emotions", "input_shape", "output_shape", "embedding_shape", "tensor_file"]}
        )

@dataclass
class MemoryConfig:
    """CONFIGURATION FOR EPISODIC MEMORY"""
    max_memories: int = 1000  # MAXIMUM NUMBER OF STORED MEMORIES
    storage_threshold: float = 0.1  # MINIMUM EMOTIONAL WEIGHT FOR STORAGE
    retrieval_k: int = 5  # NUMBER OF MEMORIES TO RETRIEVE
    decay_factor: float = 0.99  # MEMORY STRENGTH DECAY PER ACCESS
    emotional_resonance_weight: float = 0.7  # WEIGHT OF EMOTIONAL SIMILARITY IN RETRIEVAL
    
    # LANGCHAIN CONFIGURATION
    vector_store_type: str = "chroma"  # chroma, faiss
    embedding_model: str = "openai"  # openai, huggingface
    embedding_dim: int = 1536  # DIMENSION OF EMBEDDINGS
    similarity_threshold: float = 0.7  # MINIMUM SIMILARITY FOR RETRIEVAL

class LangChainMemoryStore:
    """LANGCHAIN-BASED MEMORY STORE WITH VECTOR SIMILARITY SEARCH"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.vector_store = None
        self.embeddings = None
        self.retriever = None
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """INITIALIZE LANGCHAIN VECTOR STORE AND EMBEDDINGS"""
        try:
            # CREATE EMBEDDINGS
            if self.config.embedding_model == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY NOT SET - USING FALLBACK EMBEDDINGS")
                    return
                self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            elif self.config.embedding_model == "huggingface":
                self.embeddings = HuggingFaceEmbeddings()
            else:
                logger.error(f"UNSUPPORTED EMBEDDING MODEL: {self.config.embedding_model}")
                return
            
            # CREATE VECTOR STORE
            if self.config.vector_store_type == "chroma":
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    collection_name="episodic_memory"
                )
            elif self.config.vector_store_type == "faiss":
                # FAISS NEEDS INITIAL DOCUMENTS TO CREATE INDEX
                self.vector_store = None  # WILL BE CREATED ON FIRST STORE
            else:
                logger.error(f"UNSUPPORTED VECTOR STORE: {self.config.vector_store_type}")
                return
            
            # CREATE RETRIEVER
            if self.vector_store:
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.retrieval_k}
                )
            
            logger.info(f"INITIALIZED LANGCHAIN MEMORY STORE: {self.config.vector_store_type}")
            
        except Exception as e:
            logger.error(f"FAILED TO INITIALIZE LANGCHAIN MEMORY STORE: {e}")
    
    def store(self, memory: MemoryEntry, run_dir: str) -> bool:
        """STORE MEMORY IN VECTOR STORE AND SAVE TENSORS TO DISK"""
        if not LANGCHAIN_AVAILABLE or not self.vector_store:
            return False
        
        try:
            # SAVE TENSORS TO DISK FIRST
            tensor_path = memory.save_tensors(run_dir)
            logger.debug(f"SAVED TENSORS TO: {tensor_path}")
            
            # CONVERT TO DOCUMENT
            doc = memory.to_document()
            if not doc:
                return False
            
            # STORE IN VECTOR STORE
            if self.config.vector_store_type == "faiss" and self.vector_store is None:
                # CREATE FAISS STORE ON FIRST STORE
                self.vector_store = FAISS.from_documents([doc], self.embeddings)
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": self.config.retrieval_k}
                )
            else:
                self.vector_store.add_documents([doc])
            
            return True
            
        except Exception as e:
            logger.error(f"FAILED TO STORE MEMORY IN VECTOR STORE: {e}")
            return False
    
    def retrieve(self, query_embedding: torch.Tensor, emotions: torch.Tensor, 
                 k: Optional[int] = None, run_dir: str = None) -> List[Document]:
        """RETRIEVE MEMORIES USING VECTOR SIMILARITY WITH EMOTIONAL FILTERING"""
        if not LANGCHAIN_AVAILABLE or not self.retriever:
            return []
        
        try:
            # CONVERT TENSOR TO LIST FOR LANGCHAIN
            query_vector = query_embedding.tolist()
            
            # RETRIEVE SIMILAR DOCUMENTS
            docs = self.retriever.get_relevant_documents(str(query_vector))
            
            # FILTER BY EMOTIONAL RESONANCE IF NEEDED
            if emotions is not None and run_dir:
                filtered_docs = []
                for doc in docs:
                    try:
                        # LOAD TENSORS TO CALCULATE EMOTIONAL RESONANCE
                        memory = MemoryEntry.from_document(doc, run_dir)
                        
                        # CALCULATE EMOTIONAL SIMILARITY
                        emotion_sim = torch.cosine_similarity(
                            emotions.unsqueeze(0), 
                            memory.emotions.unsqueeze(0)
                        ).item()
                        
                        # APPLY EMOTIONAL RESONANCE FILTER
                        if emotion_sim > self.config.similarity_threshold:
                            filtered_docs.append(doc)
                            
                    except Exception as e:
                        logger.warning(f"FAILED TO LOAD MEMORY FOR EMOTIONAL FILTERING: {e}")
                        # INCLUDE DOC IF EMOTIONAL FILTERING FAILS
                        filtered_docs.append(doc)
                
                docs = filtered_docs
            
            return docs[:k or self.config.retrieval_k]
            
        except Exception as e:
            logger.error(f"FAILED TO RETRIEVE MEMORIES: {e}")
            return []

class EpisodicMemory:
    """
    EPISODIC MEMORY WITH STORAGE/RETRIEVAL AND EMOTIONAL WEIGHTING
    
    MEMORY = RESTORING PRIOR WEIGHTS OF PERCEPTION AND FEELING
    HIGH-EMOTION EVENTS → HIGH PROBABILITY OF STORAGE
    RECALL RE-INJECTS BOTH SENSORY CONTEXT AND EMOTIONAL STATE
    
    ONE MEMORY TICK:
    1. INPUT: new experience (inputs, outputs, emotions, embedding)
    2. STORAGE: emotional weight > threshold? Store if yes
    3. RETRIEVAL: find similar memories using embedding + emotional resonance
    4. INTEGRATION: combine retrieved memories with current context
    5. UPDATE: store new memory, update access stats
    6. CLEANUP: remove weak memories if over capacity
    
    HYBRID STORAGE:
    - LANGCHAIN VECTOR STORE: handles similarity search with embeddings
    - TORCH FILES: store actual tensors on disk with file paths in metadata
    - FALLBACK STORE: keeps everything in memory for fast access
    """
    
    def __init__(self, config: MemoryConfig = None, run_dir: str = None):
        self.config = config or MemoryConfig()
        self.run_dir = run_dir
        
        # MEMORY STORAGE (FALLBACK)
        self.memories: List[MemoryEntry] = []
        
        # LANGCHAIN MEMORY STORE
        self.langchain_store = LangChainMemoryStore(self.config) if LANGCHAIN_AVAILABLE else None
        
        # MEMORY STATISTICS
        self.storage_count = 0
        self.retrieval_count = 0
        
        logger.info(f"INITIALIZED EPISODIC MEMORY WITH MAX {self.config.max_memories} MEMORIES")
        if self.langchain_store:
            logger.info("USING LANGCHAIN VECTOR STORE")
        else:
            logger.info("USING FALLBACK MEMORY STORE")
    
    def set_run_directory(self, run_dir: str):
        """SET RUN DIRECTORY FOR TENSOR STORAGE"""
        self.run_dir = run_dir
        logger.info(f"SET RUN DIRECTORY FOR MEMORY STORAGE: {run_dir}")
    
    def store(self, inputs: torch.Tensor, outputs: torch.Tensor, 
               emotions: torch.Tensor, embedding: torch.Tensor, 
               attention_weights: Optional[torch.Tensor] = None) -> bool:
        """
        STORE NEW MEMORY WITH EMOTIONAL WEIGHTING AND ATTENTION WEIGHTS
        
        STORAGE PROBABILITY ∝ EMOTION INTENSITY
        """
        # CALCULATE EMOTIONAL WEIGHT (INTENSITY OF EMOTIONAL STATE)
        emotional_weight = torch.norm(emotions).item()
        
        # CHECK IF EMOTIONAL WEIGHT IS HIGH ENOUGH FOR STORAGE
        if emotional_weight < self.config.storage_threshold:
            logger.debug(f"EMOTIONAL WEIGHT {emotional_weight:.3f} BELOW THRESHOLD, NOT STORING")
            return False
        
        # CREATE MEMORY ENTRY
        memory = MemoryEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            inputs=inputs.clone(),
            outputs=outputs.clone(),
            emotions=emotions.clone(),
            embedding=embedding.clone(),
            emotional_weight=emotional_weight,
            attention_weights=attention_weights.clone() if attention_weights is not None else None
        )
        
        # STORE IN LANGCHAIN VECTOR STORE IF AVAILABLE
        if self.langchain_store and self.run_dir:
            stored = self.langchain_store.store(memory, self.run_dir)
            if stored:
                logger.debug(f"STORED MEMORY {memory.id} IN LANGCHAIN VECTOR STORE")
        
        # ADD TO FALLBACK MEMORY STORE
        self.memories.append(memory)
        self.storage_count += 1
        
        # ENFORCE MEMORY LIMIT
        if len(self.memories) > self.config.max_memories:
            # REMOVE OLDEST, LEAST EMOTIONALLY WEIGHTED MEMORY
            self.memories.sort(key=lambda m: (m.emotional_weight, m.timestamp))
            removed = self.memories.pop(0)
            logger.debug(f"REMOVED MEMORY {removed.id} DUE TO CAPACITY LIMIT")
        
        logger.info(f"STORED MEMORY {memory.id} WITH EMOTIONAL WEIGHT {emotional_weight:.3f}")
        return True
    
    def retrieve(self, query_embedding: torch.Tensor, emotions: torch.Tensor, 
                 k: Optional[int] = None) -> List[MemoryEntry]:
        """
        RETRIEVE MEMORIES BASED ON SIMILARITY AND EMOTIONAL RESONANCE
        
        RETRIEVAL WEIGHTED BY SIMILARITY × EMOTIONAL RESONANCE
        """
        # TRY LANGCHAIN RETRIEVAL FIRST
        if self.langchain_store and self.run_dir:
            docs = self.langchain_store.retrieve(query_embedding, emotions, k, self.run_dir)
            if docs:
                # CONVERT DOCUMENTS BACK TO MEMORY ENTRIES
                langchain_memories = []
                for doc in docs:
                    try:
                        memory = MemoryEntry.from_document(doc, self.run_dir)
                        langchain_memories.append(memory)
                    except Exception as e:
                        logger.warning(f"FAILED TO RECONSTRUCT MEMORY FROM DOCUMENT: {e}")
                
                if langchain_memories:
                    logger.debug(f"RETRIEVED {len(langchain_memories)} MEMORIES FROM LANGCHAIN STORE")
                    # UPDATE ACCESS STATISTICS
                    for memory in langchain_memories:
                        memory.access_count += 1
                        memory.last_accessed = datetime.now()
                    
                    self.retrieval_count += 1
                    return langchain_memories
        
        # FALLBACK TO TRADITIONAL RETRIEVAL
        if not self.memories:
            return []
        
        k = k or self.config.retrieval_k
        
        # CALCULATE SIMILARITY SCORES
        similarities = []
        for memory in self.memories:
            # COSINE SIMILARITY BETWEEN EMBEDDINGS
            cos_sim = torch.cosine_similarity(
                query_embedding.unsqueeze(0), 
                memory.embedding.unsqueeze(0)
            ).item()
            
            # EMOTIONAL RESONANCE (SIMILARITY OF EMOTIONAL STATES)
            emotion_sim = torch.cosine_similarity(
                emotions.unsqueeze(0), 
                memory.emotions.unsqueeze(0)
            ).item()
            
            # COMBINED SCORE: SIMILARITY + EMOTIONAL RESONANCE
            combined_score = (1 - self.config.emotional_resonance_weight) * cos_sim + \
                           self.config.emotional_resonance_weight * emotion_sim
            
            similarities.append((combined_score, memory))
        
        # SORT BY COMBINED SCORE AND RETURN TOP K
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_memories = [memory for _, memory in similarities[:k]]
        
        # UPDATE ACCESS STATISTICS
        for memory in top_memories:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        
        self.retrieval_count += 1
        logger.debug(f"RETRIEVED {len(top_memories)} MEMORIES")
        
        return top_memories
    
    def consolidate(self, consolidation_factor: float = 0.1) -> None:
        """
        CONSOLIDATE MEMORIES - STRENGTHEN RELATED MEMORIES, WEAKEN ISOLATED ONES
        
        THIS IS CALLED DURING SLEEP CYCLES
        """
        if len(self.memories) < 2:
            return
        
        # FIND MEMORY CLUSTERS BASED ON EMBEDDING SIMILARITY
        clusters = self._find_memory_clusters()
        
        for cluster in clusters:
            if len(cluster) > 1:
                # STRENGTHEN CLUSTER MEMORIES
                for memory in cluster:
                    memory.emotional_weight *= (1 + consolidation_factor)
                    memory.emotional_weight = min(memory.emotional_weight, 1.0)
        
        # WEAKEN ISOLATED MEMORIES
        for memory in self.memories:
            if not any(memory in cluster for cluster in clusters if len(cluster) > 1):
                memory.emotional_weight *= (1 - consolidation_factor)
                memory.emotional_weight = max(memory.emotional_weight, 0.0)
        
        logger.info(f"CONSOLIDATED {len(clusters)} MEMORY CLUSTERS")
    
    def forget_weak_memories(self, threshold: float = 0.05) -> int:
        """
        REMOVE MEMORIES WITH VERY LOW EMOTIONAL WEIGHT
        
        RETURNS NUMBER OF FORGOTTEN MEMORIES
        """
        initial_count = len(self.memories)
        
        # REMOVE MEMORIES BELOW THRESHOLD
        self.memories = [m for m in self.memories if m.emotional_weight > threshold]
        
        forgotten_count = initial_count - len(self.memories)
        if forgotten_count > 0:
            logger.info(f"FORGOT {forgotten_count} WEAK MEMORIES")
        
        return forgotten_count
    
    def _find_memory_clusters(self, similarity_threshold: float = 0.7) -> List[List[MemoryEntry]]:
        """FIND CLUSTERS OF SIMILAR MEMORIES"""
        if len(self.memories) < 2:
            return []
        
        # SIMPLE CLUSTERING BASED ON EMBEDDING SIMILARITY
        clusters = []
        used = set()
        
        for i, memory1 in enumerate(self.memories):
            if i in used:
                continue
            
            cluster = [memory1]
            used.add(i)
            
            for j, memory2 in enumerate(self.memories[i+1:], i+1):
                if j in used:
                    continue
                
                similarity = torch.cosine_similarity(
                    memory1.embedding.unsqueeze(0),
                    memory2.embedding.unsqueeze(0)
                ).item()
                
                if similarity > similarity_threshold:
                    cluster.append(memory2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """RETURN MEMORY STATISTICS FOR MONITORING"""
        if not self.memories:
            return {
                "total_memories": 0,
                "avg_emotional_weight": 0.0,
                "storage_count": self.storage_count,
                "retrieval_count": self.retrieval_count
            }
        
        emotional_weights = [m.emotional_weight for m in self.memories]
        
        return {
            "total_memories": len(self.memories),
            "avg_emotional_weight": np.mean(emotional_weights),
            "max_emotional_weight": np.max(emotional_weights),
            "min_emotional_weight": np.min(emotional_weights),
            "storage_count": self.storage_count,
            "retrieval_count": self.retrieval_count
        }
    
    def clear(self) -> None:
        """CLEAR ALL MEMORIES AND TENSOR FILES"""
        # CLEAR IN-MEMORY STORES
        self.memories.clear()
        self.storage_count = 0
        self.retrieval_count = 0
        
        # CLEAR TENSOR FILES IF RUN DIRECTORY IS SET
        if self.run_dir:
            try:
                memory_dir = os.path.join(self.run_dir, "memories")
                if os.path.exists(memory_dir):
                    import shutil
                    shutil.rmtree(memory_dir)
                    logger.info(f"CLEARED TENSOR FILES FROM: {memory_dir}")
            except Exception as e:
                logger.error(f"FAILED TO CLEAR TENSOR FILES: {e}")
        
        logger.info("CLEARED ALL MEMORIES")
