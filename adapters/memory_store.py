"""
Memory Store Adapter - Unified interface for different memory backends
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class MemoryStoreAdapter(ABC):
    """
    ABSTRACT BASE CLASS FOR MEMORY STORE BACKENDS
    
    PROVIDES UNIFIED INTERFACE FOR DIFFERENT MEMORY STORAGE SYSTEMS (FAISS, QDRANT, SQLITE)
    """
    
    def __init__(self, backend_name: str, config: Dict[str, Any] = None):
        self.backend_name = backend_name
        self.config = config or {}
        self.store = None
        self.is_initialized = False
        
        logger.info(f"INITIALIZING MEMORY STORE ADAPTER FOR {backend_name}")
    
    @abstractmethod
    def initialize(self) -> None:
        """INITIALIZE THE MEMORY STORE BACKEND"""
        pass
    
    @abstractmethod
    def store_memory(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """STORE A MEMORY WITH EMBEDDING AND METADATA"""
        pass
    
    @abstractmethod
    def retrieve_memories(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """RETRIEVE K MOST SIMILAR MEMORIES"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """DELETE A MEMORY BY ID"""
        pass
    
    @abstractmethod
    def get_memory_count(self) -> int:
        """RETURN TOTAL NUMBER OF STORED MEMORIES"""
        pass
    
    def is_ready(self) -> bool:
        """CHECK IF MEMORY STORE IS READY FOR USE"""
        return self.is_initialized and self.store is not None
    
    def get_store_info(self) -> Dict[str, Any]:
        """RETURN INFORMATION ABOUT THE MEMORY STORE"""
        if not self.is_ready():
            return {"status": "store_not_initialized"}
        
        info = {
            "backend_name": self.backend_name,
            "memory_count": self.get_memory_count(),
            "store_type": type(self.store).__name__
        }
        
        return info

class FAISSMemoryStore(MemoryStoreAdapter):
    """FAISS-BASED MEMORY STORE FOR VECTOR SIMILARITY SEARCH"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("faiss", config)
        self.dimension = config.get("dimension", 128)
        self.index_type = config.get("index_type", "flat")  # FLAT, IVF, HNSW
        self.metric = config.get("metric", "cosine")  # COSINE, EUCLIDEAN
        
        # MEMORY STORAGE
        self.memories = {}  # ID -> MEMORY DATA
        self.next_id = 0
    
    def initialize(self) -> None:
        """INITIALIZE FAISS INDEX"""
        try:
            import faiss
            
            # CREATE FAISS INDEX
            if self.index_type == "flat":
                if self.metric == "cosine":
                    # COSINE SIMILARITY REQUIRES NORMALIZED VECTORS
                    self.store = faiss.IndexFlatIP(self.dimension)
                else:
                    self.store = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                # INVERTED FILE INDEX FOR LARGE DATASETS
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.store = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                # HIERARCHICAL NAVIGABLE SMALL WORLD GRAPH
                self.store = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"UNSUPPORTED INDEX TYPE: {self.index_type}")
            
            self.is_initialized = True
            logger.info(f"INITIALIZED FAISS INDEX: {self.index_type}")
            
        except ImportError:
            logger.error("FAISS NOT INSTALLED")
            raise
    
    def store_memory(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """STORE MEMORY IN FAISS INDEX"""
        if not self.is_ready():
            self.initialize()
        
        # CONVERT EMBEDDING TO NUMPY ARRAY
        if isinstance(embedding, torch.Tensor):
            embedding_np = embedding.detach().cpu().numpy()
        else:
            embedding_np = np.array(embedding)
        
        # ENSURE CORRECT SHAPE
        if embedding_np.ndim == 1:
            embedding_np = embedding_np.reshape(1, -1)
        
        # NORMALIZE FOR COSINE SIMILARITY
        if self.metric == "cosine":
            embedding_np = embedding_np / np.linalg.norm(embedding_np, axis=1, keepdims=True)
        
        # ADD TO FAISS INDEX
        self.store.add(embedding_np)
        
        # STORE METADATA
        memory_id = f"memory_{self.next_id}"
        self.memories[memory_id] = {
            "id": memory_id,
            "embedding": embedding_np,
            "metadata": metadata,
            "index_position": self.store.ntotal - 1
        }
        
        self.next_id += 1
        
        logger.debug(f"STORED MEMORY {memory_id} IN FAISS INDEX")
        return memory_id
    
    def retrieve_memories(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """RETRIEVE K MOST SIMILAR MEMORIES"""
        if not self.is_ready():
            return []
        
        # CONVERT QUERY TO NUMPY ARRAY
        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.detach().cpu().numpy()
        else:
            query_np = np.array(query_embedding)
        
        # ENSURE CORRECT SHAPE
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        
        # NORMALIZE FOR COSINE SIMILARITY
        if self.metric == "cosine":
            query_np = query_np / np.linalg.norm(query_np, axis=1, keepdims=True)
        
        # SEARCH FAISS INDEX
        distances, indices = self.store.search(query_np, min(k, self.store.ntotal))
        
        # RETRIEVE MEMORIES
        retrieved_memories = []
        for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
            if index != -1:  # VALID INDEX
                # FIND MEMORY BY INDEX POSITION
                for memory_id, memory_data in self.memories.items():
                    if memory_data["index_position"] == index:
                        retrieved_memory = {
                            "id": memory_id,
                            "embedding": memory_data["embedding"],
                            "metadata": memory_data["metadata"],
                            "similarity_score": 1.0 / (1.0 + distance)  # CONVERT DISTANCE TO SIMILARITY
                        }
                        retrieved_memories.append(retrieved_memory)
                        break
        
        # SORT BY SIMILARITY SCORE
        retrieved_memories.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.debug(f"RETRIEVED {len(retrieved_memories)} MEMORIES FROM FAISS")
        return retrieved_memories[:k]
    
    def delete_memory(self, memory_id: str) -> bool:
        """DELETE MEMORY BY ID"""
        if memory_id not in self.memories:
            return False
        
        # REMOVE FROM METADATA STORE
        del self.memories[memory_id]
        
        # NOTE: FAISS DOESN'T SUPPORT EASY DELETION
        # IN PRACTICE, YOU'D NEED TO REBUILD THE INDEX
        logger.warning("FAISS DELETION NOT FULLY SUPPORTED - MEMORY REMOVED FROM METADATA ONLY")
        
        return True
    
    def get_memory_count(self) -> int:
        """RETURN TOTAL NUMBER OF STORED MEMORIES"""
        return len(self.memories)

class QdrantMemoryStore(MemoryStoreAdapter):
    """QDRANT-BASED MEMORY STORE FOR VECTOR DATABASE"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("qdrant", config)
        self.collection_name = config.get("collection_name", "memories")
        self.vector_size = config.get("vector_size", 128)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6333)
        
        # QDRANT CLIENT
        self.client = None
    
    def initialize(self) -> None:
        """INITIALIZE QDRANT CLIENT AND COLLECTION"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # CREATE QDRANT CLIENT
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # CREATE COLLECTION IF IT DOESN'T EXIST
            collections = self.client.get_collections()
            if self.collection_name not in [col.name for col in collections.collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"CREATED QDRANT COLLECTION: {self.collection_name}")
            else:
                logger.info(f"USING EXISTING QDRANT COLLECTION: {self.collection_name}")
            
            self.is_initialized = True
            
        except ImportError:
            logger.error("QDRANT_CLIENT NOT INSTALLED")
            raise
    
    def store_memory(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """STORE MEMORY IN QDRANT"""
        if not self.is_ready():
            self.initialize()
        
        # CONVERT EMBEDDING TO LIST
        if isinstance(embedding, torch.Tensor):
            embedding_list = embedding.detach().cpu().numpy().tolist()
        else:
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        
        # GENERATE MEMORY ID
        memory_id = f"memory_{len(self.get_all_memory_ids())}"
        
        # STORE IN QDRANT
        self.client.upsert(
            collection_name=self.collection_name,
            points=[{
                "id": memory_id,
                "vector": embedding_list,
                "payload": metadata
            }]
        )
        
        logger.debug(f"STORED MEMORY {memory_id} IN QDRANT")
        return memory_id
    
    def retrieve_memories(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """RETRIEVE K MOST SIMILAR MEMORIES FROM QDRANT"""
        if not self.is_ready():
            return []
        
        # CONVERT QUERY TO LIST
        if isinstance(query_embedding, torch.Tensor):
            query_list = query_embedding.detach().cpu().numpy().tolist()
        else:
            query_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
        
        # SEARCH QDRANT
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_list,
            limit=k
        )
        
        # CONVERT TO STANDARD FORMAT
        retrieved_memories = []
        for point in search_result:
            memory = {
                "id": point.id,
                "embedding": np.array(point.vector),
                "metadata": point.payload,
                "similarity_score": point.score
            }
            retrieved_memories.append(memory)
        
        logger.debug(f"RETRIEVED {len(retrieved_memories)} MEMORIES FROM QDRANT")
        return retrieved_memories
    
    def delete_memory(self, memory_id: str) -> bool:
        """DELETE MEMORY FROM QDRANT"""
        if not self.is_ready():
            return False
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )
            logger.debug(f"DELETED MEMORY {memory_id} FROM QDRANT")
            return True
        except Exception as e:
            logger.error(f"ERROR DELETING MEMORY {memory_id}: {e}")
            return False
    
    def get_memory_count(self) -> int:
        """RETURN TOTAL NUMBER OF STORED MEMORIES"""
        if not self.is_ready():
            return 0
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"ERROR GETTING MEMORY COUNT: {e}")
            return 0
    
    def get_all_memory_ids(self) -> List[str]:
        """RETURN ALL MEMORY IDS (FOR INTERNAL USE)"""
        if not self.is_ready():
            return []
        
        try:
            # SCROLL THROUGH ALL POINTS TO GET IDS
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000  # MAX LIMIT
            )
            return [point.id for point in points[0]]
        except Exception as e:
            logger.error(f"ERROR GETTING MEMORY IDS: {e}")
            return []

class SQLiteMemoryStore(MemoryStoreAdapter):
    """SQLITE-BASED MEMORY STORE FOR SIMPLE PERSISTENCE"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("sqlite", config)
        self.db_path = config.get("db_path", "memories.db")
        self.vector_size = config.get("vector_size", 128)
        
        # SQLITE CONNECTION
        self.connection = None
    
    def initialize(self) -> None:
        """INITIALIZE SQLITE DATABASE AND TABLES"""
        try:
            import sqlite3
            
            # CREATE DATABASE CONNECTION
            self.connection = sqlite3.connect(self.db_path)
            
            # CREATE MEMORIES TABLE
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    embedding BLOB,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # CREATE INDEX FOR SIMILARITY SEARCH
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
            """)
            
            self.connection.commit()
            self.is_initialized = True
            
            logger.info(f"INITIALIZED SQLITE MEMORY STORE: {self.db_path}")
            
        except ImportError:
            logger.error("SQLITE3 NOT AVAILABLE")
            raise
    
    def store_memory(self, embedding: torch.Tensor, metadata: Dict[str, Any}) -> str:
        """STORE MEMORY IN SQLITE"""
        if not self.is_ready():
            self.initialize()
        
        # CONVERT EMBEDDING TO BYTES
        if isinstance(embedding, torch.Tensor):
            embedding_bytes = embedding.detach().cpu().numpy().tobytes()
        else:
            embedding_bytes = np.array(embedding).tobytes()
        
        # CONVERT METADATA TO JSON
        import json
        metadata_json = json.dumps(metadata)
        
        # GENERATE MEMORY ID
        memory_id = f"memory_{self.get_memory_count()}"
        
        # STORE IN SQLITE
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO memories (id, embedding, metadata)
            VALUES (?, ?, ?)
        """, (memory_id, embedding_bytes, metadata_json))
        
        self.connection.commit()
        
        logger.debug(f"STORED MEMORY {memory_id} IN SQLITE")
        return memory_id
    
    def retrieve_memories(self, query_embedding: torch.Tensor, k: int = 5) -> List[Dict[str, Any]]:
        """RETRIEVE K MOST SIMILAR MEMORIES FROM SQLITE"""
        if not self.is_ready():
            return []
        
        # CONVERT QUERY TO NUMPY ARRAY
        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.detach().cpu().numpy()
        else:
            query_np = np.array(query_embedding)
        
        # GET ALL MEMORIES (SIMPLIFIED - NO VECTOR SIMILARITY SEARCH)
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, embedding, metadata FROM memories
            ORDER BY timestamp DESC
            LIMIT ?
        """, (k,))
        
        memories = []
        for row in cursor.fetchall():
            memory_id, embedding_bytes, metadata_json = row
            
            # CONVERT EMBEDDING BACK TO NUMPY ARRAY
            embedding_np = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # PARSE METADATA
            metadata = json.loads(metadata_json)
            
            # CALCULATE SIMPLIFIED SIMILARITY (COSINE)
            similarity = np.dot(query_np, embedding_np) / (np.linalg.norm(query_np) * np.linalg.norm(embedding_np))
            
            memory = {
                "id": memory_id,
                "embedding": embedding_np,
                "metadata": metadata,
                "similarity_score": similarity
            }
            memories.append(memory)
        
        # SORT BY SIMILARITY SCORE
        memories.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.debug(f"RETRIEVED {len(memories)} MEMORIES FROM SQLITE")
        return memories[:k]
    
    def delete_memory(self, memory_id: str) -> bool:
        """DELETE MEMORY FROM SQLITE"""
        if not self.is_ready():
            return False
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.connection.commit()
            
            logger.debug(f"DELETED MEMORY {memory_id} FROM SQLITE")
            return True
        except Exception as e:
            logger.error(f"ERROR DELETING MEMORY {memory_id}: {e}")
            return False
    
    def get_memory_count(self) -> int:
        """RETURN TOTAL NUMBER OF STORED MEMORIES"""
        if not self.is_ready():
            return 0
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"ERROR GETTING MEMORY COUNT: {e}")
            return 0
    
    def close(self) -> None:
        """CLOSE SQLITE CONNECTION"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __del__(self):
        """CLEANUP WHEN ADAPTER IS DESTROYED"""
        self.close()
