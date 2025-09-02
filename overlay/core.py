"""
CORE AGENT LOOP WITH GPT-OSS-20B INTEGRATION AND ATTENTION-BASED MEMORY INFLUENCE
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os

from .emotion import EmotionEngine, EmotionConfig
from .memory import EpisodicMemory, MemoryConfig
from .lifespan import LifespanManager, LifespanConfig
from .self_model import SelfModel, SelfModelConfig
from .narrative import NarrativeEngine, NarrativeConfig
from .sleep import SleepCycle, SleepConfig
from .memory import MemoryEntry
from .utils.langchain import LangChainConfig, create_gpt_oss_llm, call_gpt_oss_with_memory

logger = logging.getLogger(__name__)

class AgentLoop:
    """MAIN AGENT LOOP WITH GPT-OSS-20B AND MEMORY INFLUENCE"""
    
    def __init__(self, 
                 run_id: str = None,
                 emotion_config: EmotionConfig = None,
                 memory_config: MemoryConfig = None,
                 lifespan_config: LifespanConfig = None,
                 self_model_config: SelfModelConfig = None,
                 narrative_config: NarrativeConfig = None,
                 sleep_config: SleepConfig = None,
                 llm_config: LangChainConfig = None):
        
        # GENERATE RUN ID IF NOT PROVIDED
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = os.path.join("runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # INITIALIZE COMPONENTS
        self.emotion_engine = EmotionEngine(emotion_config or EmotionConfig())
        self.memory_system = EpisodicMemory(memory_config or MemoryConfig())
        self.lifespan_manager = LifespanManager(lifespan_config or LifespanConfig())
        self.self_model = SelfModel(self_model_config or SelfModelConfig())
        self.narrative_engine = NarrativeEngine(narrative_config or NarrativeConfig())
        self.sleep_cycle = SleepCycle(sleep_config or SleepConfig())
        
        # SET RUN DIRECTORY FOR COMPONENTS
        self.emotion_engine.set_run_directory(self.run_dir)
        self.memory_system.set_run_directory(self.run_dir)
        
        # INITIALIZE LLM
        self.llm_config = llm_config or LangChainConfig()
        self.llm = create_gpt_oss_llm(self.llm_config)
        
        if not self.llm:
            logger.error("FAILED TO INITIALIZE GPT-OSS-20B - AGENT WILL NOT FUNCTION")
        
        # INITIALIZE EXTENSIONS
        self._initialize_extensions()
        
        # SETUP EMOTION INFERENCE COMPONENT
        self._setup_emotion_inference()
        
        # AGENT STATE
        self.current_tick = 0
        self.is_alive = True
        self.current_input = ""
        self.current_output = ""
        
        # ATTENTION MODIFICATION CONFIGURATION
        self.memory_influence_factor = 0.3  # HOW STRONGLY MEMORIES MODIFY ATTENTION
        
        # FEEDBACK LOOP TRACKING
        self.attention_evolution_history = []  # TRACK HOW ATTENTION PATTERNS EVOLVE
        self.memory_attention_map = {}  # MAP MEMORY IDS TO ATTENTION PATTERNS
        
        logger.info(f"AGENT LOOP INITIALIZED: {self.run_id}")
    
    def _initialize_extensions(self):
        """INITIALIZE COMPONENTS THAT NEED CROSS-REFERENCES"""
        # NARRATIVE ENGINE NEEDS MEMORY SYSTEM
        self.narrative_engine.set_memory_system(self.memory_system)
        
        # SLEEP CYCLE NEEDS MEMORY SYSTEM
        self.sleep_cycle.set_memory_system(self.memory_system)
    
    def _setup_emotion_inference(self):
        """SETUP EMOTION INFERENCE COMPONENT WITH LANGCHAIN"""
        try:
            from overlay.emotion import APIBasedInference
            
            # CREATE API-BASED EMOTION INFERENCE
            emotion_inference = APIBasedInference(
                config=self.emotion_engine.config,
                langchain_config=self.llm_config,
                model_name=self.llm_config.model_name
            )
            
            # SET THE INFERENCE COMPONENT
            self.emotion_engine.set_emotion_inference(emotion_inference)
            
            logger.info("SETUP EMOTION INFERENCE COMPONENT")
            
        except Exception as e:
            logger.error(f"FAILED TO SETUP EMOTION INFERENCE: {e}")
            logger.warning("EMOTION ENGINE WILL USE RANDOM EMOTIONS")
    
    def process_input(self, text_input: str) -> str:
        """PROCESS TEXT INPUT THROUGH THE AGENT LOOP"""
        if not self.is_alive:
            return "AGENT IS DEAD - CANNOT PROCESS INPUT"
        
        self.current_input = text_input
        self.current_tick += 1
        
        logger.info(f"TICK {self.current_tick}: Processing input")
        
        try:
            # CHECK LLM AVAILABILITY
            if not self.llm:
                return "LLM NOT AVAILABLE - AGENT CANNOT PROCESS INPUT"
            
            # 1. EMOTION INFERENCE
            emotion_gradient = self.emotion_engine.infer_emotion(text_input)
            # UPDATE EMOTIONS WITH INFERRED GRADIENT
            # USE ACTUAL INPUT TEXT FOR EMOTION UPDATE
            input_tensor = torch.tensor([ord(c) for c in text_input[:128]], dtype=torch.float32)
            self.emotion_engine.update(inputs=input_tensor, outputs=torch.tensor([0]), emotion_gradient=emotion_gradient)
            current_emotions = self.emotion_engine.get_state()
            
            # 2. MEMORY RETRIEVAL
            relevant_memories, memory_texts = self._retrieve_relevant_memories(text_input, current_emotions)
            
            # 3. IDENTITY UPDATE
            self.self_model.update(current_emotions, relevant_memories, text_input)
            current_identity = self.self_model.get_identity_embedding()
            
            # HANDLE CASE WHERE NO MEMORIES EXIST
            if not relevant_memories:
                logger.debug("NO RELEVANT MEMORIES FOUND - USING DEFAULT CONTEXT")
                memory_texts = ["No previous memories available"]
            
            # 4. LLM PROCESSING WITH MEMORY INFLUENCE
            response = self._process_with_llm(text_input, memory_texts, current_identity, relevant_memories)
            
            # 5. MEMORY STORAGE WITH RESPONSE ATTENTION CAPTURE
            self._store_interaction(text_input, response, current_emotions)
            
            # 6. LIFESPAN UPDATE
            self.lifespan_manager.update_tick()
            
            # 7. NARRATIVE REFLECTION
            if self.narrative_engine.should_reflect(current_emotions, len(relevant_memories)):
                try:
                    self.narrative_engine.reflect(text_input, current_emotions, self.current_tick)
                except Exception as e:
                    logger.error(f"NARRATIVE REFLECTION FAILED: {e}")
            
            # 8. SLEEP CHECK
            if self.sleep_cycle.should_sleep(self.current_tick, len(relevant_memories)):
                try:
                    self.sleep_cycle.sleep(self.memory_system, self.emotion_engine, self.self_model)
                except Exception as e:
                    logger.error(f"SLEEP CYCLE FAILED: {e}")
            
            # 9. DEATH CHECK
            if not self.lifespan_manager.is_alive():
                self.is_alive = False
                logger.warning("AGENT HAS DIED")
            
            self.current_output = response
            return response
            
        except Exception as e:
            logger.error(f"ERROR IN AGENT LOOP: {e}")
            return f"ERROR: {str(e)}"
    
    def _retrieve_relevant_memories(self, input_text: str, emotions: torch.Tensor) -> Tuple[List[MemoryEntry], List[str]]:
        """RETRIEVE RELEVANT MEMORIES FOR CONTEXT AND TEXT SUMMARIES"""
        try:
            # CREATE TEXT EMBEDDING FOR SIMILARITY SEARCH
            query_embedding = self._create_text_embedding(input_text)
            
            # GET TOP MEMORIES BASED ON SEMANTIC SIMILARITY AND EMOTIONAL RESONANCE
            memories = self.memory_system.retrieve(
                query_embedding=query_embedding,  # FIXED: PASS EMBEDDING NOT TEXT
                emotions=emotions,
                k=5
            )
            
            # CONVERT TO TEXT FOR LLM CONTEXT
            memory_texts = []
            for memory in memories:
                # EXTRACT KEY INFORMATION FROM MEMORY
                # CONVERT TENSOR BACK TO READABLE TEXT
                input_text_reconstructed = self._tensor_to_text(memory.inputs)
                output_text_reconstructed = self._tensor_to_text(memory.outputs)
                
                memory_summary = f"Input: {input_text_reconstructed[:100]}... | Output: {output_text_reconstructed[:100]}... | Emotional Weight: {memory.emotional_weight:.3f}"
                memory_texts.append(memory_summary)
            
            logger.debug(f"RETRIEVED {len(memories)} RELEVANT MEMORIES")
            return memories, memory_texts
            
        except Exception as e:
            logger.error(f"MEMORY RETRIEVAL FAILED: {e}")
            return [], []
    
    def _tensor_to_text(self, tensor: torch.Tensor) -> str:
        """CONVERT TENSOR BACK TO READABLE TEXT"""
        try:
            # CONVERT ASCII VALUES BACK TO CHARACTERS
            chars = []
            for val in tensor:
                if val > 0:  # SKIP PADDING ZEROS
                    chars.append(chr(int(val)))
            
            return ''.join(chars)
        except Exception as e:
            logger.error(f"FAILED TO CONVERT TENSOR TO TEXT: {e}")
            return "Unknown"
    
    def _process_with_llm(self, input_text: str, memory_texts: List[str], identity: torch.Tensor, memory_objects: List[MemoryEntry]) -> str:
        """PROCESS INPUT THROUGH GPT-OSS-20B WITH ATTENTION-BASED MEMORY INFLUENCE"""
        if not self.llm:
            return "LLM NOT AVAILABLE"
        
        try:
            # CONVERT IDENTITY TENSOR TO TEXT DESCRIPTION
            identity_text = self._tensor_to_identity_description(identity)
            
            # MODIFY ATTENTION WITH STORED MEMORY PATTERNS
            modified_response = self._generate_with_modified_attention(
                input_text, memory_texts, identity_text, memory_objects
            )
            
            return modified_response
            
        except Exception as e:
            logger.error(f"LLM PROCESSING FAILED: {e}")
            return f"LLM ERROR: {str(e)}"
    
    def _generate_with_modified_attention(self, input_text: str, memory_texts: List[str], identity_text: str, memory_objects: List[MemoryEntry]) -> str:
        """GENERATE RESPONSE WITH MEMORY-MODIFIED ATTENTION WEIGHTS"""
        try:
            if not self.llm or 'pipeline' not in self.llm:
                # FALLBACK TO REGULAR GENERATION
                return call_gpt_oss_with_memory(
                    prompt=input_text,
                    memories=memory_texts,
                    identity=identity_text,
                    llm_config=self.llm,
                    memory_influence=0.3
                )
            
            # GET THE ACTUAL MODEL FROM THE PIPELINE
            model = self.llm['pipeline'].model
            tokenizer = self.llm['pipeline'].tokenizer
            
            # BUILD PROMPT WITH MEMORY CONTEXT
            full_prompt = f"""Context: {input_text}

Relevant Memories:
{chr(10).join([f"- {memory}" for memory in memory_texts])}

Identity Context: {identity_text}

Generate response:"""
            
            # TOKENIZE INPUT
            inputs = tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # CAPTURE BASE ATTENTION PATTERNS
            base_attention_output = {}
            
            def base_hook_fn(module, input, output):
                base_attention_output["out"] = output
            
            # REGISTER HOOK ON LAST ATTENTION LAYER
            last_attn_layer = model.model.layers[-1].self_attn
            base_handle = last_attn_layer.register_forward_hook(base_hook_fn)
            
            try:
                # RUN BASE FORWARD PASS TO GET ATTENTION PATTERNS
                with torch.no_grad():
                    _ = model(**inputs)
                
                base_attention = base_attention_output.get("out")
                
                if base_attention is None:
                    logger.warning("FAILED TO CAPTURE BASE ATTENTION - USING REGULAR GENERATION")
                    return call_gpt_oss_with_memory(
                        prompt=input_text,
                        memories=memory_texts,
                        identity=identity_text,
                        llm_config=self.llm,
                        memory_influence=0.3
                    )
                
                # MODIFY ATTENTION WITH STORED MEMORY PATTERNS
                modified_attention = self._modify_attention_with_memories(base_attention, memory_objects)
                
                # GENERATE WITH MODIFIED ATTENTION
                logger.info(f"ATTENTION MODIFIED WITH {len(memory_objects)} MEMORY PATTERNS")
                
                # CUSTOM GENERATION WITH INJECTED ATTENTION
                response = self._generate_with_injected_attention(
                    model, tokenizer, full_prompt, modified_attention
                )
                
                return response
                
            finally:
                base_handle.remove()
                
        except Exception as e:
            logger.error(f"ATTENTION MODIFICATION FAILED: {e}")
            # FALLBACK TO REGULAR GENERATION
            return call_gpt_oss_with_memory(
                prompt=input_text,
                memories=memory_texts,
                identity=identity_text,
                llm_config=self.llm,
                memory_influence=0.3
            )
    
    def _modify_attention_with_memories(self, base_attention: torch.Tensor, memory_objects: List[MemoryEntry]) -> torch.Tensor:
        """MODIFY ATTENTION WEIGHTS WITH STORED MEMORY PATTERNS"""
        try:
            logger.debug(f"BASE ATTENTION SHAPE: {base_attention.shape}")
            logger.debug(f"MODIFYING ATTENTION WITH {len(memory_objects)} MEMORIES")
            
            if not memory_objects:
                logger.debug("NO MEMORIES TO MODIFY ATTENTION WITH")
                return base_attention
            
            # START WITH BASE ATTENTION
            modified_attention = base_attention.clone()
            
            # APPLY MEMORY ATTENTION MODIFICATIONS
            for memory in memory_objects:
                if memory.attention_weights is not None:
                    try:
                        # GET STORED ATTENTION WEIGHTS
                        stored_attention = memory.attention_weights
                        
                        # CALCULATE INFLUENCE BASED ON EMOTIONAL WEIGHT
                        memory_influence = memory.emotional_weight * self.memory_influence_factor
                        
                        logger.debug(f"MEMORY {memory.id}: EMOTIONAL WEIGHT {memory.emotional_weight:.3f}, INFLUENCE {memory_influence:.3f}")
                        
                        # ENSURE SHAPE COMPATIBILITY
                        if stored_attention.shape == base_attention.shape:
                            # MULTIPLICATIVE MODIFICATION: ATTENTION *= (1 + INFLUENCE * STORED_PATTERN)
                            attention_modification = 1 + memory_influence * stored_attention
                            modified_attention *= attention_modification
                            
                            logger.debug(f"APPLIED ATTENTION MODIFICATION FROM MEMORY {memory.id}")
                        else:
                            logger.warning(f"MEMORY {memory.id} ATTENTION SHAPE MISMATCH: {stored_attention.shape} vs {base_attention.shape}")
                            
                    except Exception as e:
                        logger.error(f"FAILED TO APPLY ATTENTION FROM MEMORY {memory.id}: {e}")
                        continue
                else:
                    logger.debug(f"MEMORY {memory.id} HAS NO STORED ATTENTION WEIGHTS")
            
            # LOG FINAL ATTENTION MODIFICATION
            attention_change = torch.norm(modified_attention - base_attention).item()
            logger.info(f"ATTENTION MODIFICATION COMPLETE: TOTAL CHANGE {attention_change:.6f}")
            
            # TRACK ATTENTION EVOLUTION FOR FEEDBACK LOOP ANALYSIS
            self._track_attention_evolution(base_attention, modified_attention, memory_objects)
            
            return modified_attention
            
        except Exception as e:
            logger.error(f"ATTENTION MODIFICATION FAILED: {e}")
            return base_attention
    
    def _track_attention_evolution(self, base_attention: torch.Tensor, modified_attention: torch.Tensor, memory_objects: List[MemoryEntry]):
        """TRACK ATTENTION EVOLUTION FOR FEEDBACK LOOP ANALYSIS"""
        try:
            # CALCULATE ATTENTION CHANGE METRICS
            attention_change = torch.norm(modified_attention - base_attention).item()
            attention_similarity = torch.cosine_similarity(
                base_attention.flatten(), 
                modified_attention.flatten(), 
                dim=0
            ).item()
            
            # RECORD EVOLUTION DATA
            evolution_entry = {
                "tick": self.current_tick,
                "timestamp": datetime.now().isoformat(),
                "attention_change": attention_change,
                "attention_similarity": attention_similarity,
                "memory_count": len(memory_objects),
                "memory_ids": [m.id for m in memory_objects if m.attention_weights is not None],
                "base_attention_shape": list(base_attention.shape),
                "modified_attention_shape": list(modified_attention.shape)
            }
            
            self.attention_evolution_history.append(evolution_entry)
            
            # UPDATE MEMORY ATTENTION MAP
            for memory in memory_objects:
                if memory.attention_weights is not None:
                    self.memory_attention_map[memory.id] = {
                        "attention_shape": list(memory.attention_weights.shape),
                        "emotional_weight": memory.emotional_weight,
                        "last_used": self.current_tick
                    }
            
            logger.debug(f"ATTENTION EVOLUTION TRACKED: CHANGE={attention_change:.6f}, SIMILARITY={attention_similarity:.6f}")
            
        except Exception as e:
            logger.error(f"ATTENTION EVOLUTION TRACKING FAILED: {e}")
    
    def get_attention_evolution_stats(self) -> Dict[str, Any]:
        """GET STATISTICS ABOUT ATTENTION EVOLUTION"""
        if not self.attention_evolution_history:
            return {"status": "No attention evolution data available"}
        
        try:
            changes = [entry["attention_change"] for entry in self.attention_evolution_history]
            similarities = [entry["attention_similarity"] for entry in self.attention_evolution_history]
            
            return {
                "total_ticks": len(self.attention_evolution_history),
                "avg_attention_change": sum(changes) / len(changes),
                "avg_attention_similarity": sum(similarities) / len(similarities),
                "max_attention_change": max(changes),
                "min_attention_change": min(changes),
                "memory_attention_patterns": len(self.memory_attention_map),
                "recent_evolution": self.attention_evolution_history[-5:] if len(self.attention_evolution_history) >= 5 else self.attention_evolution_history
            }
        except Exception as e:
            logger.error(f"FAILED TO GET ATTENTION EVOLUTION STATS: {e}")
            return {"status": "Error getting evolution stats"}
    
    def _generate_with_injected_attention(self, model, tokenizer, prompt: str, modified_attention: torch.Tensor) -> str:
        """GENERATE RESPONSE WITH INJECTED ATTENTION WEIGHTS"""
        try:
            # TOKENIZE THE PROMPT
            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            input_ids = inputs["input_ids"]
            
            # SETUP GENERATION PARAMETERS
            max_new_tokens = self.llm_config.max_tokens if hasattr(self.llm_config, 'max_tokens') else 256
            temperature = self.llm_config.temperature if hasattr(self.llm_config, 'temperature') else 0.7
            
            # STORAGE FOR GENERATED TOKENS
            generated_tokens = []
            current_input_ids = input_ids.clone()
            
            # ATTENTION INJECTION HOOK WITH BETTER MANAGEMENT
            attention_injection_data = {"modified_attention": modified_attention}
            
            def attention_injection_hook(module, input, output):
                """HOOK TO INJECT MODIFIED ATTENTION WEIGHTS"""
                try:
                    # MODIFY THE ATTENTION OUTPUT WITH OUR MEMORY-INFLUENCED PATTERNS
                    if attention_injection_data["modified_attention"] is not None:
                        # ENSURE SHAPE COMPATIBILITY
                        stored_attention = attention_injection_data["modified_attention"]
                        if stored_attention.shape == output.shape:
                            # APPLY ATTENTION MODIFICATION
                            modified_output = output * stored_attention
                            # NORMALIZE TO PREVENT EXPLODING VALUES
                            modified_output = modified_output / (torch.norm(modified_output) + 1e-8)
                            return modified_output
                        else:
                            logger.warning(f"ATTENTION SHAPE MISMATCH: {stored_attention.shape} vs {output.shape}")
                            # TRY TO RESHAPE IF POSSIBLE
                            try:
                                if len(stored_attention.shape) == len(output.shape):
                                    # ATTEMPT TO BROADCAST OR RESHAPE
                                    if stored_attention.numel() == output.numel():
                                        reshaped_attention = stored_attention.reshape(output.shape)
                                        modified_output = output * reshaped_attention
                                        modified_output = modified_output / (torch.norm(modified_output) + 1e-8)
                                        logger.info(f"RESHAPED ATTENTION FROM {stored_attention.shape} TO {output.shape}")
                                        return modified_output
                            except Exception as reshape_error:
                                logger.error(f"ATTENTION RESHAPE FAILED: {reshape_error}")
                    
                    return output
                except Exception as e:
                    logger.error(f"ATTENTION INJECTION HOOK FAILED: {e}")
                    return output
            
            # REGISTER HOOK ON LAST ATTENTION LAYER WITH BETTER MANAGEMENT
            last_attn_layer = model.model.layers[-1].self_attn
            injection_handle = last_attn_layer.register_forward_hook(attention_injection_hook)
            
            # ENSURE HOOK IS PROPERLY MANAGED
            if not hasattr(self, '_active_hooks'):
                self._active_hooks = []
            self._active_hooks.append(injection_handle)
            
            try:
                # CUSTOM GENERATION LOOP WITH ATTENTION INJECTION
                for _ in range(max_new_tokens):
                    # FORWARD PASS WITH ATTENTION INJECTION
                    with torch.no_grad():
                        # CREATE ATTENTION MASK FOR INJECTION
                        attention_mask = torch.ones_like(current_input_ids)
                        
                        # RUN MODEL WITH INJECTED ATTENTION (HOOK WILL MODIFY ATTENTION)
                        outputs = model(
                            input_ids=current_input_ids,
                            attention_mask=attention_mask,
                            use_cache=False  # DISABLE CACHE FOR ATTENTION INJECTION
                        )
                        
                        # GET LOGITS FROM OUTPUT
                        logits = outputs.logits[:, -1, :]  # LAST TOKEN LOGITS
                        
                        # APPLY TEMPERATURE AND SAMPLING
                        if temperature > 0:
                            logits = logits / temperature
                            probs = torch.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        # ADD TO GENERATED TOKENS
                        generated_tokens.append(next_token.item())
                        current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                        
                        # CHECK FOR END OF SEQUENCE
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                
                # DECODE THE GENERATED RESPONSE
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                logger.info(f"GENERATED RESPONSE WITH INJECTED ATTENTION: {len(generated_tokens)} TOKENS")
                return generated_text
                
            finally:
                # REMOVE ATTENTION INJECTION HOOK AND CLEAN UP
                injection_handle.remove()
                if injection_handle in self._active_hooks:
                    self._active_hooks.remove(injection_handle)
            
        except Exception as e:
            logger.error(f"ATTENTION INJECTION GENERATION FAILED: {e}")
            # FALLBACK TO REGULAR GENERATION
            return call_gpt_oss_with_memory(
                prompt=prompt,
                memories=[],
                identity="",
                llm_config=self.llm,
                memory_influence=0.3
            )
    
    def _tensor_to_identity_description(self, identity_tensor: torch.Tensor) -> str:
        """CONVERT IDENTITY TENSOR TO READABLE DESCRIPTION"""
        try:
            # ANALYZE IDENTITY VECTOR PATTERNS
            mean_val = torch.mean(identity_tensor).item()
            std_val = torch.std(identity_tensor).item()
            max_dims = torch.topk(identity_tensor, 3).indices.tolist()
            
            # CREATE DESCRIPTIVE TEXT
            description = f"Identity: mean={mean_val:.3f}, std={std_val:.3f}, dominant_dims={max_dims}"
            return description
            
        except Exception as e:
            logger.error(f"IDENTITY TENSOR CONVERSION FAILED: {e}")
            return "Identity: Unknown"
    
    def _store_interaction(self, input_text: str, output_text: str, emotions: torch.Tensor):
        """STORE INTERACTION IN MEMORY SYSTEM WITH PROPER EMBEDDINGS AND ATTENTION WEIGHTS"""
        try:
            # CREATE PROPER SENTENCE TRANSFORMER EMBEDDINGS
            input_embedding = self._create_text_embedding(input_text)
            
            # CAPTURE ATTENTION WEIGHTS FROM THE LLM GENERATION
            # NOTE: This captures attention from input processing, not response generation
            # For true memory influence, we'd need to capture attention during response generation
            attention_weights = self._capture_attention_weights(input_text)
            
            # CREATE TENSOR REPRESENTATIONS FOR STORAGE
            # CONVERT TEXT TO FIXED-SIZE TENSORS (PAD OR TRUNCATE)
            input_chars = [ord(c) for c in input_text[:128]]
            output_chars = [ord(c) for c in output_text[:128]]
            
            # PAD TO FIXED SIZE
            input_tensor = torch.tensor(input_chars + [0] * (128 - len(input_chars)), dtype=torch.float32)
            output_tensor = torch.tensor(output_chars + [0] * (128 - len(output_chars)), dtype=torch.float32)
            
            # STORE IN MEMORY WITH PROPER EMBEDDINGS AND ATTENTION
            stored = self.memory_system.store(
                inputs=input_tensor,
                outputs=output_tensor,
                emotions=emotions,
                embedding=input_embedding,  # USE INPUT EMBEDDING FOR SIMILARITY SEARCH
                attention_weights=attention_weights  # ADD ATTENTION WEIGHTS
            )
            
            if stored:
                logger.debug(f"STORED INTERACTION WITH EMBEDDING DIM: {input_embedding.shape} AND ATTENTION SHAPE: {attention_weights.shape if attention_weights is not None else 'None'}")
            else:
                logger.warning("FAILED TO STORE INTERACTION")
                
        except Exception as e:
            logger.error(f"FAILED TO STORE INTERACTION: {e}")
    
    def _capture_attention_weights(self, input_text: str) -> Optional[torch.Tensor]:
        """CAPTURE ATTENTION WEIGHTS FROM GPT-OSS-20B FORWARD PASS"""
        try:
            if not self.llm or 'pipeline' not in self.llm:
                logger.warning("LLM NOT AVAILABLE FOR ATTENTION CAPTURE")
                return None
            
            # GET THE ACTUAL MODEL FROM THE PIPELINE
            model = self.llm['pipeline'].model
            
            # STORAGE FOR HOOK OUTPUT
            final_attention_output = {}
            
            # HOOK FUNCTION TO CAPTURE ATTENTION
            def hook_fn(module, input, output):
                # SAVE OUTPUT OF THE ATTENTION MODULE
                final_attention_output["out"] = output
            
            # FIND THE LAST ATTENTION LAYER
            # GPT-OSS-20B ARCHITECTURE: model.model.layers[-1].self_attn
            last_attn_layer = model.model.layers[-1].self_attn
            
            # REGISTER HOOK
            handle = last_attn_layer.register_forward_hook(hook_fn)
            
            try:
                # RUN A FORWARD PASS TO CAPTURE ATTENTION
                inputs = self.llm['pipeline'].tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    _ = model(**inputs)
                
                # RETRIEVE THE CAPTURED ATTENTION OUTPUT
                attention_weights = final_attention_output.get("out")
                
                if attention_weights is not None:
                    logger.debug(f"CAPTURED ATTENTION WEIGHTS: {attention_weights.shape}")
                    return attention_weights
                else:
                    logger.warning("NO ATTENTION WEIGHTS CAPTURED")
                    return None
                    
            finally:
                # REMOVE HOOK WHEN DONE
                handle.remove()
                
        except Exception as e:
            logger.error(f"FAILED TO CAPTURE ATTENTION WEIGHTS: {e}")
            return None
    
    def _create_text_embedding(self, text: str) -> torch.Tensor:
        """CREATE SENTENCE TRANSFORMER EMBEDDING FOR TEXT"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # LAZY LOAD SENTENCE TRANSFORMER
            if not hasattr(self, '_sentence_transformer'):
                self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # CREATE EMBEDDING
            embedding = self._sentence_transformer.encode(text, convert_to_tensor=True)
            
            # ENSURE IT'S A TORCH TENSOR
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
            
            return embedding
            
        except Exception as e:
            logger.error(f"FAILED TO CREATE TEXT EMBEDDING: {e}")
            # FALLBACK TO RANDOM EMBEDDING WITH CORRECT DIMENSION
            return torch.randn(384, dtype=torch.float32) * 0.1  # 384D for all-MiniLM-L6-v2
    
    def get_status(self) -> Dict[str, Any]:
        """GET CURRENT AGENT STATUS"""
        return {
            "run_id": self.run_id,
            "tick": self.current_tick,
            "alive": self.is_alive,
            "emotions": self.emotion_engine.get_state_probs_dict(),
            "life_stage": self.lifespan_manager.get_life_stage(),
            "identity_stability": self.self_model.get_stability_factor(),
            "memory_count": len(self.memory_system.get_all_memories()),
            "narrative_count": len(self.narrative_engine.get_narrative_history()),
            "attention_evolution": self.get_attention_evolution_stats(),
            "active_hooks": len(self._active_hooks) if hasattr(self, '_active_hooks') else 0
        }
    
    def reset(self):
        """RESET AGENT TO INITIAL STATE"""
        self.current_tick = 0
        self.is_alive = True
        self.current_input = ""
        self.current_output = ""
        
        # CLEAN UP ACTIVE HOOKS
        if hasattr(self, '_active_hooks'):
            for hook in self._active_hooks:
                try:
                    hook.remove()
                except Exception as e:
                    logger.error(f"FAILED TO REMOVE HOOK: {e}")
            self._active_hooks.clear()
        
        # RESET COMPONENTS
        self.emotion_engine.reset()
        self.memory_system.clear()
        self.lifespan_manager.reset()
        self.self_model.reset()
        self.narrative_engine.reset()
        self.sleep_cycle.reset()
        
        logger.info("AGENT RESET COMPLETE")
