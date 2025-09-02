"""
LangChain Utilities for Overlay Systems

Provides unified interface for LLM interactions across emotion, memory, narrative, and other systems.
"""

import os
from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass

try:
    from langchain.llms.base import LLM
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from langchain.schema import BaseOutputParser
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
    print("✅ OLD LANGCHAIN IMPORTS SUCCESSFUL")
except ImportError as e:
    print(f"❌ OLD LANGCHAIN IMPORTS FAILED: {e}")
    # TRY ALTERNATIVE IMPORT PATHS FOR NEWER VERSIONS
    try:
        # FIRST TRY TO INSTALL MISSING PACKAGES
        import subprocess
        import sys
        print("🔧 INSTALLING MISSING LANGCHAIN PACKAGES...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-openai", "langchain-anthropic"])
        
        from langchain_core.language_models import LLM
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser, BaseOutputParser
        from langchain_core.callbacks import CallbackManagerForLLMRun
        from pydantic import BaseModel, Field
        LANGCHAIN_AVAILABLE = True
        print("✅ NEW LANGCHAIN IMPORTS SUCCESSFUL")
    except Exception as e2:
        LANGCHAIN_AVAILABLE = False
        print(f"❌ NEW LANGCHAIN IMPORTS ALSO FAILED: {e2}")
        logging.warning("LANGCHAIN NOT AVAILABLE - USING FALLBACK IMPLEMENTATIONS")

logger = logging.getLogger(__name__)

class LangChainConfig:
    """CONFIGURATION FOR LANGCHAIN LLM INTEGRATION"""
    def __init__(self, 
                 provider: str = "huggingface",
                 model_name: str = "openai/gpt-oss-20b",
                 reasoning_level: str = "medium",
                 max_tokens: int = 512,
                 temperature: float = 0.7):
        self.provider = provider
        self.model_name = model_name
        self.reasoning_level = reasoning_level  # low, medium, high
        self.max_tokens = max_tokens
        self.temperature = temperature

class EmotionOutputParser(BaseOutputParser):
    """PARSES EMOTION INFERENCE RESPONSES INTO N-DIMENSIONAL VECTORS"""
    
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
    
    def parse(self, text: str) -> List[float]:
        """PARSE TEXT RESPONSE INTO EMOTION VECTOR"""
        try:
            # EXTRACT NUMBERS FROM RESPONSE
            import re
            numbers = re.findall(r'-?\d+\.?\d*', text)
            
            if len(numbers) < self.dimension:
                logger.warning(f"INSUFFICIENT NUMBERS IN RESPONSE: {len(numbers)} < {self.dimension}")
                # PAD WITH ZEROS
                numbers.extend(['0.0'] * (self.dimension - len(numbers)))
            elif len(numbers) > self.dimension:
                logger.warning(f"TOO MANY NUMBERS IN RESPONSE: {len(numbers)} > {self.dimension}")
                # TRUNCATE
                numbers = numbers[:self.dimension]
            
            # CONVERT TO FLOATS AND VALIDATE RANGE
            emotion_vector = []
            for num_str in numbers:
                value = float(num_str)
                # CLAMP TO [-1, 1] RANGE
                value = max(-1.0, min(1.0, value))
                emotion_vector.append(value)
            
            return emotion_vector
            
        except Exception as e:
            logger.error(f"FAILED TO PARSE EMOTION RESPONSE: {e}")
            # RETURN ZERO VECTOR AS FALLBACK
            return [0.0] * self.dimension

def create_llm(config: LangChainConfig) -> Optional[LLM]:
    """CREATE LANGCHAIN LLM CLIENT BASED ON CONFIGURATION"""
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LANGCHAIN NOT AVAILABLE - CANNOT CREATE LLM CLIENT")
        return None
    
    try:
        if config.provider == "openai":
            api_key = config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY NOT SET")
            
            return ChatOpenAI(
                model_name=config.model_name,
                openai_api_key=api_key,
                temperature=config.temperature,
                max_retries=config.max_retries
            )
            
        elif config.provider == "anthropic":
            api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY NOT SET")
            
            return ChatAnthropic(
                model=config.model_name,
                anthropic_api_key=api_key,
                temperature=config.temperature,
                max_retries=config.max_retries
            )
            
        else:
            logger.error(f"UNSUPPORTED PROVIDER: {config.provider}")
            return None
            
    except Exception as e:
        logger.error(f"FAILED TO CREATE LLM CLIENT: {e}")
        return None

def create_gpt_oss_llm(config: LangChainConfig) -> Any:
    """CREATE GPT-OSS-20B LLM WITH REASONING LEVEL"""
    try:
        from transformers import pipeline
        import torch
        
        # CREATE PIPELINE WITH REASONING LEVEL
        pipe = pipeline(
            "text-generation",
            model=config.model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        
        # SET REASONING LEVEL IN SYSTEM PROMPT
        reasoning_prompt = f"Reasoning: {config.reasoning_level}\n"
        
        return {
            "pipeline": pipe,
            "reasoning_prompt": reasoning_prompt,
            "config": config
        }
    except Exception as e:
        logger.error(f"FAILED TO CREATE GPT-OSS-20B: {e}")
        return None

def call_gpt_oss_with_memory(prompt: str, 
                            memories: List[str], 
                            identity: str,
                            llm_config: dict,
                            memory_influence: float = 0.3) -> str:
    """CALL GPT-OSS-20B WITH MEMORY-INFLUENCED ATTENTION"""
    try:
        pipe = llm_config["pipeline"]
        reasoning_prompt = llm_config["reasoning_prompt"]
        
        # BUILD CONTEXT WITH MEMORIES AND IDENTITY
        full_prompt = f"""{reasoning_prompt}
Context: {prompt}

Relevant Memories:
{chr(10).join([f"- {memory}" for memory in memories])}

Identity Context: {identity}

Generate response:"""
        
        # GENERATE WITH MEMORY INFLUENCE
        # NOTE: We'll implement attention modification in core.py
        outputs = pipe(
            full_prompt,
            max_new_tokens=llm_config["config"].max_tokens,
            temperature=llm_config["config"].temperature,
            do_sample=True
        )
        
        return outputs[0]["generated_text"][-1]
        
    except Exception as e:
        logger.error(f"GPT-OSS-20B GENERATION FAILED: {e}")
        return "ERROR: Generation failed"

def create_emotion_prompt(dimension: int, labels: Optional[List[str]] = None) -> PromptTemplate:
    """CREATE PROMPT TEMPLATE FOR EMOTION INFERENCE"""
    if not LANGCHAIN_AVAILABLE:
        # FALLBACK TO STRING TEMPLATE
        return f"Analyze emotions and output {dimension} numbers between -1 and 1"
    
    if labels:
        label_text = ", ".join(labels)
        prompt_text = f"""Analyze the emotional content of this text and output exactly {dimension} numbers between -1 and 1, representing: {label_text}

Text: {{text}}

Output {dimension} numbers separated by commas:"""
    else:
        prompt_text = f"""Analyze the emotional content of this text and output exactly {dimension} numbers between -1 and 1.

Text: {{text}}

Output {dimension} numbers separated by commas:"""
    
    return PromptTemplate(
        input_variables=["text"],
        template=prompt_text
    )

def call_llm_with_prompt(llm: LLM, prompt: PromptTemplate, text: str) -> str:
    """CALL LLM WITH PROMPT AND RETURN RESPONSE"""
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LANGCHAIN NOT AVAILABLE - RETURNING FALLBACK RESPONSE")
        return "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
    
    try:
        formatted_prompt = prompt.format(text=text)
        response = llm(formatted_prompt)
        return response.strip()
        
    except Exception as e:
        logger.error(f"LLM CALL FAILED: {e}")
        # RETURN FALLBACK RESPONSE
        return "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"

# FALLBACK IMPLEMENTATIONS WHEN LANGCHAIN IS NOT AVAILABLE
if not LANGCHAIN_AVAILABLE:
    class FallbackLLM:
        """FALLBACK LLM IMPLEMENTATION WHEN LANGCHAIN IS NOT AVAILABLE"""
        
        def __call__(self, prompt: str) -> str:
            logger.warning("USING FALLBACK LLM - RETURNING DEFAULT RESPONSE")
            return "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
    
    def create_llm(config: LangChainConfig) -> FallbackLLM:
        return FallbackLLM()
    
    def create_emotion_prompt(dimension: int, labels: Optional[List[str]] = None) -> str:
        return f"Analyze emotions and output {dimension} numbers between -1 and 1"
    
    def call_llm_with_prompt(llm: FallbackLLM, prompt: str, text: str) -> str:
        return "0.0, 0.0, 0.0, 0.0, 0.0, 0.0"
