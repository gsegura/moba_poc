import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from ollama import AsyncClient, ChatResponse, embed
import time
import logging
from abc import ABC, abstractmethod

class LLMProviderError(Exception):
    """Base exception for LLM provider errors"""
    pass

class LLMResponseError(LLMProviderError):
    """Exception for malformed or invalid LLM responses"""
    pass

class LLMTimeoutError(LLMProviderError):
    """Exception for LLM request timeouts"""
    pass

class LLMNetworkError(LLMProviderError):
    """Exception for network-related errors"""
    pass

class OllamaProvider(ABC):
    def __init__(
        self, 
        host: str = 'http://localhost:11434',
        model: str = 'deepseek-r1:14b',
        max_retries: int = 3,
        timeout: float = 600.0
    ):
        self.client = AsyncClient(host=host)
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
    
    async def generate_reasoning(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[str], Optional[str]]:
        try:
            response = await self._call_async(
                messages,
                model or self.model,
                **kwargs
            )
            
            if response and response.message:
                return self._extract_thinking(response.message.content)
            
            logging.error("Empty response from LLM")
            return None, None
            
        except Exception as e:
            logging.error(f"LLM reasoning failed: {str(e)}")
            if isinstance(e, (LLMProviderError, LLMResponseError)):
                raise
            raise LLMProviderError(str(e))


    def _extract_thinking(self, content: str) -> Tuple[Optional[str], str]:
        """Extract thinking process from response"""
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            response = content[think_match.end():].strip()
            return thought, response
        return None, content


    async def _call_async(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        **kwargs
    ) -> ChatResponse:
        start_time = time.time()

        options = {}
        if 'context_window' in kwargs:
            options['num_ctx'] = kwargs.pop('context_window')
        # Add the remaining keyword arguments to options.
        options.update(kwargs)

        extra_args = {'options': options}
  
        try:
            response = await asyncio.wait_for(
                self.client.chat(
                    model=model,
                    messages=messages,
                    **extra_args
                ),
                timeout=self.timeout
            )
            return response

        except asyncio.TimeoutError as e:
            logging.error(f"LLM request timed out after {self.timeout}s")
            raise LLMTimeoutError(f"Request timed out: {str(e)}")
        except ConnectionError as e:
            logging.error(f"Network error during LLM request: {str(e)}")
            raise LLMNetworkError(f"Connection error: {str(e)}")
        except Exception as e:
            logging.error(f"Unexpected error during LLM request: {str(e)}")
            raise LLMProviderError(f"LLM request failed: {str(e)}")

    async def get_embeddings(self, model: Optional[str] = None, input_text: Union[str, List[str]] = None) -> Any:
        """
        Get embeddings for the given input using the specified model asynchronously.
        Calls the embed method on the instance's async client.

        Example usage:
          response = await self.get_embeddings(model='llama3.2', input_text='Hello, world!')
          response = await self.get_embeddings(model='llama3.2', input_text=[
              "The sky is blue because of rayleigh scattering",
              "Grass is green because of chlorophyll"
          ])
        """
        model = model or self.model
        return await self.client.embed(model=model, input=input_text)


if __name__ == "__main__":
    async def main():
        provider = OllamaProvider(host="http://ai-server:11434")
        out = await provider.get_embeddings(input_text="Hello, world!", model="bge-m3:latest")
        print(out)
    
    asyncio.run(main())
