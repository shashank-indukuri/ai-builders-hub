# flexible_llm_provider.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import json
import requests
import os
import time
import logging
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

class ProviderType(Enum):
    OPENAI = "openai"
    GEMINI = "gemini" 
    GROQ = "groq"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"

@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, int] = None
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    error: Optional[str] = None

@dataclass 
class LLMConfig:
    provider: ProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    extra_params: Dict[str, Any] = None

class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()
        if config.timeout:
            self.session.timeout = config.timeout

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Keep for compatibility but don't use if not needed

    def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        # Return error response since we're not using OpenAI
        return LLMResponse(
            content="",
            error="OpenAI provider not configured",
            provider="openai"
        )

    def is_available(self) -> bool:
        return False

class GeminiClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Gemini requires api_key")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        start_time = time.time()
        
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
                **(self.config.extra_params or {})
            }
        }

        try:
            url = f"{self.base_url}/models/{self.config.model}:generateContent"
            headers = {"Content-Type": "application/json"}
            
            response = self.session.post(
                url,
                json=payload,
                headers=headers,
                params={"key": self.config.api_key}
            )
            response.raise_for_status()
            
            data = response.json()
            latency = int((time.time() - start_time) * 1000)
            
            if "candidates" in data and data["candidates"]:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                usage = data.get("usageMetadata", {})
                
                return LLMResponse(
                    content=content,
                    usage={
                        "prompt_tokens": usage.get("promptTokenCount", 0),
                        "completion_tokens": usage.get("candidatesTokenCount", 0),
                        "total_tokens": usage.get("totalTokenCount", 0)
                    },
                    model=self.config.model,
                    provider="gemini",
                    latency_ms=latency
                )
            else:
                return LLMResponse(
                    content="",
                    error="No candidates in response",
                    provider="gemini",
                    latency_ms=latency
                )
                
        except Exception as e:
            return LLMResponse(
                content="",
                error=str(e),
                provider="gemini",
                latency_ms=int((time.time() - start_time) * 1000)
            )

    def is_available(self) -> bool:
        try:
            url = f"{self.base_url}/models"
            response = self.session.get(url, params={"key": self.config.api_key})
            return response.status_code == 200
        except:
            return False

class GroqClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Groq requires api_key")
        self.base_url = config.base_url or "https://api.groq.com/openai/v1"

    def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        start_time = time.time()
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **(self.config.extra_params or {})
        }

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            latency = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                usage=data.get("usage", {}),
                model=data.get("model", self.config.model),
                provider="groq",
                latency_ms=latency
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                error=str(e),
                provider="groq",
                latency_ms=int((time.time() - start_time) * 1000)
            )

    def is_available(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            response = self.session.get(f"{self.base_url}/models", headers=headers)
            return response.status_code == 200
        except:
            return False

class OllamaClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or None #"http://localhost:11434"

    def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        start_time = time.time()
        
        # Convert messages to prompt format for Ollama
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                **(self.config.extra_params or {})
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            latency = int((time.time() - start_time) * 1000)
            
            return LLMResponse(
                content=data.get("response", ""),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                },
                model=self.config.model,
                provider="ollama",
                latency_ms=latency
            )
            
        except Exception as e:
            return LLMResponse(
                content="",
                error=str(e),
                provider="ollama",
                latency_ms=int((time.time() - start_time) * 1000)
            )

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt for Ollama"""
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"Human: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
        
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

    def is_available(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

class AnthropicClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Keep for compatibility but don't use if not needed

    def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        # Return error response since we're not using Anthropic
        return LLMResponse(
            content="",
            error="Anthropic provider not configured",
            provider="anthropic"
        )

    def is_available(self) -> bool:
        return False

class LLMProviderManager:
    """Unified manager for multiple LLM providers with fallback support"""
    
    def __init__(self):
        self.clients: Dict[str, BaseLLMClient] = {}
        self.primary_provider: Optional[str] = None
        self.fallback_order: List[str] = []

    def add_provider(self, name: str, config: LLMConfig, is_primary: bool = False) -> None:
        """Add a provider configuration"""
        client_class = {
            ProviderType.OPENAI: OpenAIClient,
            ProviderType.GEMINI: GeminiClient,
            ProviderType.GROQ: GroqClient,
            ProviderType.OLLAMA: OllamaClient,
            ProviderType.ANTHROPIC: AnthropicClient
        }
        
        if config.provider not in client_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
            
        try:
            client = client_class[config.provider](config)
            self.clients[name] = client
            
            if is_primary:
                self.primary_provider = name
                
            logging.info(f"Added {config.provider.value} provider: {name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize {name}: {e}")

    def set_fallback_order(self, provider_names: List[str]) -> None:
        """Set the order of fallback providers"""
        self.fallback_order = [name for name in provider_names if name in self.clients]

    def generate_response(self, messages: List[Dict[str, str]], preferred_provider: Optional[str] = None) -> LLMResponse:
        """Generate response with automatic fallback"""
        
        # Determine provider order
        providers_to_try = []
        
        if preferred_provider and preferred_provider in self.clients:
            providers_to_try.append(preferred_provider)
        elif self.primary_provider:
            providers_to_try.append(self.primary_provider)
            
        # Add fallback providers
        for provider in self.fallback_order:
            if provider not in providers_to_try:
                providers_to_try.append(provider)
                
        # Add remaining providers
        for provider in self.clients:
            if provider not in providers_to_try:
                providers_to_try.append(provider)

        last_error = None
        
        for provider_name in providers_to_try:
            client = self.clients[provider_name]
            
            try:
                response = client.generate_response(messages)
                
                if response.error is None and response.content:
                    logging.info(f"Successfully used provider: {provider_name}")
                    return response
                else:
                    last_error = response.error or "Empty response"
                    logging.warning(f"Provider {provider_name} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logging.warning(f"Provider {provider_name} exception: {last_error}")
                continue

        # All providers failed
        return LLMResponse(
            content="",
            error=f"All providers failed. Last error: {last_error}",
            provider="none"
        )

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        available = []
        for name, client in self.clients.items():
            if client.is_available():
                available.append(name)
        return available

    def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        status = {}
        for name, client in self.clients.items():
            status[name] = client.is_available()
        return status

# Simplified setup focusing on your specified providers
def setup_simple_manager(
    gemini_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    ollama_model: str = "qwen2.5vl:latest",
    ollama_url: str = "http://localhost:11434"
) -> LLMProviderManager:
    """Setup a manager with only the specified providers"""
    
    manager = LLMProviderManager()
    
    # Add Gemini if key is provided
    if gemini_key:
        gemini_config = LLMConfig(
            provider=ProviderType.GEMINI,
            model="gemini-2.0-flash",
            api_key=gemini_key,
            temperature=0.1,
            max_tokens=4000
        )
        manager.add_provider("gemini", gemini_config, is_primary=True)
    
    # Add Groq if key is provided
    if groq_key:
        groq_config = LLMConfig(
            provider=ProviderType.GROQ,
            # model="llama3-70b-8192",
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_key,
            temperature=0.1,
            max_tokens=4000
        )
        manager.add_provider("groq", groq_config, is_primary=not gemini_key)
    
    # Always add Ollama with the specified model
    ollama_config = LLMConfig(
        provider=ProviderType.OLLAMA,
        model=ollama_model,
        base_url=os.environ.get("OLLAMA_URL",None) or ollama_url,
        temperature=0.1,
        max_tokens=4000
    )
    manager.add_provider("ollama", ollama_config, is_primary=not (gemini_key or groq_key))
    
    # Set fallback order
    fallback_order = []
    if gemini_key:
        fallback_order.append("gemini")
    if groq_key:
        fallback_order.append("groq")
    fallback_order.append("ollama")
    
    manager.set_fallback_order(fallback_order)
    
    return manager


def test_providers(manager: LLMProviderManager) -> None:
    """Test all configured providers"""
    test_messages = [{"role": "user", "content": "Say 'Hello, I am working correctly!'"}]
    
    print("\n=== Provider Health Check ===")
    health = manager.health_check()
    for name, status in health.items():
        print(f"{name}: {'✓ Available' if status else '✗ Unavailable'}")
    
    print(f"\n=== Testing Providers ===")
    for name in manager.clients:
        print(f"\nTesting {name}...")
        response = manager.generate_response(test_messages, preferred_provider=name)
        
        if response.error:
            print(f"  ✗ Error: {response.error}")
        else:
            print(f"  ✓ Response: {response.content[:100]}...")
            print(f"  ✓ Latency: {response.latency_ms}ms")
            if response.usage:
                print(f"  ✓ Tokens: {response.usage.get('total_tokens', 'N/A')}")

if __name__ == "__main__":
    # Example usage with environment variables
    
    manager = setup_simple_manager(
        # gemini_key=os.getenv("GEMINI_API_KEY"),
        gemini_key=os.environ.get("GEMINI_API_KEY"),
        groq_key=os.environ.get("GROQ_API_KEY"),
        # groq_key=os.getenv("GROQ_API_KEY"),
        ollama_model="qwen2.5vl:latest",
        ollama_url=os.environ.get("OLLAMA_URL")#, "http://localhost:11434")
    )

    test_providers(manager)