import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Conditional import for OpenAI and SentenceTransformer
# We'll put these in a try-except block in case not all dependencies are installed
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI library not found. OpenAI embedding models will not be available.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logging.warning("Sentence Transformers library not found. HuggingFace embedding models will not be available.")

logger = logging.getLogger(__name__)

class EmbeddingModel(ABC):
    """
    Abstract base class for all embedding models.
    Defines the interface for generating text embeddings.
    """
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a numerical vector embedding for the given text.
        """
        pass

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """
        Returns the dimensionality of the generated embeddings.
        """
        pass


class OpenRouterOpenAIEmbeddingModel(EmbeddingModel):
    """
    Embedding model using OpenAI's API through OpenRouter.
    Supports models like 'text-embedding-ada-002' or 'text-embedding-3-small'.
    """
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        if OpenAI is None:
            raise ImportError("OpenAI library is not installed. Please install it with 'pip install openai'.")
        
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Determine vector_size. For 'text-embedding-ada-002' it's 1536.
        # For 'text-embedding-3-small', it can be dynamically set,
        # but for simplicity, we'll use common defaults or assume a known size.
        if "ada-002" in model_name:
            self._vector_size = 1536
        elif "text-embedding-3-small" in model_name:
            self._vector_size = 1536 # Default for text-embedding-3-small without dimensions param
        elif "text-embedding-3-large" in model_name:
            self._vector_size = 3072 # Default for text-embedding-3-large without dimensions param
        else:
            logger.warning(f"Unknown OpenAI embedding model '{model_name}'. Defaulting vector size to 1536. "
                           "Consider providing exact size if different.")
            self._vector_size = 1536

        logger.info(f"Initialized OpenRouter OpenAI Embedding Model: {self.model_name} (vector size: {self.vector_size})")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding using the specified OpenAI model via OpenRouter.
        """
        try:
            # OpenRouter passes optional 'headers' to the underlying API client
            # For OpenAI API, the `dimensions` parameter is important for 'text-embedding-3-small/large'
            # We're not explicitly setting 'dimensions' here, relying on default output size.
            response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding with OpenRouter OpenAI model {self.model_name}: {e}")
            raise

    @property
    def vector_size(self) -> int:
        return self._vector_size


class HFEmbeddingModel(EmbeddingModel):
    """
    Embedding model using a Hugging Face Sentence Transformer model.
    """
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        if SentenceTransformer is None:
            raise ImportError("Sentence Transformers library is not installed. Please install it with 'pip install sentence-transformers'.")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._vector_size = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized Hugging Face Embedding Model: {self.model_name} (vector size: {self.vector_size})")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding using the loaded Sentence Transformer model.
        """
        try:
            # convert_to_tensor=False ensures a numpy array, then tolist() for List[float]
            return self.model.encode(text, convert_to_tensor=False).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding with HF model {self.model_name}: {e}")
            raise

    @property
    def vector_size(self) -> int:
        return self._vector_size


def create_embedding_model(config: Dict[str, Any]) -> EmbeddingModel:
    """
    Factory function to create an embedding model instance based on configuration.

    Args:
        config (Dict[str, Any]): A dictionary containing model configuration, e.g.:
            {
                "type": "openrouter_openai",
                "model_name": "text-embedding-ada-002",
                "api_key_env": "OPENROUTER_API_KEY", # Environment variable name for API key
                "base_url": "https://openrouter.ai/api/v1"
            }
            OR
            {
                "type": "hf",
                "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            }

    Returns:
        EmbeddingModel: An instance of a concrete EmbeddingModel.

    Raises:
        ValueError: If an unsupported model type is specified or required configuration is missing.
    """
    # Logic for creating concrete classes
    model_type = config.get("type")
    
    if model_type == "openrouter_openai":
        model_name = config.get("model_name")
        api_key_env = config.get("api_key_env") # Name of the environment variable holding the API key
        base_url = config.get("base_url", "https://openrouter.ai/api/v1")

        if not model_name:
            raise ValueError("Configuration for 'openrouter_openai' model missing 'model_name'.")
        if not api_key_env:
            raise ValueError("Configuration for 'openrouter_openai' model missing 'api_key_env'.")
        
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key environment variable '{api_key_env}' not set for OpenRouter OpenAI model.")
        
        return OpenRouterOpenAIEmbeddingModel(model_name=model_name, api_key=api_key, base_url=base_url)
    
    elif model_type == "hf":
        model_name = config.get("model_name", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        return HFEmbeddingModel(model_name=model_name)
    
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")

