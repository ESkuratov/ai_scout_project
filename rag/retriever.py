import logging
from typing import List, Dict, Any, Optional
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

from src.vector_store.qdrant_client import QdrantClient
from src.embeddings.models.embedding_model import create_embedding_model
from src.utils.logger_config import setup_logging  # Предполагаю, что есть

setup_logging()
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, qdrant_client: QdrantClient, embedding_model, collection_name: str):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model  # Инициализированная модель для векторизации запросов
        self.collection_name = collection_name
        logger.info("Retriever initialized.")

    def build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Строит Filter для Qdrant из словаря (e.g., {'region_id': 3, 'sector_id': 44})."""
        must_conditions = []
        for key, value in filters.items():
            if isinstance(value, (int, float)):
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            elif isinstance(value, str):
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            elif isinstance(value, list):
                must_conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                logger.warning(f"Unsupported filter type for {key}: {type(value)}")
        return Filter(must=must_conditions) if must_conditions else None

    def search(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Выполняет векторный поиск по запросу + фильтры, возвращает релевантные чанки."""
        try:
            logger.info(f"Searching for query: '{query}' with filters: {filters} (top_k: {top_k})")
            # Векторизируем запрос
            query_vector = self.embedding_model.get_embedding(query)
            if isinstance(query_vector, list):
                pass  # Уже список
            elif hasattr(query_vector, 'tolist'):
                query_vector = query_vector.tolist()
            else:
                raise TypeError("Embedding должен быть list или ndarray для Qdrant.")

            # Строим фильтр
            qdrant_filter = self.build_filter(filters) if filters else None

            # Ищем в Qdrant
            search_result = self.qdrant_client.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter
            )

            # Форматируем результаты (список dict'ов с payload и score)
            results = []
            for hit in search_result:
                results.append({
                    'id': hit.id,
                    'score': hit.score,
                    'payload': hit.payload
                })
            logger.info(f"Found {len(results)} results.")
            return results
        except Exception as e:
            logger.exception(f"Error during search for query '{query}'.")
            raise
