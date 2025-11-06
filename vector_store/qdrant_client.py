import os
import logging
from datetime import datetime # Добавлен импорт для примера использования
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient as QdrantNativeClient # Переименовал, чтобы избежать конфликта имен
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models

from src.vector_store.schemas import CasePayload, ScoredPoint # Импортируем наши схемы

logger = logging.getLogger(__name__)

class QdrantClient: # Переименован класс в QdrantClient
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "ai_cases",
                 api_key: Optional[str] = None, grpc_port: Optional[int] = None, vector_size: Optional[int] = None):
        """
        Инициализирует клиент Qdrant и настраивает коллекцию.
        :param host: Хост Qdrant сервера.
        :param port: Порт Qdrant HTTP API.
        :param collection_name: Имя коллекции для хранения кейсов.
        :param api_key: API ключ для аутентификации (если используется Qdrant Cloud или защищенный инстанс).
        :param grpc_port: Порт gRPC API Qdrant (опционально).
        :param vector_size: Размерность векторов, которые будут храниться в коллекции.
                            Обязателен при создании или пересоздании коллекции.
        """
        self.client = QdrantNativeClient( # Используем переименованный импорт
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
        )
        self.collection_name = collection_name
        self.vector_size = vector_size # Теперь vector_size передается явно

        if self.vector_size is None:
            logger.warning("QdrantClient initialized without a vector_size. You must provide it when calling recreate_collection.")
        logger.info(f"Initialized Qdrant client for {host}:{port}. Collection: '{self.collection_name}'. Expected vector size: {self.vector_size if self.vector_size else 'Not set'}")


    def recreate_collection(self, vector_size: Optional[int] = None, distance: models.Distance = models.Distance.COSINE):
        """
        Пересоздает (или создает, если нет) коллекцию с заданными параметрами.
        Если vector_size не передан, используется self.vector_size.
        :param vector_size: Размерность векторов. Если None, используется self.vector_size.
        :param distance: Метрика расстояния для векторов.
        """
        if vector_size is None and self.vector_size is None:
            raise ValueError("Vector size must be provided either during QdrantClient initialization or in recreate_collection method call.")
        
        # If vector_size is provided as an argument, update self.vector_size
        if vector_size is not None:
            self.vector_size = vector_size
        
        # Проверяем существование коллекции и её конфигурацию
        collection_exists = self.client.collection_exists(collection_name=self.collection_name)
        
        if collection_exists:
            try:
                collection_info = self.client.get_collection(collection_name=self.collection_name)
                current_vectors_config = collection_info.config.vectors
                
                # Check if dimensions and distance match
                if (current_vectors_config.size == self.vector_size and
                        current_vectors_config.distance == distance):
                    logger.info(f"Collection '{self.collection_name}' already exists with correct configuration (size: {self.vector_size}, distance: {distance.value}). Skipping recreation.")
                    return
                else:
                    logger.warning(f"Collection '{self.collection_name}' exists but configuration (size or distance) does not match. Deleting and re-creating.")
                    self.client.delete_collection(collection_name=self.collection_name)
            except Exception as e:
                logger.error(f"Error checking collection '{self.collection_name}' configuration, attempting to delete and re-create: {e}")
                self.client.delete_collection(collection_name=self.collection_name)
        
        logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size} and distance {distance.value}...")
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=distance),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000) # Пример оптимизации
        )
        logger.info(f"Collection '{self.collection_name}' created successfully.")

    def upsert_points(self, points: List[Dict[str, Any]]):
        """
        Вставляет или обновляет точки (векторы + payload) в коллекцию.
        :param points: Список словарей, каждый из которых содержит 'id', 'vector' и 'payload'.
        """
        if not points:
            logger.warning("No points provided for upsert.")
            return

        if self.vector_size is None:
            raise RuntimeError("Cannot upsert points: vector_size is not set. Call recreate_collection first.")

        # Проверяем, что все векторы имеют правильную размерность
        for i, point in enumerate(points):
            if len(point['vector']) != self.vector_size:
                raise ValueError(f"Vector at index {i} has dimension {len(point['vector'])}, expected {self.vector_size}.")

        try:
            qdrant_points = [
                models.PointStruct(
                    id=str(point['id']),
                    vector=point['vector'],
                    payload=point['payload']
                )
                for point in points
            ]
            
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=qdrant_points
            )
            if operation_info.status == models.UpdateStatus.COMPLETED:
                logger.info(f"Операция upsert завершена успешно: {len(points)} точек.")
            else:
                logger.error(f"Операция upsert завершена со статусом: {operation_info.status}. Ошибка: {operation_info.error}")

        except UnexpectedResponse as e:
            logger.error(f"Ошибка при upsert в Qdrant (UnexpectedResponse): {e}")
            raise
        except Exception as e:
            logger.error(f"Неизвестная ошибка при upsert: {e}")
            raise

    def search_points(self,
                      query_vector: List[float],
                      limit: int = 10,
                      filters: Optional[models.Filter] = None,
                      with_payload: bool = True,
                      with_vectors: bool = False) -> List[ScoredPoint]:
        """
        Выполняет поиск ближайших точек к заданному вектору.
        :param query_vector: Вектор запроса.
        :param limit: Максимальное количество возвращаемых точек.
        :param filters: Опциональные фильтры для payload (Qdrant models.Filter).
        :param with_payload: Возвращать ли payload найденных точек.
        :param with_vectors: Возвращать ли сами векторы найденных точек.
        :return: Список ScoredPoint объектов.
        """
        if self.vector_size is None:
            raise RuntimeError("Cannot search points: vector_size is not set. Call recreate_collection first.")

        if len(query_vector) != self.vector_size:
            raise ValueError(f"Query vector has dimension {len(query_vector)}, expected {self.vector_size}.")

        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filters,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            return [
                ScoredPoint(
                    id=point.id,
                    score=point.score,
                    payload=CasePayload(**point.payload) if point.payload else None,
                    vector=point.vector if with_vectors else None
                )
                for point in search_result
            ]
        except Exception as e:
            logger.error(f"Ошибка при поиске в Qdrant: {e}")
            raise

    def count_points(self, filters: Optional[models.Filter] = None) -> int:
        """
        Подсчитывает количество точек в коллекции с учетом фильтров.
        """
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
                query_filter=filters
            )
            return count_result.count
        except Exception as e:
            logger.error(f"Ошибка при подсчете точек в Qdrant: {e}")
            raise

# Пример использования:
if __name__ == "__main__":
    # Для запуска этого примера убедитесь, что Qdrant запущен локально (например, через Docker)
    # docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "test_ai_cases")

    # Предположим, что у нас есть эмбеддинги размером 4 (для примера)
    MOCK_VECTOR_SIZE = 4

    # Передаем vector_size в конструктор
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, collection_name=COLLECTION_NAME, vector_size=MOCK_VECTOR_SIZE)

    try:
        logger.info(f"Подключаемся к Qdrant на {QDRANT_HOST}:{QDRANT_PORT}, коллекция: {COLLECTION_NAME}")
        
        # 1. Создаем/пересоздаем коллекцию
        client.recreate_collection() # Теперь vector_size берется из self.vector_size
        
        # 2. Генерируем тестовые данные
        test_points = [
            {
                "id": 1,
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": CasePayload(
                    case_id=1,
                    title="AI для планирования",
                    summary="LLM для оптимизации смен.",
                    created_at=datetime.utcnow(),
                    region_id=1,
                    region_name="Регион А",
                    sector_id=10,
                    sector_name="Производство",
                    text_chunk="LLM для оптимизации смен.",
                    text_chunk_original_field="summary"
                ).model_dump() # .dict() для v1, .model_dump() для v2
            },
            {
                "id": 2,
                "vector": [0.5, 0.6, 0.7, 0.8],
                "payload": CasePayload(
                    case_id=2,
                    title="AI для финансов",
                    summary="RAG на базе документов ЦБ.",
                    created_at=datetime.utcnow(),
                    region_id=1,
                    region_name="Регион А",
                    sector_id=20,
                    sector_name="Финансы",
                    text_chunk="RAG на базе документов ЦБ.",
                    text_chunk_original_field="summary"
                ).model_dump()
            },
            {
                "id": 3,
                "vector": [0.15, 0.25, 0.35, 0.45],
                "payload": CasePayload(
                    case_id=3,
                    title="Планирование логистики",
                    summary="Оптимизация маршрутов с AI.",
                    created_at=datetime.utcnow(),
                    region_id=2,
                    region_name="Регион Б",
                    sector_id=10,
                    sector_name="Производство",
                    text_chunk="Оптимизация маршрутов с AI.",
                    text_chunk_original_field="summary"
                ).model_dump()
            },
        ]

        # 3. Вставляем тестовые точки
        client.upsert_points(test_points)

        # 4. Подсчитываем точки
        total_points = client.count_points()
        logger.info(f"Всего точек в коллекции: {total_points}")

        # 5. Ищем точки по вектору
        query_vec = [0.12, 0.22, 0.32, 0.42] # Вектор, похожий на первый и третий кейсы
        logger.info(f"\nПоиск по вектору: {query_vec}")
        search_results = client.search_points(query_vector=query_vec, limit=2)
        for res in search_results:
            logger.info(f"  ID: {res.id}, Score: {res.score}, Title: {res.payload.title}")

        # 6. Ищем точки с фильтрацией (например, только в секторе производства)
        logger.info("Поиск по вектору с фильтром (sector_id = 10):")
        production_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="sector_id",
                    match=models.MatchValue(value=10) # Более современный способ
                )
            ]
        )
        search_results_filtered = client.search_points(
            query_vector=query_vec,
            limit=2,
            filters=production_filter
        )
        for res in search_results_filtered:
            logger.info(f"  ID: {res.id}, Score: {res.score}, Title: {res.payload.title}, Sector: {res.payload.sector_name}")

        # 7. Фильтрация по строковому полю
        logger.info("Поиск по вектору с фильтром (region_name = 'Регион А'):")
        region_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="region_name",
                    match=models.MatchValue(value="Регион А")
                )
            ]
        )
        search_results_region = client.search_points(
            query_vector=query_vec,
            limit=2,
            filters=region_filter
        )
        for res in search_results_region:
            logger.info(f"  ID: {res.id}, Score: {res.score}, Title: {res.payload.title}, Region: {res.payload.region_name}")

    except Exception as e:
        logger.error(f"Произошла ошибка в примере Qdrant: {e}", exc_info=True)
