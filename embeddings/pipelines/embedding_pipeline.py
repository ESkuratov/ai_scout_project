# Оптимизировано: Импорт datetime для работы с датами и uuid4 для генерации уникальных UUID
import os
import sys
import logging
from datetime import datetime
from uuid import uuid4  # Добавленный импорт для генерации UUID

# !!! Важно: Настройка логирования должна быть одной из первых вещей в скрипте !!!
# Убираем logging.basicConfig отсюда. Оно должно быть вызвано один раз в точке входа
# приложения, например, в cli/ingest.py через utils.logger_config.setup_logging()
logger = logging.getLogger(__name__) # Получаем логгер, который уже настроен централизованно.

# Remove the project root to the Python path - it's generally better to manage
# PYTHONPATH externally or use proper package structure.
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import utility for loading config (если используется)
# from configs.utils import load_config # Этот импорт не был добавлен в структуру utils/
# Import all necessary components
# from data.database import get_db_session_maker, engine # get_db_session_maker и engine не используются напрямую здесь
from src.data_ingestion.postgres_loader import PostgresLoader
from src.data_ingestion.cleaners.case_cleaner import CaseCleaner
from src.embeddings.models.embedding_model import EmbeddingModel # Изменено на прямой импорт класса
from src.embeddings.chunkers.text_splitter import TextSplitter
from src.vector_store.qdrant_client import QdrantClient
from src.vector_store.schemas import CasePayload  # Для structured payload, если это отдельная Pydantic модель


class EmbeddingPipeline:
    def __init__(
        self,
        postgres_loader: PostgresLoader,
        case_cleaner: CaseCleaner,
        embedding_model: EmbeddingModel, # Теперь это уже инициализированный объект модели
        text_splitter: TextSplitter,
        qdrant_client: QdrantClient,
        collection_name: str
    ):
        self.postgres_loader = postgres_loader
        self.case_cleaner = case_cleaner
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        logger.info("EmbeddingPipeline initialized.")

    def run(self):
        logger.info("Starting embedding pipeline execution.")

        logger.info(f"Using embedding model: {self.embedding_model.model_name}")
        logger.info(f"Embedding model vector size: {self.embedding_model.vector_size}") # Используем метод для получения размера
        logger.info(f"Target Qdrant collection: '{self.collection_name}'")

        try:
            # 1. Load data
            logger.info("Loading cases from PostgreSQL...")
            # Используем fetch_cases, который мы обновили
            raw_cases = self.postgres_loader.fetch_cases(include_relations=True)
            if not raw_cases:
                logger.warning("No cases loaded from PostgreSQL. Exiting pipeline.")
                return

            logger.info(f"Successfully loaded {len(raw_cases)} raw cases from PostgreSQL.")

            # 2. Clean and prepare data
            logger.info(f"Cleaning and preparing {len(raw_cases)} raw cases...")
            cleaned_cases = []
            for case_data in raw_cases:
                try:
                    # Modified: Check if the 'clean' method exists before calling it; fallback to raw data if not
                    # This allows the pipeline to proceed without the method while you implement 'clean' in CaseCleaner later
                    if hasattr(self.case_cleaner, 'clean'):
                        cleaned_case = self.case_cleaner.clean(case_data)
                    else:
                        logger.warning("CaseCleaner.clean() method not found; using raw case data as fallback.")
                        cleaned_case = case_data  # Use raw data as-is

                    if cleaned_case: # Убедимся, что очистка прошла успешно и вернула данные
                        cleaned_cases.append(cleaned_case)
                        logger.debug(f"Successfully cleaned case ID: {cleaned_case.get('case_id')}")
                    else:
                        logger.warning(f"Case {case_data.get('case_id', 'N/A')} was skipped during cleaning (returned None or empty).")
                except Exception: # Используем logger.exception для ошибок в цикле
                    logger.exception(f"Error cleaning case {case_data.get('case_id', 'N/A')}. Skipping this case.")
                    continue

            if not cleaned_cases:
                logger.warning("No cases remained after cleaning. Exiting pipeline.")
                return

            logger.info(f"Successfully cleaned and prepared {len(cleaned_cases)} cases for embedding.")

            # 3. Generate embeddings and prepare for Qdrant
            points_to_upsert = []
            for i, case in enumerate(cleaned_cases):
                # case_id теперь "case_id", а не "id"
                case_id = case.get('case_id', f"unknown_{i}")

                if i % 100 == 0 or i == len(cleaned_cases) - 1: # Логгируем прогресс реже для больших объемов
                    logger.info(f"Processing case {i+1}/{len(cleaned_cases)} (Case ID: {case_id})...")

                text_to_embed_parts = [
                    case.get('title'),
                    case.get('summary'),
                    case.get('detailed_notes'),
                    case.get('key_effect_note')
                ]
                text_to_embed = " ".join(filter(None, text_to_embed_parts)).strip()

                if not text_to_embed:
                    logger.warning(f"Case {case_id} has no significant content to embed after cleaning. Skipping.")
                    continue

                chunks = self.text_splitter.split_text(text_to_embed)
                logger.debug(f"Case ID {case_id} split into {len(chunks)} chunks.")

                # Qdrant Point IDs should be unique. Using UUID for compatibility.
                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        logger.debug(f"Generating embedding for case {case_id}, chunk {chunk_idx + 1}/{len(chunks)} (length: {len(chunk)} chars)...")
                        embedding = self.embedding_model.get_embedding(chunk)

                        # Подготовка payload для Qdrant
                        # Используется CasePayload (я ее добавил в import), чтобы гарантировать схему
                        final_payload_for_qdrant = CasePayload(
                            case_id=case.get('case_id'),
                            region_id=case.get('region_id'),
                            sector_id=case.get('sector_id'),
                            company_id=case.get('company_id'),
                            title=case.get('title'),
                            summary=case.get('summary'),
                            key_effect_note=case.get('key_effect_note'),
                            detailed_notes=case.get('detailed_notes'),
                            maturity_level=case.get('maturity_level'),
                            created_at=case.get('created_at') if case.get('created_at') else datetime.utcnow(),
                            source_id=case.get('source_id'),
                            # Добавленные поля для меты из связанных таблиц
                            region_name=case.get('region_name'),
                            sector_name=case.get('sector_name'),
                            company_name=case.get('company_name'),
                            implementation_status_code=case.get('implementation_status_code'),
                            maturity_level_code=case.get('maturity_level_code'),
                            source_title=case.get('source_title'),
                            technology_drivers_names=case.get('technology_drivers_names'),
                            economic_effects_summaries=case.get('economic_effects_summaries'),
                            # Информация о чанке
                            text_chunk=chunk,
                            text_chunk_index=chunk_idx,
                            text_chunk_original_field="combined_fields" # Как указано, для отслеживания источника
                        ).model_dump(by_alias=True, exclude_none=True) # Исключаем None значения из payload

                        points_to_upsert.append({
                            # Modified: Replace string ID with UUID for Qdrant compatibility (must be unsigned integer or UUID)
                            "id": str(uuid4()),  # Генерируем уникальный UUID для каждого чанка
                            "vector": embedding if isinstance(embedding, list) else embedding.tolist(),
                            "payload": final_payload_for_qdrant
                        })
                        logger.debug(f"Prepared point '{points_to_upsert[-1]['id']}' (total {len(points_to_upsert)}).")

                    except Exception:
                        logger.exception(f"Error processing chunk {chunk_idx} for case {case_id}. Skipping this chunk.")
                        continue # Продолжаем с другими чанками или кейсами

            if not points_to_upsert:
                logger.warning("No valid points generated for upserting to Qdrant. Exiting pipeline.")
                return

            logger.info(f"Finished processing all cases. Total {len(points_to_upsert)} points prepared for upsert.")

            # 4. Upsert to Qdrant
            logger.info(f"Attempting to upsert {len(points_to_upsert)} points to Qdrant collection '{self.collection_name}'...")
            # Modified: Remove the 'collection_name' keyword as it's not expected by upsert_points; relies on client's internal collection_name
            self.qdrant_client.upsert_points(points=points_to_upsert)
            logger.info(f"Successfully upserted {len(points_to_upsert)} points to Qdrant collection '{self.collection_name}'.")

        except Exception: # Общий обработчик ошибок для всего пайплайна
            logger.exception("An unhandled error occurred during the embedding pipeline execution. Pipeline failed.")
            raise # Перевыбрасываем, чтобы вызвать аварийное завершение или обработку на вышестоящем уровне
        logger.info("Embedding pipeline finished successfully.")

