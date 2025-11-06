from typing import List, Dict, Any
import logging
from sqlalchemy.orm import sessionmaker, joinedload
from sqlalchemy.exc import SQLAlchemyError

# Импортируем SessionLocal и engine для управления сессиями
# А также импортируем модели для запросов
from src.data.database import SessionLocal, engine
from src.data.models import Case, Region, Sector, Company, ImplementationStatus, PilotMaturityLevel, Source, TechnologyDriver, EconomicEffect

# Инициализируем логгер для этого модуля
logger = logging.getLogger(__name__)

class PostgresLoader:
    def __init__(self):
        """
        Инициализирует загрузчик.
        Теперь db_url не нужен, так как SessionLocal уже сконфигурирован.
        """
        logger.info("PostgresLoader initialized.")

    def fetch_cases(self, filters: Dict[str, Any] = None, include_relations: bool = True) -> List[Dict[str, Any]]:
        """
        Извлекает кейсы из базы данных с возможностью фильтрации и включения связанных данных.
        :param filters: Словарь с фильтрами, где ключ - название колонки, значение - требуемое значение.
                        (например, {'sector_id': 1}). Поддерживаются только прямые фильтры по полям Case.
        :param include_relations: Если True, загружает связанные объекты (region, sector, company и т.д.).
                                  Это делает запрос более "тяжелым", но обогащает данные.
        :return: Список словарей с данными кейсов, где связанные объекты также конвертированы в словари.
                 Если include_relations=False, возвращает только поля Case.
        """
        if filters is None:
            filters = {}

        db = SessionLocal()
        logger.info(f"Attempting to fetch cases with filters: {filters}, include_relations: {include_relations}")

        try:
            query = db.query(Case)

            # Если нужно включить связанные данные, используем .options(joinedload(...))
            if include_relations:
                query = query.options(
                    joinedload(Case.region),
                    joinedload(Case.sector),
                    joinedload(Case.company),
                    joinedload(Case.implementation_status_obj),
                    joinedload(Case.maturity_level_obj),
                    joinedload(Case.source),
                    joinedload(Case.technology_drivers), # для many-to-many
                    joinedload(Case.economic_effects)
                )

            for key, value in filters.items():
                if hasattr(Case, key):
                    query = query.filter(getattr(Case, key) == value)
                    logger.debug(f"Applying filter: {key} = {value}")
                else:
                    logger.warning(f"Column '{key}' not found in Case model for filtering. Skipping filter.")
            
            # Фильтрация по тому, что summary не NULL
            query = query.filter(Case.summary.isnot(None))
            logger.debug("Applying filter: Case.summary is not NULL.")

            cases_orm = query.all()
            logger.info(f"Successfully retrieved {len(cases_orm)} ORM objects from the database.")

            # Преобразование ORM-объектов в словари, с учетом связанных данных
            result_cases = []
            for case in cases_orm:
                case_dict = {
                    "case_id": case.case_id,
                    "region_id": case.region_id,
                    "sector_id": case.sector_id,
                    "company_id": case.company_id,
                    "implementation_status_id": case.implementation_status_id,
                    "title": case.title,
                    "summary": case.summary,
                    "detailed_notes": case.detailed_notes,
                    "key_effect_note": case.key_effect_note,
                    "maturity_level": case.maturity_level,
                    "created_at": case.created_at,
                    "source_id": case.source_id,
                }
                
                # Добавляем данные из связанных таблиц, если они нужны для RAG или индексации
                if include_relations:
                    if case.region:
                        case_dict["region_name"] = case.region.name
                        case_dict["region_description"] = case.region.description
                    if case.sector:
                        case_dict["sector_name"] = case.sector.name
                        case_dict["sector_description"] = case.sector.description
                    if case.company:
                        case_dict["company_name"] = case.company.name
                    if case.implementation_status_obj:
                        case_dict["implementation_status_code"] = case.implementation_status_obj.code
                    if case.maturity_level_obj:
                        case_dict["maturity_level_code"] = case.maturity_level_obj.code
                    if case.source:
                        case_dict["source_title"] = case.source.title
                    
                    # Для many-to-many и one-to-many отношений
                    if case.technology_drivers:
                        case_dict["technology_drivers_names"] = [td.name for td in case.technology_drivers]
                    if case.economic_effects:
                        case_dict["economic_effects_summaries"] = [
                            f"{ee.effect_type}: {ee.value_numeric} {ee.currency} ({ee.period_note})"
                            for ee in case.economic_effects
                        ]

                result_cases.append(case_dict)
            
            logger.info(f"Successfully processed {len(result_cases)} cases into dictionary format.")
            return result_cases
        except SQLAlchemyError as e:
            logger.error(f"Database error during case fetching: {e}", exc_info=True)
            raise # Перевыбрасываем исключение, чтобы вышестоящий уровень мог его обработать
        except Exception as e:
            logger.critical(f"An unexpected error occurred in PostgresLoader.fetch_cases: {e}", exc_info=True)
            raise # Перевыбрасываем общее исключение
        finally:
            db.close() # Закрываем сессию
            logger.debug("Database session closed.")

# Пример использования:
if __name__ == "__main__":
    import os
    import json
    from data.database import create_all_tables
    from utils.logger_config import setup_logging # Предполагаем, что этот файл существует

    # Настраиваем логирование для этого примера
    setup_logging()
    
    # Убедитесь, что переменная окружения DATABASE_URL установлена
    # Например, в терминале перед запуском:
    # export DATABASE_URL="postgresql://ai_scout_user:ai_scout_password@localhost:5432/ai_scout_db"
    if "DATABASE_URL" not in os.environ:
        logger.error("DATABASE_URL environment variable is not set. Please set it before running the example.")
        exit(1)

    # Создаем таблицы, если их нет (для примера и разработки)
    logger.info("Ensuring database tables exist (for example purposes)...")
    try:
        create_all_tables()
        logger.info("Database table check completed.")
    except Exception as e:
        logger.critical(f"Failed to create/check database tables: {e}", exc_info=True)
        exit(1)

    loader = PostgresLoader()

    try:
        logger.info("\n--- Starting example: Loading all cases (summary is not NULL, with relations) ---")
        all_cases = loader.fetch_cases(include_relations=True)
        logger.info(f"Loaded {len(all_cases)} cases.")
        if all_cases:
            logger.info("First loaded case (with relations):")
            logger.info(json.dumps(all_cases[0], indent=2, default=str))

        logger.info("\n--- Starting example: Loading cases for sector_id=2 (only main Case fields) ---")
        filtered_cases_no_relations = loader.fetch_cases(filters={'sector_id': 2}, include_relations=False)
        logger.info(f"Loaded {len(filtered_cases_no_relations)} cases for sector_id=2.")
        if filtered_cases_no_relations:
            logger.info("First filtered case (no relations):")
            logger.info(json.dumps(filtered_cases_no_relations[0], indent=2, default=str))

        logger.info("\n--- Starting example: Loading cases for region_id=1 (with relations) ---")
        filtered_cases_with_relations = loader.fetch_cases(filters={'region_id': 1}, include_relations=True)
        logger.info(f"Loaded {len(filtered_cases_with_relations)} cases for region_id=1.")
        if filtered_cases_with_relations:
            logger.info("First filtered case (with relations):")
            logger.info(json.dumps(filtered_cases_with_relations[0], indent=2, default=str))

        logger.info("\n--- All examples completed successfully ---")

    except Exception as e:
        logger.critical(f"An error occurred during the example execution: {e}", exc_info=True)
