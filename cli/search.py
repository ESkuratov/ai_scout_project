import os
import logging
import click
import yaml
from typing import Dict, Any
from src.utils.logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# Inline load_config function to avoid ModuleNotFoundError for missing src.configs.load_config
def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.exception(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError:
        logger.exception(f"Error parsing YAML configuration from {config_path}")
        raise
    except Exception:
        logger.exception(f"An unexpected error occurred while loading config from {config_path}")
        raise

from src.vector_store.qdrant_client import QdrantClient
from src.embeddings.models.embedding_model import create_embedding_model
from src.rag.retriever import Retriever

@click.command(name="search")
@click.option('--qdrant-config-path', default='src/configs/qdrant.yaml', help='Путь к конфигу Qdrant.')
@click.option('--embeddings-config-path', default='src/configs/embeddings.yaml', help='Путь к конфигу embeddings.')
@click.option('--query', prompt='Введите поисковый запрос', help='Семантический запрос, e.g., "генеративный аудит для банка".')
@click.option('--top-k', default=5, type=int, help='Количество топ-результатов.')
@click.option('--region-id', type=int, help='Фильтр по region_id (число).')
@click.option('--sector-id', type=int, help='Фильтр по sector_id (число).')
@click.option('--maturity-level', help='Фильтр по maturity_level (строка).')
def search(qdrant_config_path, embeddings_config_path, query, top_k, region_id, sector_id, maturity_level):
    """CLI для семантического поиска в Qdrant."""
    logger.info("CLI command 'search' started.")

    try:
        qdrant_config = load_config(qdrant_config_path).get('qdrant', {})
        embeddings_config = load_config(embeddings_config_path).get('embedding_model', {})

        # Инициализируем модель и клиента (как в ingest.py)
        embedding_model = create_embedding_model(config=embeddings_config)
        qdrant_client = QdrantClient(
            host=qdrant_config["host"],
            port=qdrant_config.get("port"),
            grpc_port=qdrant_config.get("grpc_port"),
            api_key=os.getenv(qdrant_config.get("api_key_env", "QDRANT_API_KEY")),
            vector_size=embedding_model.vector_size
        )

        retriever = Retriever(qdrant_client, embedding_model, qdrant_config["collection_name"])

        # Собираем фильтры
        filters = {}
        if region_id:
            filters['region_id'] = region_id
        if sector_id:
            filters['sector_id'] = sector_id
        if maturity_level:
            filters['maturity_level'] = maturity_level

        results = retriever.search(query=query, top_k=top_k, filters=filters)

        click.echo(f"\nРезультаты поиска по запросу '{query}' (топ {top_k}):")
        for i, res in enumerate(results, 1):
            click.echo(f"{i}. ID: {res['id']}, Score: {res['score']:.3f}")
            click.echo(f"   Payload: {res['payload']}\n")

        logger.info("Search completed successfully.")
    except Exception as e:
        logger.exception("Search failed.")
        click.echo(f"Error: Search failed. {e}", err=True)

if __name__ == "__main__":
    search()
