import os
import logging
import click
import yaml
from typing import Dict, Any

from src.utils.logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.exception(f"Failed to load config {config_path}")
        raise

from src.vector_store.qdrant_client import QdrantClient
from src.embeddings.models.embedding_model import create_embedding_model
from src.rag.retriever import Retriever
from src.rag.formatter import Formatter
from src.rag.generator import Generator

@click.command(name="report")
@click.option('--qdrant-config-path', default='src/configs/qdrant.yaml', help='Путь к конфигу Qdrant.')
@click.option('--embeddings-config-path', default='src/configs/embeddings.yaml', help='Путь к конфигу embeddings.')
@click.option('--llm-model', default='openai/gpt-3.5-turbo', help='Имя LLM модели (e.g., openai/gpt-3.5-turbo).')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API ключ для LLM (из переменной среды OPENAI_API_KEY).')
@click.option('--base-url', envvar='OPENAI_BASE_URL', help='Base URL для LLM (из переменной среды OPENAI_BASE_URL).')
@click.option('--query', prompt='Введите запрос для отчёта', help='e.g., "Обзор AI в медицине в регионе 5".')
@click.option('--top-k', default=5, type=int, help='Количество чанков для поиска.')
@click.option('--region-id', type=int)
@click.option('--sector-id', type=int)
@click.option('--maturity-level', help='Фильтр по maturity_level.')
def report(qdrant_config_path, embeddings_config_path, llm_model, api_key, base_url, query, top_k, region_id, sector_id, maturity_level):
    """Генерирует аналитический отчёт на основе RAG."""
    logger.info("Report generation started.")

    try:
        qdrant_config = load_config(qdrant_config_path).get('qdrant', {})
        embeddings_config = load_config(embeddings_config_path).get('embedding_model', {})

        embedding_model = create_embedding_model(config=embeddings_config)
        qdrant_client = QdrantClient(
            host=qdrant_config["host"],
            port=qdrant_config.get("port"),
            grpc_port=qdrant_config.get("grpc_port"),
            api_key=os.getenv(qdrant_config.get("api_key_env", "QDRANT_API_KEY")),
            vector_size=embedding_model.vector_size
        )

        retriever = Retriever(qdrant_client, embedding_model, qdrant_config["collection_name"])
        formatter = Formatter(max_chunks=top_k)
        generator = Generator(fully_specified_name=llm_model, api_key=api_key, base_url=base_url)

        filters = {}
        if region_id: filters['region_id'] = region_id
        if sector_id: filters['sector_id'] = sector_id
        if maturity_level: filters['maturity_level'] = maturity_level

        # RAG pipeline
        search_results = retriever.search(query, top_k=top_k, filters=filters)
        context = formatter.format_context(search_results)
        answer = generator.generate_answer(query, context)

        click.echo(f"\nОтчёт по запросу: {query}\n")
        click.echo(answer or "Не удалось сгенерировать ответ.")
        logger.info("Report generated successfully.")
    except Exception as e:
        logger.exception("Report generation failed.")
        click.echo(f"Error: {e}", err=True)

if __name__ == "__main__":
    report() 