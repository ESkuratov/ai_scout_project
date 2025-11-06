import os
import sys
import logging
import click
import yaml
from typing import Dict, Any

from src.utils.logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from src.data_ingestion.postgres_loader import PostgresLoader
from src.data_ingestion.cleaners.case_cleaner import CaseCleaner
from src.embeddings.models.embedding_model import create_embedding_model
from src.embeddings.chunkers.text_splitter import TextSplitter
from src.embeddings.pipelines.embedding_pipeline import EmbeddingPipeline
from src.vector_store.qdrant_client import QdrantClient


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

@click.command(name="ingest")
@click.option('--qdrant-config-path', default='src/configs/qdrant.yaml',
              help='Path to the Qdrant configuration file.',
              type=click.Path(exists=True))
@click.option('--embeddings-config-path', default='src/configs/embeddings.yaml',
              help='Path to the embeddings configuration file.',
              type=click.Path(exists=True))
@click.option('--logging-config-path', default='src/configs/logging.yaml',
              help='Path to the logging configuration file.',
              type=click.Path(exists=True))
def ingest(qdrant_config_path: str, embeddings_config_path: str, logging_config_path: str):
    """
    Runs the full data ingestion and embedding pipeline.
    """
    logger.info("CLI command 'ingest' started.")

    qdrant_config: Dict[str, Any] = {}
    embeddings_config: Dict[str, Any] = {}

    try:
        full_qdrant_config = load_config(qdrant_config_path)
        qdrant_config = full_qdrant_config.get('qdrant', {}) # Extract 'qdrant' section
        if not qdrant_config:
            raise ValueError("Qdrant configuration section 'qdrant' not found in config file.")

        full_embeddings_config = load_config(embeddings_config_path)
        embeddings_config = full_embeddings_config.get('embedding_model', {}) # Extract 'embedding_model' section
        if not embeddings_config:
            raise ValueError("Embedding model configuration section 'embedding_model' not found in config file.")

        logger.info("All configurations loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load one or more configuration files: {e}")
        click.echo(f"Error: Failed to load configurations. Check logs for details. {e}", err=True)
        sys.exit(1)

    embedding_model = None
    vector_size: int | None = None
    logger.info("Initializing embedding model...")
    try:
        embedding_model = create_embedding_model(config=embeddings_config)
        
        # Modified: Access vector_size property directly
        vector_size = embedding_model.vector_size 
        
        model_name = embeddings_config.get("model_name", "default-embedding-model")
        logger.info(f"Embedding model '{model_name}' initialized successfully with vector size: {vector_size}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding model. Error: {e}", exc_info=True)
        click.echo("Error: Failed to initialize embedding model. Check logs for details.", err=True)
        sys.exit(1)

    qdrant_client: QdrantClient | None = None
    qdrant_collection_name: str = qdrant_config["collection_name"] # Modified: Access directly after extracting 'qdrant' section
    try:
        logger.info("Initializing Qdrant client...")
        qdrant_client = QdrantClient(
            host=qdrant_config["host"],
            port=qdrant_config.get("port"),
            grpc_port=qdrant_config.get("grpc_port"),
            api_key=os.getenv(qdrant_config.get("api_key_env", "QDRANT_API_KEY")), # Pass env var name to os.getenv
            vector_size=vector_size
        )
        logger.info(f"Ensuring Qdrant collection '{qdrant_collection_name}' exists or is created with vector size {vector_size}...")
        # Modified: Pass vector_size to recreate_collection, and removed collection_name if it's already in the client
        qdrant_client.recreate_collection(vector_size=vector_size) 
        logger.info(f"Qdrant client initialized and collection '{qdrant_collection_name}' ensured.")
    except Exception:
        logger.exception("Failed to initialize Qdrant client or ensure collection.")
        click.echo("Error: Failed to initialize Qdrant client. Check logs for details.", err=True)
        sys.exit(1)

    postgres_loader: PostgresLoader | None = None
    case_cleaner: CaseCleaner | None = None
    text_splitter: TextSplitter | None = None
    try:
        logger.info("Initializing data loading and processing components...")
        postgres_loader = PostgresLoader() 
        case_cleaner = CaseCleaner()
        text_splitter = TextSplitter(
            chunk_size=embeddings_config.get("chunk_size", 500),
            chunk_overlap=embeddings_config.get("chunk_overlap", 100)
        )
        logger.info("Data loading and processing components initialized.")
    except Exception:
        logger.exception("Failed to initialize data loading and processing components.")
        click.echo("Error: Failed to initialize data components. Check logs for details.", err=True)
        sys.exit(1)

    pipeline: EmbeddingPipeline | None = None
    try:
        logger.info("Initializing Embedding Pipeline execution...")
        if not all([postgres_loader, case_cleaner, embedding_model, text_splitter, qdrant_client]):
            raise ValueError("One or more pipeline components failed to initialize unexpectedly.")

        pipeline = EmbeddingPipeline(
            postgres_loader=postgres_loader,
            case_cleaner=case_cleaner,
            embedding_model=embedding_model,
            text_splitter=text_splitter,
            qdrant_client=qdrant_client,
            collection_name=qdrant_collection_name
        )
        pipeline.run()
        logger.info("Ingestion process completed successfully.")
        click.echo("Ingestion process completed successfully.")
    except Exception:
        logger.exception("Embedding pipeline failed during execution.")
        click.echo("Error: Embedding pipeline failed. Check logs for details.", err=True)
        sys.exit(1)

if __name__ == "__main__":
    ingest()
