import os
import logging
import logging.config
import yaml
from pathlib import Path

def setup_logging(
    default_path='src/configs/logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """
    Setup logging configuration from a YAML file.
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    
    if Path(path).exists():
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        logging.getLogger(__name__).info(f"Logging configured from {path}")
    else:
        logging.basicConfig(level=default_level)
        logging.getLogger(__name__).warning(f"Logging configuration file not found at {path}. Using basic configuration.")

# Пример использования в других модулях:
# from utils.logger_config import setup_logging
# setup_logging() # Вызвать один раз при старте приложения
# logger = logging.getLogger(__name__)
# logger.info("This is an info message from a module.")