import logging
import os  # Add this import for os.getenv and os.environ
from typing import Optional

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, fully_specified_name: str = "openai/gpt-3.5-turbo", max_tokens: int = 1000, api_key: str | None = None, base_url: str | None = None):
        """
        Инициализирует генератор с LLM по аналогии с llm_utils.py и context.py.
        :param fully_specified_name: Полное имя модели в формате 'provider/model' (e.g., 'openai/gpt-3.5-turbo').
        :param max_tokens: Максимум токенов для ответа.
        :param api_key: API ключ для authenticating (если None, берёт из OPENAI_API_KEY).
        :param base_url: Base URL для модели (если None, берёт из OPENAI_BASE_URL).
        """
        # Updated: If api_key or base_url not provided, load from os.environ
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")

        # Обновлено: Добавлен контекст доступа по аналогии с context.py
        # Устанавливаем переменные среды, если предоставлены
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        if base_url:
            os.environ.setdefault("OPENAI_BASE_URL", base_url)

        self.fully_specified_name = fully_specified_name
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url

        self._langchain_available = None
        self._load_deps()

        if not self._langchain_available:
            logger.warning("LangChain not available; initializing with mock mode.")
            self.model = None
        else:
            try:
                from src.agent.llm_utils import load_chat_model
                self.model = load_chat_model(fully_specified_name)
            except ImportError as e:
                logger.warning(f"Failed to import llm_utils or load model: {e}. Using mock mode.")
                self.model = None
            except Exception as e:
                logger.exception(f"Failed to load chat model {fully_specified_name}: {e}")
                self.model = None

        if self.model is None:
            logger.info(f"Generator initialized with model {fully_specified_name} (available: False).")
        else:
            logger.info(f"Generator initialized with model {fully_specified_name} (available: True).")

    def _load_deps(self):
        """Lazy load dependencies."""
        try:
            import langchain_core
            import os  # Already imported, but ensure
            self._langchain_available = True
            logger.info("LangChain library found; Generator can use LangChain models.")
        except ImportError:
            self._langchain_available = False
            logger.warning("LangChain library not found; Generator will use mock responses instead.")

    def generate_answer(self, query: str, context: str) -> Optional[str]:
        """
        Генерирует ответ на основе запроса и контекста с использованием LangChain.
        :param query: Пользовательский запрос.
        :param context: Отформатированный контекст из retriever.
        :return: Сгенерированный ответ или None при ошибке.
        """
        if not self._langchain_available or self.model is None:
            # Mock response to allow testing without LangChain
            return f"[Mock] Недостаточно данных для запроса '{query}'. Контекст: {context[:200]}... Установи langchain и необходимые интеграции (e.g., langchain-openai) для реальной генерации. Убедись, что OPENAI_API_KEY установлен в os.environ."

        prompt = f"""
        На основе следующих релевантных чанков из базы кейсов AI Scout ответь на запрос: "{query}".

        Контекст:
        {context}

        Если контекст недостаточен, скажи «Недостаточно данных» и предложи альтернативы. Будь аналитичным и ссылайся на case_id.
        """
        try:
            from langchain_core.messages import HumanMessage
            from src.agent.llm_utils import get_message_text

            message = HumanMessage(content=prompt)
            response = self.model.invoke([message])
            answer = get_message_text(response).strip()
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer
        except Exception as e:
            logger.exception(f"Error generating answer with LangChain: {e}")
            return None

# Example usage
if __name__ == "__main__":
    import os
    gen = Generator(fully_specified_name="openai/gpt-3.5-turbo")  # Will load api_key from os.getenv if not passed
    query = "Обзор AI в финансах"
    context = "[1] Case ID: 47, Title: RAG в финансах, Content: RAG для аудита."
    answer = gen.generate_answer(query, context)
    print(f"Answer: {answer}")