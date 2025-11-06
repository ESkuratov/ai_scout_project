import logging
from typing import List, Union, Optional

# Импортируем RecursiveCharacterTextSplitter из langchain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback to a basic implementation or raise an error
    logging.error("The 'langchain_text_splitters' library is not installed. Please install it with 'pip install langchain-text-splitters'.")
    # For now, we'll make the class unusable if not installed.
    # In a real app, you might want a basic fallback implementation or graceful exit.
    RecursiveCharacterTextSplitter = None

logger = logging.getLogger(__name__)

class TextSplitter:
    """
    Класс для разбиения текста на чанки с использованием RecursiveCharacterTextSplitter из Langchain.
    Поддерживает рекурсивное разбиение по символам с учетом различных разделителей и перекрытия.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, separators: Optional[List[str]] = None):
        """
        Инициализирует TextSplitter.
        :param chunk_size: Максимальный размер каждого чанка в символах.
        :param chunk_overlap: Количество символов, на которое перекрываются соседние чанки.
        :param separators: Список разделителей для RecursiveCharacterTextSplitter.
                           По умолчанию: ["\n\n", "\n", " ", ""] (двойной перенос, перенос, пробел, пустая строка).
        """
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError("RecursiveCharacterTextSplitter is not available. Please install 'langchain-text-splitters'.")

        if chunk_overlap >= chunk_size:
            raise ValueError(f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Разделители по умолчанию для Langchain RecursiveCharacterTextSplitter
        # P.S.: Langchain по умолчанию использует: ["\n\n", "\n", " ", ""]
        # Если вы хотите свои, передайте их
        self.separators = separators


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators # Передаем опциональные разделители
        )
        logger.info(f"Langchain TextSplitter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        if self.separators:
            logger.info(f"Custom separators used: {self.separators}")

    def split_text(self, text: Union[str, List[str]]) -> List[str]:
        """
        Разбивает один или несколько текстов на чанки с использованием Langchain text splitter.
        :param text: Один текст (string) или список текстов (List[str]) для разбиения.
        :return: Список текстовых чанков.
        """
        if isinstance(text, str):
            return self.text_splitter.split_text(text)
        elif isinstance(text, list):
            all_chunks = []
            for t in text:
                all_chunks.extend(self.text_splitter.split_text(t))
            return all_chunks
        else:
            raise TypeError(f"Input text must be a string or a list of strings, got {type(text)}")

# Пример использования (можно добавить в if __name__ == "__main__": или в отдельный тестовый файл)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Пример использования с стандартными разделителями
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

        long_text = """Это первый абзац текста. Он довольно длинный и содержит несколько предложений.
        Мы хотим разбить его на более мелкие фрагменты для обработки.

        Второй абзац может быть короче, но он также важен для сохранения контекста.
        Разделители, такие как переносы строк, должны учитываться.

        Третий, очень длинный абзац. Он предназначен для того, чтобы проверить, как сплиттер справится
        с текстом, который значительно превышает размер чанка. Здесь много информации,
        которая должна быть сохранена и разделена корректно. Этот абзац имеет много текста.
        """

        chunks = splitter.split_text(long_text)

        logger.info(f"Разбито на {len(chunks)} чанков:")
        for i, chunk in enumerate(chunks):
            logger.info(f"Чанк {i+1} (длина: {len(chunk)}):\n---\n{chunk}\n---")

        short_text = "Короткий текст."
        short_chunks = splitter.split_text(short_text)
        logger.info(f"\nРазбито на {len(short_chunks)} чанков из короткого текста:")
        for i, chunk in enumerate(short_chunks):
            logger.info(f"Чанк {i+1} (длина: {len(chunk)}):\n---\n{chunk}\n---")

        list_of_texts = ["Один короткий текст.", "Второй очень длинный текст для тестирования списка. Этот текст длиннее первого."]
        list_chunks = splitter.split_text(list_of_texts)
        logger.info(f"\nРазбито на {len(list_chunks)} чанков из списка текстов:")
        for i, chunk in enumerate(list_chunks):
            logger.info(f"Чанк {i+1} (длина: {len(chunk)}):\n---\n{chunk}\n---")

        # Пример использования с кастомными разделителями
        logger.info("\n--- Тестирование с кастомными разделителями (например, только '---') ---")
        custom_splitter = TextSplitter(chunk_size=50, chunk_overlap=10, separators=["---", "\n"])
        custom_text = "Первая часть.---Вторая часть.---Третья, длинная часть, которая должна быть разбита."
        custom_chunks = custom_splitter.split_text(custom_text)
        logger.info(f"Разбито на {len(custom_chunks)} чанков с кастомными разделителями:")
        for i, chunk in enumerate(custom_chunks):
            logger.info(f"Чанк {i+1} (длина: {len(chunk)}):\n---\n{chunk}\n---")

    except RuntimeError as e:
        logger.error(f"TextSplitter initialization failed: {e}")
    except Exception as e:
        logger.error(f"Произошла ошибка в примере: {e}", exc_info=True)
