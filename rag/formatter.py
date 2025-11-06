import logging
from typing import List, Dict, Any, Optional
from src.vector_store.schemas import ScoredPoint  # Импорт ScoredPoint из ваших схем
from datetime import datetime

logger = logging.getLogger(__name__)

class Formatter:
    def __init__(self, max_tokens_per_chunk: int = 1000, max_chunks: int = 10):
        """
        Инициализирует formatter с ограничениями на контекст для LLM.
        :param max_tokens_per_chunk: Ограничение на токены на чанк (примерно для truncate).
        :param max_chunks: Максимум чанков в контексте.
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_chunks = max_chunks
        logger.info("Formatter initialized.")

    def format_context(self, search_results: List[Dict[str, Any]], include_metadata: bool = True) -> str:
        """
        Форматирует результаты поиска в строку контекста для LLM.
        :param search_results: Список dict'ов из retriever.search() (с id, score, payload).
        :param include_metadata: Включать region_name, sector_name и т.д. в контекст.
        :return: Строковое представление контекста.
        """
        context_parts = []
        for i, result in enumerate(search_results[:self.max_chunks]):  # Ограничиваем количеством
            payload = result['payload'] if isinstance(result['payload'], dict) else result['payload'].model_dump() if result['payload'] else {}
            
            # Основной текст (chunk)
            chunk_text = payload.get('text_chunk', 'N/A')
            # Truncate если слишком длинный (простая аппроксимация по словам)
            words = chunk_text.split()
            if len(words) > self.max_tokens_per_chunk // 5:  # Примерно 5 слов на 1 токен
                chunk_text = ' '.join(words[:self.max_tokens_per_chunk // 5]) + '...'
            
            part = f"[{i+1}] Case ID: {payload.get('case_id', 'N/A')}, Score: {result['score']:.3f}\n"
            part += f"Title: {payload.get('title', 'N/A')}\n"
            part += f"Summary: {payload.get('summary', 'N/A')}\n"
            part += f"Content: {chunk_text}\n"
            
            if include_metadata:
                part += f"Region: {payload.get('region_name', 'N/A')}, Sector: {payload.get('sector_name', 'N/A')}, Maturity: {payload.get('maturity_level_code', 'N/A')}\n"
                if 'created_at' in payload and payload['created_at']:
                    part += f"Created At: {payload['created_at'].strftime('%Y-%m-%d') if isinstance(payload['created_at'], datetime) else payload['created_at']}\n"
            
            context_parts.append(part)
        
        return "\n".join(context_parts)

# Пример использования (можно добавить в конец файла для теста):
if __name__ == "__main__":
    # Mock search_results
    mock_results = [
        {'id': 'uuid1', 'score': 0.95, 'payload': {'case_id': 47, 'title': 'RAG в финансах', 'text_chunk': 'RAG для аудита документов.'}},
        {'id': 'uuid2', 'score': 0.89, 'payload': {'case_id': 53, 'title': 'LLM для контролинга', 'text_chunk': 'Генерация отчётов.'}}
    ]
    formatter = Formatter()
    context = formatter.format_context(mock_results)
    print("Formatted Context:\n" + context)
