# data_ingestion/cleaners/case_cleaner.py
import re
from typing import List, Dict, Any, Optional

class CaseCleaner:
    def __init__(self,
                 required_text_fields: List[str] = None,
                 min_text_length: int = 20,
                 deduplicate_by_fields: List[str] = None):
        """
        Инициализирует очиститель кейсов.
        :param required_text_fields: Список полей, которые должны содержать непустой текст.
                                     Кейсы без этих полей будут отброшены или помечены.
        :param min_text_length: Минимальная общая длина текста для полей, используемых для эмбеддингов.
                                Игнорируются кейсы с текстом короче этой длины.
        :param deduplicate_by_fields: Список полей, по которым нужно проверять на дубликаты.
                                      Будет оставлен только первый обнаруженный дубликат.
                                      (например, ['title', 'summary'])
        """
        self.required_text_fields = required_text_fields if required_text_fields is not None else ['title', 'summary']
        self.min_text_length = min_text_length
        self.deduplicate_by_fields = deduplicate_by_fields if deduplicate_by_fields is not None else ['title']

    def _normalize_text(self, text: Optional[str]) -> Optional[str]:
        """
        Нормализует текстовое поле: удаляет лишние пробелы, HTML-теги.
        :param text: Исходный текст.
        :return: Очищенный текст или None.
        """
        if text is None:
            return None
        text = str(text).strip() # Приводим к строке и удаляем пробелы по краям
        text = re.sub(r'<.*?>', '', text) # Удаляем HTML-теги
        text = re.sub(r'\s+', ' ', text) # Сжимаем множественные пробелы в один
        text = text.strip()
        return text if text else None

    def _validate_case(self, case: Dict[str, Any]) -> bool:
        """
        Проверяет кейс на наличие обязательных полей и их минимальную длину.
        :param case: Словарь с данными кейса.
        :return: True, если кейс валиден, False в противном случае.
        """
        combined_text_length = 0
        all_required_present = True

        for field in self.required_text_fields:
            text_value = case.get(field)
            if not text_value or len(text_value) < 5: # Минимальная длина для каждого обязательного поля
                all_required_present = False
                # print(f"Невалидный кейс (case_id: {case.get('case_id')}): Отсутствует или слишком короткое поле '{field}'")
                break
            combined_text_length += len(text_value)
        
        if not all_required_present:
            return False

        if combined_text_length < self.min_text_length:
            # print(f"Невалидный кейс (case_id: {case.get('case_id')}): Общая длина текста ({combined_text_length}) меньше минимальной ({self.min_text_length})")
            return False
            
        return True

    def clean_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Очищает список кейсов, применяя нормализацию, валидацию и дедупликацию.
        :param cases: Список словарей с данными кейсов.
        :return: Список очищенных и валидных кейсов.
        """
        print(f"Начато очистка {len(cases)} кейсов...")
        cleaned_cases = []
        seen_keys = set()
        
        for original_case in cases:
            # Создаем копию, чтобы не менять оригинальный объект при итерации
            case = original_case.copy() 

            # 1. Нормализация текстовых полей
            for key, value in case.items():
                if isinstance(value, str):
                    case[key] = self._normalize_text(value)
            
            # 2. Валидация
            if not self._validate_case(case):
                # print(f"Кейс case_id={case.get('case_id')} пропущен из-за невалидности.")
                continue

            # 3. Дедупликация (на основе хеша полей)
            if self.deduplicate_by_fields:
                dedup_key_parts = []
                for field in self.deduplicate_by_fields:
                    dedup_key_parts.append(str(case.get(field, '')).lower())
                dedup_key = "|".join(dedup_key_parts)

                if dedup_key in seen_keys:
                    # print(f"Кейс case_id={case.get('case_id')} пропущен как дубликат.")
                    continue
                seen_keys.add(dedup_key)
            
            cleaned_cases.append(case)
        
        print(f"Завершено очистку. Осталось {len(cleaned_cases)} валидных и уникальных кейсов.")
        return cleaned_cases

# Пример использования:
if __name__ == "__main__":
    test_cases_data = [
        {'case_id': 1, 'title': 'AI-управляемый планировщик', 'summary': 'LLM генерирует оптимальные сметы.', 'sector_id': 1},
        {'case_id': 2, 'title': '  Сектор  ', 'summary': '<p>Core Gen-AI tech</p>', 'sector_id': 2},
        {'case_id': 3, 'title': 'Очень коротко', 'summary': 'мало', 'sector_id': 1}, # Будет отброшен
        {'case_id': 4, 'title': None, 'summary': 'Отсутствует название', 'sector_id': 3}, # Будет отброшен
        {'case_id': 5, 'title': 'Дубликат плана', 'summary': 'LLM генерирует оптимальные сметы.', 'sector_id': 1}, # Дубликат summary
        {'case_id': 6, 'title': 'AI-решение для финансов', 'summary': 'GPT-4 на базе нормативных документов ЦБ.', 'sector_id': 4},
        {'case_id': 7, 'title': 'AI-решение для финансов', 'summary': 'GPT-4 на базе нормативных документов ЦБ', 'sector_id': 4}, # Дубликат
    ]

    cleaner = CaseCleaner(
        required_text_fields=['title', 'summary'],
        min_text_length=30, # Изменил для более жесткой фильтрации
        deduplicate_by_fields=['title', 'summary'] # Дедубликация по обоим полям
    )

    cleaned_cases = cleaner.clean_cases(test_cases_data)

    print("\nОчищенные кейсы:")
    for case in cleaned_cases:
        print(case)
