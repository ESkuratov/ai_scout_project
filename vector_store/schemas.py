# vector_store/schemas.py
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

# -----------------------------------------------------------
# Модель для PayLoad (метаданных), которые будут храниться в Qdrant
# Связанные поля могут быть "плоскими" для удобства фильтрации
# -----------------------------------------------------------

class CasePayload(BaseModel):
    case_id: int
    title: str
    summary: str
    detailed_notes: Optional[str] = None
    key_effect_note: Optional[str] = None
    maturity_level: Optional[int] = None # level_id из PilotMaturityLevel
    created_at: datetime
    source_id: Optional[int] = None

    # Денормализованные связанные поля для удобства фильтрации и отображения
    region_id: Optional[int] = None
    region_name: Optional[str] = None
    sector_id: Optional[int] = None
    sector_name: Optional[str] = None
    company_id: Optional[int] = None
    company_name: Optional[str] = None
    implementation_status_id: Optional[int] = None
    implementation_status_code: Optional[str] = None
    maturity_level_code: Optional[str] = None # code из PilotMaturityLevel
    source_title: Optional[str] = None
    technology_drivers_names: Optional[List[str]] = None
    economic_effects_summaries: Optional[List[str]] = None

    # Поле, указывающее, какой именно чанк текста был векторизован
    text_chunk: str = Field(..., description="Исходный текстовый чанк, из которого был сгенерирован данный вектор.")
    text_chunk_original_field: str = Field(..., description="Исходное поле Case, из которого взят text_chunk (e.g., 'summary', 'detailed_notes', 'title').")
    
    # Дополнительные поля для RAG и фильтрации
    # Например, комбинированный текст для отображения всего кейса при релевантности чанка
    full_case_text: Optional[str] = None


# -----------------------------------------------------------
# Модель для результата поиска из Qdrant
# -----------------------------------------------------------

class ScoredPoint(BaseModel):
    id: str # В Qdrant ID может быть Integer или UUID, здесь пока String для гибкости
    score: float
    payload: Optional[CasePayload] = None # Payload (метаданные)
    vector: Optional[List[float]] = None # Сам вектор, по умолчанию не возвращаем

# -----------------------------------------------------------
# Модель для запроса поиска в Qdrant
# -----------------------------------------------------------

class SearchRequest(BaseModel):
    query_vector: List[float]
    limit: int = 10
    offset: int = 0
    # Здесь можно добавить поля для фильтрации, например:
    # filter_by_sector_id: Optional[int] = None
    # filter_by_region_id: Optional[int] = None
    # filter_by_maturity_level_code: Optional[str] = None
    # Более сложные фильтры будут в QdrantClient.search_points