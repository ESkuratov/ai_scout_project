import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Используйте переменную окружения для URL базы данных
# Например: postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Skuratov%40pg2025@localhost/ai_scout_db")

# Создаем движок SQLAlchemy
engine = create_engine(DATABASE_URL)

# Создаем базовый класс для декларативных моделей
Base = declarative_base()

# Настраиваем класс для сессий. Каждая сессия - это "разговор" с БД.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Утилита для получения сессии (для использования в API роутерах, например)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session_maker():
    """Returns the configured SessionLocal factory."""
    return SessionLocal

# Функция для создания всех таблиц (использовать только для dev/тестов, для prod - Alembic)
def create_all_tables():
    print("Создание всех таблиц в базе данных...")
    Base.metadata.create_all(engine)
    print("Таблицы созданы.")

