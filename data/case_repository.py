import os
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
import data.models as models  # Импортируем ORM-классы


class CaseRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create_region(self, name: str, description: Optional[str] = None) -> models.Region:
        region = self.db.query(models.Region).filter_by(name=name).first()
        if not region:
            region = models.Region(name=name, description=description)
            self.db.add(region)
        return region

    def get_or_create_sector(self, name: str, description: Optional[str] = None) -> models.Sector:
        sector = self.db.query(models.Sector).filter_by(name=name).first()
        if not sector:
            sector = models.Sector(name=name, description=description)
            self.db.add(sector)
        return sector

    def get_or_create_technology_drivers(self, driver_names: List[str]) -> List[models.TechnologyDriver]:
        drivers = []
        for name in driver_names:
            driver = self.db.query(models.TechnologyDriver).filter_by(name=name).first()
            if not driver:
                driver = models.TechnologyDriver(name=name)
                self.db.add(driver)
            drivers.append(driver)
        return drivers

    def create_case(
        self,
        case_data: Dict[str, Any],
        economic_effects_data: List[Dict[str, Any]],
        driver_names: List[str]
    ) -> models.Case:
        region = self.get_or_create_region(case_data["region_name"], case_data.get("region_description"))
        sector = self.get_or_create_sector(case_data["sector_name"], case_data.get("sector_description"))
        technology_drivers = self.get_or_create_technology_drivers(driver_names)

        new_case = models.Case(
            region=region,
            sector=sector,
            title=case_data["title"],
            summary=case_data["summary"],
            detailed_notes=case_data.get("detailed_notes"),
            key_effect_note=case_data.get("key_effect_note"),
        )
        new_case.technology_drivers.extend(technology_drivers)
        self.db.add(new_case)
        self.db.flush()

        for effect_data in economic_effects_data:
            effect = models.EconomicEffect(
                case=new_case,
                effect_type=effect_data["effect_type"],
                value_numeric=effect_data["value_numeric"],
                currency=effect_data.get("currency"),
                period_note=effect_data.get("period_note"),
            )
            self.db.add(effect)

        return new_case

    def save_case(
        self,
        case_data: Dict[str, Any],
        economic_effects_data: List[Dict[str, Any]],
        driver_names: List[str]
    ) -> models.Case:
        try:
            new_case = self.create_case(case_data, economic_effects_data, driver_names)
            self.db.commit()
            print(f"Кейс '{new_case.title}' успешно сохранен с ID: {new_case.case_id}")
            return new_case
        except Exception as e:
            self.db.rollback()
            print(f"Ошибка при сохранении кейса: {e}")
            raise


if __name__ == "__main__":
    from database import SessionLocal, create_all_tables

    os.environ["DATABASE_URL"] = "postgresql://user:password@localhost/ai_scout_db"

    create_all_tables()

    db_session = SessionLocal()
    case_repo = CaseRepository(db_session)

    case_example = {
        "region_name": "Россия",
        "sector_name": "Производство",
        "title": "AI-управляемый планировщик производства (обновлено)",
        "summary": "LLM-генерирует оптимальные сметы и расписания смен, учитывая ограничения поставок и реальное состояние оборудования (через IoT-датчики).",
        "detailed_notes": "Система позволяет сократить простои и оптимизировать логистику.",
        "key_effect_note": "Сокращение простоя на 22%, экономия ≈ ₽ 1,8 млрд/год для средних заводов.",
    }

    economic_effects_example = [
        {
            "effect_type": "cost_reduction_percent",
            "value_numeric": 22.0,
            "period_note": "годовой",
        },
        {
            "effect_type": "cost_reduction_rub",
            "value_numeric": 1_800_000_000.0,
            "currency": "RUB",
            "period_note": "годовой"
        },
    ]

    technology_drivers_example = ["LLM", "RAG", "агентные системы"]

    try:
        print("Попытка сохранения первого кейса...")
        case_repo.save_case(case_example, economic_effects_example, technology_drivers_example)
    except Exception as e:
        print(f"Общая ошибка при выполнении примера: {e}")
    finally:
        db_session.close()
