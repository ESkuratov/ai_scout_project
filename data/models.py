from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Numeric,
    DateTime,
    ForeignKey,
    Table,
)
from sqlalchemy.orm import relationship

from src.data.database import Base # Импортируем 'Base' из database.py

# Ассоциативная таблица для связи Many-to-Many между Case и TechnologyDriver
case_technology_drivers_association = Table(
    'case_technology_drivers',
    Base.metadata,
    Column('case_id', Integer, ForeignKey('cases.case_id'), primary_key=True),
    Column('driver_id', Integer, ForeignKey('technology_drivers.driver_id'), primary_key=True)
)


class Region(Base):
    __tablename__ = 'regions'
    region_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)

    cases = relationship("Case", back_populates="region")


class Sector(Base):
    __tablename__ = 'sectors'
    sector_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)

    cases = relationship("Case", back_populates="sector")


class TechnologyDriver(Base):
    __tablename__ = 'technology_drivers'
    driver_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)

    cases = relationship(
        "Case", secondary=case_technology_drivers_association, back_populates="technology_drivers"
    )


class PilotMaturityLevel(Base):
    __tablename__ = 'pilot_maturity_levels'
    level_id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    description = Column(Text)

    cases = relationship("Case", back_populates="maturity_level_obj")


class ImplementationStatus(Base):
    __tablename__ = 'implementation_statuses'
    status_id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    description = Column(Text)

    cases = relationship("Case", back_populates="implementation_status_obj")


class Company(Base):
    __tablename__ = 'companies'
    company_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    headquarters = Column(String)

    cases = relationship("Case", back_populates="company")


class Source(Base):
    __tablename__ = 'sources'
    source_id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    url = Column(Text)
    publisher = Column(String)
    published_at = Column(DateTime)

    cases = relationship("Case", back_populates="source")


class Case(Base):
    __tablename__ = 'cases'
    case_id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.region_id'), nullable=False)
    sector_id = Column(Integer, ForeignKey('sectors.sector_id'), nullable=False)
    company_id = Column(Integer, ForeignKey('companies.company_id'))
    implementation_status_id = Column(Integer, ForeignKey('implementation_statuses.status_id'))
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    detailed_notes = Column(Text)
    key_effect_note = Column(Text)
    maturity_level = Column(Integer, ForeignKey('pilot_maturity_levels.level_id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    source_id = Column(Integer, ForeignKey('sources.source_id'))

    # Relationships
    region = relationship("Region", back_populates="cases")
    sector = relationship("Sector", back_populates="cases")
    company = relationship("Company", back_populates="cases")
    implementation_status_obj = relationship("ImplementationStatus", back_populates="cases")
    maturity_level_obj = relationship("PilotMaturityLevel", back_populates="cases")
    source = relationship("Source", back_populates="cases")
    technology_drivers = relationship(
        "TechnologyDriver", secondary=case_technology_drivers_association, back_populates="cases"
    )
    economic_effects = relationship("EconomicEffect", back_populates="case", cascade="all, delete-orphan")


class EconomicEffect(Base):
    __tablename__ = 'economic_effects'
    effect_id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey('cases.case_id'), nullable=False)
    effect_type = Column(String, nullable=False)
    value_numeric = Column(Numeric(18, 4))
    currency = Column(String)
    period_note = Column(String)

    case = relationship("Case", back_populates="economic_effects")