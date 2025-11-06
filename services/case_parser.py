from typing import Any, Dict, List, Optional
import re

from data.case_repository import CaseRepository
from data.database import SessionLocal

SECTION_HEADER_RE = re.compile(r"^\s*##+\s+(.+)$", re.MULTILINE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002500-\U00002BEF]+", re.UNICODE)
SCENARIO_RE = re.compile(r"\*\*(.+?)\*\*\s*[–—-]\s*(.+)")
DRIVER_SPLIT_RE = re.compile(r"\s*(?:\+|,|;|/| и | and )\s*", re.IGNORECASE)
INDEX_CELL_RE = re.compile(r"^\d+(\.\d+)?$")


def _persist_cases(agent_output: str) -> None:
    """Parse agent output and persist cases into the database."""
    session = SessionLocal()
    repo = CaseRepository(session)
    try:
        for payload in parse_agent_output(agent_output):
            repo.save_case(
                case_data=payload["case"],
                economic_effects_data=payload["economic_effects"],
                driver_names=payload["technology_drivers"],
            )
    finally:
        session.close()


def parse_agent_output(agent_output: str) -> List[Dict[str, Any]]:
    """Convert raw agent output into structured case payloads."""

    def _strip_markdown(text: str) -> str:
        cleaned = text.replace("**", "").replace("__", "").replace("`", "")
        cleaned = cleaned.replace("–", "-").replace("—", "-")
        return " ".join(cleaned.split())

    def _clean_region_name(raw: str) -> str:
        cleaned = EMOJI_RE.sub("", raw).strip()
        return " ".join(cleaned.split())

    def _extract_title_and_summary(cell: str) -> tuple[str, str]:
        match = SCENARIO_RE.search(cell)
        if match:
            title = _strip_markdown(match.group(1))
            summary = match.group(2).strip()
            return title, summary
        stripped = _strip_markdown(cell)
        parts = re.split(r"\s*-\s*", stripped, maxsplit=1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        return stripped, ""

    def _parse_technology_drivers(cell: str) -> List[str]:
        if not cell:
            return []
        normalized = cell.replace("\u202f", " ").replace(" ", " ")
        candidates = [
            re.sub(r"\s+", " ", item).strip(" .")
            for item in DRIVER_SPLIT_RE.split(normalized)
        ]
        return [c for c in candidates if c]

    def _build_effects(effect_text: str) -> List[Dict[str, Any]]:
        cleaned = _strip_markdown(effect_text)
        if not cleaned:
            return []
        return [
            {
                "effect_type": "text_note",
                "value_numeric": None,
                "currency": None,
                "period_note": cleaned,
            }
        ]

    def _normalize_header_text(header: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-ZА-Яа-я]+", " ", header.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _find_column_index(headers: List[str], keywords: List[str]) -> Optional[int]:
        for idx, header in enumerate(headers):
            if any(keyword in header for keyword in keywords):
                return idx
        return None

    def _safe_get(cells: List[str], idx: Optional[int]) -> str:
        if idx is None or idx < 0 or idx >= len(cells):
            return ""
        return cells[idx]

    payloads: List[Dict[str, Any]] = []
    section_matches = list(SECTION_HEADER_RE.finditer(agent_output))

    for idx, match in enumerate(section_matches):
        header = match.group(1).strip()
        region_name = _clean_region_name(header)
        start = match.end()
        end = (
            section_matches[idx + 1].start()
            if idx + 1 < len(section_matches)
            else len(agent_output)
        )
        body = agent_output[start:end]

        table_lines = [ln for ln in body.splitlines() if ln.strip().startswith("|")]
        if not table_lines:
            continue

        header_idx = next((i for i, ln in enumerate(table_lines) if "| # " in ln), None)
        normalized_headers: List[str] = []
        sector_idx = scenario_idx = effect_idx = drivers_idx = source_idx = None

        if header_idx is None:
            data_lines = [
                ln for ln in table_lines if "---" not in ln and ln.strip().strip("|")
            ]
            sector_idx, scenario_idx, effect_idx, source_idx = 1, 2, 3, 4
        else:
            header_cells = [
                cell.strip()
                for cell in table_lines[header_idx].strip().strip("|").split("|")
            ]
            normalized_headers = [_normalize_header_text(cell) for cell in header_cells]
            sector_idx = _find_column_index(normalized_headers, ["сектор", "sector"])
            scenario_idx = _find_column_index(
                normalized_headers, ["сценарий", "описание", "scenario", "description"]
            )
            effect_idx = _find_column_index(
                normalized_headers, ["эконом", "эффект", "effect"]
            )
            drivers_idx = _find_column_index(
                normalized_headers, ["драйвер", "driver", "технолог"]
            )
            source_idx = _find_column_index(
                normalized_headers, ["источник", "обоснован", "source", "reference"]
            )
            data_lines = [
                ln
                for i, ln in enumerate(table_lines)
                if i > header_idx and "---" not in ln and ln.strip().strip("|")
            ]

        if None in (sector_idx, scenario_idx, effect_idx):
            continue

        for line in data_lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]

            if not INDEX_CELL_RE.match(_safe_get(cells, 0)):
                continue

            sector_name = _strip_markdown(_safe_get(cells, sector_idx))
            scenario_cell = _safe_get(cells, scenario_idx)
            effect_cell = _safe_get(cells, effect_idx)
            driver_cell = _safe_get(cells, drivers_idx)
            source_cell = _safe_get(cells, source_idx)

            if not scenario_cell:
                continue

            title, summary = _extract_title_and_summary(scenario_cell)
            effect_note = _strip_markdown(effect_cell)
            source_note = _strip_markdown(source_cell)
            technology_drivers = (
                _parse_technology_drivers(driver_cell)
                if drivers_idx is not None and driver_cell
                else []
            )

            payloads.append(
                {
                    "case": {
                        "region_name": region_name,
                        "sector_name": sector_name,
                        "title": title,
                        "summary": summary or effect_note,
                        "detailed_notes": source_note or None,
                        "key_effect_note": effect_note,
                    },
                    "economic_effects": _build_effects(effect_cell),
                    "technology_drivers": technology_drivers,
                }
            )

    return payloads