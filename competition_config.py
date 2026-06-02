"""
Registro central de competiciones soportadas por BSD.

Evita duplicar listas entre BSD v1/v2 y permite aplicar reglas por formato
competitivo (clubes vs selecciones) desde el modelo y el prompt.
"""

from __future__ import annotations

import unicodedata


CLUB_COMPETITIONS = {
    "Brasileirao Serie A": 9,
    "Premier League": 1,
    "La Liga": 3,
    "Bundesliga": 5,
    "Serie A": 4,
    "Ligue 1": 6,
    "Champions League": 7,
    "Europa League": 8,
    "Copa Libertadores": 32,
    "Copa Sudamericana": 33,
}

INTERNATIONAL_COMPETITIONS = {
    "World Cup 2026": 27,
    "International Friendly Games": 31,
}

TOP_LEAGUES = {
    **CLUB_COMPETITIONS,
    **INTERNATIONAL_COMPETITIONS,
}

CLUB_CORE_IDS = [9, 3, 5, 4, 6, 1, 7, 8, 32, 33]
INTERNATIONAL_CORE_IDS = [27, 31]

COMPETITION_GROUPS = {
    "club_core": CLUB_CORE_IDS,
    "international_core": INTERNATIONAL_CORE_IDS,
    "default": CLUB_CORE_IDS + INTERNATIONAL_CORE_IDS,
}

TARGET_LEAGUE_IDS = COMPETITION_GROUPS["default"]
LEAGUE_NAMES = {v: k for k, v in TOP_LEAGUES.items()}

INTERNATIONAL_LEAGUE_IDS = set(INTERNATIONAL_CORE_IDS)
WORLD_CUP_LEAGUE_IDS = {INTERNATIONAL_COMPETITIONS["World Cup 2026"]}
FRIENDLY_LEAGUE_IDS = {INTERNATIONAL_COMPETITIONS["International Friendly Games"]}


def _norm(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    without_accents = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    return " ".join(without_accents.casefold().split())


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_competition_flags(league_id=None, league_name: str | None = None) -> dict:
    """Devuelve flags de formato para reglas de selecciones/clubes."""
    lid = _safe_int(league_id)
    name = _norm(league_name or LEAGUE_NAMES.get(lid, ""))

    is_world_cup = lid in WORLD_CUP_LEAGUE_IDS or name == "world cup 2026"
    is_friendly = (
        lid in FRIENDLY_LEAGUE_IDS
        or "international friendly" in name
        or "friendly games" in name
        or "amistosos internacionales" in name
        or "amistoso internacional" in name
    )
    is_international = (
        lid in INTERNATIONAL_LEAGUE_IDS
        or is_world_cup
        or is_friendly
        or name in {"international", "selecciones"}
    )

    if is_world_cup:
        competition_type = "world_cup"
    elif is_friendly:
        competition_type = "friendly"
    elif is_international:
        competition_type = "international"
    else:
        competition_type = "club"

    return {
        "is_international": bool(is_international),
        "is_friendly": bool(is_friendly),
        "is_world_cup": bool(is_world_cup),
        "competition_type": competition_type,
    }


def get_target_league_ids(group: str = "default") -> list[int]:
    """Lista de IDs activa para listados de partidos."""
    return list(COMPETITION_GROUPS.get(group, COMPETITION_GROUPS["default"]))
