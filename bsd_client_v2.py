"""
Cliente para la BSD API v2 (Bzzoiro Sports Data).

Endpoints integrados:
  /events/              → listar eventos (paginated, filtrable)
  /events/live/         → ventana en vivo
  /events/{id}/         → detalle de evento
  /events/{id}/stats/   → estadisticas (shotmap, momentum, xg_per_minute, ball-tracking)
  /events/{id}/metadata/→ facts pre-partido, AI preview
  /events/{id}/lineups/ → alineaciones y no disponibles (respaldo de SofaScore)
  /events/{id}/player-stats/ → stats por jugador en el partido
  /events/{id}/odds/    → consenso de cuotas BSD (comparador, no fuente oficial)
  /events/{id}/odds/comparison/ → cuotas por bookmaker
  /events/{id}/polymarket/ → probabilidades Polymarket si existen
  /events/{id}/prediction/   → prediccion ML por evento
  /managers/{id}/        → perfil completo de entrenador
  /referees/{id}/        → estadisticas de arbitro
  /referees/{id}/matches/ → partidos recientes del arbitro
  /leagues/{id}/standings/ → tabla de posiciones con xG
  /leagues/{id}/season/  → temporada actual
  /teams/{id}/squad/     → plantilla del equipo
  /teams/{id}/fixtures/  → calendario para descanso/motivacion
  /players/{id}/stats/   → estadisticas recientes por jugador
  /players/{id}/career/  → carrera agregada por temporada
  /predictions/          → predicciones ML (v2 reestructurado)

NO incluye: broadcasts, social, transfers, venues.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv
from competition_config import (
    LEAGUE_NAMES,
    TARGET_LEAGUE_IDS,
    TOP_LEAGUES,
    get_competition_flags,
)

load_dotenv()

BSD_V2_BASE = "https://sports.bzzoiro.com/api/v2"
BSD_API_KEY = os.getenv("BSD_API_KEY")

def _headers():
    if not BSD_API_KEY:
        raise ValueError("BSD_API_KEY no configurada. Agregala en el archivo .env")
    return {"Authorization": f"Token {BSD_API_KEY}"}


def _get(endpoint: str, params: dict = None) -> dict:
    url = f"{BSD_V2_BASE}/{endpoint}"
    response = requests.get(url, headers=_headers(), params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _get_paginated_all(endpoint: str, params: dict = None) -> list:
    """Recorre todas las paginas de un endpoint paginado."""
    params = dict(params or {})
    params.setdefault("limit", 200)
    resultados = []
    while True:
        data = _get(endpoint, params=params)
        resultados.extend(data.get("results", []))
        next_url = data.get("next")
        if not next_url:
            break
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(next_url)
        qs = parse_qs(parsed.query)
        params.update({k: v[0] for k, v in qs.items()})
    return resultados


# ═══════════════════════════════════════════════════════════════════════════════
# EVENTOS
# ═══════════════════════════════════════════════════════════════════════════════

def listar_eventos(league_ids: list = None, solo_no_iniciados: bool = True) -> list:
    """
    Lista eventos via /events/ (v2 paginado).

    Args:
        league_ids: IDs de liga. Default: TARGET_LEAGUE_IDS.
        solo_no_iniciados: Filtrar solo notstarted.
    """
    if league_ids is None:
        league_ids = TARGET_LEAGUE_IDS

    today = date.today().isoformat()
    next_week = (date.today() + timedelta(days=10)).isoformat()

    def _fetch(liga_id):
        try:
            params = {
                "league_id": liga_id,
                "date_from": today,
                "date_to": next_week,
                "limit": 200,
            }
            if solo_no_iniciados:
                params["status"] = "notstarted"
            data = _get("events/", params=params)
            for p in data.get("results", []):
                p["_league_name"] = LEAGUE_NAMES.get(liga_id, f"Liga {liga_id}")
            return data.get("results", [])
        except requests.RequestException as e:
            print(f"  [AVISO v2] {LEAGUE_NAMES.get(liga_id, f'Liga {liga_id}')}: {e}")
            return []

    todos = []
    with ThreadPoolExecutor(max_workers=min(len(league_ids), 6)) as pool:
        futures = {pool.submit(_fetch, lid): lid for lid in league_ids}
        for future in as_completed(futures):
            todos.extend(future.result())

    todos.sort(key=lambda p: p.get("event_date", ""))
    return todos


def obtener_eventos_en_vivo(league_id: int = None, team_id: int = None) -> dict:
    """
    Ventana en vivo: partidos en curso o a punto de empezar.
    Cachado en Redis (TTL 30s); no pollear mas rapido que eso.
    """
    params = {}
    if league_id:
        params["league_id"] = league_id
    if team_id:
        params["team_id"] = team_id
    return _get("events/live/", params=params)


def obtener_detalle_evento(event_id: int) -> dict:
    """Detalle ligero del evento (IDs, weather, neutral ground, etc.)."""
    return _get(f"events/{event_id}/")


def obtener_stats_evento(event_id: int) -> dict:
    """Stats: shotmap, momentum, xg_per_minute, average_positions, ball-tracking."""
    return _get(f"events/{event_id}/stats/")


def obtener_metadata_evento(event_id: int) -> dict:
    """Facts pre-partido, AI preview, jerseys."""
    return _get(f"events/{event_id}/metadata/")


def obtener_lineups_evento(event_id: int) -> dict:
    """
    Alineaciones y jugadores no disponibles.

    SofaScore sigue siendo la fuente principal; este endpoint se usa como
    respaldo y para detectar conflictos entre fuentes.
    """
    return _get(f"events/{event_id}/lineups/")


def obtener_player_stats_evento(event_id: int) -> dict:
    """Stats por jugador en este partido (goles, xG, pases, tackles, rating...)."""
    return _get(f"events/{event_id}/player-stats/")


def obtener_odds_evento(event_id: int) -> dict:
    """Consenso de cuotas BSD para el evento. Betano sigue siendo la fuente oficial."""
    return _get(f"events/{event_id}/odds/")


def obtener_odds_comparison_evento(event_id: int) -> dict:
    """Cuotas side-by-side por bookmaker para un evento."""
    return _get(f"events/{event_id}/odds/comparison/")


def obtener_polymarket_evento(event_id: int) -> dict:
    """Probabilidades Polymarket para el evento, si BSD tiene cobertura."""
    try:
        return _get(f"events/{event_id}/polymarket/")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {}
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# MANAGERS
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_manager(manager_id: int) -> dict:
    """Perfil completo del entrenador con metricas agregadas."""
    return _get(f"managers/{manager_id}/")

# ═══════════════════════════════════════════════════════════════════════════════
# REFEREES
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_arbitro(referee_id: int = None, league_id: int = None, referee_name: str = None) -> dict:
    """Estadisticas del arbitro. Si se provee league_id+name, filtra por liga."""
    if league_id and referee_name:
        params = {"league_id": league_id, "name": referee_name, "limit": 5}
        data = _get("referees/", params=params)
        results = data.get("results", [])
        if results:
            return results[0]
        return {"_error": "No se encontro al arbitro en esa liga"}
    if referee_id:
        return _get(f"referees/{referee_id}/")
    return {"_error": "Faltan parametros"}


def obtener_partidos_arbitro(
    referee_id: int,
    league_id: int = None,
    date_from: str = None,
    date_to: str = None,
    status: str = "finished",
    limit: int = 20,
) -> list:
    """Partidos recientes/proximos dirigidos por un arbitro."""
    params = {"limit": limit}
    if league_id:
        params["league_id"] = league_id
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if status:
        params["status"] = status
    data = _get(f"referees/{referee_id}/matches/", params=params)
    return data if isinstance(data, list) else data.get("results", [])


# ═══════════════════════════════════════════════════════════════════════════════
# LEAGUES
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_temporada_actual(league_id: int) -> dict:
    """Temporada actual de la liga, util para filtrar fixtures/player stats."""
    return _get(f"leagues/{league_id}/season/")


def obtener_standings(league_id: int, season_id: int = None) -> dict:
    """Tabla de posiciones con xG (xgf, xga, xgd)."""
    params = {}
    if season_id:
        params["season_id"] = season_id
    return _get(f"leagues/{league_id}/standings/", params=params)


# ═══════════════════════════════════════════════════════════════════════════════
# TEAMS
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_squad(team_id: int) -> dict:
    """Plantilla completa del equipo (nombre, posicion, nacionalidad, dorsal, DOB)."""
    return _get(f"teams/{team_id}/squad/")


def obtener_fixtures_equipo(
    team_id: int,
    league_id: int = None,
    date_from: str = None,
    date_to: str = None,
    status: str = None,
    limit: int = 200,
) -> list:
    """Calendario de un equipo. Sirve para descanso, rotacion y partidos restantes."""
    params = {"limit": limit}
    if league_id:
        params["league_id"] = league_id
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if status:
        params["status"] = status
    data = _get(f"teams/{team_id}/fixtures/", params=params)
    return data if isinstance(data, list) else data.get("results", [])


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYERS
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_stats_jugador(
    player_id: int,
    season_id: int = None,
    team_id: int = None,
    league_id: int = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 12,
) -> list:
    """Stats partido a partido de un jugador, nuevo a viejo."""
    params = {"limit": limit}
    if season_id:
        params["season_id"] = season_id
    if team_id:
        params["team_id"] = team_id
    if league_id:
        params["league_id"] = league_id
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    data = _get(f"players/{player_id}/stats/", params=params)
    return data if isinstance(data, list) else data.get("results", [])


def obtener_carrera_jugador(player_id: int) -> dict:
    """Carrera agregada por temporada (partidos, goles, asistencias, rating)."""
    return _get(f"players/{player_id}/career/")


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICCIONES
# ═══════════════════════════════════════════════════════════════════════════════

def obtener_predicciones_v2(
    league_id: int = None,
    team_id: int = None,
    status: str = "upcoming",
    min_confidence: float = None,
    recommended: bool = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 200,
) -> list:
    """Predicciones ML CatBoost (v2 reestructurado)."""
    params = {"status": status, "limit": limit}
    if league_id:
        params["league_id"] = league_id
    if team_id:
        params["team_id"] = team_id
    if min_confidence is not None:
        params["min_confidence"] = min_confidence
    if recommended is not None:
        params["recommended"] = str(recommended).lower()
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    data = _get("predictions/", params=params)
    return data.get("results", [])


def obtener_best_odds(
    market: str = "1x2",
    league_id: int = None,
    season_id: int = None,
    team_id: int = None,
    date_from: str = None,
    date_to: str = None,
    limit: int = 200,
) -> list:
    """Mejores cuotas upcoming por mercado a traves de bookmakers BSD."""
    params = {"market": market, "limit": limit}
    if league_id:
        params["league_id"] = league_id
    if season_id:
        params["season_id"] = season_id
    if team_id:
        params["team_id"] = team_id
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    data = _get("odds/best/", params=params)
    return data if isinstance(data, list) else data.get("results", [])


def obtener_prediccion_por_evento(event_id: int) -> dict:
    """Prediccion ML para un evento especifico. 404 si no hay."""
    try:
        return _get(f"events/{event_id}/prediction/")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {}
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE ENRIQUECIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def _as_list(data) -> list:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("results", "fixtures", "events", "matches", "players", "player_stats", "seasons"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _to_float(value, default=None):
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ".").strip())
        except ValueError:
            return default
    return default


def _first_value(data: dict, *keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return default


def _parse_iso_datetime(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.fromisoformat(str(value)[:10])
        except ValueError:
            return None


def _iso(dt) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _event_window(event_date: str, hours: int = 12) -> tuple[str | None, str | None]:
    parsed = _parse_iso_datetime(event_date)
    if not parsed:
        return None, None
    return _iso(parsed - timedelta(hours=hours)), _iso(parsed + timedelta(hours=hours))


def _season_payload(season_data: dict) -> dict:
    if not isinstance(season_data, dict):
        return {}
    season = season_data.get("season")
    return season if isinstance(season, dict) else season_data


def _extract_season_id(*sources) -> int | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        season = source.get("season")
        if isinstance(season, dict) and season.get("id"):
            return season["id"]
        if source.get("season_id"):
            return source["season_id"]
        if source.get("id") and source.get("is_current") is not None:
            return source["id"]
    return None


def _extract_season_end(season_data: dict) -> str | None:
    payload = _season_payload(season_data)
    return payload.get("end_date") if isinstance(payload, dict) else None


def _extract_standing_rows(standings: dict) -> list:
    """Soporta standings planos y copas con grupos."""
    if not isinstance(standings, dict):
        return []

    rows = standings.get("standings")
    if isinstance(rows, list):
        return rows
    if isinstance(rows, dict):
        flattened = []
        for group_name, group_rows in rows.items():
            items = group_rows
            if isinstance(group_rows, dict):
                items = group_rows.get("standings") or group_rows.get("rows") or group_rows.get("table")
            for row in items or []:
                if isinstance(row, dict):
                    flattened.append({**row, "_group": group_name})
        if flattened:
            return flattened

    for key in ("groups", "tables"):
        groups = standings.get(key)
        if not isinstance(groups, (dict, list)):
            continue
        flattened = []
        iterable = groups.items() if isinstance(groups, dict) else enumerate(groups)
        for group_name, group in iterable:
            items = group
            if isinstance(group, dict):
                items = group.get("standings") or group.get("rows") or group.get("table") or group.get("teams")
                group_name = group.get("name") or group.get("group") or group_name
            for row in items or []:
                if isinstance(row, dict):
                    flattened.append({**row, "_group": group_name})
        if flattened:
            return flattened
    return []


def _find_standing_row(rows: list, team_id: int = None, team_name: str = "") -> dict:
    if team_id:
        for row in rows:
            if row.get("team_id") == team_id or (row.get("team") or {}).get("id") == team_id:
                return row
    norm_name = _norm(team_name)
    if norm_name:
        for row in rows:
            candidate = _norm(row.get("team_name") or (row.get("team") or {}).get("name"))
            if norm_name == candidate or norm_name in candidate or candidate in norm_name:
                return row
    return {}


def _norm(value: str) -> str:
    return " ".join(str(value or "").casefold().replace(".", "").split())


def _event_id(item: dict):
    if not isinstance(item, dict):
        return None
    event = item.get("event")
    return item.get("event_id") or (event.get("id") if isinstance(event, dict) else None) or item.get("id")


def _event_team_ids(item: dict) -> set:
    if not isinstance(item, dict):
        return set()
    event = item.get("event") if isinstance(item.get("event"), dict) else {}
    ids = {
        item.get("home_team_id"),
        item.get("away_team_id"),
        event.get("home_team_id"),
        event.get("away_team_id"),
    }
    return {x for x in ids if x}


def _filter_same_league_fixtures(fixtures: list, league_id: int = None) -> list:
    if not league_id:
        return fixtures
    filtered = []
    for item in fixtures:
        event = item.get("event") if isinstance(item, dict) and isinstance(item.get("event"), dict) else {}
        if item.get("league_id") == league_id or event.get("league_id") == league_id:
            filtered.append(item)
    return filtered


def _fixture_count(fixtures: list, match_id: int = None) -> tuple[int, int]:
    current = 0
    after = 0
    for item in fixtures:
        if match_id and _event_id(item) == match_id:
            current += 1
        else:
            after += 1
    return current + after, after


def _lineup_side_data(lineups: dict, side: str) -> dict:
    if not isinstance(lineups, dict):
        return {}
    nested = lineups.get("lineups")
    if isinstance(nested, dict):
        lineups = nested
    keys = {
        "home": ("home", "local", "home_lineup", "homeTeam"),
        "away": ("away", "visitante", "away_lineup", "awayTeam"),
    }[side]
    for key in keys:
        value = lineups.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _player_obj(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {}
    for key in ("player", "player_obj", "footballer"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def _player_name(raw: dict) -> str:
    player = _player_obj(raw)
    return (
        player.get("name")
        or player.get("short_name")
        or player.get("shortName")
        or raw.get("player_name")
        or raw.get("name")
        or "?"
    )


def _player_id(raw: dict):
    player = _player_obj(raw)
    return player.get("id") or player.get("player_id") or raw.get("player_id") or raw.get("id")


def _player_position(raw: dict) -> str:
    player = _player_obj(raw)
    return player.get("position") or raw.get("position") or raw.get("position_code") or "?"


def _player_entry(raw: dict, side: str, source: str, starter_default: bool | None = None) -> dict:
    if not isinstance(raw, dict):
        return {}
    substitute = _first_value(raw, "substitute", "is_substitute", "bench", default=None)
    if substitute is None:
        role = str(raw.get("role") or raw.get("type") or "").casefold()
        if role in {"substitute", "bench", "sub"}:
            substitute = True
        elif starter_default is False:
            substitute = True
        elif starter_default is True:
            substitute = False
        else:
            substitute = False
    starter = (not bool(substitute)) if starter_default is None else starter_default
    player = _player_obj(raw)
    return {
        "id": _player_id(raw),
        "name": _player_name(raw),
        "position": _player_position(raw),
        "jersey_number": player.get("jersey_number") or player.get("jerseyNumber") or raw.get("jersey_number"),
        "rating": _first_value(raw, "rating", "avg_rating"),
        "side": side,
        "source": source,
        "starter": bool(starter),
        "substitute": bool(substitute),
    }


def _extract_bsd_lineup_players(lineups: dict, side: str) -> list:
    side_data = _lineup_side_data(lineups, side)
    players = []
    if not side_data:
        nested = lineups.get("lineups") if isinstance(lineups, dict) else None
        if isinstance(nested, list):
            side_markers = {"home": {"home", "local", True}, "away": {"away", "visitante", False}}[side]
            for raw in nested:
                marker = raw.get("side", raw.get("team_side", raw.get("is_home"))) if isinstance(raw, dict) else None
                if marker in side_markers:
                    entry = _player_entry(raw, side=side, source="lineup", starter_default=None)
                    if entry and entry.get("name") != "?":
                        players.append(entry)
        return players

    for key, starter_default in (
        ("starting_xi", True),
        ("starters", True),
        ("lineup", True),
        ("players", None),
        ("substitutes", False),
        ("bench", False),
    ):
        value = side_data.get(key)
        if not isinstance(value, list):
            continue
        for raw in value:
            entry = _player_entry(raw, side=side, source="lineup", starter_default=starter_default)
            if entry and entry.get("name") != "?":
                players.append(entry)

    seen = set()
    unique = []
    for p in players:
        key = p.get("id") or _norm(p.get("name"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _extract_bsd_unavailable(lineups: dict, side: str) -> list:
    side_data = _lineup_side_data(lineups, side)
    candidates = []
    if isinstance(side_data, dict):
        for key in ("unavailable", "unavailable_players", "missingPlayers", "missing_players", "absent_players"):
            value = side_data.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    root = lineups.get("unavailable_players") if isinstance(lineups, dict) else None
    if isinstance(root, dict):
        for side_key in (("home", "local") if side == "home" else ("away", "visitante")):
            value = root.get(side_key)
            if isinstance(value, list):
                candidates.extend(value)
    elif isinstance(root, list):
        side_markers = {"home": {"home", "local", True}, "away": {"away", "visitante", False}}[side]
        for raw in root:
            marker = raw.get("side", raw.get("team_side", raw.get("is_home"))) if isinstance(raw, dict) else None
            if marker in side_markers:
                candidates.append(raw)

    result = []
    for raw in candidates:
        if not isinstance(raw, dict):
            result.append({"name": str(raw), "position": "?", "status": "missing", "side": side})
            continue
        status = (
            raw.get("status")
            or raw.get("type")
            or raw.get("reason")
            or raw.get("description")
            or "missing"
        )
        result.append({
            "id": _player_id(raw),
            "name": _player_name(raw),
            "position": _player_position(raw),
            "status": str(status),
            "reason": raw.get("description") or raw.get("reason_text") or raw.get("reason"),
            "side": side,
        })
    return result


def _extract_squad_players(squad: dict, side: str) -> list:
    players = _as_list(squad)
    result = []
    for raw in players:
        entry = _player_entry(raw, side=side, source="squad", starter_default=False)
        if entry and entry.get("name") != "?":
            result.append(entry)
    return result


def _pos_group(position: str) -> str:
    pos = str(position or "").upper()
    if pos.startswith("G") or pos in {"GK", "POR"}:
        return "G"
    if pos.startswith("D") or pos in {"CB", "LB", "RB", "LWB", "RWB"}:
        return "D"
    if pos.startswith("M") or pos in {"CM", "DM", "AM", "LW", "RW"}:
        return "M"
    if pos.startswith("F") or pos in {"ST", "CF", "FW"}:
        return "F"
    return "M"


def _select_players_for_impact(players: list, max_per_team: int = 8) -> list:
    selected = []
    seen = set()

    def add_group(group: str, limit: int):
        count = 0
        ordered = sorted(
            [p for p in players if _pos_group(p.get("position")) == group],
            key=lambda p: (not p.get("starter"), p.get("name") or ""),
        )
        for p in ordered:
            key = p.get("id") or _norm(p.get("name"))
            if key in seen:
                continue
            selected.append(p)
            seen.add(key)
            count += 1
            if count >= limit or len(selected) >= max_per_team:
                break

    add_group("G", 1)
    add_group("F", 3)
    add_group("M", 4)
    add_group("D", 4)
    for p in sorted(players, key=lambda x: (not x.get("starter"), x.get("name") or "")):
        if len(selected) >= max_per_team:
            break
        key = p.get("id") or _norm(p.get("name"))
        if key not in seen:
            selected.append(p)
            seen.add(key)
    return selected


def _select_lineup_players_for_profile(players: list, max_per_team: int = 11) -> list:
    selected = []
    seen = set()
    for p in sorted(players, key=lambda x: (not x.get("starter"), x.get("name") or "")):
        if len(selected) >= max_per_team:
            break
        key = p.get("id") or _norm(p.get("name"))
        if key in seen:
            continue
        selected.append(p)
        seen.add(key)
    return selected


def _aggregate_player_stats(rows: list) -> dict:
    rows = _as_list(rows)
    if not rows:
        return {}
    minutes = sum(_to_float(_first_value(r, "minutes_played", "minutes"), 0) or 0 for r in rows)
    apps = sum(1 for r in rows if (_to_float(_first_value(r, "minutes_played", "minutes"), 0) or 0) > 0) or len(rows)

    def total(*keys):
        return sum(_to_float(_first_value(r, *keys), 0) or 0 for r in rows)

    def per90(value):
        return round(value * 90 / minutes, 2) if minutes > 0 else None

    rating_values = []
    rating_weights = []
    for row in rows:
        rating = _to_float(_first_value(row, "rating", "avg_rating"))
        mins = _to_float(_first_value(row, "minutes_played", "minutes"), 0) or 0
        if rating is not None:
            rating_values.append(rating)
            rating_weights.append(max(mins, 1))
    avg_rating = None
    if rating_values:
        avg_rating = round(sum(v * w for v, w in zip(rating_values, rating_weights)) / sum(rating_weights), 2)

    xg = total("expected_goals", "xg")
    xa = total("expected_assists", "xa")
    shots = total("total_shots", "shots")
    shots_on_target = total("shots_on_target", "on_target_scoring_att", "on_target_scoring_attempt")
    key_passes = total("key_pass", "key_passes")
    yellow = total("yellow_card", "yellow_cards")
    saves = total("saves", "goalkeeper_saves")
    tackles = total("total_tackle", "tackles")
    interceptions = total("interception", "interceptions")
    fouls = total("fouls", "fouls_committed")

    return {
        "apps": apps,
        "minutes": round(minutes),
        "avg_rating": avg_rating,
        "goals": round(total("goals"), 2),
        "assists": round(total("goal_assist", "assists"), 2),
        "xg": round(xg, 2),
        "xa": round(xa, 2),
        "shots": round(shots, 2),
        "shots_on_target": round(shots_on_target, 2),
        "key_passes": round(key_passes, 2),
        "yellow_cards": round(yellow, 2),
        "red_cards": round(total("red_card", "red_cards"), 2),
        "saves": round(saves, 2),
        "tackles": round(tackles, 2),
        "interceptions": round(interceptions, 2),
        "fouls": round(fouls, 2),
        "xg_p90": per90(xg),
        "xa_p90": per90(xa),
        "shots_p90": per90(shots),
        "shots_on_target_p90": per90(shots_on_target),
        "key_passes_p90": per90(key_passes),
        "yellow_p90": per90(yellow),
        "saves_p90": per90(saves),
        "tackles_p90": per90(tackles),
        "interceptions_p90": per90(interceptions),
        "fouls_p90": per90(fouls),
    }


def _aggregate_career(career: dict, league_id: int = None, season_id: int = None) -> dict:
    seasons = _as_list(career)
    if not seasons:
        return {}
    chosen = None
    for row in seasons:
        if season_id and row.get("season_id") == season_id:
            chosen = row
            break
    if chosen is None and league_id:
        chosen = next((row for row in seasons if row.get("league_id") == league_id), None)
    chosen = chosen or seasons[0]
    matches = _to_float(chosen.get("matches"), 0) or 0
    minutes = _to_float(chosen.get("minutes"), 0) or 0
    return {
        "apps": int(matches),
        "minutes": int(minutes),
        "goals": _to_float(chosen.get("goals"), 0),
        "assists": _to_float(chosen.get("assists"), 0),
        "avg_rating": _to_float(chosen.get("avg_rating")),
        "career_fallback": True,
    }


def _fetch_player_profile_stats(player: dict, league_id: int = None, season_id: int = None) -> dict:
    player_id = player.get("id")
    if not player_id:
        return {}
    try:
        rows = obtener_stats_jugador(player_id, season_id=season_id, league_id=league_id, limit=12)
    except requests.HTTPError:
        rows = obtener_stats_jugador(player_id, limit=12)
    summary = _aggregate_player_stats(rows)
    if summary:
        return summary
    try:
        return _aggregate_career(obtener_carrera_jugador(player_id), league_id=league_id, season_id=season_id)
    except Exception:
        return {}


def _build_player_impact(v2_data: dict, league_id: int = None, season_id: int = None) -> dict:
    lineups = v2_data.get("lineups", {})
    has_lineups = isinstance(lineups, dict) and "_error" not in lineups and any(
        _extract_bsd_lineup_players(lineups, side) for side in ("home", "away")
    )
    sources = {}
    unavailable = {}
    candidates = []

    for side, squad_key in [("home", "squad_home"), ("away", "squad_away")]:
        players = _extract_bsd_lineup_players(lineups, side) if has_lineups else []
        source = "lineups" if players else "squad"
        if not players:
            players = _extract_squad_players(v2_data.get(squad_key, {}), side)
        selected = (
            _select_lineup_players_for_profile(players, max_per_team=11)
            if source == "lineups"
            else _select_players_for_impact(players, max_per_team=10)
        )
        sources[side] = {"source": source, "players": selected}
        candidates.extend(selected)

        missing = _extract_bsd_unavailable(lineups, side) if isinstance(lineups, dict) else []
        unavailable[side] = missing
        for p in missing[:4]:
            if p.get("id"):
                candidates.append({**p, "source": "unavailable", "starter": False})

    if not candidates:
        return {}

    unique_candidates = []
    seen_ids = set()
    for player in candidates:
        pid = player.get("id")
        if not pid or pid in seen_ids:
            continue
        unique_candidates.append(player)
        seen_ids.add(pid)
    if not unique_candidates:
        return {}

    impacts = []
    with ThreadPoolExecutor(max_workers=min(10, len(unique_candidates))) as pool:
        futures = {
            pool.submit(_fetch_player_profile_stats, player, league_id, season_id): player
            for player in unique_candidates
        }
        for future in as_completed(futures):
            player = futures[future]
            try:
                stats = future.result()
            except Exception as exc:
                stats = {"_error": str(exc)}
            impacts.append({**player, "stats": stats})

    return {
        "season_id": season_id,
        "selection": sources,
        "unavailable": unavailable,
        "players": impacts,
    }


def _build_motivation_context(
    v2_data: dict,
    match_id: int,
    league_id: int,
    home_id: int,
    away_id: int,
    home_name: str,
    away_name: str,
) -> dict:
    flags = get_competition_flags(league_id, LEAGUE_NAMES.get(league_id))
    if flags.get("is_friendly"):
        return {
            "competition_flags": flags,
            "special_context": (
                "Amistoso internacional: sin tabla competitiva; motivacion, "
                "minutos y XI pueden responder a pruebas/rotacion."
            ),
        }

    standings = v2_data.get("standings", {})
    rows = _extract_standing_rows(standings)
    if not rows:
        if flags.get("is_world_cup"):
            return {
                "competition_flags": flags,
                "special_context": (
                    "World Cup 2026: torneo internacional de maxima presion. "
                    "Si no hay tabla/fase confiable, no inventar escenarios."
                ),
            }
        return {}

    home_row = _find_standing_row(rows, home_id, home_name)
    away_row = _find_standing_row(rows, away_id, away_name)
    if not home_row or not away_row:
        return {}

    home_fixtures = _filter_same_league_fixtures(_as_list(v2_data.get("fixtures_home")), league_id)
    away_fixtures = _filter_same_league_fixtures(_as_list(v2_data.get("fixtures_away")), league_id)
    home_remaining, home_after = _fixture_count(home_fixtures, match_id)
    away_remaining, away_after = _fixture_count(away_fixtures, match_id)

    same_group_rows = rows
    group = home_row.get("_group")
    if group is not None and group == away_row.get("_group"):
        same_group_rows = [r for r in rows if r.get("_group") == group]

    sorted_rows = sorted(same_group_rows, key=lambda r: r.get("position") or 999)
    total_teams = len(sorted_rows)
    if league_id in {7, 8, 32, 33} and group is None and total_teams > 8:
        # En copas, BSD puede devolver una tabla global mezclando grupos.
        # Si no podemos aislar el grupo, es mejor omitir esta capa.
        return {}

    expected = None
    if total_teams == 4:
        expected = 6
    elif total_teams >= 10:
        expected = (total_teams - 1) * 2

    def team_summary(row, remaining, after):
        played = _first_value(row, "played", "matches", default=0) or 0
        fallback_remaining = max(expected - played, 0) if expected is not None else None
        effective_remaining = remaining or fallback_remaining or 0
        pts = _first_value(row, "pts", "points", default=0) or 0
        return {
            "team_id": row.get("team_id"),
            "team_name": row.get("team_name"),
            "position": row.get("position"),
            "points": pts,
            "played": played,
            "goal_difference": _first_value(row, "gd", "goal_difference"),
            "xgd": row.get("xgd"),
            "form": row.get("form"),
            "group": row.get("_group"),
            "remaining_including_current": effective_remaining,
            "remaining_after_current": after,
            "max_points": pts + 3 * effective_remaining,
        }

    leader = sorted_rows[0] if sorted_rows else {}
    top2 = sorted_rows[1] if len(sorted_rows) >= 2 else {}
    top4 = sorted_rows[3] if len(sorted_rows) >= 4 else {}
    relegation_cut = sorted_rows[-3] if len(sorted_rows) >= 10 else {}

    result = {
        "competition_flags": flags,
        "format_expected_matches": expected,
        "total_teams_scope": total_teams,
        "scope_group": group,
        "home": team_summary(home_row, home_remaining, home_after),
        "away": team_summary(away_row, away_remaining, away_after),
        "cutoffs": {
            "leader_points": _first_value(leader, "pts", "points"),
            "top2_points": _first_value(top2, "pts", "points"),
            "top4_points": _first_value(top4, "pts", "points"),
            "relegation_line_points": _first_value(relegation_cut, "pts", "points"),
        },
    }
    if flags.get("is_world_cup"):
        result["special_context"] = (
            "World Cup 2026: usar tabla/grupo o fase real si esta disponible; "
            "en eliminatoria pesa prorroga/penales y el 1T suele ser mas tactico."
        )
    return result


def _extract_best_odds_for_event(best_rows: list, event_id: int) -> dict:
    for row in best_rows or []:
        if _event_id(row) == event_id:
            return row
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# ENRIQUECIMIENTO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════════

def enriquecer_con_v2(datos_resumidos: dict, match_id: int) -> dict:
    """
    Enriquece los datos de un partido con todos los endpoints v2.

    Se ejecuta despues de obtener el detalle y prediccion base.
    Hace llamadas en paralelo a todos los sub-recursos.

    Args:
        datos_resumidos: Dict base con al menos home_team_id, away_team_id,
                         league (nombre), y IDs de coachs/arbitro/venue.
        match_id: ID del evento en BSD.

    Returns:
        El mismo dict con clave '_bsd_v2' agregada.
    """
    v2_data = {}

    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return {"_error": str(e)}

    # 1. Obtener detalle v2 para tener IDs de coach, referee, venue, etc.
    detalle = _safe(obtener_detalle_evento, match_id)

    home_id = datos_resumidos.get("home_team_id") or (detalle or {}).get("home_team_id")
    away_id = datos_resumidos.get("away_team_id") or (detalle or {}).get("away_team_id")
    league_id = datos_resumidos.get("league_id") or (detalle or {}).get("league_id")
    home_coach_id = datos_resumidos.get("home_coach_id") or (detalle or {}).get("home_coach_id")
    away_coach_id = datos_resumidos.get("away_coach_id") or (detalle or {}).get("away_coach_id")
    referee_id = datos_resumidos.get("referee_id") or (detalle or {}).get("referee_id")
    event_date = (detalle or {}).get("event_date") or datos_resumidos.get("fecha")
    partido = datos_resumidos.get("partido", "")
    home_name, away_name = partido.split(" vs ", 1) if " vs " in partido else ((detalle or {}).get("home_team", ""), (detalle or {}).get("away_team", ""))

    # Actualizar datos basicos desde v2 (weather, neutral_ground, etc.)
    if detalle and "_error" not in detalle:
        datos_resumidos["_v2_detail"] = {
            "weather": detalle.get("weather"),
            "is_local_derby": detalle.get("is_local_derby"),
            "is_neutral_ground": detalle.get("is_neutral_ground"),
            "travel_distance_km": detalle.get("travel_distance_km"),
            "pitch_condition": detalle.get("pitch_condition"),
            "attendance": detalle.get("attendance"),
            "round_number": detalle.get("round_number"),
        }

    # 2. Lanzar todas las sub-consultas en paralelo
    futures = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures["stats"] = pool.submit(_safe, obtener_stats_evento, match_id)
        futures["metadata"] = pool.submit(_safe, obtener_metadata_evento, match_id)
        futures["lineups"] = pool.submit(_safe, obtener_lineups_evento, match_id)
        futures["player_stats"] = pool.submit(_safe, obtener_player_stats_evento, match_id)
        futures["odds"] = pool.submit(_safe, obtener_odds_evento, match_id)
        futures["odds_comparison"] = pool.submit(_safe, obtener_odds_comparison_evento, match_id)
        futures["polymarket"] = pool.submit(_safe, obtener_polymarket_evento, match_id)
        futures["prediction"] = pool.submit(_safe, obtener_prediccion_por_evento, match_id)

        if home_coach_id:
            futures["manager_home"] = pool.submit(_safe, obtener_manager, home_coach_id)
        if away_coach_id:
            futures["manager_away"] = pool.submit(_safe, obtener_manager, away_coach_id)
        if referee_id:
            arb_name = (datos_resumidos.get("arbitro") or {}).get("nombre", "")
            futures["referee"] = pool.submit(_safe, obtener_arbitro, referee_id, league_id, arb_name)
            futures["referee_matches"] = pool.submit(_safe, obtener_partidos_arbitro, referee_id, league_id)
        if league_id:
            futures["season"] = pool.submit(_safe, obtener_temporada_actual, league_id)
            futures["standings"] = pool.submit(_safe, obtener_standings, league_id)
        if home_id:
            futures["squad_home"] = pool.submit(_safe, obtener_squad, home_id)
        if away_id:
            futures["squad_away"] = pool.submit(_safe, obtener_squad, away_id)

        for key, future in futures.items():
            v2_data[key] = future.result()

    # 2.b. Consultas que dependen de temporada/fecha: fixtures restantes,
    # mejores cuotas y perfil historico de jugadores.
    season_id = _extract_season_id(v2_data.get("standings"), _season_payload(v2_data.get("season")))
    if season_id:
        v2_data["_season_id"] = season_id

    date_from = f"{date.today().isoformat()}T00:00:00Z"
    season_end = _extract_season_end(v2_data.get("season"))
    date_to = f"{season_end}T23:59:59Z" if season_end else f"{(date.today() + timedelta(days=210)).isoformat()}T23:59:59Z"
    odds_from, odds_to = _event_window(event_date)

    extra_futures = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        if home_id:
            extra_futures["fixtures_home"] = pool.submit(_safe, obtener_fixtures_equipo, home_id, league_id, date_from, date_to, "notstarted")
        if away_id:
            extra_futures["fixtures_away"] = pool.submit(_safe, obtener_fixtures_equipo, away_id, league_id, date_from, date_to, "notstarted")
        if league_id and odds_from and odds_to:
            extra_futures["best_odds_1x2_raw"] = pool.submit(_safe, obtener_best_odds, "1x2", league_id, season_id, None, odds_from, odds_to)
        for key, future in extra_futures.items():
            v2_data[key] = future.result()

    best_rows = v2_data.pop("best_odds_1x2_raw", [])
    if isinstance(best_rows, list):
        v2_data["best_odds_1x2"] = _extract_best_odds_for_event(best_rows, match_id)

    motivation = _build_motivation_context(
        v2_data=v2_data,
        match_id=match_id,
        league_id=league_id,
        home_id=home_id,
        away_id=away_id,
        home_name=home_name,
        away_name=away_name,
    )
    if motivation:
        v2_data["motivation"] = motivation

    try:
        impact = _build_player_impact(v2_data, league_id=league_id, season_id=season_id)
        if impact:
            v2_data["player_impact"] = impact
    except Exception as e:
        v2_data["player_impact"] = {"_error": str(e)}

    # 3. Enriquecer entrenadores con datos detallados de manager
    for lado, mkey in [("home", "manager_home"), ("away", "manager_away")]:
        manager = v2_data.get(mkey, {})
        if manager and "_error" not in manager:
            coach_key = f"entrenador_{'local' if lado == 'home' else 'visitante'}"
            existing = datos_resumidos.get(coach_key, {})
            existing.update({
                "win_pct": manager.get("win_pct"),
                "avg_goals_scored": manager.get("avg_goals_scored"),
                "avg_goals_conceded": manager.get("avg_goals_conceded"),
                "avg_possession": manager.get("avg_possession"),
                "clean_sheet_pct": manager.get("clean_sheet_pct"),
                "btts_pct": manager.get("btts_pct"),
                "over_25_pct": manager.get("over_25_pct"),
                "tactical_profile": manager.get("tactical_profile"),
                "preferred_formation": manager.get("preferred_formation"),
                "matches_total": manager.get("matches_total"),
            })
            datos_resumidos[coach_key] = existing

    # 4. Enriquecer arbitro con datos detallados de referee
    # SofaScore tiene prioridad en YC/RC (datos por torneo, mas completos que BSD)
    referee = v2_data.get("referee", {})
    if referee and "_error" not in referee:
        existing_arb = datos_resumidos.get("arbitro") or {}
        # Preservar YC/RC de SofaScore si ya existen
        ss_yc = existing_arb.get("avg_yellow_per_match") if existing_arb.get("_fuente_yc") == "SofaScore" else referee.get("avg_yellow_per_match")
        ss_rc = existing_arb.get("avg_red_per_match") if existing_arb.get("_fuente_yc") == "SofaScore" else referee.get("avg_red_per_match")
        datos_resumidos["arbitro"] = {
            **existing_arb,
            "id": referee.get("id") or existing_arb.get("id"),
            "nombre": referee.get("name") or existing_arb.get("nombre"),
            "nacionalidad": referee.get("country") or existing_arb.get("nacionalidad"),
            "total_yellow_cards": referee.get("total_yellow_cards"),
            "total_red_cards": referee.get("total_red_cards"),
            "avg_yellow_per_match": ss_yc,
            "avg_red_per_match": ss_rc,
            "avg_goals_per_match": referee.get("avg_goals_per_match"),
            "avg_fouls_per_match": referee.get("avg_fouls_per_match"),
        }

    # 5. Guardar todo en _bsd_v2
    v2_data.pop("manager_home", None)
    v2_data.pop("manager_away", None)
    v2_data.pop("referee", None)
    datos_resumidos["_bsd_v2"] = v2_data

    # 6. Actualizar prediccion si v2 devolvio una
    pred = v2_data.get("prediction", {})
    if pred and "_error" not in pred and pred:
        datos_resumidos["_v2_prediction_raw"] = pred

    return datos_resumidos


# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN PARA EL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

def resumir_stats_v2_para_prompt(v2_data: dict) -> str:
    """Formatea las stats v2 para incluir en el prompt de IA.
    Retorna string vacio si no hay datos reales (pre-match)."""
    stats = v2_data.get("stats", {})
    if not stats or "_error" in stats:
        return ""

    per_team = stats.get("stats", {})
    home_stats = per_team.get("home", {})
    away_stats = per_team.get("away", {})

    has_data = (
        home_stats.get("total_shots") is not None
        or away_stats.get("total_shots") is not None
        or stats.get("shotmap")
        or stats.get("xg_per_minute")
        or stats.get("momentum")
    )
    if not has_data:
        return ""

    partes = []

    # Per-team stats
    for lado, label in [("home", "LOCAL"), ("away", "VISITANTE")]:
        s = per_team.get(lado, {})
        if not s or s.get("total_shots") is None:
            continue
        lineas = [f"\n### STATS {label} (BSD v2)"]
        lineas.append(f"- Tiros totales: {s.get('total_shots', 'N/D')}")
        lineas.append(f"- Posesion: {s.get('ball_possession', 'N/D')}%")
        lineas.append(f"- Precision de pases: {s.get('pass_accuracy_pct', 'N/D')}%")
        lineas.append(f"- Ataques: {s.get('attack', 'N/D')}")
        lineas.append(f"- Ataques peligrosos: {s.get('dangerous_attack', 'N/D')}")
        lineas.append(f"- Balon seguro: {s.get('ball_safe', 'N/D')}")

        # Ratio stats
        for key, label_r in [("crosses", "Centros"), ("dribbles", "Regates"),
                             ("long_balls", "Balones largos")]:
            r = s.get(key, {})
            if r:
                lineas.append(f"- {label_r}: {r.get('value', 0)}/{r.get('total', 0)} ({r.get('pct', 0)}%)")

        # xG
        xg = s.get("xg", {})
        if xg:
            lineas.append(f"- xG real: {xg.get('actual', 'N/D')}")
        partes.append("\n".join(lineas))

    # xG per minute
    xgpm = stats.get("xg_per_minute", [])
    if xgpm:
        partes.append(f"\n### xG POR MINUTO ({len(xgpm)} puntos)")
        partes.append("(xG acumulado minuto a minuto para detectar patrones de gol)")

    # Shotmap summary
    shotmap = stats.get("shotmap", [])
    if shotmap:
        goles = [s for s in shotmap if s.get("goal")]
        partes.append(f"\n### SHOTMAP ({len(shotmap)} disparos, {len(goles)} goles)")
        for s in goles[:5]:
            partes.append(f"  - Gol {s.get('player')}: min {s.get('minute')}, xG {s.get('xg')}")

    # Momentum summary
    momentum = stats.get("momentum", [])
    if momentum:
        partes.append(f"\n### MOMENTUM ({len(momentum)} puntos)")

    return "\n".join(partes)


def resumir_metadata_v2_para_prompt(v2_data: dict) -> str:
    """Formatea metadata para el prompt."""
    meta = v2_data.get("metadata", {})
    if not meta or "_error" in meta:
        return ""

    partes = []

    funfacts = meta.get("funfacts", [])
    if funfacts:
        partes.append("\n### DATOS PRE-PARTIDO (BSD v2)")
        for f in funfacts[:5]:
            partes.append(f"- {f.get('sentence', '')}")

    ai_preview = meta.get("ai_preview", {})
    if ai_preview and ai_preview.get("text"):
        partes.append(f"\n### PREVIEW IA\n{ai_preview['text'][:500]}")

    return "\n".join(partes)


def resumir_player_stats_v2_para_prompt(v2_data: dict, limit: int = 14) -> str:
    """Formatea stats de jugadores para el prompt (top por rating)."""
    ps = v2_data.get("player_stats", {})
    if not ps or "_error" in ps:
        return ""

    players = ps.get("player_stats", [])
    if not players:
        return ""

    sorted_players = sorted(players, key=lambda p: p.get("rating") or 0, reverse=True)
    partes = ["\n### JUGADORES DESTACADOS (BSD v2) - ordenados por rating"]
    partes.append(f"{'Jugador':<22} | Rat | G | xG  | Pases | Entradas | Min")
    partes.append("-" * 70)

    for p in sorted_players[:limit]:
        pid = p.get("player_id", "?")
        rating = p.get("rating") or "-"
        goals = p.get("goals", 0)
        xg = p.get("expected_goals")
        passes = f"{p.get('accurate_pass', 0)}/{p.get('total_pass', 0)}"
        tackles = p.get("total_tackle", 0)
        mins = p.get("minutes_played", 0)
        partes.append(
            f"{str(pid):<22} | {str(rating):>3} | {goals} | {str(xg):>4} | {passes:<9} | {str(tackles):>8} | {mins}"
        )

    return "\n".join(partes)


def resumir_standings_v2_para_prompt(v2_data: dict, home_id: int = None, away_id: int = None) -> str:
    """Formatea standings con xG para el prompt, destacando los 2 equipos."""
    standings = v2_data.get("standings", {})
    if not standings or "_error" in standings:
        return ""

    rows = _extract_standing_rows(standings)
    if not rows:
        return ""

    def _first(row: dict, *keys, default=0):
        for key in keys:
            if row.get(key) is not None:
                return row.get(key)
        return default

    partes = ["\n### TABLA DE POSICIONES (BSD v2) - con xG"]
    if standings.get("grouped") or any(row.get("_group") is not None for row in rows):
        grupos = sorted({str(row.get("_group")) for row in rows if row.get("_group") is not None})
        if grupos:
            partes.append(f"Grupos detectados: {', '.join(grupos[:8])}")
    partes.append(
        f"{'#':>3} {'Equipo':<24} {'Grp':<6} {'PJ':>3} {'V':>2} {'E':>2} {'D':>2} "
        f"{'Pts':>4} {'GF':>3} {'GC':>3} {'DG':>4} {'xGF':>6} {'xGA':>6} {'xGD':>6} {'Forma':>6}"
    )
    partes.append("-" * 111)

    destacados = set()
    if home_id:
        destacados.add(home_id)
    if away_id:
        destacados.add(away_id)

    for row in rows[:20]:
        tid = row.get("team_id")
        marker = ">" if tid in destacados else " "
        team = row.get("team_name") or (row.get("team") or {}).get("name") or "?"
        gf = _first(row, "gf", "goals_for")
        ga = _first(row, "ga", "goals_against")
        gd = _first(row, "gd", "goal_difference")
        if gd is None:
            gd = (_to_float(gf, 0) or 0) - (_to_float(ga, 0) or 0)
        xgf = _to_float(row.get("xgf"), 0) or 0
        xga = _to_float(row.get("xga"), 0) or 0
        xgd = _to_float(row.get("xgd"), 0) or 0
        partes.append(
            f"{marker}{row.get('position', '?'):>2} "
            f"{team:<24} "
            f"{str(row.get('_group') or '-')[:6]:<6} "
            f"{_first(row, 'played', 'matches'):>3} "
            f"{_first(row, 'wins', 'won'):>2} "
            f"{_first(row, 'draws', 'drawn'):>2} "
            f"{_first(row, 'losses', 'lost'):>2} "
            f"{_first(row, 'pts', 'points'):>4} "
            f"{gf:>3} "
            f"{ga:>3} "
            f"{gd:>4} "
            f"{xgf:>6.1f} "
            f"{xga:>6.1f} "
            f"{xgd:>6.1f} "
            f"{row.get('form', ''):>6}"
        )

    return "\n".join(partes)


def resumir_squads_v2_para_prompt(v2_data: dict) -> str:
    """Formatea plantillas para el prompt."""
    squads = []
    for key, label in [("squad_home", "LOCAL"), ("squad_away", "VISITANTE")]:
        s = v2_data.get(key, {})
        if not s or "_error" in s:
            continue
        players = s.get("players", [])
        if not players:
            continue
        squads.append(f"\n### PLANTILLA {label} ({len(players)} jugadores - BSD v2)")
        for p in players[:18]:
            squads.append(
                f"  - {p.get('name', '?')} ({p.get('position', '?')}) "
                f"#{p.get('jersey_number', '?')} | {p.get('nationality', '?')}"
            )
    return "\n".join(squads)


def resumir_manager_v2_para_prompt(datos_resumidos: dict) -> str:
    """Formatea datos de entrenador enriquecidos con v2."""
    partes = []
    for lado, key in [("LOCAL", "entrenador_local"), ("VISITANTE", "entrenador_visitante")]:
        e = datos_resumidos.get(key, {})
        if not e:
            continue
        lineas = [f"\n### DT {lado}: {e.get('nombre', '?')}"]
        if e.get("formacion"):
            lineas.append(f"- Formacion: {e.get('formacion')}")
        if e.get("perfil"):
            lineas.append(f"- Perfil tactico: {e.get('perfil')}")
        if e.get("tactical_profile"):
            lineas.append(f"- Perfil BSD: {e.get('tactical_profile')}")
        if e.get("win_pct") is not None:
            lineas.append(f"- Win%: {e.get('win_pct')}%")
        if e.get("avg_goals_scored") is not None:
            lineas.append(f"- Goles/prom: {e.get('avg_goals_scored')}")
        if e.get("avg_goals_conceded") is not None:
            lineas.append(f"- Goles recibidos/prom: {e.get('avg_goals_conceded')}")
        if e.get("avg_possession") is not None:
            lineas.append(f"- Posesion prom: {e.get('avg_possession')}%")
        if e.get("clean_sheet_pct") is not None:
            lineas.append(f"- Clean sheets: {e.get('clean_sheet_pct')}%")
        if e.get("btts_pct") is not None:
            lineas.append(f"- BTTS: {e.get('btts_pct')}%")
        if e.get("over_25_pct") is not None:
            lineas.append(f"- Over 2.5: {e.get('over_25_pct')}%")
        partes.append("\n".join(lineas))
    return "\n".join(partes)


def resumir_arbitro_v2_para_prompt(datos_resumidos: dict) -> str:
    """Formatea datos de arbitro enriquecidos con v2."""
    arb = datos_resumidos.get("arbitro")
    if not arb:
        return "\n### ARBITRO\nNo asignado aun."

    fuente = arb.get("_fuente_yc") or "BSD v2"
    partes = [f"\n### ARBITRO ({fuente}): {arb.get('nombre', '?')}"]
    if arb.get("nacionalidad"):
        partes.append(f"- Nacionalidad: {arb.get('nacionalidad')}")
    comp = arb.get("_sofascore_competicion")
    if comp:
        partes.append(
            f"- Competición actual ({comp.get('nombre', '?')}): "
            f"{comp.get('partidos', '?')} part, {comp.get('yc_pp', '?')} YC/part, "
            f"{comp.get('rc_pp', '?')} RC/part"
        )
    if arb.get("matches"):
        partes.append(f"- Partidos dirigidos: {arb.get('matches')}")
    if arb.get("avg_yellow_per_match") is not None:
        partes.append(f"- Amarillas/partido: {arb.get('avg_yellow_per_match')}")
    if arb.get("avg_red_per_match") is not None:
        partes.append(f"- Rojas/partido: {arb.get('avg_red_per_match')}")
    if arb.get("avg_fouls_per_match") is not None:
        partes.append(f"- Faltas/partido: {arb.get('avg_fouls_per_match')}")
    if arb.get("avg_goals_per_match") is not None:
        partes.append(f"- Goles/partido: {arb.get('avg_goals_per_match')}")

    ref_data = arb.get("_sofascore_ref") or {}
    torneos = ref_data.get("torneos", []) if isinstance(ref_data, dict) else []
    if torneos:
        partes.append("- Promedios por competición:")
        for t in torneos[:8]:
            tn = t.get("nombre", "?")
            apps = t.get("partidos")
            yc = t.get("yc_pp", "?")
            rc = t.get("rc_pp", "?")
            pen = t.get("penaltis")
            detalle = f"{yc} YC/part, {rc} RC/part"
            if apps is not None:
                detalle = f"{apps} part, {detalle}"
            if pen is not None:
                detalle += f", {pen} pen"
            partes.append(f"    {tn}: {detalle}")

    recent = _as_list((datos_resumidos.get("_bsd_v2") or {}).get("referee_matches"))
    finished_recent = [m for m in recent if str(m.get("status", "")).lower() in {"finished", "ft", "ended", ""}]
    if finished_recent:
        partes.append("- Partidos recientes arbitrados (BSD v2):")
        for m in finished_recent[:5]:
            home = m.get("home_team") or (m.get("event") or {}).get("home_team") or "Local"
            away = m.get("away_team") or (m.get("event") or {}).get("away_team") or "Visitante"
            score = ""
            if m.get("home_score") is not None and m.get("away_score") is not None:
                score = f" {m.get('home_score')}-{m.get('away_score')}"
            partes.append(f"    {str(m.get('event_date', ''))[:10]} {home} vs {away}{score}")
    return "\n".join(partes)


def _team_labels(datos_resumidos: dict) -> tuple[str, str]:
    partido = datos_resumidos.get("partido", "")
    if " vs " in partido:
        return partido.split(" vs ", 1)
    return "Local", "Visitante"


def _names_from_sofascore_lineup(datos_resumidos: dict, side: str) -> set:
    ss = (datos_resumidos.get("_sofascore") or {}).get("alineaciones") or {}
    key = "local" if side == "home" else "visitante"
    data = ss.get(key) or {}
    names = set()
    for bucket in ("titulares", "suplentes"):
        for p in data.get(bucket, []) or []:
            names.add(_norm(p.get("nombre")))
    return {n for n in names if n}


def _names_from_sofascore_missing(datos_resumidos: dict, side: str) -> set:
    ss = (datos_resumidos.get("_sofascore") or {}).get("alineaciones") or {}
    key = "local" if side == "home" else "visitante"
    data = ss.get(key) or {}
    names = set()
    bajas = data.get("bajas", {}) if isinstance(data, dict) else {}
    for bucket in ("confirmadas", "dudas"):
        for p in bajas.get(bucket, []) or []:
            names.add(_norm(p.get("nombre")))
    return {n for n in names if n}


def _format_names(players: list, limit: int = 14) -> str:
    names = []
    for p in players[:limit]:
        pos = p.get("position") or "?"
        names.append(f"{p.get('name', '?')} ({pos})")
    extra = len(players) - limit
    if extra > 0:
        names.append(f"+{extra} mas")
    return ", ".join(names) if names else "-"


def _lineup_conflicts(datos_resumidos: dict, lineups: dict) -> list:
    conflicts = []
    labels = {"home": _team_labels(datos_resumidos)[0], "away": _team_labels(datos_resumidos)[1]}
    for side in ("home", "away"):
        ss_available = _names_from_sofascore_lineup(datos_resumidos, side)
        ss_missing = _names_from_sofascore_missing(datos_resumidos, side)
        bsd_players = _extract_bsd_lineup_players(lineups, side)
        bsd_available = {_norm(p.get("name")) for p in bsd_players}
        bsd_missing = {_norm(p.get("name")) for p in _extract_bsd_unavailable(lineups, side)}

        for name in sorted(bsd_missing & ss_available):
            conflicts.append(f"{labels[side]}: BSD lo marca no disponible, pero SofaScore lo lista en XI/banco ({name}). SofaScore manda.")
        for name in sorted(ss_missing & bsd_available):
            conflicts.append(f"{labels[side]}: SofaScore lo marca baja/duda, pero BSD lo lista en XI/banco ({name}). Revisar; por regla manda SofaScore.")
    return conflicts


def resumir_lineups_v2_para_prompt(datos_resumidos: dict) -> str:
    """Lineups BSD como respaldo compacto. SofaScore conserva prioridad."""
    v2 = datos_resumidos.get("_bsd_v2", {})
    lineups = v2.get("lineups", {}) if isinstance(v2, dict) else {}
    if not lineups or "_error" in lineups:
        return ""

    has_ss_lineup = bool((datos_resumidos.get("_sofascore") or {}).get("alineaciones"))
    conflicts = _lineup_conflicts(datos_resumidos, lineups)
    if has_ss_lineup:
        if not conflicts:
            return ""
        return "\n".join([
            "\n### CONTROL DE DISPONIBILIDAD BSD v2 vs SofaScore",
            "SofaScore sigue siendo la fuente principal de alineaciones/bajas.",
            *[f"- {c}" for c in conflicts[:10]],
        ])

    local_team, away_team = _team_labels(datos_resumidos)
    confirmed = lineups.get("confirmed")
    if confirmed is None:
        confirmed = lineups.get("is_confirmed")
    lineup_status = str(lineups.get("lineup_status") or "").casefold()
    if confirmed is None and lineup_status:
        confirmed = lineup_status in {"confirmed", "official", "available"}
    estado = "CONFIRMADA" if confirmed else (lineups.get("lineup_status") or "PRELIMINAR/POSIBLE")
    has_bsd_payload = any(
        _extract_bsd_lineup_players(lineups, side) or _extract_bsd_unavailable(lineups, side)
        for side in ("home", "away")
    )
    if not has_bsd_payload:
        return "\n".join([
            "\n### ALINEACIONES BSD v2 (respaldo)",
            f"Estado BSD: {estado}. Sin XI ni no disponibles utiles todavia.",
        ])
    partes = [
        "\n### ALINEACIONES BSD v2 (respaldo)",
        f"Estado: {estado}. Usa esta seccion solo porque SofaScore no trajo alineacion.",
    ]

    for side, label in [("home", local_team), ("away", away_team)]:
        side_data = _lineup_side_data(lineups, side)
        players = _extract_bsd_lineup_players(lineups, side)
        starters = [p for p in players if p.get("starter")]
        subs = [p for p in players if p.get("substitute")]
        formation = side_data.get("formation") or side_data.get("formation_name") or "?"
        partes.append(f"\n**{label}** ({formation})")
        if starters:
            partes.append(f"  Titulares BSD: {_format_names(starters, 12)}")
        if subs:
            partes.append(f"  Suplentes BSD: {_format_names(subs, 10)}")
        missing = _extract_bsd_unavailable(lineups, side)
        if missing:
            partes.append("  No disponibles/dudas BSD:")
            for p in missing[:8]:
                reason = f" - {p.get('reason')}" if p.get("reason") else ""
                partes.append(f"    - {p.get('name', '?')} ({p.get('position', '?')}): {p.get('status', '?')}{reason}")

    partes.append("IMPORTANTE: si SofaScore aparece luego y contradice esta seccion, SofaScore manda.")
    return "\n".join(partes)


def _stat(stats: dict, key: str, default=0):
    if not isinstance(stats, dict):
        return default
    return stats.get(key, default)


def _has_any_stat(player: dict, keys: tuple[str, ...]) -> bool:
    stats = player.get("stats") or {}
    for key in keys:
        value = _stat(stats, key, 0)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _impact_line(player: dict, mode: str = "attack") -> str:
    stats = player.get("stats") or {}
    name = player.get("name", "?")
    pos = player.get("position", "?")
    apps = _stat(stats, "apps", "-")
    minutes = _stat(stats, "minutes", "-")
    rating = _stat(stats, "avg_rating", "-")
    base = f"{name} ({pos}) {apps}p/{minutes}min, rating {rating}"
    if mode == "gk":
        return f"{base}, atajadas {stats.get('saves', 0)} ({stats.get('saves_p90', '-')}/90)"
    if mode == "cards":
        return f"{base}, YC {stats.get('yellow_cards', 0)} ({stats.get('yellow_p90', '-')}/90), entradas {stats.get('tackles', 0)}"
    if mode == "creator":
        return f"{base}, key passes {stats.get('key_passes', 0)} ({stats.get('key_passes_p90', '-')}/90), asist {stats.get('assists', 0)}, xA {stats.get('xa', 0)}"
    return f"{base}, xG {stats.get('xg', 0)} ({stats.get('xg_p90', '-')}/90), tiros {stats.get('shots', 0)} ({stats.get('shots_p90', '-')}/90), goles {stats.get('goals', 0)}"


def resumir_player_avgs_v2_para_prompt(datos_resumidos: dict, limit_per_team: int = 11) -> str:
    """Lista medias generales por jugador desde BSD v2 para complementar SofaScore."""
    v2 = datos_resumidos.get("_bsd_v2", {})
    impact = v2.get("player_impact", {}) if isinstance(v2, dict) else {}
    if not impact or "_error" in impact:
        return ""

    players = [
        p for p in impact.get("players", [])
        if p.get("source") != "unavailable"
        and isinstance(p.get("stats"), dict)
        and "_error" not in p.get("stats", {})
        and p.get("stats")
    ]
    if not players:
        return ""

    local_team = datos_resumidos.get("partido", "").split(" vs ")[0] if " vs " in datos_resumidos.get("partido", "") else "Local"
    away_team = datos_resumidos.get("partido", "").split(" vs ")[1] if " vs " in datos_resumidos.get("partido", "") else "Visitante"
    side_names = {"home": local_team, "away": away_team}

    partes = ["\n### MEDIAS POR JUGADOR (BSD v2)"]
    if impact.get("season_id"):
        partes.append(f"Base: ultimos partidos filtrados por temporada {impact.get('season_id')} cuando BSD los entrega; career como respaldo si no hay partido-a-partido.")
    else:
        partes.append("Base: ultimos partidos BSD; career como respaldo si no hay partido-a-partido.")

    any_content = False
    for side in ("home", "away"):
        rows = [p for p in players if p.get("side") == side]
        if not rows:
            continue
        any_content = True
        rows.sort(key=lambda p: (not p.get("starter"), _pos_group(p.get("position")), p.get("name") or ""))
        partes.append(f"\n**{side_names[side]}**")
        for player in rows[:limit_per_team]:
            partes.append(f"  - {_player_avg_line_v2(player)}")

    if not any_content:
        return ""
    partes.append("Uso: tiros/90 y SOT/90 para volumen/calidad individual; xG/90 para amenaza; xA/key/90 para creacion; YC/90/faltas/90/duelos para tarjetas.")
    return "\n".join(partes)


def _player_avg_line_v2(player: dict) -> str:
    stats = player.get("stats") or {}
    role = "titular" if player.get("starter") else ("suplente" if player.get("substitute") else "plantilla")
    if stats.get("career_fallback"):
        role += ", career"

    base = [f"{player.get('name', '?')} ({player.get('position', '?')}, {role})"]
    apps = _stat(stats, "apps", "-")
    minutes = _stat(stats, "minutes", "-")
    rating = _stat(stats, "avg_rating", "-")
    details = [f"{apps}p/{minutes}min"]
    if rating != "-":
        details.append(f"rating {_fmt_player_avg_v2(rating)}")
    details.append(f"G/A {_fmt_player_avg_v2(stats.get('goals', 0))}/{_fmt_player_avg_v2(stats.get('assists', 0))}")

    attack = []
    if stats.get("shots_p90") is not None:
        attack.append(f"tiros {_fmt_player_avg_v2(stats.get('shots_p90'))}/90")
    if stats.get("shots_on_target_p90") is not None:
        attack.append(f"SOT {_fmt_player_avg_v2(stats.get('shots_on_target_p90'))}/90")
    if stats.get("xg_p90") is not None:
        attack.append(f"xG {_fmt_player_avg_v2(stats.get('xg_p90'))}/90")

    creation = []
    if stats.get("xa_p90") is not None:
        creation.append(f"xA {_fmt_player_avg_v2(stats.get('xa_p90'))}/90")
    if stats.get("key_passes_p90") is not None:
        creation.append(f"key {_fmt_player_avg_v2(stats.get('key_passes_p90'))}/90")

    discipline = []
    if stats.get("yellow_p90") is not None:
        discipline.append(f"YC {_fmt_player_avg_v2(stats.get('yellow_p90'))}/90")
    if stats.get("fouls_p90") is not None:
        discipline.append(f"faltas {_fmt_player_avg_v2(stats.get('fouls_p90'))}/90")
    defensive = (_stat(stats, "tackles_p90", 0) or 0) + (_stat(stats, "interceptions_p90", 0) or 0)
    if defensive:
        discipline.append(f"def {_fmt_player_avg_v2(defensive)}/90")
    if stats.get("saves_p90") is not None:
        discipline.append(f"atajadas {_fmt_player_avg_v2(stats.get('saves_p90'))}/90")

    sections = [" | ".join(details)]
    for group in (attack, creation, discipline):
        if group:
            sections.append(", ".join(group))
    return f"{base[0]}: " + " | ".join(sections)


def _fmt_player_avg_v2(value) -> str:
    num = _to_float(value)
    if num is None:
        return "-"
    if num == int(num):
        return str(int(num))
    return f"{num:.2f}"


def resumir_player_impact_v2_para_prompt(datos_resumidos: dict) -> str:
    """Resume impacto reciente por jugador desde /players/{id}/stats/ y career fallback."""
    v2 = datos_resumidos.get("_bsd_v2", {})
    impact = v2.get("player_impact", {}) if isinstance(v2, dict) else {}
    if not impact or "_error" in impact:
        return ""

    players = [p for p in impact.get("players", []) if isinstance(p.get("stats"), dict) and "_error" not in p.get("stats", {})]
    if not players:
        return ""

    available = [p for p in players if p.get("source") != "unavailable"]
    unavailable = [p for p in players if p.get("source") == "unavailable"]

    attackers = sorted(
        [
            p for p in available
            if _pos_group(p.get("position")) in {"F", "M"}
            and _has_any_stat(p, ("xg_p90", "shots_p90", "xg", "shots", "goals"))
        ],
        key=lambda p: (_stat(p.get("stats"), "xg_p90", 0) or 0, _stat(p.get("stats"), "shots_p90", 0) or 0),
        reverse=True,
    )[:6]
    creators = sorted(
        [
            p for p in available
            if _pos_group(p.get("position")) in {"M", "F"}
            and _has_any_stat(p, ("key_passes_p90", "key_passes", "assists", "xa"))
        ],
        key=lambda p: (_stat(p.get("stats"), "key_passes_p90", 0) or 0, _stat(p.get("stats"), "assists", 0) or 0),
        reverse=True,
    )[:5]
    card_risk = sorted(
        [
            p for p in available
            if _pos_group(p.get("position")) in {"D", "M"}
            and _has_any_stat(p, ("yellow_p90", "yellow_cards", "tackles", "interceptions"))
        ],
        key=lambda p: (_stat(p.get("stats"), "yellow_p90", 0) or 0, _stat(p.get("stats"), "yellow_cards", 0) or 0),
        reverse=True,
    )[:5]
    keepers = sorted(
        [
            p for p in available
            if _pos_group(p.get("position")) == "G"
            and _has_any_stat(p, ("saves_p90", "saves"))
        ],
        key=lambda p: (_stat(p.get("stats"), "saves_p90", 0) or 0),
        reverse=True,
    )[:2]

    partes = ["\n### IMPACTO POR JUGADOR (BSD v2)"]
    if impact.get("season_id"):
        partes.append(f"Base: stats recientes filtradas por temporada {impact.get('season_id')} cuando BSD las entrega; career como respaldo.")
    else:
        partes.append("Base: stats recientes BSD; career como respaldo si no hay partido-a-partido.")

    if attackers:
        partes.append("- Amenaza de gol/xG:")
        partes.extend(f"  - {_impact_line(p, 'attack')}" for p in attackers)
    if creators:
        partes.append("- Creacion:")
        partes.extend(f"  - {_impact_line(p, 'creator')}" for p in creators)
    if card_risk:
        partes.append("- Riesgo tarjetas/duelos:")
        partes.extend(f"  - {_impact_line(p, 'cards')}" for p in card_risk)
    if keepers:
        partes.append("- Porteros:")
        partes.extend(f"  - {_impact_line(p, 'gk')}" for p in keepers)
    notable_missing = [
        p for p in unavailable
        if (_stat(p.get("stats"), "minutes", 0) or 0) >= 300
        or (_stat(p.get("stats"), "xg", 0) or 0) >= 1
        or (_stat(p.get("stats"), "goals", 0) or 0) >= 1
    ][:6]
    if not any([attackers, creators, card_risk, keepers, notable_missing]):
        return ""

    if notable_missing:
        partes.append("- Bajas/dudas BSD con peso estadistico (usar solo si SofaScore no contradice):")
        partes.extend(f"  - {_impact_line(p, 'attack')}" for p in notable_missing)
    return "\n".join(partes)


def _fmt_odd(value) -> str:
    num = _to_float(value)
    return f"{num:.2f}" if num is not None else "-"


def _odds_payload(odds: dict) -> dict:
    if not isinstance(odds, dict):
        return {}
    nested = odds.get("odds")
    return nested if isinstance(nested, dict) else odds


def _compact_json(data: dict, limit: int = 650) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return text[:limit] + ("..." if len(text) > limit else "")


def resumir_odds_v2_para_prompt(v2_data: dict) -> str:
    """Resumen de mercado BSD como comparador. Betano sigue siendo oficial."""
    if not isinstance(v2_data, dict):
        return ""
    partes = []

    odds = v2_data.get("odds", {})
    if isinstance(odds, dict) and odds and "_error" not in odds:
        payload = _odds_payload(odds)
        partes.append("\n### MERCADO BSD v2 (comparador, NO reemplaza Betano)")
        partes.append(
            "- Consenso 1X2: "
            f"L {_fmt_odd(payload.get('home_win'))} / "
            f"E {_fmt_odd(payload.get('draw'))} / "
            f"V {_fmt_odd(payload.get('away_win'))}"
        )
        partes.append(
            "- Goles consenso: "
            f"O1.5 {_fmt_odd(payload.get('over_15_goals'))} / U1.5 {_fmt_odd(payload.get('under_15_goals'))}; "
            f"O2.5 {_fmt_odd(payload.get('over_25_goals'))} / U2.5 {_fmt_odd(payload.get('under_25_goals'))}; "
            f"O3.5 {_fmt_odd(payload.get('over_35_goals'))} / U3.5 {_fmt_odd(payload.get('under_35_goals'))}; "
            f"BTTS si {_fmt_odd(payload.get('btts_yes'))} / no {_fmt_odd(payload.get('btts_no'))}"
        )

    comparison = v2_data.get("odds_comparison", {})
    markets = comparison.get("markets") if isinstance(comparison, dict) else None
    if isinstance(markets, dict) and markets:
        if not partes:
            partes.append("\n### MERCADO BSD v2 (comparador, NO reemplaza Betano)")
        partes.append(f"- Comparación por bookmaker disponible para mercados: {', '.join(list(markets.keys())[:8])}")

    best = v2_data.get("best_odds_1x2", {})
    if isinstance(best, dict) and best:
        if not partes:
            partes.append("\n### MERCADO BSD v2 (comparador, NO reemplaza Betano)")
        partes.append(f"- Mejores cuotas 1X2 BSD para este evento: {_compact_json(best, 420)}")

    polymarket = v2_data.get("polymarket", {})
    if isinstance(polymarket, dict) and polymarket and "_error" not in polymarket:
        if not partes:
            partes.append("\n### MERCADO BSD v2 (comparador, NO reemplaza Betano)")
        partes.append(f"- Polymarket disponible (probabilidades 0-1): {_compact_json(polymarket, 420)}")

    if partes:
        partes.append("Usa BSD/Polymarket solo para detectar sesgo o desalineación de mercado; las picks se comparan contra Betano.")
    return "\n".join(partes)


def resumir_motivacion_v2_para_prompt(datos_resumidos: dict) -> str:
    """Contexto competitivo derivado de standings BSD + fixtures restantes."""
    motivation = ((datos_resumidos.get("_bsd_v2") or {}).get("motivation") or {})
    if not motivation:
        return ""
    local_team, away_team = _team_labels(datos_resumidos)
    home = motivation.get("home", {})
    away = motivation.get("away", {})
    cutoffs = motivation.get("cutoffs", {})
    fmt = motivation.get("format_expected_matches")
    scope = motivation.get("scope_group")
    flags = motivation.get("competition_flags", {})
    special_context = motivation.get("special_context")

    partes = ["\n### MOTOR DE MOTIVACION BSD v2"]
    if special_context:
        partes.append(f"- {special_context}")
    if flags.get("is_friendly"):
        partes.append("- Sin tabla competitiva: trata motivacion y rotacion como inciertas; baja confianza prepartido.")
        return "\n".join(partes)
    if scope is not None:
        partes.append(f"- Alcance de tabla: grupo {scope}.")
    if fmt:
        partes.append(f"- Formato estimado: {fmt} PJ por equipo.")
    for label, info in [(local_team, home), (away_team, away)]:
        if not info:
            continue
        partes.append(
            f"- {label}: #{info.get('position')} con {info.get('points')} pts, PJ {info.get('played')}, "
            f"DG {info.get('goal_difference')}, xGD {info.get('xgd')}; "
            f"restan {info.get('remaining_including_current')} incl. este partido "
            f"({info.get('remaining_after_current')} despues), max {info.get('max_points')} pts."
        )
    useful_cutoffs = {k: v for k, v in cutoffs.items() if v is not None}
    if useful_cutoffs:
        partes.append(f"- Cortes de referencia BSD: {useful_cutoffs}")
    partes.append("Interpreta urgencia con estos limites; no inventes clasificacion/descenso si el formato no queda claro.")
    return "\n".join(partes)


def resumir_prediccion_v2(prediccion: dict) -> dict:
    """Convierte una prediccion v2 al formato resumido que espera el analyzer."""
    if not prediccion or "_error" in prediccion:
        return {}

    markets = prediccion.get("markets", {})
    match_result = markets.get("match_result", {})
    expected_goals = markets.get("expected_goals", {})
    over_under = markets.get("over_under", {})
    btts = markets.get("btts", {})
    score = markets.get("score", {})
    recs = prediccion.get("recommendations", {})
    model = prediccion.get("model", {})

    return {
        "prob_local": match_result.get("prob_home"),
        "prob_empate": match_result.get("prob_draw"),
        "prob_visitante": match_result.get("prob_away"),
        "resultado_predicho": match_result.get("predicted"),
        "xG_local": expected_goals.get("home"),
        "xG_visitante": expected_goals.get("away"),
        "prob_over_15": over_under.get("prob_over_15"),
        "prob_over_25": over_under.get("prob_over_25"),
        "prob_over_35": over_under.get("prob_over_35"),
        "prob_btts": btts.get("prob_yes"),
        "marcador_probable": score.get("most_likely"),
        "confianza_modelo": model.get("confidence"),
        "version_modelo": model.get("version"),
        "recomienda_over_15": recs.get("over_15"),
        "recomienda_over_25": recs.get("over_25"),
        "recomienda_over_35": recs.get("over_35"),
        "recomienda_btts": recs.get("btts"),
        "favorite": recs.get("favorite"),
        "favorite_prob": recs.get("favorite_prob"),
        "bet_favorite": recs.get("bet_favorite"),
    }
