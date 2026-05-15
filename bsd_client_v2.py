"""
Cliente para la BSD API v2 (Bzzoiro Sports Data).

Endpoints integrados:
  /events/              → listar eventos (paginated, filtrable)
  /events/live/         → ventana en vivo
  /events/{id}/         → detalle de evento
  /events/{id}/stats/   → estadisticas (shotmap, momentum, xg_per_minute, ball-tracking)
  /events/{id}/metadata/→ facts pre-partido, AI preview
  /events/{id}/player-stats/ → stats por jugador en el partido
  /events/{id}/prediction/   → prediccion ML por evento
  /managers/{id}/        → perfil completo de entrenador
  /referees/{id}/        → estadisticas de arbitro
  /leagues/{id}/standings/ → tabla de posiciones con xG
  /teams/{id}/squad/     → plantilla del equipo
  /players/{id}/career/  → carrera agregada por temporada
  /predictions/          → predicciones ML (v2 reestructurado)

NO incluye: odds (seguimos con Betsafe), lineups (SofaScore es mas fiable),
broadcasts, social, transfers, venues.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

BSD_V2_BASE = "https://sports.bzzoiro.com/api/v2"
BSD_API_KEY = os.getenv("BSD_API_KEY")

TOP_LEAGUES = {
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
TARGET_LEAGUE_IDS = [9, 3, 5, 4, 6, 1, 7, 8, 32, 33]
LEAGUE_NAMES = {v: k for k, v in TOP_LEAGUES.items()}


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


def obtener_player_stats_evento(event_id: int) -> dict:
    """Stats por jugador en este partido (goles, xG, pases, tackles, rating...)."""
    return _get(f"events/{event_id}/player-stats/")


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


# ═══════════════════════════════════════════════════════════════════════════════
# LEAGUES
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# PLAYERS
# ═══════════════════════════════════════════════════════════════════════════════

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


def obtener_prediccion_por_evento(event_id: int) -> dict:
    """Prediccion ML para un evento especifico. 404 si no hay."""
    try:
        return _get(f"events/{event_id}/prediction/")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {}
        raise


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
        futures["player_stats"] = pool.submit(_safe, obtener_player_stats_evento, match_id)
        futures["prediction"] = pool.submit(_safe, obtener_prediccion_por_evento, match_id)

        if home_coach_id:
            futures["manager_home"] = pool.submit(_safe, obtener_manager, home_coach_id)
        if away_coach_id:
            futures["manager_away"] = pool.submit(_safe, obtener_manager, away_coach_id)
        if referee_id:
            arb_name = (datos_resumidos.get("arbitro") or {}).get("nombre", "")
            futures["referee"] = pool.submit(_safe, obtener_arbitro, referee_id, league_id, arb_name)
        if league_id:
            futures["standings"] = pool.submit(_safe, obtener_standings, league_id)
        if home_id:
            futures["squad_home"] = pool.submit(_safe, obtener_squad, home_id)
        if away_id:
            futures["squad_away"] = pool.submit(_safe, obtener_squad, away_id)

        for key, future in futures.items():
            v2_data[key] = future.result()

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
    referee = v2_data.get("referee", {})
    if referee and "_error" not in referee:
        existing_arb = datos_resumidos.get("arbitro") or {}
        datos_resumidos["arbitro"] = {
            **existing_arb,
            "id": referee.get("id") or existing_arb.get("id"),
            "nombre": referee.get("name") or existing_arb.get("nombre"),
            "nacionalidad": referee.get("country") or existing_arb.get("nacionalidad"),
            "total_yellow_cards": referee.get("total_yellow_cards"),
            "total_red_cards": referee.get("total_red_cards"),
            "avg_yellow_per_match": referee.get("avg_yellow_per_match"),
            "avg_red_per_match": referee.get("avg_red_per_match"),
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

    rows = standings.get("standings", [])
    if not rows:
        return ""

    partes = ["\n### TABLA DE POSICIONES (BSD v2) - con xG"]
    partes.append(f"{'#':>3} {'Equipo':<24} {'PJ':>3} {'Pts':>4} {'GF':>3} {'GA':>3} {'xGF':>6} {'xGA':>6} {'xGD':>6} {'Forma':>6}")
    partes.append("-" * 85)

    destacados = set()
    if home_id:
        destacados.add(home_id)
    if away_id:
        destacados.add(away_id)

    for row in rows[:20]:
        tid = row.get("team_id")
        marker = ">" if tid in destacados else " "
        partes.append(
            f"{marker}{row.get('position', '?'):>2} "
            f"{row.get('team_name', '?'):<24} "
            f"{row.get('played', 0):>3} "
            f"{row.get('pts', 0):>4} "
            f"{row.get('gf', 0):>3} "
            f"{row.get('ga', 0):>3} "
            f"{row.get('xgf', 0) or 0:>6.1f} "
            f"{row.get('xga', 0) or 0:>6.1f} "
            f"{row.get('xgd', 0) or 0:>6.1f} "
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

    partes = [f"\n### ARBITRO (BSD v2): {arb.get('nombre', '?')}"]
    if arb.get("nacionalidad"):
        partes.append(f"- Nacionalidad: {arb.get('nacionalidad')}")
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
