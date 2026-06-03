"""
Cliente para enriquecer datos de partidos via SofaScore API no-oficial.

SofaScore tiene una API JSON interna (api.sofascore.com) que requiere
bypass de TLS fingerprinting. Usamos curl_cffi con impersonacion Chrome.

Endpoints utilizados:
  - /sport/football/scheduled-events/{date}  → buscar partidos por fecha
  - /event/{id}                              → detalle completo (arbitro, entrenadores, suplentes, lesiones)
  - /event/{id}/lineups                      → alineaciones confirmadas
  - /event/{id}/statistics                   → estadisticas (xG, posesion, etc.)
  - /event/{id}/h2h                          → head-to-head historico
  - /team/{id}/statistics                    → stats de temporada + rating + top jugadores
  - /tournament/{id}/standings               → tabla de posiciones (standings)
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from utils import normalizar_nombre

load_dotenv()

SOFASCORE_API = "https://api.sofascore.com/api/v1"

# Headers minimos para no ser bloqueado
SOFASCORE_HEADERS = {
    "Origin": "https://www.sofascore.com",
    "Referer": "https://www.sofascore.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}

# Mapeo BSD league name → SofaScore tournament IDs (los mas usados en top ligas)
# No es estrictamente necesario porque buscamos por fecha+equipos,
# pero ayuda a reducir falsos positivos en la busqueda.
SOFASCORE_TOURNAMENT_IDS = {
    "Premier League": 17,
    "La Liga": 8,
    "Serie A": 23,
    "Bundesliga": 35,
    "Ligue 1": 34,
    "Brasileirão Serie A": 325,
    "Champions League": 7,
    "Europa League": 679,
    "Copa Libertadores": 586,
    "Copa Sudamericana": 587,
}

# Ligas que SOLO estan en SofaScore (BSD no las cubre)
SOFASCORE_ONLY_LEAGUES: dict[str, int] = {
}

# Mapeo para standings: BSD league name → (unique_tournament_id, season_id)
STANDINGS_MAP = {
    "La Liga":                (8,   77559),
    "Premier League":         (17,  76986),
    "Bundesliga":             (35,  77333),
    "Serie A":                (23,  76457),
    "Ligue 1":                (34,  77356),
    "Brasileirao Serie A":    (325, 87678),
    "Champions League":       (7,   77817),
    "Europa League":          (679, 78071),
    "Copa Libertadores":      (586, 80738),
    "Copa Sudamericana":      (587, 80739),
}


def _crear_sesion_sofascore():
    """
    Crea una sesion curl_cffi con impersonacion de Chrome 131.
    Necesaria para bypasear Akamai Bot Manager de SofaScore.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        raise ImportError(
            "curl_cffi no esta instalado. Instalalo con: pip install curl_cffi>=0.6"
        )

    session = cffi_requests.Session(impersonate="chrome")
    # Mantener User-Agent consistente con la version impersonada
    session.headers.update(SOFASCORE_HEADERS)
    return session


def _buscar_evento_por_fecha(
    session, fecha_str: str, home_team: str, away_team: str
) -> dict | None:
    """
    Busca un evento de SofaScore que coincida con los equipos y la fecha.

    Args:
        session: Sesion curl_cffi activa.
        fecha_str: Fecha en formato ISO (ej: "2026-04-28T21:00:00+02:00").
        home_team: Nombre del equipo local (de BSD).
        away_team: Nombre del equipo visitante (de BSD).

    Returns:
        Dict del evento SofaScore o None si no se encuentra.
    """
    fechas = _fechas_candidatas_sofascore(fecha_str)
    if not fechas:
        return None

    home_norm = normalizar_nombre(home_team)
    away_norm = normalizar_nombre(away_team)

    for fecha_solo in fechas:
        url = f"{SOFASCORE_API}/sport/football/scheduled-events/{fecha_solo}"
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
        except Exception:
            continue

        try:
            data = resp.json()
        except Exception:
            continue

        events = data.get("events", [])

        for event in events:
            e_home = normalizar_nombre(
                event.get("homeTeam", {}).get("name", "")
            )
            e_away = normalizar_nombre(
                event.get("awayTeam", {}).get("name", "")
            )

            # Coincidencia exacta o substring (equipo local)
            if home_norm and away_norm:
                if (home_norm == e_home and away_norm == e_away):
                    return event
                # Coincidencia parcial (ej: "Real Madrid" vs "Real Madrid CF")
                if (home_norm in e_home or e_home in home_norm) and \
                   (away_norm in e_away or e_away in away_norm):
                    return event

    return None


def _fechas_candidatas_sofascore(fecha_str: str) -> list[str]:
    """Devuelve fechas candidatas para SofaScore, tolerando offsets horarios.

    BSD a veces trae fecha ISO con zona europea para partidos CONMEBOL. SofaScore
    lista el calendario por la fecha real del evento; por eso probamos la fecha
    original, la fecha UTC equivalente y un día alrededor.
    """
    try:
        raw = str(fecha_str or "")
        base = raw.split("T")[0] if "T" in raw else raw[:10]
        dates = []
        if base:
            dates.append(datetime.fromisoformat(base).date())
        if "T" in raw:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            dates.append(dt.date())
            if dt.tzinfo:
                dates.append(dt.astimezone(timezone.utc).date())
    except (TypeError, ValueError):
        return []

    ordered = []
    seen = set()
    for day in dates:
        for candidate in (day, day - timedelta(days=1), day + timedelta(days=1)):
            value = candidate.isoformat()
            if value not in seen:
                ordered.append(value)
                seen.add(value)
    return ordered


def obtener_alineaciones(session, event_id: int) -> dict | None:
    """
    Obtiene las alineaciones de un partido desde SofaScore.

    Returns:
        Dict con alineaciones de ambos equipos, o None si no disponibles.
    """
    url = f"{SOFASCORE_API}/event/{event_id}/lineups"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_h2h_sofascore(session, event_id: int) -> dict | None:
    """
    Obtiene head-to-head desde SofaScore como fuente alternativa.

    Returns:
        Dict con datos H2H o None.
    """
    url = f"{SOFASCORE_API}/event/{event_id}/h2h"
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_evento_detalle(session, event_id: int) -> dict | None:
    url = f"{SOFASCORE_API}/event/{event_id}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("event") if data else None
    except Exception:
        return None


def obtener_performance_equipo(session, team_id: int) -> dict | None:
    """
    Obtiene el historial de rendimiento de un equipo desde SofaScore.
  
    Returns:
        Dict con eventos recientes o None si falla.
    """
    url = f"{SOFASCORE_API}/team/{team_id}/performance"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_shotmap(session, event_id: int) -> dict | None:
    """Obtiene el mapa de tiros (shotmap) con xG y xGOT por disparo."""
    url = f"{SOFASCORE_API}/event/{event_id}/shotmap"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_momentum(session, event_id: int) -> dict | None:
    """Obtiene el grafico de momentum (dominio minuto a minuto)."""
    url = f"{SOFASCORE_API}/event/{event_id}/graph"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_incidents(session, event_id: int) -> dict | None:
    """Obtiene el timeline de incidentes (goles, tarjetas, sustituciones)."""
    url = f"{SOFASCORE_API}/event/{event_id}/incidents"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_avg_positions(session, event_id: int) -> dict | None:
    """Obtiene posiciones promedio de los jugadores."""
    url = f"{SOFASCORE_API}/event/{event_id}/average-positions"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_team_info(session, team_id: int) -> dict | None:
    """Obtiene info del equipo (rating, manager, stadium)."""
    url = f"{SOFASCORE_API}/team/{team_id}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_team_season_stats(session, team_id: int, tournament_uid: int, season_id: int) -> dict | None:
    """Obtiene estadisticas de temporada del equipo desde SofaScore."""
    url = f"{SOFASCORE_API}/team/{team_id}/unique-tournament/{tournament_uid}/season/{season_id}/statistics/overall"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_player_profile(session, player_id: int) -> dict | None:
    """Obtiene perfil base de jugador desde SofaScore."""
    if not player_id:
        return None
    try:
        resp = session.get(f"{SOFASCORE_API}/player/{player_id}", timeout=12)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_player_recent_events(session, player_id: int) -> dict | None:
    """Obtiene partidos recientes de jugador con minutos/rating/banca."""
    if not player_id:
        return None
    try:
        resp = session.get(f"{SOFASCORE_API}/player/{player_id}/events/last/0", timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_player_season_stats(session, player_id: int, tournament_uid: int, season_id: int, split: str = "overall") -> dict | None:
    """Obtiene estadisticas del jugador para torneo/temporada actual."""
    if not player_id or not tournament_uid or not season_id:
        return None
    try:
        resp = session.get(
            f"{SOFASCORE_API}/player/{player_id}/unique-tournament/{tournament_uid}/season/{season_id}/statistics/{split}",
            timeout=12,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_player_characteristics(session, player_id: int) -> dict | None:
    """Obtiene fortalezas/debilidades codificadas y posiciones del jugador."""
    if not player_id:
        return None
    try:
        resp = session.get(f"{SOFASCORE_API}/player/{player_id}/characteristics", timeout=12)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_player_attribute_overviews(session, player_id: int) -> dict | None:
    """Obtiene radar de atributos del jugador cuando SofaScore lo ofrece."""
    if not player_id:
        return None
    try:
        resp = session.get(f"{SOFASCORE_API}/player/{player_id}/attribute-overviews", timeout=12)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_estadisticas_evento(session, event_id: int) -> dict | None:
    """Obtiene estadisticas detalladas del partido (por periodo)."""
    url = f"{SOFASCORE_API}/event/{event_id}/statistics"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def _extraer_form_string(events: list, team_id: int) -> str:
    result = []
    for ev in events:
        wc = ev.get("winnerCode", 0)
        home_id = ev.get("homeTeam", {}).get("id")
        if wc == 1:
            result.append("W" if home_id == team_id else "L")
        elif wc == 2:
            result.append("L" if home_id == team_id else "W")
        else:
            result.append("D")
    return "".join(result)


def _extraer_performance(data: dict, team_id: int) -> dict:
    events = data.get("events", [])[:10]
    if not events:
        return {}

    total_goles_favor = 0
    total_goles_contra = 0
    puntos = 0
    detalle = []

    for ev in events:
        hs = ev.get("homeScore", {}).get("current", 0) or 0
        as_ = ev.get("awayScore", {}).get("current", 0) or 0
        wc = ev.get("winnerCode", 0)
        home_id = ev.get("homeTeam", {}).get("id")
        home_name = ev.get("homeTeam", {}).get("name", "")
        away_name = ev.get("awayTeam", {}).get("name", "")

        team_home = (home_id == team_id)
        gf = hs if team_home else as_
        gc = as_ if team_home else hs
        rival = away_name if team_home else home_name
        total_goles_favor += gf
        total_goles_contra += gc

        if (team_home and wc == 1) or (not team_home and wc == 2):
            puntos += 3
        elif wc == 3:
            puntos += 1

        detalle.append({
            "rival": rival,
            "local": team_home,
            "gf": gf,
            "gc": gc,
            "event_id": ev.get("id"),
            "torneo": (ev.get("tournament") or {}).get("name", ""),
        })

    n = len(events)
    return {
        "partidos": n,
        "forma_string": _extraer_form_string(events, team_id),
        "goles_favor": total_goles_favor,
        "goles_contra": total_goles_contra,
        "promedio_goles_favor": round(total_goles_favor / n, 2),
        "promedio_goles_contra": round(total_goles_contra / n, 2),
        "puntos": puntos,
        "ppg": round(puntos / n, 2),
        "detalle": detalle,
    }


def _extraer_stats_minimas(stats_data: dict, team_id: int) -> dict | None:
    """Extrae solo las stats clave del periodo ALL para un equipo."""
    stats_list = stats_data.get("statistics", [])
    for period_stats in stats_list:
        if period_stats.get("period") != "ALL":
            continue
        result = {}
        for group in period_stats.get("groups", []):
            for item in group.get("statisticsItems", []):
                key = item.get("key", "")
                home_val = item.get("home", 0) or 0
                away_val = item.get("away", 0) or 0
                team_home = item.get("homeTeamId") == team_id if "homeTeamId" in item else None
                # Store both values
                if key in ("totalShotsOnGoal", "totalShots"):
                    result["tiros_total"] = {"home": home_val, "away": away_val}
                elif key == "shotsOnGoal":
                    result["tiros_arco"] = {"home": home_val, "away": away_val}
                elif key == "cornerKicks":
                    result["corners"] = {"home": home_val, "away": away_val}
                elif key == "yellowCards":
                    result["amarillas"] = {"home": home_val, "away": away_val}
                elif key == "redCards":
                    result["rojas"] = {"home": home_val, "away": away_val}
                elif key == "fouls":
                    result["faltas"] = {"home": home_val, "away": away_val}
        return result if result else None
    return None


def _fetch_performance_stats(session, perf_data: list, team_id: int, max_eventos: int = 5) -> list:
    """Obtiene stats minimas para los ultimos N eventos de performance."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    eventos = perf_data[:max_eventos]
    resultados = [None] * len(eventos)

    def _fetch(idx, eid):
        try:
            r = session.get(
                f"{SOFASCORE_API}/event/{eid}/statistics",
                impersonate="chrome131",
                timeout=15,
            )
            return (idx, _extraer_stats_minimas(r.json(), team_id))
        except Exception:
            return (idx, None)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_fetch, i, ev.get("event_id")): i
            for i, ev in enumerate(eventos)
            if ev.get("event_id")
        }
        for future in as_completed(futures):
            idx, stats = future.result()
            if stats:
                resultados[idx] = stats

    return resultados


# ── Nuevas funciones de extraccion para endpoints avanzados ──

def _extraer_shotmap(shotmap_data: dict) -> list:
    """Procesa el shotmap: xG y xGOT por disparo."""
    shots = []
    if not shotmap_data:
        return shots
    shotmap_list = shotmap_data.get("shotmap", [])
    for s in shotmap_list:
        shots.append({
            "player": (s.get("player") or {}).get("name", ""),
            "player_id": (s.get("player") or {}).get("id"),
            "is_home": s.get("isHome"),
            "shot_type": s.get("shotType"),
            "situation": s.get("situation"),
            "body_part": s.get("bodyPart"),
            "xg": s.get("xg"),
            "xgot": s.get("xgot"),
            "player_coordinates": {
                "x": (s.get("playerCoordinates") or {}).get("x"),
                "y": (s.get("playerCoordinates") or {}).get("y"),
            },
            "minute": s.get("time"),
            "is_blocked": s.get("isBlocked"),
            "is_on_target": s.get("isOnTarget"),
        })
    return shots


def _extraer_momentum(momentum_data: dict) -> dict:
    """Procesa el grafico de momentum."""
    if not momentum_data:
        return {}
    graph_points = momentum_data.get("graphPoints") or []
    points = [{"minute": gp.get("minute"), "value": gp.get("value")} for gp in graph_points]
    period_time = momentum_data.get("periodTime") or 45
    return {
        "period_time": period_time,
        "total_points": len(points),
        "graph": points,
        "home_dominance": round(sum(1 for p in points if (p.get("value") or 0) > 0) / max(len(points), 1) * 100, 1),
        "away_dominance": round(sum(1 for p in points if (p.get("value") or 0) < 0) / max(len(points), 1) * 100, 1),
    }


def _extraer_incidents(incidents_data: dict) -> dict:
    """Procesa timeline de incidentes."""
    if not incidents_data:
        return {"goals": [], "cards": [], "substitutions": []}
    incidents = incidents_data.get("incidents", [])
    goals = []
    cards = []
    substitutions = []

    for inc in incidents:
        inc_type = inc.get("incidentType")
        if inc_type == "goal":
            goals.append({
                "minute": inc.get("time"), "added_time": inc.get("addedTime"),
                "player": (inc.get("player") or {}).get("name", ""),
                "assist": (inc.get("assist1") or {}).get("name"),
                "home_score": inc.get("homeScore"), "away_score": inc.get("awayScore"),
                "goal_type": inc.get("incidentClass"), "is_home": inc.get("isHome"),
            })
        elif inc_type == "card":
            cards.append({
                "minute": inc.get("time"),
                "player": (inc.get("player") or {}).get("name", ""),
                "card_type": inc.get("incidentClass"),
                "is_home": inc.get("isHome"),
                "reason": (inc.get("reason") or {}).get("shortDescription", ""),
            })
        elif inc_type == "substitution":
            substitutions.append({
                "minute": inc.get("time"),
                "player_in": (inc.get("playerIn") or {}).get("name", ""),
                "player_out": (inc.get("playerOut") or {}).get("name", ""),
            })

    return {
        "goals": goals, "cards": cards, "substitutions": substitutions,
        "total_goals": len(goals), "total_cards": len(cards),
    }


def _extraer_avg_positions(avg_data: dict) -> dict:
    """Procesa posiciones promedio."""
    if not avg_data:
        return {}
    result = {}
    for side in ("home", "away"):
        side_data = avg_data.get(side, [])
        players = []
        for entry in side_data:
            p = entry.get("player", {})
            players.append({
                "player": p.get("name", ""),
                "position": p.get("position"),
                "jersey_number": p.get("jerseyNumber"),
                "average_x": entry.get("averageX"),
                "average_y": entry.get("averageY"),
                "points_count": entry.get("pointsCount"),
            })
        result[side] = players
    return result


def _extraer_players_stats(lineups_data: dict, shotmap_list: list, event_id: int = None) -> dict:
    """Procesa lineups con estadisticas detalladas por jugador (rating, tiros, pases, duelos)."""
    result = {"home": [], "away": [], "confirmed": False}
    if not lineups_data:
        return result

    result["confirmed"] = lineups_data.get("confirmed", False)

    for team_key in ("home", "away"):
        team = lineups_data.get(team_key, {})
        if not team:
            continue
        formation = team.get("formation", "")
        result[f"{team_key}_formation"] = formation

        for p_entry in team.get("players", []):
            player = p_entry.get("player", {})
            stats = p_entry.get("statistics", {})

            player_name = player.get("name", "")
            player_shots_xg = sum(
                s.get("xg", 0) or 0 for s in shotmap_list
                if (s.get("player") or {}).get("name") == player_name
            )
            player_shots_xgot = sum(
                s.get("xgot", 0) or 0 for s in shotmap_list
                if (s.get("player") or {}).get("name") == player_name
            )

            duel_won = stats.get("duelWon") or 0
            duel_lost = stats.get("duelLost") or 0
            aerial_won = stats.get("aerialWon") or 0
            aerial_lost = stats.get("aerialLost") or 0

            total_passes = stats.get("totalPass") or 0
            accurate_passes = stats.get("accuratePass") or 0

            result[team_key].append({
                "name": player_name,
                "short_name": player.get("shortName"),
                "position": p_entry.get("position") or player.get("position"),
                "jersey_number": p_entry.get("jerseyNumber") or player.get("jerseyNumber"),
                "is_substitute": p_entry.get("substitute", False),
                "captain": p_entry.get("captain", False),
                "rating": stats.get("rating"),
                "minutes_played": stats.get("minutesPlayed"),
                "xG": round(player_shots_xg, 4) if player_shots_xg else None,
                "xG_sofascore": stats.get("expectedGoals"),
                "xGOT": round(player_shots_xgot, 4) if player_shots_xgot else None,
                "xGOT_sofascore": stats.get("expectedGoalsOnTarget"),
                "expected_assists": stats.get("expectedAssists"),
                "goal_assist": stats.get("goalAssist"),
                "goals": stats.get("goals"),
                "big_chance_created": stats.get("bigChanceCreated"),
                "big_chance_missed": stats.get("bigChanceMissed"),
                "total_shots": stats.get("totalShots"),
                "shots_on_target": stats.get("onTargetScoringAttempt"),
                "shots_off_target": stats.get("shotOffTarget"),
                "blocked_shots": stats.get("blockedScoringAttempt"),
                "hit_woodwork": stats.get("hitWoodwork"),
                "accurate_passes": accurate_passes,
                "total_passes": total_passes,
                "pass_accuracy": round(accurate_passes / max(total_passes, 1) * 100, 1) if total_passes else None,
                "key_passes": stats.get("keyPass"),
                "crosses_accurate": stats.get("accurateCross"),
                "crosses_total": stats.get("totalCross"),
                "long_balls_accurate": stats.get("accurateLongBalls"),
                "long_balls_total": stats.get("totalLongBalls"),
                "duels_won": duel_won,
                "duels_lost": duel_lost,
                "duels_total": duel_won + duel_lost,
                "aerial_duels_won": aerial_won,
                "aerial_duels_lost": aerial_lost,
                "aerial_duels_total": aerial_won + aerial_lost,
                "was_fouled": stats.get("wasFouled"),
                "fouls_committed": stats.get("fouls"),
                "tackles_won": stats.get("wonTackle"),
                "tackles_total": stats.get("totalTackle"),
                "interceptions": stats.get("interceptionWon"),
                "clearances": stats.get("totalClearance"),
                "blocks": stats.get("outfielderBlock"),
                "ball_recovery": stats.get("ballRecovery"),
                "possession_lost": stats.get("possessionLostCtrl"),
                "dispossessed": stats.get("dispossessed"),
                "touches": stats.get("touches"),
                "offsides": stats.get("totalOffside"),
                "saves": stats.get("saves"),
                "goals_prevented": stats.get("goalsPrevented"),
            })

    return result


def _extraer_team_season_stats(team_stats_data: dict) -> dict:
    """Extrae estadisticas de temporada del equipo."""
    if not team_stats_data:
        return {}
    s = team_stats_data.get("statistics", {})
    if not s:
        return {}
    return {
        "goals_scored": s.get("goalsScored"),
        "goals_conceded": s.get("goalsConceded"),
        "clean_sheets": s.get("cleanSheets"),
        "avg_possession": s.get("averageBallPossession"),
        "matches_played": s.get("matchesPlayed"),
        "wins": s.get("wins"),
        "draws": s.get("draws"),
        "losses": s.get("losses"),
        "shots": s.get("shots"),
        "shots_on_target": s.get("shotsOnTarget"),
        "big_chances": s.get("bigChances"),
        "big_chances_created": s.get("bigChancesCreated"),
        "big_chances_missed": s.get("bigChancesMissed"),
        "total_passes": s.get("totalPasses"),
        "accurate_passes_pct": s.get("accuratePassesPercentage"),
        "total_crosses": s.get("totalCrosses"),
        "accurate_crosses_pct": s.get("accurateCrossesPercentage"),
        "corners": s.get("corners"),
        "duels_won_pct": s.get("duelsWonPercentage"),
        "aerial_duels_won_pct": s.get("aerialDuelsWonPercentage"),
        "successful_dribbles": s.get("successfulDribbles"),
        "dribble_attempts": s.get("dribbleAttempts"),
        "tackles": s.get("tackles"),
        "interceptions": s.get("interceptions"),
        "clearances": s.get("clearances"),
        "saves": s.get("saves"),
        "yellow_cards": s.get("yellowCards"),
        "red_cards": s.get("redCards"),
        "fouls": s.get("fouls"),
        "offsides": s.get("offsides"),
    }


def _extraer_match_statistics(estadisticas_data: dict) -> dict:
    """Extrae estadisticas del partido por periodo desde el endpoint /statistics."""
    if not estadisticas_data:
        return {}
    stats_list = estadisticas_data.get("statistics", [])
    if not stats_list:
        return {}

    stat_keys = {
        "possession": ("Match overview", "ballPossession"),
        "expected_goals": ("Match overview", "expectedGoals"),
        "big_chances": ("Match overview", "bigChanceCreated"),
        "total_shots": ("Shots", "totalShotsOnGoal"),
        "shots_on_target": ("Shots", "shotsOnGoal"),
        "shots_off_target": ("Shots", "shotsOffGoal"),
        "blocked_shots": ("Shots", "blockedScoringAttempt"),
        "shots_inside_box": ("Shots", "totalShotsInsideBox"),
        "shots_outside_box": ("Shots", "totalShotsOutsideBox"),
        "hit_woodwork": ("Shots", "hitWoodwork"),
        "corner_kicks": ("Match overview", "cornerKicks"),
        "fouls": ("Match overview", "fouls"),
        "passes": ("Match overview", "passes"),
        "accurate_passes": ("Passes", "accuratePasses"),
        "crosses": ("Passes", "accurateCross"),
        "yellow_cards": ("Match overview", "yellowCards"),
        "red_cards": ("Match overview", "redCards"),
        "offsides": ("Attack", "offsides"),
        "goalkeeper_saves": ("Match overview", "goalkeeperSaves"),
        "touches_in_penalty": ("Attack", "touchesInOppBox"),
    }

    result = {}

    for period in ["ALL", "1ST", "2ND"]:
        period_key = "match" if period == "ALL" else ("first_half" if period == "1ST" else "second_half")
        has_period = any(p.get("period") == period for p in stats_list)
        if not has_period:
            continue

        for output_key, (group, stat_key) in stat_keys.items():
            found = None
            for period_data in stats_list:
                if period_data.get("period") == period:
                    for group_data in period_data.get("groups", []):
                        if group_data.get("groupName") == group:
                            for item in group_data.get("statisticsItems", []):
                                if item.get("key") == stat_key:
                                    found = item
                                    break
            if found:
                result[f"{period_key}_{output_key}"] = {
                    "home": found.get("homeValue"), "away": found.get("awayValue"),
                    "display_home": found.get("home"), "display_away": found.get("away"),
                }

    if result.get("match_expected_goals"):
        result["xG_home"] = result["match_expected_goals"]["home"]
        result["xG_away"] = result["match_expected_goals"]["away"]
    if result.get("match_possession"):
        result["possession_home"] = result["match_possession"]["home"]
        result["possession_away"] = result["match_possession"]["away"]

    return result


def _extraer_h2h_sofascore(h2h_data: dict) -> dict:
    """
    Extrae head-to-head desde SofaScore.
    Formato nuevo (post-cambio): {"teamDuel": {homeWins, awayWins, draws}, "managerDuel": {...}}
    Formato antiguo (pre-cambio):  {"events": [...]}
    """
    # Nuevo formato
    td = h2h_data.get("teamDuel")
    if td:
        total = td.get("homeWins", 0) + td.get("awayWins", 0) + td.get("draws", 0)
        return {
            "total_partidos": total,
            "victorias_local": td.get("homeWins", 0),
            "empates": td.get("draws", 0),
            "victorias_visitante": td.get("awayWins", 0),
            "goles_local": None,
            "goles_visitante": None,
            "promedio_goles": None,
            "ultimos_enfrentamientos": [],
            "_formato_h2h": "nuevo (teamDuel resumido)",
        }

    # Formato antiguo (por si vuelve)
    events = h2h_data.get("events", [])
    if events:
        total = len(events)
        vic_local = sum(1 for e in events
                        if e.get("homeScore", {}).get("current", 0) is not None
                        and e.get("awayScore", {}).get("current", 0) is not None
                        and e["homeScore"]["current"] > e["awayScore"]["current"])
        vic_visit = sum(1 for e in events
                        if e.get("homeScore", {}).get("current", 0) is not None
                        and e.get("awayScore", {}).get("current", 0) is not None
                        and e["awayScore"]["current"] > e["homeScore"]["current"])
        empates = total - vic_local - vic_visit
        goles_local = sum(e.get("homeScore", {}).get("current", 0) or 0 for e in events)
        goles_visit = sum(e.get("awayScore", {}).get("current", 0) or 0 for e in events)

        enfrentamientos = []
        for e in events[:10]:
            home = e.get("homeTeam", {}).get("name", "?")
            away = e.get("awayTeam", {}).get("name", "?")
            hs = e.get("homeScore", {}).get("current", 0)
            as_ = e.get("awayScore", {}).get("current", 0)
            ts = e.get("startTimestamp", 0)
            fecha = ""
            if ts:
                from datetime import datetime as dt
                fecha = dt.fromtimestamp(ts).strftime("%d/%m/%Y")
            enfrentamientos.append(f"{fecha}: {home} {hs}-{as_} {away}" if fecha else f"{home} {hs}-{as_} {away}")

        return {
            "total_partidos": total,
            "victorias_local": vic_local,
            "empates": empates,
            "victorias_visitante": vic_visit,
            "goles_local": goles_local,
            "goles_visitante": goles_visit,
            "promedio_goles": round((goles_local + goles_visit) / total, 2) if total else 0,
            "ultimos_enfrentamientos": enfrentamientos,
            "_formato_h2h": "antiguo (events detallado)",
        }

    return {}


def _extraer_info_evento_detalle(evento_detalle: dict) -> dict:
    """
    Extrae arbitro y managers desde el detalle del evento.

    NOTA: SofaScore retiro injuredPlayers/suspendedPlayers de este endpoint.
    Las lesiones ahora vienen de BSD como fuente secundaria.
    Las alineaciones de SofaScore siguen siendo autoritativas para quien JUEGA.

    Returns:
        Dict con {arbitro, manager_local, manager_visitante}.
    """
    resultado = {}

    referee = evento_detalle.get("referee", {})
    if referee and referee.get("name"):
        resultado["arbitro"] = {
            "nombre": referee.get("name"),
            "nacionalidad": referee.get("slug", "").replace("-", " ").title() if referee.get("slug") else "",
            "id": referee.get("id"),
            "total_partidos": referee.get("games"),
            "yc_total": referee.get("yellowCards"),
            "rc_total": referee.get("redCards"),
            "ycrc_total": referee.get("yellowRedCards"),
        }

    home_manager = evento_detalle.get("homeTeam", {}).get("manager", {})
    if home_manager and home_manager.get("name"):
        resultado["manager_local"] = {"nombre": home_manager.get("name")}

    away_manager = evento_detalle.get("awayTeam", {}).get("manager", {})
    if away_manager and away_manager.get("name"):
        resultado["manager_visitante"] = {"nombre": away_manager.get("name")}

    return resultado


def _extraer_info_alineacion(lineups_data: dict, side: str) -> dict:
    """
    Extrae info resumida de una alineacion (home/away).

    Args:
        lineups_data: Respuesta completa del endpoint /lineups.
        side: "home" o "away".

    Returns:
        Dict con {confirmada, formacion, titulares, suplentes, ausentes_notables}.
    """
    team_data = lineups_data.get(side, {})
    if not team_data:
        return {}

    formation = team_data.get("formation", "")

    # Jugadores titulares
    players = team_data.get("players", [])
    titulares = []
    suplentes = []
    for p in players:
        player_info = p.get("player", {})
        nombre = player_info.get("name", player_info.get("shortName", "?"))
        posicion = player_info.get("position", "?")
        jersey = player_info.get("jerseyNumber", "?")
        es_suplente = p.get("substitute", False)
        entry = {
            "player_id": player_info.get("id"),
            "nombre": nombre,
            "posicion": posicion,
            "dorsal": jersey,
        }
        valor_mercado = _extraer_valor_mercado_sofascore(player_info)
        edad = _extraer_edad_sofascore(player_info)
        if valor_mercado:
            entry["valor_mercado"] = valor_mercado
        if edad:
            entry["edad"] = edad
        if es_suplente:
            suplentes.append(entry)
        else:
            titulares.append(entry)

    return {
        "confirmada": bool(lineups_data.get("confirmed", False)),
        "formacion": formation,
        "titulares": titulares,
        "suplentes": suplentes,
        "bajas": _extraer_missing_players_sofascore(team_data.get("missingPlayers", [])),
    }


def _extraer_missing_players_sofascore(missing_players: list) -> dict:
    """Extrae lesionados/suspendidos/dudas desde lineups.missingPlayers."""
    bajas = {"confirmadas": [], "dudas": []}
    if not missing_players:
        return bajas

    for item in missing_players:
        player = item.get("player", {}) or {}
        tipo = (item.get("type") or "").lower()
        desc = item.get("description") or ""
        estado = _estado_missing_player(item)
        entry = {
            "player_id": player.get("id"),
            "nombre": player.get("name") or player.get("shortName") or "?",
            "posicion": player.get("position", "?"),
            "dorsal": player.get("jerseyNumber", "?"),
            "estado": estado,
            "motivo": desc or _motivo_missing_player(item),
        }
        valor_mercado = _extraer_valor_mercado_sofascore(player)
        edad = _extraer_edad_sofascore(player)
        if valor_mercado:
            entry["valor_mercado"] = valor_mercado
        if edad:
            entry["edad"] = edad
        if item.get("expectedEndDate"):
            entry["fecha_fin_estimada"] = item["expectedEndDate"][:10]

        if tipo == "doubtful" or estado == "doubtful":
            bajas["dudas"].append(entry)
        else:
            bajas["confirmadas"].append(entry)

    return bajas


def _extraer_valor_mercado_sofascore(player: dict) -> str | None:
    """Extrae valor de mercado si SofaScore lo incluye en el objeto player."""
    if not player:
        return None

    raw = player.get("proposedMarketValueRaw") or player.get("marketValueRaw")
    currency = player.get("marketValueCurrency")
    if isinstance(raw, dict):
        currency = raw.get("currency") or currency
        formatted = _formatear_valor_mercado(raw.get("value"), currency)
        if formatted:
            return formatted

    for key in ("proposedMarketValue", "marketValue"):
        value = player.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        formatted = _formatear_valor_mercado(value, currency)
        if formatted:
            return formatted

    return None


def _formatear_valor_mercado(value, currency: str | None = None) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    suffix = f" {currency}" if currency else ""
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M{suffix}"
    if number >= 1_000:
        return f"{number / 1_000:.0f}k{suffix}"
    return f"{number:.0f}{suffix}"


def _extraer_edad_sofascore(player: dict) -> int | None:
    dob_ts = player.get("dateOfBirthTimestamp") if player else None
    if not dob_ts:
        return None
    try:
        born = datetime.fromtimestamp(int(dob_ts), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None
    today = datetime.now(timezone.utc)
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _to_float_or_none(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _to_int_or_zero(value) -> int:
    number = _to_float_or_none(value)
    return int(number) if number is not None else 0


def _extraer_valor_mercado_num_sofascore(player: dict) -> float | None:
    if not player:
        return None
    raw = player.get("proposedMarketValueRaw") or player.get("marketValueRaw")
    if isinstance(raw, dict):
        value = _to_float_or_none(raw.get("value"))
        if value is not None:
            return value
    for key in ("proposedMarketValue", "marketValue"):
        value = _to_float_or_none(player.get(key))
        if value is not None:
            return value
    return None


def _formatear_ratio_90(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.2f}/90"


def _estado_missing_player(item: dict) -> str:
    tipo = (item.get("type") or "").lower()
    desc = (item.get("description") or "").lower()
    reason = item.get("reason")
    if tipo == "doubtful":
        return "doubtful"
    if reason == 12 or "suspension" in desc or "red_card" in desc or "yellow_or_red" in desc:
        return "suspended"
    if reason == 1 or "injury" in desc or "injured" in desc:
        return "injured"
    return tipo or "missing"


def _motivo_missing_player(item: dict) -> str:
    estado = _estado_missing_player(item)
    if estado == "suspended":
        return "Suspension"
    if estado in {"injured", "doubtful"}:
        return "Injury"
    return "No disponible"


def _enriquecer_impacto_bajas_sofascore(session, alineaciones: dict, tournament_uid: int | None, season_id: int | None) -> None:
    """Agrega impacto estimado a las bajas de SofaScore in-place."""
    refs = []
    for side in ("local", "visitante"):
        bajas = ((alineaciones.get(side) or {}).get("bajas") or {})
        for bucket in ("confirmadas", "dudas"):
            for baja in bajas.get(bucket, []) or []:
                if baja.get("player_id"):
                    refs.append(baja)

    if not refs:
        return

    def _fetch(baja: dict) -> tuple[dict, dict | None]:
        return baja, _calcular_impacto_baja_sofascore(session, baja, tournament_uid, season_id)

    max_workers = min(5, max(1, len(refs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, baja) for baja in refs[:18]]
        for future in as_completed(futures):
            try:
                baja, impacto = future.result()
            except Exception:
                continue
            if impacto:
                baja["impacto_baja"] = impacto


def _enriquecer_impacto_xi_sofascore(session, alineaciones: dict, tournament_uid: int | None, season_id: int | None) -> None:
    """Agrega impacto estimado a titulares probables/confirmados de SofaScore in-place."""
    refs = []
    for side in ("local", "visitante"):
        for jugador in (alineaciones.get(side) or {}).get("titulares", []) or []:
            if jugador.get("player_id"):
                refs.append(jugador)

    if not refs:
        return

    def _fetch(jugador: dict) -> tuple[dict, dict | None]:
        return jugador, _calcular_impacto_baja_sofascore(
            session,
            jugador,
            tournament_uid,
            season_id,
            include_traits=False,
        )

    # Solo titulares: maximo 22 perfiles. Threads moderados para no castigar la API.
    max_workers = min(6, max(1, len(refs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch, jugador) for jugador in refs[:22]]
        for future in as_completed(futures):
            try:
                jugador, impacto = future.result()
            except Exception:
                continue
            if impacto:
                jugador["impacto_jugador"] = impacto


def _calcular_impacto_baja_sofascore(
    session,
    baja: dict,
    tournament_uid: int | None,
    season_id: int | None,
    include_traits: bool = True,
) -> dict | None:
    player_id = baja.get("player_id")
    if not player_id:
        return None

    profile_data = obtener_player_profile(session, player_id) or {}
    profile = profile_data.get("player", {}) if isinstance(profile_data, dict) else {}
    if profile:
        if not baja.get("valor_mercado"):
            valor = _extraer_valor_mercado_sofascore(profile)
            if valor:
                baja["valor_mercado"] = valor
        if not baja.get("edad"):
            edad = _extraer_edad_sofascore(profile)
            if edad:
                baja["edad"] = edad
        if not baja.get("posicion") or baja.get("posicion") == "?":
            baja["posicion"] = profile.get("position", "?")
        if not baja.get("dorsal") or baja.get("dorsal") == "?":
            baja["dorsal"] = profile.get("jerseyNumber") or profile.get("shirtNumber") or "?"

    events_data = obtener_player_recent_events(session, player_id) or {}
    current_stats_data = obtener_player_season_stats(session, player_id, tournament_uid, season_id) or {}
    characteristics = obtener_player_characteristics(session, player_id) or {} if include_traits else {}
    attrs = obtener_player_attribute_overviews(session, player_id) or {} if include_traits else {}

    recent = _resumir_recent_events_jugador(events_data)
    stats = (current_stats_data.get("statistics") or {}) if isinstance(current_stats_data, dict) else {}
    return _score_impacto_baja_sofascore(baja, profile, recent, stats, characteristics, attrs)


def _resumir_recent_events_jugador(events_data: dict, max_events: int = 10) -> dict:
    events = events_data.get("events", []) if isinstance(events_data, dict) else []
    stats_map = events_data.get("statisticsMap", {}) if isinstance(events_data, dict) else {}
    bench_map = events_data.get("onBenchMap", {}) if isinstance(events_data, dict) else {}
    incidents_map = events_data.get("incidentsMap", {}) if isinstance(events_data, dict) else {}

    rows = []
    for event in events:
        event_id = str(event.get("id"))
        status = event.get("status", {}) or {}
        status_type = status.get("type")
        status_desc = status.get("description")
        stat = stats_map.get(event_id) or {}
        on_bench = bool(bench_map.get(event_id))
        if status_type != "finished" and status_desc not in {"Ended", "AP", "AET"} and not stat and not on_bench:
            continue
        rows.append({
            "timestamp": event.get("startTimestamp") or 0,
            "minutes": _to_int_or_zero(stat.get("minutesPlayed")),
            "rating": _to_float_or_none(stat.get("rating")),
            "on_bench": on_bench,
            "incidents": incidents_map.get(event_id) or {},
        })

    rows.sort(key=lambda row: row.get("timestamp") or 0, reverse=True)
    rows = rows[:max_events]
    appearances = [r for r in rows if (r.get("minutes") or 0) > 0]
    ratings = [r["rating"] for r in rows if r.get("rating") is not None]
    starts_like = [r for r in appearances if (r.get("minutes") or 0) >= 60]
    bench_only = [r for r in rows if r.get("on_bench") and not r.get("minutes")]
    goals = sum(_to_int_or_zero((r.get("incidents") or {}).get("goals")) for r in rows)
    assists = sum(_to_int_or_zero((r.get("incidents") or {}).get("assists")) for r in rows)
    yellow_cards = sum(_to_int_or_zero((r.get("incidents") or {}).get("yellowCards")) for r in rows)
    minutes_total = sum(r.get("minutes") or 0 for r in rows)

    return {
        "sample": len(rows),
        "appearances": len(appearances),
        "starts_like": len(starts_like),
        "bench_only": len(bench_only),
        "minutes_total": minutes_total,
        "avg_minutes_all": round(minutes_total / len(rows), 1) if rows else None,
        "avg_minutes_apps": round(minutes_total / len(appearances), 1) if appearances else None,
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "goals": goals,
        "assists": assists,
        "yellow_cards": yellow_cards,
    }


def _score_impacto_baja_sofascore(
    baja: dict,
    profile: dict,
    recent: dict,
    stats: dict,
    characteristics: dict,
    attrs: dict,
) -> dict:
    score = 0.0
    razones = []
    posicion = (baja.get("posicion") or profile.get("position") or "").upper()
    market_value = _extraer_valor_mercado_num_sofascore(profile)

    recent_sample = recent.get("sample") or 0
    recent_apps = recent.get("appearances") or 0
    recent_starts = recent.get("starts_like") or 0
    recent_bench = recent.get("bench_only") or 0
    recent_avg_minutes = recent.get("avg_minutes_apps")
    recent_rating = recent.get("avg_rating")

    if recent_apps >= 8:
        score += 2.2
    elif recent_apps >= 5:
        score += 1.5
    elif recent_apps >= 3:
        score += 0.8
    elif recent_apps >= 1:
        score += 0.3

    if recent_avg_minutes is not None:
        if recent_avg_minutes >= 75:
            score += 1.5
        elif recent_avg_minutes >= 45:
            score += 1.0
        elif recent_avg_minutes >= 20:
            score += 0.4

    if recent_starts >= 5:
        score += 1.2
    elif recent_starts >= 2:
        score += 0.6

    if recent_sample >= 4 and recent_apps == 0 and recent_bench >= 4:
        score -= 1.4
        razones.append(f"suplente habitual: 0/{recent_sample} apariciones recientes")
    elif recent_sample:
        razones.append(f"uso reciente: {recent_apps}/{recent_sample} PJ, {recent_avg_minutes or 0} min/prom")

    minutes = _to_int_or_zero(stats.get("minutesPlayed"))
    count_rating = _to_int_or_zero(stats.get("countRating"))
    rating = _to_float_or_none(stats.get("rating"))
    if minutes >= 900:
        score += 2.5
    elif minutes >= 600:
        score += 1.8
    elif minutes >= 300:
        score += 1.1
    elif minutes >= 120:
        score += 0.6

    if count_rating >= 10:
        score += 1.0
    elif count_rating >= 5:
        score += 0.6
    elif count_rating >= 2:
        score += 0.3

    if minutes:
        razones.append(f"temporada actual: {minutes} min, {count_rating} PJ rating")

    effective_rating = rating if rating is not None else recent_rating
    if effective_rating is not None:
        if effective_rating >= 7.1:
            score += 0.8
        elif effective_rating >= 6.8:
            score += 0.4
        elif effective_rating < 6.35:
            score -= 0.3
        razones.append(f"rating {effective_rating:.2f}")

    if market_value is not None:
        if market_value >= 15_000_000:
            score += 1.2
        elif market_value >= 5_000_000:
            score += 0.8
        elif market_value >= 1_000_000:
            score += 0.4
        raw_market = profile.get("proposedMarketValueRaw") or profile.get("marketValueRaw") or {}
        currency = raw_market.get("currency") if isinstance(raw_market, dict) else profile.get("marketValueCurrency")
        razones.append(f"valor {_formatear_valor_mercado(market_value, currency)}")

    minutes_base = max(minutes, recent.get("minutes_total") or 0)

    def per90(value):
        number = _to_float_or_none(value)
        return number * 90 / minutes_base if number is not None and minutes_base else None
    goals = _to_int_or_zero(stats.get("goals"))
    assists = _to_int_or_zero(stats.get("assists") or stats.get("goalAssist"))
    shots_total = _to_float_or_none(stats.get("totalShots"))
    sot_total = _to_float_or_none(stats.get("shotsOnTarget") or stats.get("onTargetScoringAttempt"))
    xg_total = _to_float_or_none(stats.get("expectedGoals"))
    xa_total = _to_float_or_none(stats.get("expectedAssists"))
    key_total = _to_float_or_none(stats.get("keyPasses") or stats.get("keyPass"))
    crosses_total = _to_float_or_none(stats.get("accurateCrosses") or stats.get("accurateCross"))
    fouls_total = _to_float_or_none(stats.get("fouls"))
    yellow_total = _to_float_or_none(stats.get("yellowCards") or stats.get("yellowCard"))
    tackles_total = _to_float_or_none(stats.get("tackles") or stats.get("totalTackle"))
    interceptions_total = _to_float_or_none(stats.get("interceptions") or stats.get("interceptionWon"))
    saves_total = _to_float_or_none(stats.get("saves"))

    shots90 = per90(stats.get("totalShots"))
    sot90 = per90(stats.get("shotsOnTarget") or stats.get("onTargetScoringAttempt"))
    xg90 = per90(stats.get("expectedGoals"))
    xa90 = per90(stats.get("expectedAssists"))
    key90 = per90(stats.get("keyPasses") or stats.get("keyPass"))
    crosses90 = per90(stats.get("accurateCrosses") or stats.get("accurateCross"))
    defensive90 = per90((_to_float_or_none(stats.get("tackles")) or 0) + (_to_float_or_none(stats.get("interceptions")) or 0))
    saves90 = per90(stats.get("saves"))

    if posicion in {"F", "FW"}:
        if (xg90 is not None and xg90 >= 0.25) or (shots90 is not None and shots90 >= 2.0):
            score += 1.0
            razones.append(f"amenaza ofensiva {(_formatear_ratio_90(xg90) or _formatear_ratio_90(shots90))}")
        if goals + assists >= 3:
            score += 0.8
            razones.append(f"G+A {goals + assists}")
    elif posicion in {"M", "AM", "DM"}:
        if (key90 is not None and key90 >= 1.2) or (xa90 is not None and xa90 >= 0.12):
            score += 1.0
            razones.append(f"creacion {(_formatear_ratio_90(key90) or _formatear_ratio_90(xa90))}")
    elif posicion == "D":
        if defensive90 is not None and defensive90 >= 2.0:
            score += 0.6
            razones.append(f"duelo/defensa {_formatear_ratio_90(defensive90)}")
        if crosses90 is not None and crosses90 >= 0.8:
            score += 0.4
            razones.append(f"centros {_formatear_ratio_90(crosses90)}")
    elif posicion == "G":
        if minutes >= 450 or (recent_starts >= 5 and recent_avg_minutes and recent_avg_minutes >= 80):
            score += 1.2
            razones.append("portero de uso alto")
        if saves90 is not None and saves90 >= 2.5:
            score += 0.4
            razones.append(f"atajadas {_formatear_ratio_90(saves90)}")

    attr_reason = _resumir_attribute_edge(attrs)
    if attr_reason:
        score += 0.3
        razones.append(attr_reason)

    char_positions = characteristics.get("positions") if isinstance(characteristics, dict) else None
    if char_positions and not posicion:
        razones.append(f"rol {','.join(char_positions[:2])}")

    if recent_sample == 0 and minutes == 0:
        nivel = "DESCONOCIDA"
        if market_value and market_value >= 5_000_000:
            razones.append("valor alto, pero sin evidencia reciente de minutos")
        elif not razones:
            razones.append("sin minutos/stats recientes suficientes")
    elif score >= 5.0:
        nivel = "ALTA"
    elif score >= 3.0:
        nivel = "MEDIA"
    else:
        nivel = "BAJA"

    # Mantener el prompt compacto y accionable.
    razones = [r for r in razones if r][:4]
    has_stats_context = bool(
        minutes
        or count_rating
        or recent_sample
        or rating is not None
        or recent_rating is not None
        or any(
            value is not None
            for value in (
                shots_total,
                sot_total,
                xg_total,
                xa_total,
                key_total,
                crosses_total,
                fouls_total,
                yellow_total,
                tackles_total,
                interceptions_total,
                saves_total,
            )
        )
    )
    stats_resumen = {
        "sample_recent": recent_sample,
        "apps_recent": recent_apps,
        "avg_minutes_recent": recent_avg_minutes,
        "apps": count_rating or recent_apps or None,
        "minutes": minutes or recent.get("minutes_total") or None,
        "avg_rating": rating if rating is not None else recent_rating,
        "goals": goals,
        "assists": assists,
        "shots": _round_or_none(shots_total),
        "shots_p90": _round_or_none(shots90),
        "shots_on_target": _round_or_none(sot_total),
        "shots_on_target_p90": _round_or_none(sot90),
        "xg": _round_or_none(xg_total),
        "xg_p90": _round_or_none(xg90),
        "xa": _round_or_none(xa_total),
        "xa_p90": _round_or_none(xa90),
        "key_passes": _round_or_none(key_total),
        "key_passes_p90": _round_or_none(key90),
        "accurate_crosses": _round_or_none(crosses_total),
        "accurate_crosses_p90": _round_or_none(crosses90),
        "fouls": _round_or_none(fouls_total),
        "fouls_p90": _round_or_none(per90(fouls_total)),
        "yellow_cards": _round_or_none(yellow_total),
        "yellow_cards_p90": _round_or_none(per90(yellow_total)),
        "tackles": _round_or_none(tackles_total),
        "interceptions": _round_or_none(interceptions_total),
        "defensive_actions_p90": _round_or_none(defensive90),
        "saves": _round_or_none(saves_total),
        "saves_p90": _round_or_none(saves90),
    }
    stats_resumen = {k: v for k, v in stats_resumen.items() if v is not None}
    if has_stats_context and stats_resumen:
        stats_resumen["source"] = "SofaScore temporada actual"
    else:
        stats_resumen = {}
    return {
        "nivel": nivel,
        "score": round(score, 1),
        "razones": razones,
        "stats_resumen": stats_resumen,
    }


def _round_or_none(value, digits: int = 2):
    number = _to_float_or_none(value)
    return round(number, digits) if number is not None else None


def _resumir_attribute_edge(attrs: dict) -> str | None:
    if not isinstance(attrs, dict):
        return None
    player_attrs = (attrs.get("playerAttributeOverviews") or [])
    avg_attrs = (attrs.get("averageAttributeOverviews") or [])
    if not player_attrs or not avg_attrs:
        return None
    player = player_attrs[0] or {}
    avg = avg_attrs[0] or {}
    diffs = []
    for key, label in [
        ("attacking", "ATT"),
        ("technical", "TEC"),
        ("tactical", "TAC"),
        ("defending", "DEF"),
        ("creativity", "CRE"),
    ]:
        p_val = _to_float_or_none(player.get(key))
        a_val = _to_float_or_none(avg.get(key))
        if p_val is not None and a_val is not None:
            diffs.append((p_val - a_val, label))
    if not diffs:
        return None
    best_diff, label = max(diffs, key=lambda x: abs(x[0]))
    if abs(best_diff) < 6:
        return None
    sign = "+" if best_diff > 0 else ""
    return f"atributo {label} {sign}{best_diff:.0f} vs media pos."


def enriquecer_datos_partido(datos_bsd: dict) -> dict:
    """
    Toma los datos resumidos de BSD e intenta enriquecerlos con SofaScore.

    Busca el partido por fecha + equipos, obtiene alineaciones (si disponibles),
    estadisticas de equipo, standings, H2H, top jugadores, rating, detalle del partido.

    Args:
        datos_bsd: Diccionario de datos resumidos desde bsd_client.resumir_datos_partido().

    Returns:
        El mismo dict con campos adicionales bajo la clave "_sofascore".
        Si falla, devuelve los datos originales sin modificar con
        _sofascore: {"disponible": False, "error": "..."}.
    """
    home_team = datos_bsd.get("partido", "").split(" vs ")[0] if " vs " in datos_bsd.get("partido", "") else ""
    away_team = datos_bsd.get("partido", "").split(" vs ")[1] if " vs " in datos_bsd.get("partido", "") else ""
    fecha = datos_bsd.get("fecha", "")

    if not home_team or not away_team:
        datos_bsd["_sofascore"] = {"disponible": False, "error": "Faltan nombres de equipos"}
        return datos_bsd

    try:
        session = _crear_sesion_sofascore()
    except ImportError as e:
        datos_bsd["_sofascore"] = {"disponible": False, "error": str(e)}
        return datos_bsd

    evento = _buscar_evento_por_fecha(session, fecha, home_team, away_team)

    if not evento:
        datos_bsd["_sofascore"] = {"disponible": False, "error": "Partido no encontrado en SofaScore"}
        return datos_bsd

    event_id = evento.get("id")
    home_team_id = evento.get("homeTeam", {}).get("id")
    away_team_id = evento.get("awayTeam", {}).get("id")
    tournament_id = evento.get("tournament", {}).get("id")

    enriquecido = {
        "disponible": True,
        "event_id": event_id,
        "torneo": evento.get("tournament", {}).get("name", ""),
        "status": evento.get("status", {}).get("type", ""),
        "form_performance": False,
    }

    # Determinar tournament_uid y season_id para team stats
    tournament_uid = evento.get("tournament", {}).get("uniqueTournament", {}).get("id")
    season_id = evento.get("season", {}).get("id")

    futures = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures[pool.submit(obtener_evento_detalle, session, event_id)] = "detalle"
        futures[pool.submit(obtener_alineaciones, session, event_id)] = "lineups"
        futures[pool.submit(obtener_h2h_sofascore, session, event_id)] = "h2h"
        if home_team_id:
            futures[pool.submit(obtener_performance_equipo, session, home_team_id)] = "perf_local"
        if away_team_id:
            futures[pool.submit(obtener_performance_equipo, session, away_team_id)] = "perf_visitante"
        # Standings
        liga = datos_bsd.get("liga", "")
        if liga:
            team_ids = set()
            if home_team_id:
                team_ids.add(home_team_id)
            if away_team_id:
                team_ids.add(away_team_id)
            futures[pool.submit(_obtener_standings, session, liga, team_ids, tournament_uid, season_id)] = "standings"
        # Nuevos endpoints avanzados
        futures[pool.submit(obtener_shotmap, session, event_id)] = "shotmap"
        futures[pool.submit(obtener_momentum, session, event_id)] = "momentum"
        futures[pool.submit(obtener_incidents, session, event_id)] = "incidents"
        futures[pool.submit(obtener_avg_positions, session, event_id)] = "avg_positions"
        futures[pool.submit(obtener_estadisticas_evento, session, event_id)] = "estadisticas"
        # Team info y season stats
        if home_team_id:
            futures[pool.submit(obtener_team_info, session, home_team_id)] = "team_info_local"
        if away_team_id:
            futures[pool.submit(obtener_team_info, session, away_team_id)] = "team_info_visitante"
        if home_team_id and tournament_uid and season_id:
            futures[pool.submit(obtener_team_season_stats, session, home_team_id, tournament_uid, season_id)] = "team_stats_local"
        if away_team_id and tournament_uid and season_id:
            futures[pool.submit(obtener_team_season_stats, session, away_team_id, tournament_uid, season_id)] = "team_stats_visitante"

        shotmap_raw = None

        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception:
                continue

            if result is None:
                continue

            if key == "detalle":
                info_detalle = _extraer_info_evento_detalle(result)
                if info_detalle:
                    enriquecido["detalle_evento"] = info_detalle
            elif key == "lineups":
                enriquecido["alineaciones"] = {
                    "local": _extraer_info_alineacion(result, "home"),
                    "visitante": _extraer_info_alineacion(result, "away"),
                }
                enriquecido["_lineups_raw"] = result
            elif key == "h2h":
                h2h_info = _extraer_h2h_sofascore(result)
                if h2h_info:
                    enriquecido["h2h"] = h2h_info
            elif key == "perf_local":
                enriquecido["form_performance_local"] = _extraer_performance(result, home_team_id)
                enriquecido["form_performance"] = True
            elif key == "perf_visitante":
                enriquecido["form_performance_visitante"] = _extraer_performance(result, away_team_id)
                enriquecido["form_performance"] = True
            elif key == "standings":
                enriquecido["standings"] = result
            elif key == "shotmap":
                shotmap_raw = result
                enriquecido["shotmap"] = _extraer_shotmap(result)
            elif key == "momentum":
                enriquecido["momentum"] = _extraer_momentum(result)
            elif key == "incidents":
                enriquecido["incidents"] = _extraer_incidents(result)
            elif key == "avg_positions":
                enriquecido["avg_positions"] = _extraer_avg_positions(result)
            elif key == "estadisticas":
                enriquecido["estadisticas_partido"] = _extraer_match_statistics(result)
            elif key == "team_info_local":
                t = result.get("team", {})
                enriquecido["team_info_local"] = {
                    "name": t.get("name"), "country": (t.get("country") or {}).get("name"),
                    "manager": (t.get("manager") or {}).get("name"),
                    "venue": (t.get("venue") or {}).get("name"),
                }
            elif key == "team_info_visitante":
                t = result.get("team", {})
                enriquecido["team_info_visitante"] = {
                    "name": t.get("name"), "country": (t.get("country") or {}).get("name"),
                    "manager": (t.get("manager") or {}).get("name"),
                    "venue": (t.get("venue") or {}).get("name"),
                }
            elif key == "team_stats_local":
                enriquecido["team_stats_local"] = _extraer_team_season_stats(result)
            elif key == "team_stats_visitante":
                enriquecido["team_stats_visitante"] = _extraer_team_season_stats(result)

        # Extraer players stats combinando lineups + shotmap
        if enriquecido.get("_lineups_raw") and (shotmap_raw or enriquecido.get("shotmap")):
            shotmap_list = shotmap_raw.get("shotmap", []) if shotmap_raw else []
            enriquecido["players_stats"] = _extraer_players_stats(
                enriquecido["_lineups_raw"], shotmap_list, event_id
            )

        # Obtener stats por partido para los eventos de performance
        for pkey, tid in [("form_performance_local", home_team_id), ("form_performance_visitante", away_team_id)]:
            perf = enriquecido.get(pkey, {})
            detalle = perf.get("detalle", [])
            if detalle and tid:
                try:
                    stats_per_match = _fetch_performance_stats(session, detalle, tid, max_eventos=10)
                    if any(s is not None for s in stats_per_match):
                        enriquecido[f"{pkey}_stats_per_match"] = stats_per_match
                except Exception:
                    pass

        if enriquecido.get("alineaciones"):
            try:
                _enriquecer_impacto_bajas_sofascore(
                    session,
                    enriquecido["alineaciones"],
                    tournament_uid,
                    season_id,
                )
            except Exception:
                pass
            try:
                _enriquecer_impacto_xi_sofascore(
                    session,
                    enriquecido["alineaciones"],
                    tournament_uid,
                    season_id,
                )
            except Exception:
                pass

    # Enriquecer arbitro desde API de SofaScore (stats por torneo)
    detalle_ev = enriquecido.get("detalle_evento", {})
    arb_ss = detalle_ev.get("arbitro", {}) if isinstance(detalle_ev, dict) else {}
    arb_id = arb_ss.get("id")
    if arb_id:
        try:
            arb_stats = obtener_datos_arbitro_sofascore(session, arb_id)
            if arb_stats:
                enriquecido["arbitro"] = arb_stats
                # Merge en datos_bsd para que el analyzer lo use
                existing = datos_bsd.get("arbitro") or {}
                torneo_actual = enriquecido.get("torneo", "")
                ss_yc = arb_stats.get("yc_pp")
                ss_rc = arb_stats.get("rc_pp")
                torneo_ref = _buscar_torneo_arbitro_actual(arb_stats, torneo_actual)
                # Buscar YC/part especifica del torneo actual
                if torneo_ref:
                    ss_yc = torneo_ref.get("yc_pp", ss_yc)
                    ss_rc = torneo_ref.get("rc_pp", ss_rc)
                datos_bsd["arbitro"] = {
                    **existing,
                    "nombre": arb_stats.get("nombre") or existing.get("nombre"),
                    "nacionalidad": arb_stats.get("pais") or existing.get("nacionalidad"),
                    "avg_yellow_per_match": ss_yc or existing.get("avg_yellow_per_match"),
                    "avg_red_per_match": ss_rc or existing.get("avg_red_per_match"),
                    "total_yellow_cards": arb_stats.get("yc_total") or existing.get("total_yellow_cards"),
                    "total_red_cards": arb_stats.get("rc_total") or existing.get("total_red_cards"),
                    "_fuente_yc": "SofaScore",
                    "_sofascore_ref": arb_stats,
                    "_sofascore_competicion": torneo_ref,
                }
        except Exception:
            pass

    datos_bsd["_sofascore"] = enriquecido
    return datos_bsd


def _formatear_form_performance_para_prompt(datos: dict) -> str:
    """
    Formatea los datos de performance (form reciente) de SofaScore para el prompt.
    Incluye desglose por partido para que la IA evalue consistencia (varianza).
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible") or not sofas.get("form_performance"):
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### FORMA RECIENTE (SofaScore Performance)"]

    def _num_stat(value):
        try:
            parsed = float(str(value).replace(",", "."))
            return int(parsed) if parsed.is_integer() else parsed
        except (TypeError, ValueError):
            return 0

    # Detectar el torneo actual del partido que se analiza
    torneo_actual = sofas.get("torneo", "")
    for equipo, key in [(local_team, "form_performance_local"), (away_team, "form_performance_visitante")]:
        perf = sofas.get(key, {})
        if not perf:
            continue
        partes.append(
            f"\n**{equipo}**: {perf.get('forma_string', '?')} | "
            f"PTS: {perf.get('puntos', '?')}/{perf.get('partidos', 0)*3} ({perf.get('ppg', '?')} ppg) | "
            f"GF: {perf.get('goles_favor', '?')} | GC: {perf.get('goles_contra', '?')} | "
            f"Prom GF: {perf.get('promedio_goles_favor', '?')} | Prom GC: {perf.get('promedio_goles_contra', '?')}"
        )
        detalle = perf.get("detalle", [])
        stats_list = sofas.get(f"{key}_stats_per_match", [])
        if detalle:
            # Agrupar por torneo
            torneos = {}
            for m in detalle[:10]:
                tn = m.get("torneo", "Otros")
                torneos.setdefault(tn, []).append(m)

            # Orden: primero el torneo actual, despues los demas
            orden = []
            if torneo_actual and torneo_actual in torneos:
                orden.append(torneo_actual)
            for tn in torneos:
                if tn != torneo_actual:
                    orden.append(tn)

            for tn in orden[:3]:
                matches = torneos[tn]
                es_copa = any(w in tn.lower() for w in ["copa", "libertadores", "champions", "sudamericana", "concacaf", "europa", "uefa"])
                label = "Copa/Torneo" if es_copa else "Liga"
                # Para copa/torneo: mostrar TODOS los partidos de esta temporada
                # Para liga: ampliar muestra sin volver enorme el prompt
                if not es_copa:
                    matches = matches[:8]
                partes.append(f"  Ultimos en {label} ({tn}):")
                for m in matches:
                    idx = next((i for i, d in enumerate(detalle) if d.get("event_id") == m.get("event_id")), -1)
                    loc = "CASA" if m.get("local") else "FUERA"
                    linea = f"    vs {m.get('rival', '?')} ({loc}): {m.get('gf', 0)}-{m.get('gc', 0)}"
                    st = stats_list[idx] if 0 <= idx < len(stats_list) and stats_list[idx] else None
                    if st:
                        th = _num_stat(st.get("tiros_total", {}).get("home", 0))
                        ta = _num_stat(st.get("tiros_total", {}).get("away", 0))
                        ah = _num_stat(st.get("tiros_arco", {}).get("home", 0))
                        aa = _num_stat(st.get("tiros_arco", {}).get("away", 0))
                        yh = _num_stat(st.get("amarillas", {}).get("home", 0))
                        ya = _num_stat(st.get("amarillas", {}).get("away", 0))
                        ch = _num_stat(st.get("corners", {}).get("home", 0))
                        ca = _num_stat(st.get("corners", {}).get("away", 0))
                        fh = _num_stat(st.get("faltas", {}).get("home", 0))
                        fa = _num_stat(st.get("faltas", {}).get("away", 0))
                        extras = []
                        if th or ta:
                            extras.append(f"Tiros: {th}-{ta}")
                        if ah or aa:
                            extras.append(f"Arco: {ah}-{aa}")
                        if yh or ya:
                            extras.append(f"YC: {yh}-{ya}")
                        if ch or ca:
                            extras.append(f"Corners: {ch}-{ca}")
                        if fh or fa:
                            extras.append(f"Faltas: {fh}-{fa} (Tot {fh + fa})")
                        if extras:
                            linea += " | " + " | ".join(extras)
                    partes.append(linea)
            partes.append("  NOTA: Evaluar si los resultados fueron consistentes o si hubo outliers.")
    return "\n".join(partes)


def _formatear_h2h_sofascore_para_prompt(datos: dict) -> str:
    """
    Formatea H2H de SofaScore para el prompt.
    Soporta formato nuevo (teamDuel resumido) y antiguo (events detallado).
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible"):
        return ""
    h2h = sofas.get("h2h", {})
    if not h2h:
        return ""

    partes = ["\n### H2H SOFASCORE"]
    partes.append(f"- Total partidos: {h2h.get('total_partidos', '?')} | "
                  f"Local: {h2h.get('victorias_local', 0)}V | "
                  f"Empates: {h2h.get('empates', 0)} | "
                  f"Visitante: {h2h.get('victorias_visitante', 0)}V")

    if h2h.get("goles_local") is not None and h2h.get("goles_visitante") is not None:
        partes.append(f"- Goles: Local {h2h.get('goles_local', '?')} - Visitante {h2h.get('goles_visitante', '?')} | "
                      f"Promedio: {h2h.get('promedio_goles', '?')}")
    else:
        partes.append("- NOTA: H2H en formato resumido (sin detalle de goles ni fechas). Usa los totales W/D/L como referencia.")

    ultimos = h2h.get("ultimos_enfrentamientos", [])
    if ultimos:
        partes.append("\nUltimos enfrentamientos:")
        for u in ultimos[:10]:
            partes.append(f"  {u}")

    return "\n".join(partes)


def _formatear_detalle_evento_para_prompt(datos: dict) -> str:
    """
    Formatea arbitro, managers, lesiones de SofaScore para el prompt.
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible"):
        return ""
    detalle = sofas.get("detalle_evento", {})
    if not detalle:
        return ""

    partes = ["\n### DETALLE DEL PARTIDO (SofaScore)"]

    if detalle.get("arbitro"):
        a = detalle["arbitro"]
        partes.append(f"\n**Árbitro** (SofaScore): {a.get('nombre', '?')} ({a.get('nacionalidad', '?')})")
        partes.append("IMPORTANTE: El árbitro de SofaScore es la fuente autoritativa. Usa este, no el de BSD si hay conflicto.")

    if detalle.get("manager_local") or detalle.get("manager_visitante"):
        local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
        away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"
        if detalle.get("manager_local"):
            partes.append(f"\nManager {local_team}: {detalle['manager_local'].get('nombre', '?')}")
        if detalle.get("manager_visitante"):
            partes.append(f"Manager {away_team}: {detalle['manager_visitante'].get('nombre', '?')}")

    partes.append("\n**NOTA SOBRE DISPONIBILIDAD**: SofaScore es la fuente principal para alineaciones y bajas cuando `missingPlayers` está disponible en lineups.")
    partes.append("Usa BAJAS BSD solo como respaldo/comparación si SofaScore no lista bajas.")

    # Detectar conflictos BSD vs SofaScore
    conflictos = _detectar_conflictos(datos, detalle)
    if conflictos:
        partes.append("\n**\u26a0\ufe0f CONFLICTOS BSD vs SofaScore:**")
        partes.append("Los siguientes jugadores aparecen como lesionados/suspendidos en BSD pero estan en la alineacion titular de SofaScore.")
        partes.append("Como SofaScore es la fuente autoritativa, asume que SI juegan. Pero ten en cuenta que pueden no estar al 100%:")
        for c in conflictos:
            partes.append(f"  - {c['equipo']}: {c['jugador']} (BSD: {c['estado_bsd']}, SofaScore: TITULAR)")
        partes.append("IMPORTANTE: Estos jugadores JUEGAN segun SofaScore. No los consideres bajas.")

    return "\n".join(partes)


def _detectar_conflictos(datos: dict, detalle: dict) -> list:
    """Compara bajas de BSD contra alineacion de SofaScore y detecta conflictos."""
    bajas_bsd = datos.get("bajas_bsd", {})
    alin = (datos.get("_sofascore") or {}).get("alineaciones") or {}
    if not bajas_bsd or not alin:
        return []

    local_team_name = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team_name = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    conflictos = []

    def _nombres_alineacion(side):
        titulares = alin.get(side, {}).get("titulares", [])
        suplentes = alin.get(side, {}).get("suplentes", [])
        return {t["nombre"].lower().strip() for t in titulares + suplentes}

    nombres_local = _nombres_alineacion("local")
    nombres_visitante = _nombres_alineacion("visitante")

    for lado, side_key, team_name in [("local", "local", local_team_name), ("visitante", "visitante", away_team_name)]:
        side_bajas = bajas_bsd.get(side_key, {})
        confirmadas = side_bajas.get("confirmadas", [])
        alin_nombres = nombres_local if lado == "local" else nombres_visitante

        for baja in confirmadas:
            nombre_baja = baja["nombre"].lower().strip()
            if nombre_baja in alin_nombres:
                conflictos.append({
                    "equipo": team_name,
                    "jugador": baja["nombre"],
                    "estado_bsd": baja.get("estado", "?"),
                })

    return conflictos


def _formatear_alineaciones_para_prompt(datos: dict) -> str:
    """
    Formatea los datos de alineaciones de SofaScore para incluirlos en el prompt.

    Args:
        datos: Dict completo de datos (con clave _sofascore).

    Returns:
        String formateado para el prompt, o cadena vacia si no hay alineaciones.
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible") or "alineaciones" not in sofas:
        return ""

    alin = sofas["alineaciones"]
    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = []
    partes.append("\n### ALINEACIONES (SofaScore)")
    partes.append("ATENCION: Estas alineaciones y bajas de SofaScore son la fuente principal de disponibilidad. No uses BSD si contradice SofaScore.")

    for lado, equipo, data in [("local", local_team, alin.get("local", {})),
                                ("visitante", away_team, alin.get("visitante", {}))]:
        if not data:
            continue
        estado_alineacion = "CONFIRMADA" if data.get("confirmada") else "PRELIMINAR/POSIBLE"
        partes.append(f"\n**{equipo}** ({data.get('formacion', '?')}) - {estado_alineacion}")

        titulares = data.get("titulares", [])
        if titulares:
            nombres = [f"{t['nombre']} ({t['posicion']})" for t in titulares]
            partes.append(f"  Titulares: {', '.join(nombres)}")

        suplentes = data.get("suplentes", [])
        if suplentes:
            nombres = [f"{t['nombre']} ({t['posicion']})" for t in suplentes]
            partes.append(f"  Suplentes: {', '.join(nombres)}")

        bajas = data.get("bajas", {})
        confirmadas = bajas.get("confirmadas", [])
        dudas = bajas.get("dudas", [])
        if confirmadas:
            partes.append("  No disponibles SofaScore:")
            for baja in confirmadas:
                motivo = f" - {baja.get('motivo')}" if baja.get("motivo") else ""
                fecha = f" (fin est.: {baja.get('fecha_fin_estimada')})" if baja.get("fecha_fin_estimada") else ""
                extra = _formatear_extra_baja_sofascore(baja)
                partes.append(f"    - {baja.get('nombre', '?')} ({baja.get('posicion', '?')}): {baja.get('estado', '?')}{motivo}{fecha}{extra}")
        if dudas:
            partes.append("  Dudas SofaScore:")
            for baja in dudas:
                motivo = f" - {baja.get('motivo')}" if baja.get("motivo") else ""
                fecha = f" (fin est.: {baja.get('fecha_fin_estimada')})" if baja.get("fecha_fin_estimada") else ""
                extra = _formatear_extra_baja_sofascore(baja)
                partes.append(f"    - {baja.get('nombre', '?')} ({baja.get('posicion', '?')}): {baja.get('estado', '?')}{motivo}{fecha}{extra}")

    partes.append("\nIMPORTANTE: Si la alineacion es CONFIRMADA, los titulares/suplentes listados por SofaScore son los disponibles para jugar. Si es PRELIMINAR/POSIBLE, usalos solo como probable XI. Los jugadores en 'No disponibles SofaScore' NO deben contarse como disponibles; las 'Dudas' tienen disponibilidad incierta.")
    return "\n".join(partes)


def _formatear_bajas_sofascore_para_prompt(datos: dict) -> str:
    """Formatea bajas/dudas extraídas de SofaScore lineups.missingPlayers."""
    sofas = datos.get("_sofascore", {})
    alin = sofas.get("alineaciones", {}) if sofas.get("disponible") else {}
    if not alin:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = []
    for side, team_name in [("local", local_team), ("visitante", away_team)]:
        bajas = (alin.get(side, {}) or {}).get("bajas", {})
        confirmadas = bajas.get("confirmadas", [])
        dudas = bajas.get("dudas", [])
        if not confirmadas and not dudas:
            continue
        partes.append(f"\n**{team_name}**")
        if confirmadas:
            partes.append("No disponibles:")
            for baja in confirmadas:
                motivo = f" - {baja.get('motivo')}" if baja.get("motivo") else ""
                fecha = f" (fin est.: {baja.get('fecha_fin_estimada')})" if baja.get("fecha_fin_estimada") else ""
                extra = _formatear_extra_baja_sofascore(baja)
                partes.append(f"  - {baja.get('nombre', '?')} ({baja.get('posicion', '?')}): {baja.get('estado', '?')}{motivo}{fecha}{extra}")
        if dudas:
            partes.append("Dudas:")
            for baja in dudas:
                motivo = f" - {baja.get('motivo')}" if baja.get("motivo") else ""
                fecha = f" (fin est.: {baja.get('fecha_fin_estimada')})" if baja.get("fecha_fin_estimada") else ""
                extra = _formatear_extra_baja_sofascore(baja)
                partes.append(f"  - {baja.get('nombre', '?')} ({baja.get('posicion', '?')}): {baja.get('estado', '?')}{motivo}{fecha}{extra}")

    if not partes:
        return "(SofaScore no lista bajas/dudas en lineups para este partido.)"

    return "\n".join([
        "### BAJAS SOFASCORE (fuente principal)",
        "Usa estas bajas/dudas por encima de BSD. Si la alineación está confirmada, titulares/suplentes de SofaScore juegan; si está preliminar, trátalos como probable XI.",
        *partes,
    ])


def _formatear_xi_impact_sofascore_para_prompt(datos: dict) -> str:
    """Resume peso interno de titulares probables/confirmados sin inflar el prompt."""
    sofas = datos.get("_sofascore", {})
    alin = sofas.get("alineaciones", {}) if sofas.get("disponible") else {}
    if not alin:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### IMPACTO XI SOFASCORE (quienes SI juegan)"]
    partes.append(
        "Uso: media reciente/temporada de titulares probables o confirmados. "
        "Mide peso interno/continuidad dentro del equipo, NO calidad absoluta entre equipos. "
        "Si la alineación es preliminar, úsalo solo como perfil probable."
    )

    any_content = False
    for side, team_name in [("local", local_team), ("visitante", away_team)]:
        team = alin.get(side, {}) or {}
        starters = [p for p in team.get("titulares", []) or [] if p.get("impacto_jugador")]
        if not starters:
            continue
        any_content = True
        confirmed = "confirmado" if team.get("confirmada") else "probable"
        levels = {"ALTA": 0, "MEDIA": 0, "BAJA": 0, "DESCONOCIDA": 0}
        scores = []
        for p in starters:
            impacto = p.get("impacto_jugador") or {}
            levels[impacto.get("nivel", "DESCONOCIDA")] = levels.get(impacto.get("nivel", "DESCONOCIDA"), 0) + 1
            if impacto.get("score") is not None:
                scores.append(float(impacto["score"]))
        avg_score = round(sum(scores) / len(scores), 1) if scores else "?"
        partes.append(
            f"\n**{team_name}** ({confirmed}) | datos XI: {len(starters)}/11 | "
            f"peso interno A/M/B/D: {levels.get('ALTA',0)}/{levels.get('MEDIA',0)}/{levels.get('BAJA',0)}/{levels.get('DESCONOCIDA',0)} | peso prom {avg_score}"
        )

        top = sorted(starters, key=lambda p: (p.get("impacto_jugador") or {}).get("score", -99), reverse=True)[:4]
        if top:
            partes.append("  Titulares con mayor peso interno:")
            for p in top:
                partes.append(f"    - {_linea_impacto_jugador_sofascore(p)}")

        gk = next((p for p in starters if (p.get("posicion") or "").upper() == "G"), None)
        weak = [
            p for p in starters
            if (p.get("impacto_jugador") or {}).get("nivel") in {"BAJA", "DESCONOCIDA"}
            and p is not gk
        ][:3]
        alertas = []
        if gk:
            alertas.append(f"Portero: {_linea_impacto_jugador_sofascore(gk, compact=True)}")
        if len(weak) >= 2:
            alertas.append("XI con varios perfiles de bajo peso: " + ", ".join(
                f"{p.get('nombre','?')} {((p.get('impacto_jugador') or {}).get('nivel','?'))}"
                for p in weak
            ))
        if alertas:
            partes.append("  Alertas:")
            partes.extend(f"    - {a}" for a in alertas[:3])

    return "\n".join(partes) if any_content else ""


def _formatear_player_avgs_sofascore_para_prompt(datos: dict) -> str:
    """Formatea medias de temporada por jugador desde SofaScore para el prompt."""
    sofas = datos.get("_sofascore", {})
    alin = sofas.get("alineaciones", {}) if sofas.get("disponible") else {}
    if not alin:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### MEDIAS POR JUGADOR (SofaScore)"]
    partes.append(
        "Base: temporada/torneo actual cuando SofaScore la entrega. "
        "Las medias /90 usan minutos jugados; si la alineacion es preliminar, leer como probable XI."
    )

    any_content = False
    for side, team_name in [("local", local_team), ("visitante", away_team)]:
        team = alin.get(side, {}) or {}
        starters = team.get("titulares", []) or []
        rows = [
            _linea_media_jugador_sofascore(p)
            for p in starters
            if (p.get("impacto_jugador") or {}).get("stats_resumen")
        ]
        bajas = (team.get("bajas") or {})
        missing_rows = []
        for bucket, label in [("confirmadas", "baja"), ("dudas", "duda")]:
            for player in bajas.get(bucket, []) or []:
                if (player.get("impacto_baja") or {}).get("stats_resumen"):
                    missing_rows.append(_linea_media_jugador_sofascore(player, impact_key="impacto_baja", estado=label))
        if not rows and not missing_rows:
            continue
        any_content = True
        confirmed = "CONFIRMADA" if team.get("confirmada") else "PRELIMINAR/POSIBLE"
        if rows:
            partes.append(f"\n**{team_name}** - titulares {confirmed}")
            partes.extend(f"  - {row}" for row in rows[:11])
        elif missing_rows:
            partes.append(f"\n**{team_name}** - titulares {confirmed}")
        if missing_rows:
            partes.append("  Bajas/dudas con media:")
            partes.extend(f"    - {row}" for row in missing_rows[:8])

    if not any_content:
        return ""
    partes.append(
        "Uso recomendado: para tiros de jugador/equipo mira tiros/90, SOT/90 y xG/90; "
        "para corners/volumen mira creacion, key passes y centros; para tarjetas mira YC/90, faltas y acciones defensivas."
    )
    return "\n".join(partes)


def _linea_media_jugador_sofascore(player: dict, impact_key: str = "impacto_jugador", estado: str = "") -> str:
    stats = ((player.get(impact_key) or {}).get("stats_resumen") or {})
    name = player.get("nombre") or player.get("name") or "?"
    pos = player.get("posicion") or player.get("position") or "?"
    prefix = f"{name} ({pos})"
    if estado:
        prefix += f" [{estado}]"

    base = []
    apps = stats.get("apps")
    minutes = stats.get("minutes")
    if apps or minutes:
        base.append(f"{apps or '?'}PJ/{minutes or '?'}min")
    if stats.get("avg_rating") is not None:
        base.append(f"rating {_fmt_player_stat(stats.get('avg_rating'))}")
    if stats.get("goals") is not None or stats.get("assists") is not None:
        base.append(f"G/A {stats.get('goals', 0)}/{stats.get('assists', 0)}")

    attack = []
    if stats.get("shots_p90") is not None:
        attack.append(f"tiros {_fmt_player_stat(stats.get('shots_p90'))}/90")
    if stats.get("shots_on_target_p90") is not None:
        attack.append(f"SOT {_fmt_player_stat(stats.get('shots_on_target_p90'))}/90")
    if stats.get("xg_p90") is not None:
        attack.append(f"xG {_fmt_player_stat(stats.get('xg_p90'))}/90")

    creation = []
    if stats.get("xa_p90") is not None:
        creation.append(f"xA {_fmt_player_stat(stats.get('xa_p90'))}/90")
    if stats.get("key_passes_p90") is not None:
        creation.append(f"key {_fmt_player_stat(stats.get('key_passes_p90'))}/90")
    if stats.get("accurate_crosses_p90") is not None:
        creation.append(f"centros {_fmt_player_stat(stats.get('accurate_crosses_p90'))}/90")

    discipline = []
    if stats.get("yellow_cards_p90") is not None:
        discipline.append(f"YC {_fmt_player_stat(stats.get('yellow_cards_p90'))}/90")
    if stats.get("fouls_p90") is not None:
        discipline.append(f"faltas {_fmt_player_stat(stats.get('fouls_p90'))}/90")
    if stats.get("defensive_actions_p90") is not None:
        discipline.append(f"def {_fmt_player_stat(stats.get('defensive_actions_p90'))}/90")
    if stats.get("saves_p90") is not None:
        discipline.append(f"atajadas {_fmt_player_stat(stats.get('saves_p90'))}/90")

    sections = [" | ".join(part for part in base if part)]
    for group in (attack, creation, discipline):
        if group:
            sections.append(", ".join(group))
    return prefix + ": " + " | ".join(s for s in sections if s)


def _fmt_player_stat(value) -> str:
    number = _to_float_or_none(value)
    if number is None:
        return "-"
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}"


def _linea_impacto_jugador_sofascore(player: dict, compact: bool = False) -> str:
    impacto = player.get("impacto_jugador") or {}
    nivel = impacto.get("nivel", "DESCONOCIDA")
    score = impacto.get("score")
    score_txt = f" {score}" if score is not None else ""
    base = f"{player.get('nombre', '?')} ({player.get('posicion', '?')}) peso interno {nivel}{score_txt}"
    extras = []
    if player.get("valor_mercado") and not compact:
        extras.append(f"valor {player['valor_mercado']}")
    razones = impacto.get("razones") or []
    extras.extend(razones[:1 if compact else 2])
    return base + (": " + "; ".join(extras) if extras else "")


def _formatear_extra_baja_sofascore(baja: dict) -> str:
    detalles = []
    if baja.get("valor_mercado"):
        detalles.append(f"valor {baja['valor_mercado']}")
    if baja.get("edad"):
        detalles.append(f"edad {baja['edad']}")
    impacto = baja.get("impacto_baja") or {}
    if impacto:
        razones = impacto.get("razones") or []
        razon_txt = "; ".join(razones[:3])
        nivel = impacto.get("nivel", "DESCONOCIDA")
        score = impacto.get("score")
        score_txt = f" {score}" if score is not None else ""
        detalles.append(f"impacto {nivel}{score_txt}" + (f": {razon_txt}" if razon_txt else ""))
    return f" [{', '.join(detalles)}]" if detalles else ""


def _obtener_standings(session, liga: str, team_ids: set, tournament_uid: int = None, season_id: int = None) -> dict:
    """Obtiene la tabla de posiciones relevante para el partido.

    En torneos con grupos, SofaScore devuelve una lista de tablas. Para no
    mezclar todos los grupos, se conserva solo el grupo donde aparecen los
    equipos del partido. En ligas de tabla única, se conserva la tabla completa.
    """
    candidates = []
    if tournament_uid and season_id:
        candidates.append((tournament_uid, season_id))

    mapping = STANDINGS_MAP.get(liga)
    if mapping and mapping not in candidates:
        candidates.append(mapping)

    if not candidates:
        return {}

    for uid, sid in candidates:
        try:
            resp = session.get(f"{SOFASCORE_API}/unique-tournament/{uid}/season/{sid}/standings/total", timeout=15)
            if resp.status_code != 200:
                continue
            data = resp.json()
        except Exception:
            continue

        result = _extraer_standings_relevantes(data, liga, team_ids)
        if result:
            return result

    return {}


def _extraer_standings_relevantes(data: dict, liga: str, team_ids: set) -> dict:
    standings = data.get("standings", [])
    if not standings:
        return {}

    selected_standings = standings
    if len(standings) > 1 and team_ids:
        groups_with_all_match_teams = []
        groups_with_any_match_team = []
        for st in standings:
            rows = st.get("rows", [])
            row_team_ids = {row.get("team", {}).get("id") for row in rows}
            if team_ids.issubset(row_team_ids):
                groups_with_all_match_teams.append(st)
            elif row_team_ids & team_ids:
                groups_with_any_match_team.append(st)
        if groups_with_all_match_teams:
            selected_standings = groups_with_all_match_teams
        elif groups_with_any_match_team:
            selected_standings = groups_with_any_match_team

    result = {}
    total_rows = 0
    table_names = []
    tie_rules = []
    for st in selected_standings:
        rows = st.get("rows", [])
        if st.get("name"):
            table_names.append(st["name"])
        if st.get("tieBreakingRule", {}).get("text"):
            tie_rules.append(st["tieBreakingRule"]["text"])
        total_rows += len(rows)
        for row in rows:
            team = row.get("team", {})
            promotion = row.get("promotion") or {}
            info = {
                "posicion": row.get("position"),
                "puntos": row.get("points"),
                "partidos_jugados": row.get("matches"),
                "goles_favor": row.get("scoresFor"),
                "goles_contra": row.get("scoresAgainst"),
                "diferencia": (row.get("scoresFor", 0) or 0) - (row.get("scoresAgainst", 0) or 0),
                "victorias": row.get("wins"),
                "empates": row.get("draws"),
                "derrotas": row.get("losses"),
                "forma": row.get("form", ""),
                "zona": promotion.get("text", ""),
                "descripcion": ", ".join(row.get("descriptions", []) or []),
            }
            result[team.get("name", "")] = info

    result["_liga_info"] = {
        "total_equipos": total_rows,
        "nombre_tabla": " / ".join(table_names) if table_names else liga,
        "fuente": "SofaScore",
        "regla_desempate": tie_rules[0] if tie_rules else "",
    }
    return result


def _formatear_standings_para_prompt(datos: dict) -> str:
    """Formatea los datos de standings/tabla de posiciones para el prompt, incluyendo tabla completa."""
    ss = datos.get("_sofascore", {})
    standings = ss.get("standings", {})
    if not standings:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    liga_info = standings.get("_liga_info", {})
    total_equipos = liga_info.get("total_equipos", "?")
    league_name = liga_info.get("nombre_tabla", datos.get("liga", "?"))
    fuente = liga_info.get("fuente")
    regla_desempate = liga_info.get("regla_desempate")

    def _find_team(name, standings):
        if name in standings:
            return name, standings[name]
        for k, v in standings.items():
            if k.startswith("_liga"):
                continue
            if name.lower() in k.lower() or k.lower() in name.lower():
                return k, v
        return None, None

    local_key, local_info = _find_team(local_team, standings)
    away_key, away_info = _find_team(away_team, standings)

    all_rows = [(k, v) for k, v in standings.items() if not k.startswith("_liga")]
    all_rows.sort(key=lambda x: (x[1].get("posicion") or 999))

    partes = [f"\n### TABLA DE POSICIONES COMPLETA ({league_name})"]
    partes.append(f"Total equipos: {total_equipos}")
    if fuente:
        partes.append(f"Fuente tabla: {fuente}")
    if regla_desempate:
        primera_linea_regla = regla_desempate.splitlines()[0]
        partes.append(f"Regla desempate: {primera_linea_regla}")

    if local_info:
        desc = f" ({local_info.get('descripcion')})" if local_info.get("descripcion") else ""
        form_str = f" | Forma: {local_info.get('forma')}" if local_info.get("forma") else ""
        zona_str = f" | Zona: {local_info.get('zona')}" if local_info.get("zona") else ""
        partes.append(
            f"\n**LOCAL: {local_key}** "
            f"#{local_info.get('posicion', '?')} de {total_equipos} | "
            f"PTS: {local_info.get('puntos', '?')} | "
            f"PJ: {local_info.get('partidos_jugados', '?')} | "
            f"V: {local_info.get('victorias', '?')} | "
            f"E: {local_info.get('empates', '?')} | "
            f"D: {local_info.get('derrotas', '?')} | "
            f"GF: {local_info.get('goles_favor', '?')} | "
            f"GC: {local_info.get('goles_contra', '?')} | "
            f"DG: {local_info.get('diferencia', '?')}{zona_str}{desc}{form_str}"
        )
    if away_info:
        desc = f" ({away_info.get('descripcion')})" if away_info.get("descripcion") else ""
        form_str = f" | Forma: {away_info.get('forma')}" if away_info.get("forma") else ""
        zona_str = f" | Zona: {away_info.get('zona')}" if away_info.get("zona") else ""
        partes.append(
            f"**VISITANTE: {away_key}** "
            f"#{away_info.get('posicion', '?')} de {total_equipos} | "
            f"PTS: {away_info.get('puntos', '?')} | "
            f"PJ: {away_info.get('partidos_jugados', '?')} | "
            f"V: {away_info.get('victorias', '?')} | "
            f"E: {away_info.get('empates', '?')} | "
            f"D: {away_info.get('derrotas', '?')} | "
            f"GF: {away_info.get('goles_favor', '?')} | "
            f"GC: {away_info.get('goles_contra', '?')} | "
            f"DG: {away_info.get('diferencia', '?')}{zona_str}{desc}{form_str}"
        )

    partes.append(f"\n**TABLA COMPLETA ({league_name})**")
    match_keys = set()
    if local_key:
        match_keys.add(local_key)
    if away_key:
        match_keys.add(away_key)

    for name, info in all_rows:
        marker = ">" if name in match_keys else " "
        pos = str(info.get("posicion", "?")).rjust(2)
        pts = str(info.get("puntos", "?")).rjust(2)
        pj = str(info.get("partidos_jugados", "?")).rjust(2)
        g = f"{info.get('goles_favor', '?')}:{info.get('goles_contra', '?')}"
        diff = info.get("diferencia", 0) or 0
        dg = f"+{diff}" if diff > 0 else str(diff)
        parts = [
            f"[#{pos}]",
            f"{marker}",
            f"{name:<22}",
            f"PTS:{pts}",
            f"PJ:{pj}",
            f"GF:GC {g}",
            f"DG:{dg}",
            f"Zona:{info.get('zona', '-') or '-'}",
            f"Forma:{info.get('forma', '-')}",
        ]
        extra = f" {info.get('descripcion')}" if info.get("descripcion") else ""
        partes.append("  ".join(parts) + extra)

    return "\n".join(partes)


def _formatear_players_stats_para_prompt(datos: dict) -> str:
    """Formatea estadisticas detalladas por jugador (xG, pases, duelos, rating)."""
    ss = datos.get("_sofascore", {})
    players = ss.get("players_stats", {})
    if not players or not players.get("confirmed"):
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### ESTADISTICAS POR JUGADOR (SofaScore)"]

    for side, team_name in [("home", local_team), ("away", away_team)]:
        form = players.get(f"{side}_formation", "")
        partes.append(f"\n**{team_name}** ({form})")
        team_players = players.get(side, [])
        if not team_players:
            continue

        starters = [p for p in team_players if not p.get("is_substitute")]
        subs = [p for p in team_players if p.get("is_substitute")]

        for p in starters:
            extra_parts = []
            if p.get("rating") is not None:
                extra_parts.append(f"Rating: {p['rating']}")
            if p.get("xG") is not None:
                extra_parts.append(f"xG: {p['xG']}")
            if p.get("goals"):
                extra_parts.append(f"Goles: {p['goals']}")
            if p.get("total_shots"):
                extra_parts.append(f"Tiros: {p['total_shots']}/{p.get('shots_on_target', 0)}")
            if p.get("key_passes"):
                extra_parts.append(f"KeyPass: {p['key_passes']}")
            if p.get("total_passes"):
                extra_parts.append(f"Pases: {p.get('accurate_passes', 0)}/{p['total_passes']} ({p.get('pass_accuracy', '?')}%)")
            if p.get("duels_total"):
                extra_parts.append(f"Duelos: {p.get('duels_won', 0)}/{p['duels_total']}")
            if p.get("tackles_total"):
                extra_parts.append(f"Entradas: {p.get('tackles_won', 0)}/{p['tackles_total']}")

            extra = " | ".join(extra_parts)
            partes.append(f"  #{p.get('jersey_number', '?')} {p.get('name', '?')} ({p.get('position', '?')}): {extra}")

        if subs:
            sub_names = [f"#{p.get('jersey_number', '?')} {p.get('name', '?')}" for p in subs[:5]]
            partes.append(f"  Suplentes: {', '.join(sub_names)}{' (+' + str(len(subs) - 5) + ')' if len(subs) > 5 else ''}")

    return "\n".join(partes)


def _formatear_shotmap_para_prompt(datos: dict) -> str:
    """Formatea el mapa de tiros para el prompt."""
    ss = datos.get("_sofascore", {})
    shots = ss.get("shotmap", [])
    if not shots:
        return ""

    home_xg = round(sum(s.get("xg", 0) or 0 for s in shots if s.get("is_home")), 4)
    away_xg = round(sum(s.get("xg", 0) or 0 for s in shots if not s.get("is_home")), 4)
    home_xgot = round(sum(s.get("xgot", 0) or 0 for s in shots if s.get("is_home")), 4)
    away_xgot = round(sum(s.get("xgot", 0) or 0 for s in shots if not s.get("is_home")), 4)

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### MAPA DE TIROS (SofaScore Shotmap)"]
    partes.append(f"**xG Total**: {local_team} {home_xg} - {away_xg} {away_team}")
    partes.append(f"**xGOT Total**: {local_team} {home_xgot} - {away_xgot} {away_team}")
    partes.append(f"**Total disparos**: {len(shots)}")

    if len(shots) <= 20:
        partes.append(f"\nDetalle de disparos:")
        for s in shots:
            side = local_team if s.get("is_home") else away_team
            partes.append(
                f"  {s.get('minute', '?')}' {s.get('player', '?')} ({side}) | "
                f"xG: {s.get('xg', '?')} | xGOT: {s.get('xgot', '?')} | "
                f"{s.get('shot_type', '?')} [{s.get('body_part', '?')}] ({s.get('situation', '?')})"
            )
    else:
        partes.append("(Demasiados disparos para mostrar detalle completo)")

    return "\n".join(partes)


def _formatear_momentum_para_prompt(datos: dict) -> str:
    """Formatea el grafico de momentum."""
    ss = datos.get("_sofascore", {})
    momentum = ss.get("momentum", {})
    if not momentum:
        return ""

    partes = ["\n### MOMENTUM (SofaScore)"]
    partes.append(f"- Periodo: {momentum.get('period_time', '?')} min")
    partes.append(f"- Dominio Local: {momentum.get('home_dominance', '?')}% | Dominio Visitante: {momentum.get('away_dominance', '?')}%")

    graph = momentum.get("graph", [])
    if graph:
        partes.append(f"- Puntos de momentum: {len(graph)}")
        partes.append("IMPORTANTE: El momentum mide el dominio acumulado minuto a minuto. Valores positivos = dominio local, negativos = dominio visitante.")

    return "\n".join(partes)


def _formatear_avg_positions_para_prompt(datos: dict) -> str:
    """Formatea posiciones promedio."""
    ss = datos.get("_sofascore", {})
    avg = ss.get("avg_positions", {})
    if not avg:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### POSICIONES PROMEDIO (SofaScore)"]

    for side, team_name in [("home", local_team), ("away", away_team)]:
        players = avg.get(side, [])
        if not players:
            continue
        partes.append(f"\n**{team_name}**:")
        for p in players:
            partes.append(
                f"  #{p.get('jersey_number', '?')} {p.get('player', '?')} ({p.get('position', '?')}) "
                f"X:{p.get('average_x', '?')} Y:{p.get('average_y', '?')}"
            )

    return "\n".join(partes)


def _formatear_incidents_para_prompt(datos: dict) -> str:
    """Formatea timeline de incidentes (goles, tarjetas, sustituciones)."""
    ss = datos.get("_sofascore", {})
    incidents = ss.get("incidents", {})
    if not incidents:
        return ""

    goals = incidents.get("goals", [])
    cards = incidents.get("cards", [])
    subs = incidents.get("substitutions", [])
    if not goals and not cards and not subs:
        return ""

    partes = ["\n### TIMELINE SOFASCORE"]

    if goals:
        partes.append(f"\n**Goles ({len(goals)}):**")
        for g in goals:
            asist = f" (asist: {g['assist']})" if g.get("assist") else ""
            partes.append(
                f"  {g.get('minute', '?')}' {g.get('player', '?')}{asist} "
                f"[{g.get('home_score', '?')}-{g.get('away_score', '?')}]"
            )

    if cards:
        partes.append(f"\n**Tarjetas ({len(cards)}):**")
        for c in cards:
            partes.append(f"  {c.get('minute', '?')}' {c.get('card_type', '?')} - {c.get('player', '?')}")

    if subs:
        partes.append(f"\n**Sustituciones ({len(subs)}):**")
        for s in subs[:10]:
            partes.append(f"  {s.get('minute', '?')}' IN: {s.get('player_in', '?')} OUT: {s.get('player_out', '?')}")

    return "\n".join(partes)


def _formatear_match_statistics_para_prompt(datos: dict) -> str:
    """Formatea estadisticas del partido (posesion, tiros, corners, faltas, etc)."""
    ss = datos.get("_sofascore", {})
    stats = ss.get("estadisticas_partido", {})
    if not stats:
        return ""

    stat_labels = {
        "match_possession": "Posesion",
        "match_expected_goals": "xG",
        "match_big_chances": "Ocasiones claras",
        "match_total_shots": "Tiros totales",
        "match_shots_on_target": "Tiros al arco",
        "match_shots_off_target": "Tiros desviados",
        "match_blocked_shots": "Tiros bloqueados",
        "match_shots_inside_box": "Tiros dentro del area",
        "match_shots_outside_box": "Tiros fuera del area",
        "match_hit_woodwork": "Al palo",
        "match_corner_kicks": "Corners",
        "match_fouls": "Faltas",
        "match_yellow_cards": "Amarillas",
        "match_red_cards": "Rojas",
        "match_offsides": "Offsides",
        "match_passes": "Pases totales",
        "match_accurate_passes": "Pases precisos",
        "match_crosses": "Centros",
        "match_goalkeeper_saves": "Paradas del portero",
        "match_touches_in_penalty": "Toques en area rival",
    }

    partes = ["\n### ESTADISTICAS DEL PARTIDO (SofaScore)"]
    for key, label in stat_labels.items():
        v = stats.get(key)
        if v:
            partes.append(f"  {label}: {v.get('home', '?')} - {v.get('away', '?')}")

    return "\n".join(partes)


def _formatear_team_season_stats_para_prompt(datos: dict) -> str:
    """Formatea estadisticas de temporada de los equipos."""
    ss = datos.get("_sofascore", {})
    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### ESTADISTICAS DE TEMPORADA (SofaScore)"]

    def _has_value(value):
        return value is not None and value != ""

    def _fmt_value(value, decimals: int = 1):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            rounded = round(float(value), decimals)
            if rounded.is_integer():
                return int(rounded)
            return rounded
        return value

    def _append_pair_line(lines: list, label_a: str, value_a, label_b: str, value_b, suffix: str = ""):
        chunks = []
        if _has_value(value_a):
            chunks.append(f"{label_a}: {_fmt_value(value_a)}{suffix}")
        if _has_value(value_b):
            chunks.append(f"{label_b}: {_fmt_value(value_b)}{suffix}")
        if chunks:
            lines.append("  " + " | ".join(chunks))

    for side, team_name in [("team_stats_local", local_team), ("team_stats_visitante", away_team)]:
        s = ss.get(side, {})
        if not s:
            continue

        team_lines = [f"\n**{team_name}**:"]
        record = []
        for label, key in [("PJ", "matches_played"), ("V", "wins"), ("E", "draws"), ("D", "losses")]:
            value = s.get(key)
            if _has_value(value):
                record.append(f"{label}: {value}")
        if record:
            team_lines.append("  " + " | ".join(record))

        goals = []
        for label, key in [("GF", "goals_scored"), ("GC", "goals_conceded"), ("CS", "clean_sheets")]:
            value = s.get(key)
            if _has_value(value):
                goals.append(f"{label}: {value}")
        if goals:
            team_lines.append("  " + " | ".join(goals))

        if _has_value(s.get("avg_possession")):
            team_lines.append(f"  Posesion prom: {_fmt_value(s.get('avg_possession'))}%")
        _append_pair_line(team_lines, "Tiros", s.get("shots"), "A puerta", s.get("shots_on_target"))
        _append_pair_line(team_lines, "Pases prec", s.get("accurate_passes_pct"), "Centros prec", s.get("accurate_crosses_pct"), "%")
        _append_pair_line(team_lines, "Duelos gan", s.get("duels_won_pct"), "Aereos gan", s.get("aerial_duels_won_pct"), "%")
        if _has_value(s.get("successful_dribbles")):
            team_lines.append(f"  Regates: {s.get('successful_dribbles')}")
        _append_pair_line(team_lines, "Corners", s.get("corners"), "Faltas", s.get("fouls"))
        _append_pair_line(team_lines, "Amarillas", s.get("yellow_cards"), "Rojas", s.get("red_cards"))

        chances = []
        for label, key in [("Big chances", "big_chances"), ("Creadas", "big_chances_created"), ("Falladas", "big_chances_missed")]:
            value = s.get(key)
            if _has_value(value):
                chances.append(f"{label}: {value}")
        if chances:
            team_lines.append("  " + " | ".join(chances))

        if len(team_lines) > 1:
            partes.extend(team_lines)

    if len(partes) == 1:
        return ""
    return "\n".join(partes)
    """Formatea los datos de standings/tabla de posiciones para el prompt, incluyendo tabla completa."""
    ss = datos.get("_sofascore", {})
    standings = ss.get("standings", {})
    if not standings:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    liga_info = standings.get("_liga_info", {})
    total_equipos = liga_info.get("total_equipos", "?")
    league_name = liga_info.get("nombre_tabla", datos.get("liga", "?"))

    # ── Encontrar los equipos del partido en la tabla ──
    def _find_team(name, standings):
        if name in standings:
            return name, standings[name]
        for k, v in standings.items():
            if k.startswith("_liga"):
                continue
            if name.lower() in k.lower() or k.lower() in name.lower():
                return k, v
        return None, None

    local_key, local_info = _find_team(local_team, standings)
    away_key, away_info = _find_team(away_team, standings)

    # ── Construir filas ordenadas de toda la tabla ──
    all_rows = [(k, v) for k, v in standings.items() if not k.startswith("_liga")]
    all_rows.sort(key=lambda x: (x[1].get("posicion") or 999))

    partes = [f"\n### TABLA DE POSICIONES COMPLETA ({league_name})"]
    partes.append(f"Total equipos: {total_equipos}")

    # Highlight de los equipos del partido
    if local_info:
        desc = f" ({local_info.get('descripcion')})" if local_info.get("descripcion") else ""
        form_str = f" | Forma: {local_info.get('forma')}" if local_info.get("forma") else ""
        partes.append(
            f"\n▶ **LOCAL: {local_key}** "
            f"#{local_info.get('posicion', '?')} de {total_equipos} | "
            f"PTS: {local_info.get('puntos', '?')} | "
            f"PJ: {local_info.get('partidos_jugados', '?')} | "
            f"V: {local_info.get('victorias', '?')} | "
            f"E: {local_info.get('empates', '?')} | "
            f"D: {local_info.get('derrotas', '?')} | "
            f"GF: {local_info.get('goles_favor', '?')} | "
            f"GC: {local_info.get('goles_contra', '?')} | "
            f"DG: {local_info.get('diferencia', '?')}{desc}{form_str}"
        )
    if away_info:
        desc = f" ({away_info.get('descripcion')})" if away_info.get("descripcion") else ""
        form_str = f" | Forma: {away_info.get('forma')}" if away_info.get("forma") else ""
        partes.append(
            f"▶ **VISITANTE: {away_key}** "
            f"#{away_info.get('posicion', '?')} de {total_equipos} | "
            f"PTS: {away_info.get('puntos', '?')} | "
            f"PJ: {away_info.get('partidos_jugados', '?')} | "
            f"V: {away_info.get('victorias', '?')} | "
            f"E: {away_info.get('empates', '?')} | "
            f"D: {away_info.get('derrotas', '?')} | "
            f"GF: {away_info.get('goles_favor', '?')} | "
            f"GC: {away_info.get('goles_contra', '?')} | "
            f"DG: {away_info.get('diferencia', '?')}{desc}{form_str}"
        )

    # ── Tabla completa compacta ──
    partes.append(f"\n**TABLA COMPLETA ({league_name})**")
    match_keys = set()
    if local_key:
        match_keys.add(local_key)
    if away_key:
        match_keys.add(away_key)

    for name, info in all_rows:
        marker = "▶" if name in match_keys else " "
        pos = str(info.get("posicion", "?")).rjust(2)
        pts = str(info.get("puntos", "?")).rjust(2)
        pj = str(info.get("partidos_jugados", "?")).rjust(2)
        g = f"{info.get('goles_favor', '?')}:{info.get('goles_contra', '?')}"
        diff = info.get("diferencia", 0) or 0
        dg = f"+{diff}" if diff > 0 else str(diff)
        parts = [
            f"[#{pos}]",
            f"{marker}",
            f"{name:<22}",
            f"PTS:{pts}",
            f"PJ:{pj}",
            f"GF:GC {g}",
            f"DG:{dg}",
            f"Forma:{info.get('forma', '-')}",
        ]
        extra = f" {info.get('descripcion')}" if info.get("descripcion") else ""
        partes.append("  ".join(parts) + extra)

    return "\n".join(partes)


def verificar_salud_sofascore(detallado: bool = True) -> dict:
    """
    Verifica que los endpoints de SofaScore sigan accesibles y con la estructura esperada.

    Prueba cada endpoint usado por el sistema contra un partido real de hoy.
    Si algun endpoint cambio URL, devuelve 4xx/5xx o cambio su schema JSON,
    lo reporta como fallo.

    Args:
        detallado: Si True, imprime el resultado a consola.

    Returns:
        Dict con status general y detalle por endpoint:
        {
            "ok": bool,
            "endpoints": {
                "scheduled_events": {"ok": bool, "error": str|None, "campos_ok": bool},
                "event_detail":     {...},
                "lineups":          {...},
                "performance":      {...},
                "h2h":              {...},
            }
        }
    """
    resultado = {"ok": True, "endpoints": {}}
    hoy = datetime.now().strftime("%Y-%m-%d")

    try:
        session = _crear_sesion_sofascore()
    except ImportError as e:
        resultado["ok"] = False
        resultado["error"] = f"curl_cffi no instalado: {e}"
        if detallado:
            print(f"\n  SOFASCORE HEALTH: FAIL - {resultado['error']}")
        return resultado
    except Exception as e:
        resultado["ok"] = False
        resultado["error"] = f"No se pudo crear sesion: {e}"
        if detallado:
            print(f"\n  SOFASCORE HEALTH: FAIL - {resultado['error']}")
        return resultado

    # ── 1. Scheduled events ──
    ep = {"ok": True, "error": None, "campos_ok": True}
    try:
        url = f"{SOFASCORE_API}/sport/football/scheduled-events/{hoy}"
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            ep["ok"] = False
            ep["error"] = f"HTTP {resp.status_code}"
        else:
            data = resp.json()
            events = data.get("events", [])
            if not events:
                ep["error"] = "Sin eventos hoy (puede ser normal)"
            else:
                ev = events[0]
                if not ev.get("homeTeam") or not ev.get("awayTeam") or not ev.get("id"):
                    ep["campos_ok"] = False
                    ep["error"] = "Estructura del evento cambio (falta homeTeam/awayTeam/id)"
    except Exception as e:
        ep["ok"] = False
        ep["error"] = str(e)[:100]
    resultado["endpoints"]["scheduled_events"] = ep
    if not ep["ok"]:
        resultado["ok"] = False

    # ── 2. Buscar un event_id y team_id valido ──
    event_id = None
    team_id = None
    try:
        resp = session.get(f"{SOFASCORE_API}/sport/football/scheduled-events/{hoy}", timeout=15)
        if resp.status_code == 200:
            events = resp.json().get("events", [])
            if events:
                event_id = events[0].get("id")
                team_id = events[0].get("homeTeam", {}).get("id")
    except Exception:
        pass

    if not event_id:
        for ep_name in ["event_detail", "lineups", "h2h", "performance"]:
            resultado["endpoints"][ep_name] = {
                "ok": False, "error": "No se encontro event_id para testear", "campos_ok": False
            }
        if detallado:
            _imprimir_resultado_salud(resultado)
        return resultado

    # ── 3. Event detail ──
    ep = _testear_event_detail(session, event_id)
    resultado["endpoints"]["event_detail"] = ep
    if not ep["ok"]:
        resultado["ok"] = False

    # ── 4. Lineups ──
    ep = _testear_lineups(session, event_id)
    resultado["endpoints"]["lineups"] = ep
    if not ep["ok"]:
        resultado["ok"] = False

    # ── 5. H2H ──
    ep = _testear_h2h(session, event_id)
    resultado["endpoints"]["h2h"] = ep
    if not ep["ok"]:
        resultado["ok"] = False

    # ── 6. Performance ──
    ep = _testear_performance(session, team_id) if team_id else {"ok": False, "error": "Sin team_id", "campos_ok": False}
    resultado["endpoints"]["performance"] = ep
    if not ep["ok"]:
        resultado["ok"] = False

    if detallado:
        _imprimir_resultado_salud(resultado)

    return resultado


def _testear_event_detail(session, event_id: int) -> dict:
    ep = {"ok": True, "error": None, "campos_ok": True}
    try:
        resp = session.get(f"{SOFASCORE_API}/event/{event_id}", timeout=15)
        if resp.status_code != 200:
            ep["ok"] = False
            ep["error"] = f"HTTP {resp.status_code}"
        else:
            data = resp.json()
            ev = data.get("event") or data
            if not ev.get("homeTeam"):
                ep["campos_ok"] = False
                ep["error"] = "Falta homeTeam en event detail"
    except Exception as e:
        ep["ok"] = False
        ep["error"] = str(e)[:100]
    return ep


def _testear_lineups(session, event_id: int) -> dict:
    ep = {"ok": True, "error": None, "campos_ok": True}
    try:
        resp = session.get(f"{SOFASCORE_API}/event/{event_id}/lineups", timeout=15)
        if resp.status_code != 200:
            ep["ok"] = False
            ep["error"] = f"HTTP {resp.status_code}"
        else:
            data = resp.json()
            if not data.get("home") and not data.get("away"):
                ep["campos_ok"] = False
                ep["error"] = "Faltan home/away en lineups (posible partido sin alineacion aun)"
            else:
                home = data.get("home", {})
                if home and not home.get("players"):
                    ep["campos_ok"] = False
                    ep["error"] = "Falta players[] en lineups/home"
    except Exception as e:
        ep["ok"] = False
        ep["error"] = str(e)[:100]
    return ep


def _testear_h2h(session, event_id: int) -> dict:
    ep = {"ok": True, "error": None, "campos_ok": True}
    try:
        resp = session.get(f"{SOFASCORE_API}/event/{event_id}/h2h", timeout=20)
        if resp.status_code != 200:
            ep["ok"] = False
            ep["error"] = f"HTTP {resp.status_code}"
        else:
            data = resp.json()
            if data and "teamDuel" not in data:
                ep["campos_ok"] = False
                ep["error"] = "Falta teamDuel en respuesta h2h (schema cambio)"
    except Exception as e:
        ep["ok"] = False
        ep["error"] = str(e)[:100]
    return ep


def _testear_performance(session, team_id: int) -> dict:
    ep = {"ok": True, "error": None, "campos_ok": True}
    try:
        resp = session.get(f"{SOFASCORE_API}/team/{team_id}/performance", timeout=15)
        if resp.status_code != 200:
            ep["ok"] = False
            ep["error"] = f"HTTP {resp.status_code}"
        else:
            data = resp.json()
            events = data.get("events", [])
            if events:
                ev = events[0]
                if "winnerCode" not in ev or "homeScore" not in ev:
                    ep["campos_ok"] = False
                    ep["error"] = "Estructura de performance/events cambio (falta winnerCode/homeScore)"
    except Exception as e:
        ep["ok"] = False
        ep["error"] = str(e)[:100]
    return ep


def _imprimir_resultado_salud(resultado: dict):
    """Imprime el resultado del health check en consola."""
    status_icon = "OK" if resultado["ok"] else "FAIL"
    print(f"\n  SOFASCORE HEALTH CHECK: {status_icon}")
    for name, ep in resultado.get("endpoints", {}).items():
        icon = "\u2713" if ep["ok"] else "\u2717"
        extra = f" ({ep['error']})" if ep.get("error") else ""
        campos = " [schema OK]" if ep.get("campos_ok") else " [schema CAMBIO]"
        print(f"    {icon} {name}{campos}{extra}")
    if not resultado["ok"]:
        print("  ADVERTENCIA: Algunos endpoints de SofaScore fallaron. Los datos pueden estar incompletos.")


# ── Datos de arbitro via API de SofaScore ──

def _extraer_id_arbitro_desde_url(url: str) -> int | None:
    """Extrae el ID del arbitro desde una URL de SofaScore.
    Ej: https://www.sofascore.com/football/referee/herrera-alexis/786550 -> 786550"""
    m = re.search(r"/referee/[^/]+/(\d+)", url)
    return int(m.group(1)) if m else None


def obtener_datos_arbitro_sofascore(session, arb_id: int) -> dict | None:
    """Obtiene estadisticas del arbitro desde la API de SofaScore."""
    try:
        r = session.get(
            f"{SOFASCORE_API}/referee/{arb_id}",
            timeout=15,
        )
        profile = r.json().get("referee", {})
    except Exception:
        return None

    if not profile:
        return None

    result = {
        "nombre": profile.get("name", ""),
        "pais": (profile.get("country") or {}).get("name", ""),
        "total_partidos": profile.get("games"),
        "yc_total": profile.get("yellowCards"),
        "rc_total": profile.get("redCards"),
        "ycrc_total": profile.get("yellowRedCards"),
    }

    # Calcular promedios
    games = result["total_partidos"] or 1
    if result["yc_total"] is not None:
        result["yc_pp"] = round(result["yc_total"] / games, 2)
    if result["rc_total"] is not None:
        result["rc_pp"] = round(result["rc_total"] / games, 2)

    # Obtener desglose por torneo
    try:
        r2 = session.get(
            f"{SOFASCORE_API}/referee/{arb_id}/statistics",
            timeout=15,
        )
        stats = r2.json().get("statistics", [])
    except Exception:
        stats = []

    torneos = []
    for s in stats:
        t = s.get("uniqueTournament", {})
        apps = s.get("appearances", 0)
        yc = s.get("yellowCards", 0)
        rc = s.get("redCards", 0)
        ycrc = s.get("yellowRedCards", 0)
        pen = s.get("penalty", 0)
        if apps > 0:
            torneos.append({
                "nombre": t.get("name", "?"),
                "partidos": apps,
                "yc_total": yc,
                "yc_pp": round(yc / apps, 2),
                "rc_total": rc + ycrc,
                "rc_pp": round((rc + ycrc) / apps, 2),
                "penaltis": pen,
            })
    if torneos:
        result["torneos"] = torneos

    return result


def _normalizar_torneo_ref(nombre: str) -> str:
    """Normaliza nombres de torneo para comparar evento vs stats de árbitro."""
    n = normalizar_nombre(nombre or "").lower()
    n = re.sub(r",?\s*group\s+[a-z0-9]+$", "", n).strip()
    n = re.sub(r",?\s*grupo\s+[a-z0-9]+$", "", n).strip()
    return n


def _buscar_torneo_arbitro_actual(arb_stats: dict, torneo_actual: str) -> dict | None:
    """Devuelve la fila de stats arbitral que corresponde al torneo del partido."""
    torneos = arb_stats.get("torneos", []) if isinstance(arb_stats, dict) else []
    if not torneos or not torneo_actual:
        return None

    actual = _normalizar_torneo_ref(torneo_actual)
    if not actual:
        return None

    for t in torneos:
        tn = _normalizar_torneo_ref(t.get("nombre", ""))
        if tn == actual:
            return t

    for t in torneos:
        tn = _normalizar_torneo_ref(t.get("nombre", ""))
        if tn and (tn in actual or actual in tn):
            return t

    return None


def enriquecer_arbitro_sofascore(datos_resumidos: dict, url: str) -> dict:
    """Enriquece datos del arbitro desde URL de SofaScore."""
    arb_id = _extraer_id_arbitro_desde_url(url)
    if not arb_id:
        return datos_resumidos

    try:
        session = _crear_sesion_sofascore()
        sf_data = obtener_datos_arbitro_sofascore(session, arb_id)
        if sf_data:
            merged = {**(datos_resumidos.get("arbitro") or {}), "_sofascore_ref": sf_data}
            datos_resumidos["arbitro"] = merged
    except Exception:
        pass

    return datos_resumidos


def formatear_arbitro_sofascore_para_prompt(datos_resumidos: dict) -> str:
    """Formatea datos del arbitro de SofaScore para el prompt."""
    arb = datos_resumidos.get("arbitro", {})
    sf = arb.get("_sofascore_ref", {})
    if not sf:
        return ""

    partes = [f"\n### ARBITRO (SofaScore): {sf.get('nombre', arb.get('nombre', '?'))}"]

    if sf.get("pais"):
        partes[0] += f" ({sf['pais']})"

    if sf.get("total_partidos"):
        partes.append(f"- Partidos dirigidos: {sf['total_partidos']}")

    yc = sf.get("yc_pp")
    if yc is not None:
        partes.append(f"- Amarillas/partido: {yc} (total: {sf.get('yc_total', '?')})")

    rc = sf.get("rc_pp")
    if rc is not None:
        partes.append(f"- Rojas/partido: {rc} (total: {sf.get('rc_total', '?')} incl. doble amarilla: {sf.get('ycrc_total', 0)})")

    torneos = sf.get("torneos", [])
    if torneos:
        partes.append(f"\n  Por torneo ({len(torneos)} competiciones):")
        for t in torneos:
            partes.append(
                f"    {t['nombre']}: {t.get('partidos', '?')} part, "
                f"{t.get('yc_pp', '?')} YC/part ({t.get('yc_total', 0)} total), "
                f"{t.get('rc_pp', '?')} RC/part ({t.get('rc_total', 0)} total)"
            )

    partes.append("(Fuente: SofaScore API - datos completos de carrera)")
    return "\n".join(partes)


def obtener_partidos_sofascore_only() -> list:
    """
    Obtiene proximos partidos de ligas que solo estan en SofaScore (sin BSD).

    Returns:
        Lista de dicts con estructura compatible con obtener_proximos_partidos() de BSD:
        {id, home_team, away_team, event_date, _league_name, league: {id, name}, _source: "sofascore"}
    """
    from datetime import date, datetime as dt_import, timedelta

    todos = []
    seen_ids = set()
    try:
        session = _crear_sesion_sofascore()
    except Exception:
        return todos

    for league_name, tournament_id in SOFASCORE_ONLY_LEAGUES.items():
        today = date.today()
        for offset in range(10):
            dt = (today + timedelta(days=offset)).isoformat()
            try:
                resp = session.get(f"{SOFASCORE_API}/sport/football/scheduled-events/{dt}", timeout=15)
                if resp.status_code != 200:
                    continue
                events = resp.json().get("events", [])
                for ev in events:
                    tour = ev.get("tournament", {})
                    if tour.get("id") != tournament_id:
                        continue
                    if ev.get("status", {}).get("type") != "notstarted":
                        continue
                    eid = ev.get("id")
                    if eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    ts = ev.get("startTimestamp", 0)
                    fecha = dt_import.fromtimestamp(ts).isoformat() if ts else ""
                    todos.append({
                        "id": eid,
                        "home_team": ev.get("homeTeam", {}).get("name", "?"),
                        "away_team": ev.get("awayTeam", {}).get("name", "?"),
                        "event_date": fecha,
                        "_league_name": league_name,
                        "league": {"id": tournament_id, "name": league_name},
                        "_source": "sofascore_only",
                    })
            except Exception:
                continue

    todos.sort(key=lambda p: p.get("event_date", "") if isinstance(p.get("event_date"), str) else "")
    return todos


def obtener_datos_completos_sofascore(partido: dict) -> dict:
    """
    Obtiene todos los datos de un partido que solo existe en SofaScore.
    Como no hay BSD, construye el dict datos_resumidos con lo que SofaScore provee.

    Args:
        partido: Dict con id (event_id de SofaScore), home_team, away_team, _league_name.

    Returns:
        Dict datos_resumidos compatible con analyzer.py, con clave _sofascore.
    """
    match_id = partido.get("id")
    home_team = partido.get("home_team", "?")
    away_team = partido.get("away_team", "?")
    liga = partido.get("_league_name", "?")

    resultado = {
        "id": match_id,
        "match_id": match_id,
        "partido": f"{home_team} vs {away_team}",
        "liga": liga,
        "fecha": partido.get("event_date", ""),
        "cuotas": {},
        "forma_local": {},
        "forma_visitante": {},
        "h2h": {},
        "entrenador_local": {},
        "entrenador_visitante": {},
        "arbitro": None,
        "estadio": None,
        "bajas_bsd": {"local": {"confirmadas": [], "dudas": []}, "visitante": {"confirmadas": [], "dudas": []}},
        "_sofascore": {"disponible": False, "error": ""},
        "_source": "sofascore_only",
    }

    try:
        session = _crear_sesion_sofascore()
    except Exception as e:
        resultado["_sofascore"]["error"] = str(e)
        return resultado

    # Reutilizar enriquecer_datos_partido con datos minimos
    datos_minimos = {
        "partido": f"{home_team} vs {away_team}",
        "fecha": partido.get("event_date", ""),
        "bajas_bsd": {"local": {"confirmadas": [], "dudas": []}, "visitante": {"confirmadas": [], "dudas": []}},
    }
    datos_enriquecidos = enriquecer_datos_partido(datos_minimos)

    resultado["_sofascore"] = datos_enriquecidos.get("_sofascore", {})
    resultado["liga"] = liga
    resultado["partido"] = f"{home_team} vs {away_team}"
    resultado["fecha"] = partido.get("event_date", "")

    # Extraer info del evento detalle
    try:
        detalle_evento = resultado["_sofascore"].get("detalle_evento", {})
        if detalle_evento.get("arbitro"):
            resultado["arbitro"] = detalle_evento["arbitro"]
        if detalle_evento.get("manager_local"):
            resultado["entrenador_local"] = {"nombre": detalle_evento["manager_local"].get("nombre")}
        if detalle_evento.get("manager_visitante"):
            resultado["entrenador_visitante"] = {"nombre": detalle_evento["manager_visitante"].get("nombre")}
    except Exception:
        pass

    return resultado
