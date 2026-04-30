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
import unicodedata
from datetime import datetime

from dotenv import load_dotenv

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
    "Champions League": 7,
    "Europa League": 679,
    "Copa Libertadores": 586,
    "Copa Sudamericana": 587,
}


def _normalizar_nombre(nombre: str) -> str:
    """
    Normaliza un nombre de equipo para matching entre BSD y SofaScore.
    Elimina acentos, pasa a minusculas, quita sufijos/prefijos genericos
    (FC, CF, SC, etc.) que no forman parte de la identidad del equipo.

    NO elimina palabras como "United", "City", "Town" ya que son parte
    del nombre real del club.
    """
    if not nombre:
        return ""

    # Mapa de abreviaturas conocidas a nombre canónico
    ABREVIATURAS = {
        "psg": "paris saint-germain",
        "fcb": "barcelona",
        "rma": "real madrid",
        "fc bayern": "bayern munich",
    }

    n = unicodedata.normalize("NFKD", nombre)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip()

    # Sufijos genericos (quitar SOLO si NO son parte del nombre identitario)
    sufijos_genericos = [
        " football club",
        " futbol club",
        " association football club",
    ]
    for s in sufijos_genericos:
        if n.endswith(s):
            n = n[:-len(s)].strip()
            break

    # Eliminar FC/CF/SC/AFC como sufijo (ej: "Arsenal FC" -> "arsenal")
    for s in [" fc", " cf", " sc", " ac", " afc"]:
        if n.endswith(s):
            remaining = n[:-len(s)].strip()
            # Solo quitar si lo que queda tiene al menos 5 caracteres
            # y no termina en una palabra que podria ser parte del nombre
            if len(remaining) >= 5:
                n = remaining
                break

    # Eliminar FC/CF/SC como prefijo (ej: "FC Bayern" -> "bayern")
    for p in ["fc ", "cf ", "sc ", "ac "]:
        if n.startswith(p):
            n = n[len(p):].strip()
            break

    # Normalizar nombres con guiones (ej: "Paris Saint-Germain" vs "Paris Saint Germain")
    n = n.replace("saint germain", "saint-germain")

    # Contraer espacios multiples
    n = re.sub(r"\s+", " ", n)

    # Resolver abreviaturas conocidas
    if n in ABREVIATURAS:
        n = ABREVIATURAS[n]

    return n


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
    # Extraer fecha YYYY-MM-DD
    try:
        if "T" in fecha_str:
            fecha_solo = fecha_str.split("T")[0]
        else:
            fecha_solo = fecha_str[:10]
    except (IndexError, TypeError):
        return None

    home_norm = _normalizar_nombre(home_team)
    away_norm = _normalizar_nombre(away_team)

    url = f"{SOFASCORE_API}/sport/football/scheduled-events/{fecha_solo}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    events = data.get("events", [])

    for event in events:
        e_home = _normalizar_nombre(
            event.get("homeTeam", {}).get("name", "")
        )
        e_away = _normalizar_nombre(
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


def obtener_estadisticas(session, event_id: int) -> dict | None:
    """
    Obtiene estadisticas del partido (si ya se jugo) para cross-validacion.

    Returns:
        Dict con estadisticas o None.
    """
    url = f"{SOFASCORE_API}/event/{event_id}/statistics"
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


def obtener_estadisticas_equipo(session, team_id: int) -> dict | None:
    """
    Obtiene estadisticas de temporada de un equipo desde SofaScore.

    Args:
        session: Sesion curl_cffi activa.
        team_id: ID del equipo en SofaScore.

    Returns:
        Dict con estadisticas o None si falla.
    """
    url = f"{SOFASCORE_API}/team/{team_id}/statistics"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def obtener_evento_detalle(session, event_id: int) -> dict | None:
    """
    Obtiene el detalle completo de un evento SofaScore (arbitro, managers, venue, sustitutos, lesiones).

    Returns:
        Dict con datos del evento o None si falla.
    """
    url = f"{SOFASCORE_API}/event/{event_id}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("event") if data else None
    except Exception:
        return None


def obtener_standings(session, tournament_id: int) -> dict | None:
    """
    Obtiene la tabla de posiciones (standings) de un torneo.

    Returns:
        Dict con standings (overall y home/away) o None si falla.
    """
    url = f"{SOFASCORE_API}/tournament/{tournament_id}/standings"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def _extraer_h2h_sofascore(h2h_data: dict) -> dict:
    """
    Extrae y resume datos de H2H desde la respuesta de SofaScore.

    Returns:
        Dict con {total_partidos, victorias_local, empates, victorias_visitante, goles_local, goles_visitante, promedio_goles, ultimos_enfrentamientos}.
    """
    resultado = {}
    events = h2h_data.get("events", [])

    if not events:
        return resultado

    resultado["total_partidos"] = len(events)
    vic_local = sum(1 for e in events if e.get("homeScore", {}).get("current", 0) is not None and
                    e.get("awayScore", {}).get("current", 0) is not None and
                    e.get("homeScore", {}).get("current", 0) > e.get("awayScore", {}).get("current", 0))
    vic_visit = sum(1 for e in events if e.get("homeScore", {}).get("current", 0) is not None and
                    e.get("awayScore", {}).get("current", 0) is not None and
                    e.get("awayScore", {}).get("current", 0) > e.get("homeScore", {}).get("current", 0))
    empates = resultado["total_partidos"] - vic_local - vic_visit

    goles_local = sum(e.get("homeScore", {}).get("current", 0) or 0 for e in events)
    goles_visit = sum(e.get("awayScore", {}).get("current", 0) or 0 for e in events)

    resultado["victorias_local"] = vic_local
    resultado["empates"] = empates
    resultado["victorias_visitante"] = vic_visit
    resultado["goles_local"] = goles_local
    resultado["goles_visitante"] = goles_visit
    resultado["promedio_goles"] = round((goles_local + goles_visit) / resultado["total_partidos"], 2) if resultado["total_partidos"] > 0 else 0

    ultimos = events[:5]
    enfrentamientos = []
    for e in ultimos:
        home = e.get("homeTeam", {}).get("name", "?")
        away = e.get("awayTeam", {}).get("name", "?")
        hs = e.get("homeScore", {}).get("current", 0)
        as_ = e.get("awayScore", {}).get("current", 0)
        fecha_ts = e.get("startTimestamp", 0)
        fecha_str = ""
        if fecha_ts:
            from datetime import datetime as dt
            fecha_str = dt.fromtimestamp(fecha_ts).strftime("%d/%m/%Y")
        enfrentamientos.append(f"{fecha_str}: {home} {hs}-{as_} {away}" if fecha_str else f"{home} {hs}-{as_} {away}")
    resultado["ultimos_enfrentamientos"] = enfrentamientos

    return resultado


def _extraer_standings(standings_data: dict, local_id: int, visitante_id: int) -> dict | None:
    """
    Extrae posiciones de ambos equipos en la tabla.

    Returns:
        Dict con {local: {posicion, pts, pj, v, e, d, gf, gc}, visitante: {...}} o None.
    """
    tables = standings_data.get("standings", [])
    if not tables:
        return None

    resultado = {}

    for table in tables:
        rows = table.get("rows", [])
        for row in rows:
            tid = row.get("team", {}).get("id")
            if not tid:
                continue
            info = {
                "posicion": row.get("position", "?"),
                "pts": row.get("points"),
                "pj": row.get("matches"),
                "v": row.get("wins"),
                "e": row.get("draws"),
                "d": row.get("losses"),
                "gf": row.get("scoresFor"),
                "gc": row.get("scoresAgainst"),
            }
            if tid == local_id:
                resultado["local"] = info
            elif tid == visitante_id:
                resultado["visitante"] = info

    return resultado if resultado else None


def _extraer_top_jugadores(stats_data: dict) -> dict:
    """
    Extrae rating promedio y top jugadores de las estadisticas de equipo SofaScore.

    Returns:
        Dict con {rating_promedio, top_jugadores: [{nombre, posicion, rating}]}.
    """
    resultado = {}

    stats_list = stats_data.get("statistics", [])
    if not isinstance(stats_list, list):
        return resultado

    for bloque in stats_list:
        # Buscar rating promedio del equipo
        for k, v in bloque.items():
            if isinstance(v, (int, float)) and "rating" in k.lower():
                resultado["rating_promedio_equipo"] = round(float(v), 2)

        items_explorar = bloque.get("groups") or bloque.get("statistics") or []
        if not isinstance(items_explorar, list):
            continue
        for grupo in items_explorar:
            stats_items = grupo.get("statisticsItems") or grupo.get("items") or []
            if not isinstance(stats_items, list):
                continue
            for item in stats_items:
                nombre = (item.get("name") or item.get("label") or "").lower()
                if "average sofascore rating" in nombre or "average rating" in nombre:
                    valor = item.get("value") or item.get("displayValue") or item.get("text")
                    if valor is not None:
                        try:
                            resultado["rating_promedio_equipo"] = round(float(valor), 2)
                        except (ValueError, TypeError):
                            resultado["rating_promedio_equipo"] = valor

    # Extraer top jugadores
    top_players = stats_data.get("topPlayers", [])
    if top_players:
        jugadores = []
        for tp in top_players[:5]:
            player = tp.get("player") or tp
            name = player.get("name", player.get("shortName", "?"))
            pos = player.get("position", "?")
            rating = tp.get("rating")
            jugadores.append({"nombre": name, "posicion": pos, "rating": rating})
        if jugadores:
            resultado["top_jugadores"] = jugadores

    return resultado


def obtener_top_jugadores(session, team_id: int) -> dict | None:
    """
    Obtiene estadisticas del equipo incluyendo rating y top jugadores.

    Returns:
        Dict con rating y top jugadores o None.
    """
    url = f"{SOFASCORE_API}/team/{team_id}/statistics"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return _extraer_top_jugadores(data) if data else None
    except Exception:
        return None


def _extraer_info_evento_detalle(evento_detalle: dict) -> dict:
    """
    Extrae arbitro, managers, suplentes nombrados, lesiones desde el detalle del evento.

    Returns:
        Dict con {arbitro, manager_local, manager_visitante, lesiones}.
    """
    resultado = {}

    referee = evento_detalle.get("referee", {})
    if referee and referee.get("name"):
        resultado["arbitro"] = {
            "nombre": referee.get("name"),
            "nacionalidad": referee.get("slug", "").replace("-", " ").title() if referee.get("slug") else "",
        }

    home_manager = evento_detalle.get("homeTeam", {}).get("manager", {})
    if home_manager and home_manager.get("name"):
        resultado["manager_local"] = {"nombre": home_manager.get("name")}

    away_manager = evento_detalle.get("awayTeam", {}).get("manager", {})
    if away_manager and away_manager.get("name"):
        resultado["manager_visitante"] = {"nombre": away_manager.get("name")}

    # Lesiones y suspensiones desde los datos de equipo
    lesiones = []
    for lado, key in [("local", "homeTeam"), ("visitante", "awayTeam")]:
        team_info = evento_detalle.get(key, {})
        injured = team_info.get("injuredPlayers", [])
        suspended = team_info.get("suspendedPlayers", [])
        for p in injured:
            name = p.get("player", p).get("name", p.get("shortName", "?"))
            lesiones.append({"equipo": lado, "jugador": name, "tipo": "lesion"})
        for p in suspended:
            name = p.get("player", p).get("name", p.get("shortName", "?"))
            lesiones.append({"equipo": lado, "jugador": name, "tipo": "suspension"})

    if lesiones:
        resultado["lesiones"] = lesiones

    return resultado


def _extraer_stats_equipo(stats_data: dict) -> dict:
    """
    Extrae estadisticas relevantes de la respuesta de team/statistics de SofaScore.

    Busca en la estructura (que puede variar entre ligas) los valores de:
    - remates promedio por partido
    - remates al arco promedio por partido
    - tarjetas amarillas promedio por partido
    - faltas promedio por partido
    - corners promedio por partido (si disponible)
    - posesion promedio (si disponible)

    Args:
        stats_data: Respuesta completa de /team/{id}/statistics.

    Returns:
        Dict con estadisticas extraidas, o dict vacio si no se encuentra info.
    """
    resultado = {}
    stats_list = stats_data.get("statistics") or stats_data.get("stats") or []

    if not isinstance(stats_list, list):
        return resultado

    for bloque in stats_list:
        grupos = bloque.get("groups") or bloque.get("statistics") or []
        if not isinstance(grupos, list):
            continue
        for grupo in grupos:
            items = grupo.get("statisticsItems") or grupo.get("items") or []
            if not isinstance(items, list):
                continue
            for item in items:
                nombre = (item.get("name") or item.get("label") or "").lower()
                valor = item.get("value") or item.get("displayValue") or item.get("text")
                if not nombre or valor is None:
                    continue

                # Remates totales por partido
                if "shots" in nombre and "on target" not in nombre and "off target" not in nombre and "blocked" not in nombre:
                    try:
                        resultado["remates_promedio_sf"] = round(float(valor), 2)
                    except (ValueError, TypeError):
                        resultado["remates_promedio_sf"] = valor

                # Remates al arco
                if "shots on" in nombre or "on target" in nombre:
                    try:
                        resultado["remates_arco_promedio_sf"] = round(float(valor), 2)
                    except (ValueError, TypeError):
                        resultado["remates_arco_promedio_sf"] = valor

                # Amarillas
                if "yellow" in nombre and "card" in nombre:
                    try:
                        resultado["amarillas_promedio_sf"] = round(float(valor), 2)
                    except (ValueError, TypeError):
                        resultado["amarillas_promedio_sf"] = valor

                # Faltas
                if "foul" in nombre and "suffered" not in nombre and "won" not in nombre:
                    try:
                        resultado["faltas_promedio_sf"] = round(float(valor), 2)
                    except (ValueError, TypeError):
                        resultado["faltas_promedio_sf"] = valor

                # Corners
                if "corner" in nombre:
                    try:
                        resultado["corners_promedio_sf"] = round(float(valor), 2)
                    except (ValueError, TypeError):
                        resultado["corners_promedio_sf"] = valor

                # Posesion
                if "possession" in nombre or "posesion" in nombre:
                    try:
                        resultado["posesion_promedio_sf"] = round(float(valor), 2)
                    except (ValueError, TypeError):
                        resultado["posesion_promedio_sf"] = valor

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
            "nombre": nombre,
            "posicion": posicion,
            "dorsal": jersey,
        }
        if es_suplente:
            suplentes.append(entry)
        else:
            titulares.append(entry)

    return {
        "confirmada": True,
        "formacion": formation,
        "titulares": titulares,
        "suplentes": suplentes,
    }


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
    }

    # Detalle del evento (arbitro, managers, lesiones, suplentes)
    time.sleep(0.5)
    detalle = obtener_evento_detalle(session, event_id)
    if detalle:
        info_detalle = _extraer_info_evento_detalle(detalle)
        if info_detalle:
            enriquecido["detalle_evento"] = info_detalle

    # Alineaciones (solo disponibles ~1h antes del partido)
    time.sleep(0.5)
    lineups = obtener_alineaciones(session, event_id)
    if lineups:
        enriquecido["alineaciones"] = {
            "local": _extraer_info_alineacion(lineups, "home"),
            "visitante": _extraer_info_alineacion(lineups, "away"),
        }

    # Estadisticas de equipo (remates, tarjetas, corners, posesion, rating, top jugadores)
    if home_team_id and away_team_id:
        time.sleep(0.5)
        stats_local = obtener_estadisticas_equipo(session, home_team_id)
        if stats_local:
            enriquecido["stats_equipo_local"] = _extraer_stats_equipo(stats_local)
            top_local = _extraer_top_jugadores(stats_local)
            if top_local:
                enriquecido["top_jugadores_local"] = top_local

        time.sleep(0.5)
        stats_visitante = obtener_estadisticas_equipo(session, away_team_id)
        if stats_visitante:
            enriquecido["stats_equipo_visitante"] = _extraer_stats_equipo(stats_visitante)
            top_visitante = _extraer_top_jugadores(stats_visitante)
            if top_visitante:
                enriquecido["top_jugadores_visitante"] = top_visitante

    # Standings (tabla de posiciones del torneo)
    if tournament_id:
        time.sleep(0.5)
        standings = obtener_standings(session, tournament_id)
        if standings:
            info_standings = _extraer_standings(standings, home_team_id, away_team_id)
            if info_standings:
                enriquecido["standings"] = info_standings

    # Estadisticas (solo si el partido ya se jugo - util para backtesting)
    time.sleep(0.5)
    stats = obtener_estadisticas(session, event_id)
    if stats:
        enriquecido["estadisticas_disponibles"] = True

    # H2H (parseado completo, no solo flag)
    time.sleep(0.5)
    h2h = obtener_h2h_sofascore(session, event_id)
    if h2h:
        h2h_info = _extraer_h2h_sofascore(h2h)
        if h2h_info:
            enriquecido["h2h"] = h2h_info

    datos_bsd["_sofascore"] = enriquecido
    return datos_bsd


def _formatear_stats_sofascore_para_prompt(datos: dict) -> str:
    """
    Formatea las estadisticas de equipo de SofaScore para incluir en el prompt.

    Args:
        datos: Dict completo de datos (con clave _sofascore).

    Returns:
        String formateado para el prompt, o cadena vacia si no hay stats de SofaScore.
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible"):
        return ""

    partes = []
    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    for lado, equipo, key in [("local", local_team, "stats_equipo_local"),
                                ("visitante", away_team, "stats_equipo_visitante")]:
        stats = sofas.get(key, {})
        if not stats:
            continue

        partes.append(f"\n**{equipo} — Estadisticas SofaScore (temporada actual):**")
        if "remates_promedio_sf" in stats:
            partes.append(f"  - Remates promedio: {stats['remates_promedio_sf']}")
        if "remates_arco_promedio_sf" in stats:
            partes.append(f"  - Remates al arco promedio: {stats['remates_arco_promedio_sf']}")
        if "amarillas_promedio_sf" in stats:
            partes.append(f"  - Amarillas promedio: {stats['amarillas_promedio_sf']}")
        if "faltas_promedio_sf" in stats:
            partes.append(f"  - Faltas promedio: {stats['faltas_promedio_sf']}")
        if "corners_promedio_sf" in stats:
            partes.append(f"  - Corners promedio: {stats['corners_promedio_sf']}")
        if "posesion_promedio_sf" in stats:
            partes.append(f"  - Posesion promedio: {stats['posesion_promedio_sf']}%")

    if partes:
        partes.insert(0, "\n### ESTADISTICAS DE EQUIPO (SofaScore, temporada actual)")
        partes.append("\nIMPORTANTE: Estas estadisticas de SofaScore son la FUENTE PRINCIPAL para remates, tarjetas y corners. Pesan mas que los promedios de BSD.")
        return "\n".join(partes)
    return ""


def _formatear_standings_para_prompt(datos: dict) -> str:
    """
    Formatea la tabla de posiciones de SofaScore para incluir en el prompt.
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible"):
        return ""
    standings = sofas.get("standings", {})
    if not standings:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### STANDINGS (SofaScore - Tabla de Posiciones)"]
    for lado, equipo, key in [("local", local_team, "local"), ("visitante", away_team, "visitante")]:
        info = standings.get(key, {})
        if not info:
            continue
        partes.append(
            f"\n**{equipo}**: Pos #{info.get('posicion', '?')} | "
            f"PTS: {info.get('pts', '?')} | PJ: {info.get('pj', '?')} | "
            f"V: {info.get('v', '?')} | E: {info.get('e', '?')} | D: {info.get('d', '?')} | "
            f"GF: {info.get('gf', '?')} | GC: {info.get('gc', '?')}"
        )
    return "\n".join(partes)


def _formatear_h2h_sofascore_para_prompt(datos: dict) -> str:
    """
    Formatea H2H de SofaScore para el prompt.
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible"):
        return ""
    h2h = sofas.get("h2h", {})
    if not h2h:
        return ""

    partes = ["\n### H2H SOFASCORE (detalle completo)"]
    partes.append(f"- Total partidos: {h2h.get('total_partidos', '?')} | "
                  f"Local: {h2h.get('victorias_local', 0)}V | "
                  f"Empates: {h2h.get('empates', 0)} | "
                  f"Visitante: {h2h.get('victorias_visitante', 0)}V")
    partes.append(f"- Goles: Local {h2h.get('goles_local', '?')} - Visitante {h2h.get('goles_visitante', '?')} | "
                  f"Promedio: {h2h.get('promedio_goles', '?')}")

    ultimos = h2h.get("ultimos_enfrentamientos", [])
    if ultimos:
        partes.append("\nÚltimos enfrentamientos:")
        for u in ultimos:
            partes.append(f"  {u}")

    return "\n".join(partes)


def _formatear_top_jugadores_para_prompt(datos: dict) -> str:
    """
    Formatea rating y top jugadores de SofaScore para el prompt.
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible"):
        return ""
    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### RATINGS Y TOP JUGADORES (SofaScore)"]
    for lado, equipo, key_stats, key_top in [
        ("local", local_team, "stats_equipo_local", "top_jugadores_local"),
        ("visitante", away_team, "stats_equipo_visitante", "top_jugadores_visitante"),
    ]:
        stats = sofas.get(key_stats, {})
        top = sofas.get(key_top, {})

        rating = stats.get("rating_promedio_equipo") or (top.get("rating_promedio_equipo") if top else None)
        jugadores = top.get("top_jugadores", []) if top else []

        if rating or jugadores:
            partes.append(f"\n**{equipo}**")
            if rating:
                partes.append(f"  Rating promedio SofaScore: {rating}")
            if jugadores:
                partes.append("  Top 5 jugadores:")
                for j in jugadores:
                    r = j.get("rating", "?")
                    partes.append(f"    - {j['nombre']} ({j['posicion']}) Rating: {r}")

    if len(partes) == 1:
        return ""
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

    if detalle.get("lesiones"):
        partes.append("\n**Lesiones y suspensiones (SofaScore):**")
        local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
        away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"
        for l in detalle["lesiones"]:
            team_name = local_team if l["equipo"] == "local" else away_team
            partes.append(f"  - {team_name}: {l['jugador']} ({l['tipo']})")
        partes.append("IMPORTANTE: Estas lesiones/suspensiones de SofaScore son la fuente autoritativa de bajas.")

    return "\n".join(partes)


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
    partes.append("ATENCION: Estas alineaciones son la UNICA fuente de disponibilidad de jugadores. No uses datos de bajas de BSD.")

    for lado, equipo, data in [("local", local_team, alin.get("local", {})),
                                ("visitante", away_team, alin.get("visitante", {}))]:
        if not data.get("confirmada"):
            continue
        partes.append(f"\n**{equipo}** ({data.get('formacion', '?')})")

        titulares = data.get("titulares", [])
        if titulares:
            nombres = [f"{t['nombre']} ({t['posicion']})" for t in titulares]
            partes.append(f"  Titulares: {', '.join(nombres)}")

        suplentes = data.get("suplentes", [])
        if suplentes:
            nombres = [f"{t['nombre']} ({t['posicion']})" for t in suplentes]
            partes.append(f"  Suplentes: {', '.join(nombres)}")

    partes.append("\nIMPORTANTE: Esta es la alineacion oficial. Los jugadores listados aqui SON los que juegan. No asumas bajas adicionales de otras fuentes.")
    return "\n".join(partes)
