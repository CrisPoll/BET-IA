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
from datetime import datetime

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
SOFASCORE_ONLY_LEAGUES = {
    "Liga 1 Peru": 34467,  # Liga 1, Apertura 2026
}

# Mapeo para standings: BSD league name → (unique_tournament_id, season_id)
STANDINGS_MAP = {
    "La Liga":                (8,   77559),
    "Premier League":         (17,  76986),
    "Bundesliga":             (35,  77333),
    "Brasileirão Serie A":    (325, 87678),
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
    # Extraer fecha YYYY-MM-DD
    try:
        if "T" in fecha_str:
            fecha_solo = fecha_str.split("T")[0]
        else:
            fecha_solo = fecha_str[:10]
    except (IndexError, TypeError):
        return None

    home_norm = normalizar_nombre(home_team)
    away_norm = normalizar_nombre(away_team)

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

    for ev in events:
        hs = ev.get("homeScore", {}).get("current", 0) or 0
        as_ = ev.get("awayScore", {}).get("current", 0) or 0
        wc = ev.get("winnerCode", 0)
        home_id = ev.get("homeTeam", {}).get("id")

        team_home = (home_id == team_id)
        gf = hs if team_home else as_
        gc = as_ if team_home else hs
        total_goles_favor += gf
        total_goles_contra += gc

        if (team_home and wc == 1) or (not team_home and wc == 2):
            puntos += 3
        elif wc == 3:
            puntos += 1

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
    }


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
        "form_performance": False,
    }

    futures = {}
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures[pool.submit(obtener_evento_detalle, session, event_id)] = "detalle"
        futures[pool.submit(obtener_alineaciones, session, event_id)] = "lineups"
        futures[pool.submit(obtener_h2h_sofascore, session, event_id)] = "h2h"
        if home_team_id:
            futures[pool.submit(obtener_performance_equipo, session, home_team_id)] = "perf_local"
        if away_team_id:
            futures[pool.submit(obtener_performance_equipo, session, away_team_id)] = "perf_visitante"
        # Standings (tabla de posiciones)
        liga = datos_bsd.get("liga", "")
        if liga:
            team_ids = set()
            if home_team_id:
                team_ids.add(home_team_id)
            if away_team_id:
                team_ids.add(away_team_id)
            futures[pool.submit(_obtener_standings, session, liga, team_ids)] = "standings"

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

    datos_bsd["_sofascore"] = enriquecido
    return datos_bsd


def _formatear_form_performance_para_prompt(datos: dict) -> str:
    """
    Formatea los datos de performance (form reciente) de SofaScore para el prompt.
    """
    sofas = datos.get("_sofascore", {})
    if not sofas.get("disponible") or not sofas.get("form_performance"):
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    partes = ["\n### FORMA RECIENTE (SofaScore Performance)"]
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

    # NOTA: SofaScore ya no expone lesiones via API (endpoint removido).
    # Las lesiones ahora vienen de BSD (seccion BAJAS BSD en el prompt).
    # Las alineaciones de SofaScore siguen siendo autoritativas para quien JUEGA.
    partes.append("\n**NOTA SOBRE LESIONES**: SofaScore ya no proporciona datos de lesiones via API.")
    partes.append("Usa los datos de BAJAS BSD como referencia de lesionados/suspendidos, pero con PRECAUCION.")
    partes.append("La ALINEACION de SofaScore es la unica fuente confiable de quien JUEGA.")

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


def _obtener_standings(session, liga: str, team_ids: set) -> dict:
    """Obtiene la tabla de posiciones y extrae los datos de los equipos del partido."""
    mapping = STANDINGS_MAP.get(liga)
    if not mapping:
        return {}
    uid, sid = mapping
    try:
        resp = session.get(f"{SOFASCORE_API}/unique-tournament/{uid}/season/{sid}/standings/total", timeout=15)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        result = {}
        for st in data.get("standings", []):
            rows = st.get("rows", [])
            for row in rows:
                team = row.get("team", {})
                tid = team.get("id")
                if tid in team_ids or len(rows) > 0:
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
                        "descripcion": ", ".join(row.get("descriptions", [])) if row.get("descriptions") else "",
                    }
                    result[team.get("name", "")] = info
            # Info general de la liga
            result["_liga_info"] = {
                "total_equipos": len(rows),
                "nombre_tabla": st.get("name", liga),
            }
        return result
    except Exception:
        return {}


def _formatear_standings_para_prompt(datos: dict) -> str:
    """Formatea los datos de standings/tabla de posiciones para el prompt."""
    ss = datos.get("_sofascore", {})
    standings = ss.get("standings", {})
    if not standings:
        return ""

    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    liga_info = standings.get("_liga_info", {})
    total_equipos = liga_info.get("total_equipos", "?")

    partes = [f"\n### TABLA DE POSICIONES ({liga_info.get('nombre_tabla', datos.get('liga', '?'))})"]
    partes.append(f"Total equipos: {total_equipos}")

    for team_name, label in [(local_team, "LOCAL"), (away_team, "VISITANTE")]:
        info = standings.get(team_name)
        if not info:
            # Fuzzy match
            for k, v in standings.items():
                if k.startswith("_liga"):
                    continue
                if team_name.lower() in k.lower() or k.lower() in team_name.lower():
                    info = v
                    break
        if info:
            desc = f" ({info.get('descripcion')})" if info.get("descripcion") else ""
            form_str = f" | Forma: {info.get('forma')}" if info.get("forma") else ""
            partes.append(
                f"\n**{team_name}** ({label}): "
                f"#{info.get('posicion', '?')} de {total_equipos} | "
                f"PTS: {info.get('puntos', '?')} | "
                f"PJ: {info.get('partidos_jugados', '?')} | "
                f"V: {info.get('victorias', '?')} | "
                f"E: {info.get('empates', '?')} | "
                f"D: {info.get('derrotas', '?')} | "
                f"GF: {info.get('goles_favor', '?')} | "
                f"GC: {info.get('goles_contra', '?')} | "
                f"DG: {info.get('diferencia', '?')}{desc}{form_str}"
            )
        else:
            partes.append(f"\n**{team_name}** ({label}): No encontrado en la tabla")

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
