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
    "Champions League": 7,
    "Europa League": 679,
    "Copa Libertadores": 586,
    "Copa Sudamericana": 587,
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
    events = h2h_data.get("events", [])
    if not events:
        return {}

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
    for e in events[:5]:
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
    }


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
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures[pool.submit(obtener_evento_detalle, session, event_id)] = "detalle"
        futures[pool.submit(obtener_alineaciones, session, event_id)] = "lineups"
        futures[pool.submit(obtener_h2h_sofascore, session, event_id)] = "h2h"
        if home_team_id:
            futures[pool.submit(obtener_performance_equipo, session, home_team_id)] = "perf_local"
        if away_team_id:
            futures[pool.submit(obtener_performance_equipo, session, away_team_id)] = "perf_visitante"

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
