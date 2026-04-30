"""
Cliente para enriquecer datos de partidos via 365Score API (no-oficial).

365score.com tiene una API JSON interna que puede proporcionar datos
complementarios a BSD y SofaScore: lineups, stats de equipo, noticias.

Endpoints probados (mejor esfuerzo, pueden cambiar):
  - /soccer/matches?date=YYYY-MM-DD  → buscar partidos por fecha
  - /soccer/match/{id}               → detalle del partido
"""

import time
from datetime import datetime

from dotenv import load_dotenv

from utils import normalizar_nombre

load_dotenv()

SCORE365_API = "https://api.365scores.com"

SCORE365_HEADERS = {
    "Origin": "https://www.365scores.com",
    "Referer": "https://www.365scores.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


def _crear_sesion():
    """
    Crea una sesion curl_cffi para 365score.

    Returns:
        Sesion o None si curl_cffi no esta instalado.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError:
        return None

    session = cffi_requests.Session()
    session.headers.update(SCORE365_HEADERS)
    return session


def _buscar_partido(session, fecha_str: str, home_team: str, away_team: str) -> dict | None:
    """
    Busca un partido en 365score por fecha y equipos.

    Args:
        session: Sesion curl_cffi activa.
        fecha_str: Fecha ISO del partido.
        home_team: Nombre equipo local.
        away_team: Nombre equipo visitante.

    Returns:
        Dict del partido o None.
    """
    if not session:
        return None

    try:
        if "T" in fecha_str:
            fecha_solo = fecha_str.split("T")[0]
        else:
            fecha_solo = fecha_str[:10]
    except (IndexError, TypeError):
        return None

    home_norm = normalizar_nombre(home_team)
    away_norm = normalizar_nombre(away_team)

    # Intentar endpoint de busqueda por fecha
    try:
        url = f"{SCORE365_API}/soccer/matches?date={fecha_solo}&langId=31"
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    matches = data.get("matches") or data.get("games") or data.get("competitions") or []
    if isinstance(matches, dict):
        matches = []
        for comp in data.get("competitions", []):
            matches.extend(comp.get("games") or comp.get("matches") or [])

    for m in matches:
        m_home = normalizar_nombre(m.get("homeCompetitor", {}).get("name") or m.get("homeName", ""))
        m_away = normalizar_nombre(m.get("awayCompetitor", {}).get("name") or m.get("awayName", ""))
        if (home_norm in m_home or m_home in home_norm) and (away_norm in m_away or m_away in away_norm):
            return m

    return None


def _obtener_match_detail(session, match_id: int) -> dict | None:
    """
    Obtiene el detalle de un partido de 365score.

    Returns:
        Dict con datos del partido o None.
    """
    if not session or not match_id:
        return None
    try:
        url = f"{SCORE365_API}/soccer/match/{match_id}?langId=31"
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if data else None
    except Exception:
        return None


def _extraer_stats_equipo_365(match_detail: dict, side: str) -> dict:
    """
    Extrae estadisticas de equipo desde el detalle de partido de 365score.

    Args:
        match_detail: Respuesta del endpoint /soccer/match/{id}.
        side: "home" o "away".

    Returns:
        Dict con estadisticas extraidas.
    """
    resultado = {}
    team_data = match_detail.get(f"{side}Team") or match_detail.get(f"{side}Competitor") or {}

    # Stats de temporada (si vienen en el match detail)
    stats_block = team_data.get("statistics") or team_data.get("stats") or match_detail.get(f"{side}Stats") or {}

    if not stats_block and isinstance(match_detail.get("statistics"), dict):
        stats_block = match_detail["statistics"].get(side, {})

    for key, valor in stats_block.items() if isinstance(stats_block, dict) else []:
        key_lower = key.lower()
        try:
            v = round(float(valor), 2)
        except (ValueError, TypeError):
            v = valor

        if "shot" in key_lower and "on" not in key_lower:
            resultado["remates_promedio_365"] = v
        if "shot" in key_lower and ("on" in key_lower or "target" in key_lower):
            resultado["remates_arco_promedio_365"] = v
        if "yellow" in key_lower:
            resultado["amarillas_promedio_365"] = v
        if "foul" in key_lower:
            resultado["faltas_promedio_365"] = v
        if "corner" in key_lower:
            resultado["corners_promedio_365"] = v

    return resultado


def enriquecer_datos_partido(datos_bsd: dict) -> dict:
    """
    Intenta enriquecer datos con 365score.

    Busca el partido por fecha + equipos y extrae estadisticas
    de equipo si estan disponibles.

    Args:
        datos_bsd: Datos resumidos del partido (con _sofascore ya poblado).

    Returns:
        El mismo dict con clave "_365score" agregada.
    """
    home_team = datos_bsd.get("partido", "").split(" vs ")[0] if " vs " in datos_bsd.get("partido", "") else ""
    away_team = datos_bsd.get("partido", "").split(" vs ")[1] if " vs " in datos_bsd.get("partido", "") else ""
    fecha = datos_bsd.get("fecha", "")

    if not home_team or not away_team:
        datos_bsd["_365score"] = {"disponible": False, "error": "Faltan nombres de equipos"}
        return datos_bsd

    session = _crear_sesion()
    if not session:
        datos_bsd["_365score"] = {"disponible": False, "error": "curl_cffi no instalado"}
        return datos_bsd

    partido = _buscar_partido(session, fecha, home_team, away_team)
    if not partido:
        datos_bsd["_365score"] = {"disponible": False, "error": "Partido no encontrado en 365score"}
        return datos_bsd

    match_id = partido.get("id") or partido.get("gameId")
    enriquecido = {
        "disponible": True,
        "match_id": match_id,
    }

    # Intentar detalle del partido con stats de equipo
    if match_id:
        time.sleep(0.5)
        detail = _obtener_match_detail(session, match_id)
        if detail:
            enriquecido["stats_equipo_local_365"] = _extraer_stats_equipo_365(detail, "home")
            enriquecido["stats_equipo_visitante_365"] = _extraer_stats_equipo_365(detail, "away")

    datos_bsd["_365score"] = enriquecido
    return datos_bsd


def _formatear_stats_365_para_prompt(datos: dict) -> str:
    """
    Formatea datos de 365score para incluir en el prompt de la IA.

    Returns:
        String formateado o cadena vacia.
    """
    sc = datos.get("_365score", {})
    if not sc.get("disponible"):
        return ""

    partes = []
    local_team = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else "Local"
    away_team = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else "Visitante"

    for equipo, key in [(local_team, "stats_equipo_local_365"),
                          (away_team, "stats_equipo_visitante_365")]:
        stats = sc.get(key, {})
        if not stats:
            continue
        partes.append(f"\n**{equipo} — Estadisticas 365score (adicional):**")
        for k, v in stats.items():
            partes.append(f"  - {k}: {v}")

    if partes:
        partes.insert(0, "\n### DATOS 365SCORE (fuente adicional)")
        partes.append("\nNOTA: Datos de 365score son complementarios. Si contradicen SofaScore, prefiere SofaScore.")
        return "\n".join(partes)
    return ""
