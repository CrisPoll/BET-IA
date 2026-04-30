"""
Cliente para obtener cuotas frescas via The Odds API (free tier).

La API gratuita permite 500 req/mes (~16 por dia) y cubre las
principales ligas europeas. Usamos las cuotas de Pinnacle/Bet365
por ser las mas ajustadas al mercado real.

Registro gratuito: https://the-odds-api.com/#get-access
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

# Mapeo de BSD league name → The Odds API sport key (solo ligas cubiertas gratis)
BSD_TO_ODDS_SPORT = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Champions League": "soccer_uefa_champs_league",
    "Europa League": "soccer_uefa_europa_league",
    # Libertadores y Sudamericana NO estan en el free tier de The Odds API
}

# Bookmakers preferidos: Pinnacle (mejores cuotas) + Bet365 (mercado mas liquido)
BOOKMAKERS = "pinnacle,bet365"
# Para minimizar uso de requests, pedimos todos los mercados de una vez
MARKETS = "h2h,totals,both_teams_to_score"
# Region EU para bookmakers europeos
REGIONS = "eu"

# Cache interno para no repetir requests en la misma ejecucion
_cache_cuotas: dict = {}
_cache_timestamp: float = 0.0
_CACHE_TTL = 300  # 5 minutos


def _obtener_eventos_deportivos(sport_key: str) -> list:
    """
    Obtiene todos los eventos proximos para un deporte/liga.

    Args:
        sport_key: Clave del deporte en The Odds API (ej: "soccer_epl").

    Returns:
        Lista de eventos con cuotas.
    """
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY no configurada en .env")

    url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code == 401:
        raise ValueError(
            "ODDS_API_KEY invalida. Obten una key gratis en https://the-odds-api.com"
        )
    if resp.status_code == 422:
        # Limite de requests excedido (free tier: 500/mes)
        remaining = resp.headers.get("x-requests-remaining", "?")
        raise RuntimeError(
            f"Limite de requests de The Odds API excedido. "
            f"Requests restantes este mes: {remaining}"
        )
    resp.raise_for_status()

    # Mostrar cuota de uso para monitoreo
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    if remaining != "?":
        print(f"  [Odds API] Requests restantes este mes: {remaining}/{int(remaining) + int(used) if used != '?' else '?'}")

    return resp.json()


def _buscar_cuotas_por_equipos(
    eventos: list, home_team: str, away_team: str
) -> dict | None:
    """
    Busca en la lista de eventos de The Odds API el partido que
    coincide con los nombres de equipos.

    Args:
        eventos: Lista de eventos desde The Odds API.
        home_team: Nombre del equipo local.
        away_team: Nombre del equipo visitante.

    Returns:
        Dict con cuotas del partido o None.
    """
    home_lower = home_team.lower().strip()
    away_lower = away_team.lower().strip()

    for event in eventos:
        e_home = event.get("home_team", "").lower().strip()
        e_away = event.get("away_team", "").lower().strip()

        if (home_lower in e_home or e_home in home_lower) and \
           (away_lower in e_away or e_away in away_lower):
            return event

    return None


def _extraer_cuotas_h2h(bookmakers: list, home_team: str, away_team: str) -> dict | None:
    """
    Extrae las mejores cuotas 1X2 de los bookmakers.

    Prioriza Pinnacle, luego Bet365.

    Returns:
        Dict con {local, empate, visitante, bookmaker}.
    """
    if not bookmakers:
        return None

    def _parse_outcomes(outcomes):
        """Parsea los outcomes de h2h en {local, empate, visitante}."""
        result = {}
        for o in outcomes:
            name = o.get("name", "").strip()
            price = o.get("price")
            name_lower = name.lower()
            home_lower = home_team.lower()
            away_lower = away_team.lower()
            if name_lower == "draw":
                result["empate"] = price
            elif home_lower in name_lower or name_lower in home_lower:
                result["local"] = price
            elif away_lower in name_lower or name_lower in away_lower:
                result["visitante"] = price
        return result if len(result) == 3 else None

    # Preferir Pinnacle si esta disponible
    for bm in bookmakers:
        if bm.get("key") == "pinnacle":
            for m in bm.get("markets", []):
                if m.get("key") == "h2h":
                    cuotas = _parse_outcomes(m.get("outcomes", []))
                    if cuotas:
                        cuotas["bookmaker"] = "Pinnacle"
                        return cuotas

    # Fallback: cualquier bookmaker
    for bm in bookmakers:
        for m in bm.get("markets", []):
            if m.get("key") == "h2h":
                cuotas = _parse_outcomes(m.get("outcomes", []))
                if cuotas:
                    cuotas["bookmaker"] = bm.get("title", "Desconocido")
                    return cuotas

    return None


def _extraer_cuotas_totales(bookmakers: list) -> dict | None:
    """
    Extrae cuotas de Over/Under 2.5 de los bookmakers.

    Returns:
        Dict con {over_25, under_25, bookmaker}.
    """
    if not bookmakers:
        return None

    for bm in bookmakers:
        markets = bm.get("markets", [])
        for m in markets:
            if m.get("key") == "totals":
                outcomes = m.get("outcomes", [])
                over_price = None
                under_price = None
                for o in outcomes:
                    if o.get("name") == "Over" and o.get("point") == 2.5:
                        over_price = o.get("price")
                    elif o.get("name") == "Under" and o.get("point") == 2.5:
                        under_price = o.get("price")
                if over_price and under_price:
                    return {
                        "over_25": over_price,
                        "under_25": under_price,
                        "bookmaker": bm.get("title", "Desconocido"),
                    }
    return None


def _extraer_cuotas_btts(bookmakers: list) -> dict | None:
    """
    Extrae cuotas BTTS (Both Teams to Score).

    Returns:
        Dict con {btts_si, btts_no, bookmaker}.
    """
    if not bookmakers:
        return None

    for bm in bookmakers:
        markets = bm.get("markets", [])
        for m in markets:
            if m.get("key") == "both_teams_to_score":
                outcomes = m.get("outcomes", [])
                yes_price = None
                no_price = None
                for o in outcomes:
                    if o.get("name") == "Yes":
                        yes_price = o.get("price")
                    elif o.get("name") == "No":
                        no_price = o.get("price")
                if yes_price and no_price:
                    return {
                        "btts_si": yes_price,
                        "btts_no": no_price,
                        "bookmaker": bm.get("title", "Desconocido"),
                    }
    return None


def obtener_cuotas_frescas(datos_bsd: dict) -> dict | None:
    """
    Obtiene cuotas actualizadas desde The Odds API.

    Args:
        datos_bsd: Datos resumidos del partido desde BSD.

    Returns:
        Dict con las cuotas frescas de los 3 mercados,
        o None si la liga no esta soportada o falla la API.

    Raises:
        ValueError: Si ODDS_API_KEY no esta configurada.
    """
    if not ODDS_API_KEY:
        return None

    liga = datos_bsd.get("liga", "")
    sport_key = BSD_TO_ODDS_SPORT.get(liga)
    if not sport_key:
        # Liga no soportada en el free tier (ej: Libertadores)
        return None

    home_team = datos_bsd.get("partido", "").split(" vs ")[0] if " vs " in datos_bsd.get("partido", "") else ""
    away_team = datos_bsd.get("partido", "").split(" vs ")[1] if " vs " in datos_bsd.get("partido", "") else ""

    if not home_team or not away_team:
        return None

    # Check cache
    global _cache_cuotas, _cache_timestamp
    cache_key = sport_key
    now = time.time()
    if cache_key in _cache_cuotas and (now - _cache_timestamp) < _CACHE_TTL:
        eventos = _cache_cuotas[cache_key]
    else:
        eventos = _obtener_eventos_deportivos(sport_key)
        _cache_cuotas[cache_key] = eventos
        _cache_timestamp = now

    partido = _buscar_cuotas_por_equipos(eventos, home_team, away_team)
    if not partido:
        return None

    bookmakers = partido.get("bookmakers", [])

    cuotas_h2h = _extraer_cuotas_h2h(bookmakers, home_team, away_team)
    cuotas_totales = _extraer_cuotas_totales(bookmakers)
    cuotas_btts = _extraer_cuotas_btts(bookmakers)

    if not cuotas_h2h:
        return None

    frescas = {
        "fuente": "The Odds API",
        "fecha_consulta": time.strftime("%Y-%m-%d %H:%M:%S"),
        "local": cuotas_h2h.get("local"),
        "empate": cuotas_h2h.get("empate"),
        "visitante": cuotas_h2h.get("visitante"),
        "bookmaker_h2h": cuotas_h2h.get("bookmaker"),
    }

    if cuotas_totales:
        frescas["over_25"] = cuotas_totales.get("over_25")
        frescas["under_25"] = cuotas_totales.get("under_25")
        frescas["bookmaker_totales"] = cuotas_totales.get("bookmaker")

    if cuotas_btts:
        frescas["btts_si"] = cuotas_btts.get("btts_si")
        frescas["btts_no"] = cuotas_btts.get("btts_no")
        frescas["bookmaker_btts"] = cuotas_btts.get("bookmaker")

    return frescas


def enriquecer_cuotas(datos_bsd: dict) -> dict:
    """
    Intenta reemplazar las cuotas de BSD con cuotas frescas de The Odds API.

    Si las cuotas frescas estan disponibles, las inyecta en datos_bsd["cuotas"]
    y deja las originales en datos_bsd["_cuotas_bsd_original"].

    Args:
        datos_bsd: Dict con datos del partido (mutable, se modifica in-place).

    Returns:
        El mismo dict (modificado in-place).
    """
    try:
        cuotas_frescas = obtener_cuotas_frescas(datos_bsd)
    except Exception as e:
        datos_bsd["_odds_api"] = {"disponible": False, "error": str(e)}
        return datos_bsd

    if not cuotas_frescas:
        datos_bsd["_odds_api"] = {"disponible": False, "error": "Liga no soportada o partido no encontrado"}
        return datos_bsd

    # Guardar cuotas originales de BSD
    datos_bsd["_cuotas_bsd_original"] = datos_bsd.get("cuotas", {}).copy()

    # Inyectar cuotas frescas (solo las que tenemos)
    cuotas_actuales = datos_bsd.get("cuotas", {})
    for key in ["local", "empate", "visitante", "over_25", "under_25", "btts_si", "btts_no"]:
        if key in cuotas_frescas and cuotas_frescas[key] is not None:
            cuotas_actuales[key] = cuotas_frescas[key]

    datos_bsd["cuotas"] = cuotas_actuales
    datos_bsd["_odds_api"] = {
        "disponible": True,
        "fuente": cuotas_frescas["fuente"],
        "fecha_consulta": cuotas_frescas["fecha_consulta"],
        "bookmakers": ", ".join(
            filter(None, [
                cuotas_frescas.get("bookmaker_h2h"),
                cuotas_frescas.get("bookmaker_totales"),
                cuotas_frescas.get("bookmaker_btts"),
            ])
        ),
    }

    return datos_bsd


def _formatear_odds_para_prompt(datos: dict) -> str:
    """
    Genera una nota para el prompt indicando si las cuotas fueron actualizadas.

    Args:
        datos: Dict completo de datos (con clave _odds_api).

    Returns:
        String formateado para anadir al prompt.
    """
    odds_info = datos.get("_odds_api", {})
    if not odds_info.get("disponible"):
        return ""

    return (
        "\n### CUOTAS ACTUALIZADAS\n"
        f"NOTA: Las cuotas mostradas arriba fueron actualizadas via {odds_info.get('fuente')} "
        f"el {odds_info.get('fecha_consulta', '?')}. "
        f"Bookmakers: {odds_info.get('bookmakers', '?')}. "
        "Estas cuotas reflejan el mercado REAL al momento del analisis.\n"
    )
