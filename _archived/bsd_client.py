"""
Cliente para la BSD API (Bzzoiro Sports Data).

Proporciona funciones para obtener datos de partidos, estadísticas,
cuotas y predicciones de fútbol europeo.

Documentación: https://sports.bzzoiro.com/docs/
"""

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

BSD_BASE_URL = "https://sports.bzzoiro.com/api"
BSD_API_KEY = os.getenv("BSD_API_KEY")

# IDs de ligas y torneos según BSD
TOP_LEAGUES = {
    # Grandes ligas europeas
    "Premier League": 1,
    "Liga Portugal": 2,   # Fuera del top 5, incluida como referencia
    "La Liga": 3,
    "Serie A": 4,
    "Bundesliga": 5,
    "Ligue 1": 6,
    # Torneos internacionales
    "Champions League": 7,
    "Europa League": 8,
    # Torneos sudamericanos
    "Copa Libertadores": 32,
    "Copa Sudamericana": 33,
}

# IDs de las ligas y torneos objetivo para el análisis
TARGET_LEAGUE_IDS = [1, 3, 4, 5, 6, 7, 8, 32, 33]

# Mapeo de ID de liga a nombre
LEAGUE_NAMES = {v: k for k, v in TOP_LEAGUES.items()}


def _headers():
    """Encabezados de autenticación para la BSD API."""
    if not BSD_API_KEY:
        raise ValueError(
            "BSD_API_KEY no configurada. Agrégala en el archivo .env"
        )
    return {"Authorization": f"Token {BSD_API_KEY}"}


def _get(endpoint: str, params: dict = None) -> dict:
    """
    Realiza una petición GET a la BSD API.

    Args:
        endpoint: Ruta del endpoint (sin la base URL).
        params: Parámetros de consulta opcionales.

    Returns:
        Respuesta JSON como diccionario.
    """
    url = f"{BSD_BASE_URL}/{endpoint}"
    response = requests.get(url, headers=_headers(), params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def obtener_proximos_partidos(league_ids: list = None) -> list:
    """
    Obtiene los próximos partidos de las ligas especificadas.

    La BSD API devuelve por defecto partidos desde hace 3 horas
    hasta 7 días en el futuro. Filtramos solo los no iniciados.

    Args:
        league_ids: Lista de IDs de liga. Si es None, usa las 5 grandes.

    Returns:
        Lista de partidos próximos (status='notstarted').
    """
    if league_ids is None:
        league_ids = TARGET_LEAGUE_IDS

    today = date.today().isoformat()
    next_week = (date.today() + timedelta(days=10)).isoformat()

    todos_partidos = []

    def _fetch_league(liga_id):
        try:
            data = _get("events/", params={
                "date_from": today,
                "date_to": next_week,
                "league": liga_id,
                "tz": "Europe/Madrid",
            })
            resultados = data.get("results", [])
            league_matches = []
            for partido in resultados:
                if partido.get("status") == "notstarted":
                    partido["_league_name"] = LEAGUE_NAMES.get(
                        liga_id, f"Liga {liga_id}"
                    )
                    league_matches.append(partido)
            return league_matches
        except requests.RequestException as e:
            print(f"  [AVISO] No se pudieron obtener partidos de "
                  f"{LEAGUE_NAMES.get(liga_id, f'Liga {liga_id}')}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=min(len(league_ids), 6)) as pool:
        futures = {pool.submit(_fetch_league, liga_id): liga_id for liga_id in league_ids}
        for future in as_completed(futures):
            todos_partidos.extend(future.result())

    # Ordenar por fecha
    todos_partidos.sort(key=lambda p: p.get("event_date", ""))
    return todos_partidos


def obtener_detalle_partido(match_id: int) -> dict:
    """
    Obtiene el detalle completo de un partido.

    Incluye: form de equipos, head-to-head, lesiones/suspensiones,
    cuotas, entrenadores y más.

    Args:
        match_id: ID interno del partido en BSD.

    Returns:
        Diccionario con el detalle completo del partido.
    """
    return _get(f"events/{match_id}/", params={"tz": "Europe/Madrid"})


def obtener_predicciones(match_id: int = None, league_id: int = None) -> list:
    """
    Obtiene predicciones ML (CatBoost) de la BSD API.

    Args:
        match_id: ID del partido para filtrar una predicción específica.
        league_id: ID de liga para filtrar predicciones.

    Returns:
        Lista de predicciones.
    """
    params = {}
    if league_id:
        params["league"] = league_id

    data = _get("predictions/", params=params)
    predicciones = data.get("results", [])

    if match_id:
        predicciones = [
            p for p in predicciones
            if p.get("event", {}).get("id") == match_id
        ]

    return predicciones


def _clasificar_bajas(jugadores: list) -> dict:
    """
    Clasifica los jugadores no disponibles según su estado real.

    La API de BSD no siempre distingue entre lesionado, sancionado,
    duda o rotación. Esta función separa las bajas confirmadas de
    las dudosas para que la IA no asuma automáticamente que TODOS
    los jugadores listados están fuera del partido.

    Args:
        jugadores: Lista de jugadores (str o dict con name, status, reason).

    Returns:
        Dict con tres listas: confirmadas, dudas, desconocido.
    """
    confirmadas = []
    dudas = []
    desconocido = []

    for p in jugadores:
        if isinstance(p, dict):
            nombre = p.get("name", p.get("full_name", str(p)))
            estado = (p.get("status") or "").lower()
            motivo = (p.get("reason") or "").lower()

            if estado in ("injured", "suspended", "banned"):
                confirmadas.append({
                    "nombre": nombre,
                    "estado": estado,
                    "motivo": motivo,
                })
            elif estado in ("doubtful", "questionable", "doubt"):
                dudas.append({
                    "nombre": nombre,
                    "estado": estado,
                    "motivo": motivo,
                })
            else:
                dudas.append({
                    "nombre": nombre,
                    "estado": estado or "desconocido",
                    "motivo": motivo,
                })
        elif isinstance(p, str) and p.strip():
            desconocido.append(p)

    return {
        "confirmadas": confirmadas,
        "dudas": dudas,
        "desconocido": desconocido,
    }


def _filtrar_h2h_reciente(h2h: dict, anos_max: int = 3) -> dict:
    """
    Filtra los enfrentamientos H2H a los últimos N años.
    Evita que partidos de hace 5+ temporadas sesguen el análisis.

    Args:
        h2h: Diccionario H2H desde BSD.
        anos_max: Máximo de años hacia atrás a considerar.

    Returns:
        Dict con los mismos campos pero solo con partidos recientes.
    """
    from datetime import datetime, timedelta

    fecha_limite = datetime.utcnow() - timedelta(days=anos_max * 365)
    recientes = h2h.get("recent_matches") or []

    filtrados = []
    for m in recientes:
        fecha_str = m.get("date") or m.get("event_date") or m.get("match_date", "")
        try:
            if fecha_str:
                fecha_partido = datetime.fromisoformat(fecha_str.replace("Z", "+00:00").replace("+00:00", ""))
                if fecha_partido >= fecha_limite:
                    filtrados.append(m)
        except (ValueError, TypeError):
            filtrados.append(m)

    if not filtrados:
        return h2h

    total = len(filtrados)
    home_w = sum(1 for m in filtrados if m.get("home_goals", 0) > m.get("away_goals", 0))
    draws = sum(1 for m in filtrados if m.get("home_goals") == m.get("away_goals"))
    away_w = total - home_w - draws
    home_g = sum(m.get("home_goals", 0) for m in filtrados)
    away_g = sum(m.get("away_goals", 0) for m in filtrados)

    return {
        "total_partidos": total,
        "victorias_local": home_w,
        "empates": draws,
        "victorias_visitante": away_w,
        "goles_local": home_g,
        "goles_visitante": away_g,
        "promedio_goles": round((home_g + away_g) / total, 2) if total else None,
        "ultimos_enfrentamientos": filtrados[:5],
        "_filtrado_reciente": f"ultimos_{anos_max}_anos",
    }


def resumir_datos_partido(evento: dict) -> dict:
    """
    Resume los datos relevantes de un partido para enviar a la IA.

    Extrae solo la información necesaria para el análisis de value betting
    y la formatea de manera compacta para ahorrar tokens.

    Args:
        evento: Diccionario completo del evento desde BSD.

    Returns:
        Diccionario con los datos resumidos y formateados.
    """
    resumen = {}

    # Información básica del partido
    resumen["partido"] = f"{evento.get('home_team', '?')} vs {evento.get('away_team', '?')}"
    resumen["liga"] = evento.get("_league_name") or evento.get("league", {}).get("name", "Desconocida")
    resumen["fecha"] = evento.get("event_date", "Desconocida")

    # Equipos (objetos con ID)
    home_obj = evento.get("home_team_obj") or {}
    away_obj = evento.get("away_team_obj") or {}
    resumen["home_team_id"] = home_obj.get("id")
    resumen["away_team_id"] = away_obj.get("id")

    # Cuotas del bookmaker (1X2)
    resumen["cuotas"] = {
        "local": evento.get("odds_home"),
        "empate": evento.get("odds_draw"),
        "visitante": evento.get("odds_away"),
        "over_25": evento.get("odds_over_25"),
        "under_25": evento.get("odds_under_25"),
        "btts_si": evento.get("odds_btts_yes"),
        "btts_no": evento.get("odds_btts_no"),
    }

    # Forma reciente de los equipos
    home_form = evento.get("home_form") or {}
    away_form = evento.get("away_form") or {}
    resumen["forma_local"] = {
        "partidos_jugados": home_form.get("matches_played"),
        "forma_string": home_form.get("form_string"),  # ej: "WWDWL"
        "victorias": home_form.get("wins"),
        "empates": home_form.get("draws"),
        "derrotas": home_form.get("losses"),
        "goles_favor_ultimos_n": home_form.get("goals_scored_last_n"),
        "goles_contra_ultimos_n": home_form.get("goals_conceded_last_n"),
        "xG_promedio": home_form.get("avg_xg"),
        "xG_contra_promedio": home_form.get("avg_xg_conceded"),
        "porterias_cero": home_form.get("clean_sheets"),
        "ppg_local": home_form.get("home_ppg"),
        "ppg_visitante": away_form.get("away_ppg") if away_form else None,
        "goles_local_casa": home_form.get("home_goals_scored"),
        "goles_local_contra_casa": home_form.get("home_goals_conceded"),
        "conversion_gol": home_form.get("goal_conversion_rate"),
        "remates_promedio": home_form.get("avg_shots"),
        "remates_arco_promedio": home_form.get("avg_shots_on_target"),
        "amarillas_promedio": home_form.get("avg_yellow_cards"),
        "faltas_promedio": home_form.get("avg_fouls"),
    }
    resumen["forma_visitante"] = {
        "partidos_jugados": away_form.get("matches_played"),
        "forma_string": away_form.get("form_string"),
        "victorias": away_form.get("wins"),
        "empates": away_form.get("draws"),
        "derrotas": away_form.get("losses"),
        "goles_favor_ultimos_n": away_form.get("goals_scored_last_n"),
        "goles_contra_ultimos_n": away_form.get("goals_conceded_last_n"),
        "xG_promedio": away_form.get("avg_xg"),
        "xG_contra_promedio": away_form.get("avg_xg_conceded"),
        "porterias_cero": away_form.get("clean_sheets"),
        "ppg_local": home_form.get("home_ppg") if home_form else None,
        "ppg_visitante": away_form.get("away_ppg"),
        "goles_visitante_fuera": away_form.get("away_goals_scored"),
        "goles_visitante_contra_fuera": away_form.get("away_goals_conceded"),
        "conversion_gol": away_form.get("goal_conversion_rate"),
        "remates_promedio": away_form.get("avg_shots"),
        "remates_arco_promedio": away_form.get("avg_shots_on_target"),
        "amarillas_promedio": away_form.get("avg_yellow_cards"),
        "faltas_promedio": away_form.get("avg_fouls"),
    }

    # Head to Head histórico (filtrado a últimos 3 años)
    h2h = evento.get("head_to_head") or {}
    h2h_filtrado = _filtrar_h2h_reciente(h2h, anos_max=3)
    resumen["h2h"] = h2h_filtrado

    # Entrenadores
    home_coach = evento.get("home_coach") or {}
    away_coach = evento.get("away_coach") or {}
    resumen["entrenador_local"] = {
        "nombre": home_coach.get("name"),
        "formacion": home_coach.get("preferred_formation"),
        "perfil": home_coach.get("profile"),
        "presion": home_coach.get("pressing_intensity"),
        "linea_defensiva": home_coach.get("defensive_line"),
    }
    resumen["entrenador_visitante"] = {
        "nombre": away_coach.get("name"),
        "formacion": away_coach.get("preferred_formation"),
        "perfil": away_coach.get("profile"),
        "presion": away_coach.get("pressing_intensity"),
        "linea_defensiva": away_coach.get("defensive_line"),
    }

    # Arbitro
    referee = evento.get("referee")
    if isinstance(referee, dict):
        resumen["arbitro"] = {
            "nombre": referee.get("name"),
            "nacionalidad": referee.get("nationality"),
            "id": referee.get("id"),
        }
    elif isinstance(referee, str) and referee.strip():
        resumen["arbitro"] = {"nombre": referee}
    else:
        resumen["arbitro"] = None

    # Estadio
    venue = evento.get("venue") or {}
    if venue:
        resumen["estadio"] = {
            "nombre": venue.get("name"),
            "ciudad": venue.get("city"),
            "capacidad": venue.get("capacity"),
        }

    return resumen


def resumir_prediccion(prediccion: dict) -> dict:
    """
    Resume una predicción ML de BSD para enviar a la IA.

    Args:
        prediccion: Diccionario de predicción desde BSD.

    Returns:
        Diccionario resumido con solo los campos relevantes.
    """
    if not prediccion:
        return {}

    return {
        "prob_local": prediccion.get("prob_home_win"),
        "prob_empate": prediccion.get("prob_draw"),
        "prob_visitante": prediccion.get("prob_away_win"),
        "resultado_predicho": prediccion.get("predicted_result"),
        "xG_local": prediccion.get("expected_home_goals"),
        "xG_visitante": prediccion.get("expected_away_goals"),
        "prob_over_25": prediccion.get("prob_over_25"),
        "prob_btts": prediccion.get("prob_btts_yes"),
        "confianza_modelo": prediccion.get("confidence"),
        "marcador_probable": prediccion.get("most_likely_score"),
        "recomienda_over_25": prediccion.get("over_25_recommend"),
        "recomienda_btts": prediccion.get("btts_recommend"),
        "version_modelo": prediccion.get("model_version"),
    }


def depurar_partido(datos_resumidos: dict, prediccion_resumida: dict = None):
    """
    Vuelca todos los datos recibidos del partido en formato legible.

    Args:
        datos_resumidos: Datos resumidos desde resumir_datos_partido().
        prediccion_resumida: Prediccion ML resumida (opcional).
    """
    import json

    sep = "\n" + "─" * 66 + "\n"
    print(sep + "  📦 DATOS RECIBIDOS — BSD API (resumen)" + sep)

    print(f"  ⚽ Partido:      {datos_resumidos.get('partido', '?')}")
    print(f"  🏟  Liga:         {datos_resumidos.get('liga', '?')}")
    print(f"  📅 Fecha:        {datos_resumidos.get('fecha', '?')}")

    # Estadio
    estadio = datos_resumidos.get("estadio")
    if estadio:
        print(f"  🏟  Estadio:      {estadio.get('nombre', '?')} ({estadio.get('ciudad', '?')}, cap: {estadio.get('capacidad', '?')})")

    # Arbitro
    arb = datos_resumidos.get("arbitro")
    if arb:
        extra = f" ({arb.get('nacionalidad', '')})" if arb.get('nacionalidad') else ""
        print(f"  👨‍⚖️  Arbitro:      {arb.get('nombre', '?')}{extra}")
    else:
        print(f"  👨‍⚖️  Arbitro:      NO ASIGNADO (se asignara mas cerca del partido)")

    # Cuotas
    print(f"\n  💰 CUOTAS:")
    c = datos_resumidos.get("cuotas", {})
    for k, v in c.items():
        if v is not None:
            print(f"     {k}: {v}")

    fuente_odds = datos_resumidos.get("_odds_api")
    if fuente_odds and fuente_odds.get("disponible"):
        print(f"     ⚡ Fuente: {fuente_odds.get('fuente')} ({fuente_odds.get('bookmakers')})")
        print(f"     ⚡ Actualizadas: {fuente_odds.get('fecha_consulta')}")

    # Forma local y visitante
    ss = datos_resumidos.get("_sofascore", {})
    for lado, key, idx in [("🏠 LOCAL", "forma_local", "stats_equipo_local"), ("🚩 VISITANTE", "forma_visitante", "stats_equipo_visitante")]:
        f = datos_resumidos.get(key, {})
        if not f:
            continue
        print(f"\n  {lado}: {f.get('forma_string', '?')} ({f.get('victorias', 0)}V/{f.get('empates', 0)}E/{f.get('derrotas', 0)}D)")
        print(f"     xG prom: {f.get('xG_promedio', '?')}  |  xG contra: {f.get('xG_contra_promedio', '?')}")
        print(f"     Goles (ult N): {f.get('goles_favor_ultimos_n', '?')}F / {f.get('goles_contra_ultimos_n', '?')}C")
        print(f"     Clean sheets: {f.get('porterias_cero', '?')}  |  PPGCasa: {f.get('ppg_local', '?')} / PPGFuera: {f.get('ppg_visitante', '?')}")

        # Stats de SofaScore (fuente principal para remates/tiros/tarjetas)
        sf_stats = ss.get(idx, {}) if ss.get("disponible") else {}
        if sf_stats:
            print(f"     [SofaScore] Remates prom: {sf_stats.get('remates_promedio_sf', '?')}  |  Al arco: {sf_stats.get('remates_arco_promedio_sf', '?')}")
            print(f"     [SofaScore] Amarillas prom: {sf_stats.get('amarillas_promedio_sf', '?')}  |  Faltas prom: {sf_stats.get('faltas_promedio_sf', '?')}")
            if sf_stats.get('corners_promedio_sf'):
                print(f"     [SofaScore] Corners prom: {sf_stats.get('corners_promedio_sf')}")
            if sf_stats.get('posesion_promedio_sf'):
                print(f"     [SofaScore] Posesion prom: {sf_stats.get('posesion_promedio_sf')}%")
        else:
            # Fallback a BSD si no hay SofaScore stats
            print(f"     [BSD] Remates prom: {f.get('remates_promedio', '?')}  |  Al arco: {f.get('remates_arco_promedio', '?')}")
            print(f"     [BSD] Amarillas prom: {f.get('amarillas_promedio', '?')}  |  Faltas prom: {f.get('faltas_promedio', '?')}")

    # H2H
    h2h = datos_resumidos.get("h2h", {})
    if h2h:
        print(f"\n  📊 HEAD-TO-HEAD: {h2h.get('total_partidos', '?')} partidos")
        print(f"     Local: {h2h.get('victorias_local', '?')}V  Empates: {h2h.get('empates', '?')}  Visitante: {h2h.get('victorias_visitante', '?')}V")
        print(f"     Goles: {h2h.get('goles_local', '?')}L / {h2h.get('goles_visitante', '?')}V  |  Prom: {h2h.get('promedio_goles', '?')}")
        print(f"     Filtro: {h2h.get('_filtrado_reciente', '?')}")

    # ALINEACION SofaScore (fuente autoritativa)
    ss = datos_resumidos.get("_sofascore", {})
    alin = ss.get("alineaciones", {}) if ss.get("disponible") else {}
    if alin.get("local", {}).get("confirmada"):
        print(f"\n  📋 ALINEACION CONFIRMADA (SofaScore)")
        loc = alin["local"]
        vis = alin["visitante"]
        home_team = datos_resumidos.get("partido", "").split(" vs ")[0] if " vs " in datos_resumidos.get("partido", "") else "Local"
        away_team = datos_resumidos.get("partido", "").split(" vs ")[1] if " vs " in datos_resumidos.get("partido", "") else "Visitante"

        for equipo_nombre, data, lado in [(home_team, loc, "🏠"), (away_team, vis, "🚩")]:
            print(f"\n  {lado} {equipo_nombre} ({data.get('formacion', '?')})")
            titulares = data.get("titulares", [])
            for i, t in enumerate(titulares, 1):
                print(f"     {i:>2}. {t['nombre']} ({t['posicion']}) #{t['dorsal']}")
            suplentes = data.get("suplentes", [])
            if suplentes:
                nombres_sup = [f"{s['nombre']} ({s['posicion']})" for s in suplentes[:5]]
                print(f"     SUP: {', '.join(nombres_sup)}{' (+' + str(len(suplentes)-5) + ')' if len(suplentes) > 5 else ''}")
    elif ss.get("disponible"):
        print(f"\n  📋 SOFASCORE: partido encontrado pero alineaciones NO disponibles aun")
        print(f"     (suelen salir ~1h antes del partido)")
    else:
        print(f"\n  📋 SOFASCORE: no disponible ({ss.get('error', '?')})")
        print(f"     ⚠️  Sin alineacion de referencia.")

    # Entrenadores
    for lado, key in [("🏠 LOCAL", "entrenador_local"), ("🚩 VISITANTE", "entrenador_visitante")]:
        e = datos_resumidos.get(key, {})
        if e.get("nombre"):
            print(f"\n  👔 DT {lado}: {e['nombre']} ({e.get('formacion', '?')}, {e.get('perfil', '?')})")

    # SofaScore - datos adicionales
    if ss.get("disponible"):
        extras_sf = []
        if ss.get("stats_equipo_local"):
            extras_sf.append("stats equipo (remates, tarjetas, corners)")
        if ss.get("stats_equipo_visitante"):
            pass  # ya contado con stats_equipo_local
        if ss.get("estadisticas_disponibles"):
            extras_sf.append("stats post-partido")
        if ss.get("h2h"):
            extras_sf.append("H2H SofaScore")
        if extras_sf:
            print(f"\n  🔍 SOFASCORE (datos adicionales): {', '.join(extras_sf)}")

    # Prediccion ML
    if prediccion_resumida:
        print(f"\n  🤖 PREDICCION ML (BSD CatBoost):")
        print(f"     1X2: L={prediccion_resumida.get('prob_local')}% / E={prediccion_resumida.get('prob_empate')}% / V={prediccion_resumida.get('prob_visitante')}%")
        print(f"     xG: {prediccion_resumida.get('xG_local')}L - {prediccion_resumida.get('xG_visitante')}V")
        print(f"     Over 2.5: {prediccion_resumida.get('prob_over_25')}%  |  BTTS: {prediccion_resumida.get('prob_btts')}%")
        print(f"     Marcador probable: {prediccion_resumida.get('marcador_probable')}")
        print(f"     Confianza: {prediccion_resumida.get('confianza_modelo')}")

    print(sep)


if __name__ == "__main__":
    # Prueba rápida del cliente
    print("=== Probando BSD Client ===\n")
    try:
        partidos = obtener_proximos_partidos()
        print(f"Próximos partidos encontrados: {len(partidos)}")
        for i, p in enumerate(partidos[:5], 1):
            print(f"  {i}. {p.get('home_team')} vs {p.get('away_team')} "
                  f"({p.get('_league_name')}) - {p.get('event_date')}")
    except Exception as e:
        print(f"Error: {e}")
