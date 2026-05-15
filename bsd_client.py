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
BSD_V2_BASE = "https://sports.bzzoiro.com/api/v2"
BSD_API_KEY = os.getenv("BSD_API_KEY")

# IDs de ligas y torneos según BSD
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

# IDs de las ligas y torneos objetivo para el análisis
TARGET_LEAGUE_IDS = [9, 3, 5, 4, 6, 1, 7, 8, 32, 33]

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
    Realiza una peticion GET a la BSD API v1.

    Args:
        endpoint: Ruta del endpoint (sin la base URL).
        params: Parametros de consulta opcionales.

    Returns:
        Respuesta JSON como diccionario.
    """
    url = f"{BSD_BASE_URL}/{endpoint}"
    response = requests.get(url, headers=_headers(), params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _get_v2(endpoint: str, params: dict = None) -> dict:
    """Peticion GET a la BSD API v2."""
    url = f"{BSD_V2_BASE}/{endpoint}"
    response = requests.get(url, headers=_headers(), params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def obtener_proximos_partidos(league_ids: list = None) -> list:
    """
    Obtiene los próximos partidos de las ligas especificadas via API v2.

    La BSD API v2 devuelve partidos con status=notstarted.
    Usa paginacion nativa (limit=200).

    Compatibilidad: agrega campo sintetico 'league' como dict {id, name}
    para mantener compatibilidad con codigo que espera el formato v1.

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
            data = _get_v2("events/", params={
                "league_id": liga_id,
                "date_from": today,
                "date_to": next_week,
                "status": "notstarted",
                "limit": 200,
            })
            resultados = data.get("results", [])
            league_name = LEAGUE_NAMES.get(liga_id, f"Liga {liga_id}")
            for p in resultados:
                p["_league_name"] = league_name
                # Compatibilidad v1: emular el objeto 'league'
                p["league"] = {"id": p.get("league_id", liga_id), "name": league_name}
            return resultados
        except requests.RequestException as e:
            print(f"  [AVISO] No se pudieron obtener partidos de "
                  f"{LEAGUE_NAMES.get(liga_id, f'Liga {liga_id}')}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=min(len(league_ids), 6)) as pool:
        futures = {pool.submit(_fetch_league, liga_id): liga_id for liga_id in league_ids}
        for future in as_completed(futures):
            todos_partidos.extend(future.result())

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
    Obtiene predicciones ML (CatBoost) de la BSD API v2.

    Args:
        match_id: ID del partido para filtrar una prediccion especifica.
        league_id: ID de liga para filtrar predicciones.

    Returns:
        Lista de predicciones en formato v2.
    """
    params = {"status": "upcoming", "limit": 200}
    if league_id:
        params["league_id"] = league_id

    data = _get_v2("predictions/", params=params)
    predicciones = data.get("results", [])

    if match_id:
        # Filtrado por event.id (v2 usa event.id, no event_id directo)
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

    def _parse_score(m):
        score_str = m.get("score", "0-0") or "0-0"
        parts = str(score_str).split("-")
        hg = int(parts[0]) if len(parts) >= 1 else 0
        ag = int(parts[1]) if len(parts) >= 2 else 0
        h2h_home = m.get("home", "")
        h2h_away = m.get("away", "")
        return hg, ag, h2h_home, h2h_away

    # Determinar el equipo local actual (de este partido)
    current_home = h2h.get("_home_team", "")
    if not current_home:
        # Inferir del primer enfrentamiento
        first_away = filtrados[0].get("away", "")
        first_home = filtrados[0].get("home", "")
        # fallback
        current_home = first_home

    total = len(filtrados)
    home_w = 0
    draws = 0
    away_w = 0
    home_g = 0
    away_g = 0

    for m in filtrados:
        hg, ag, h2h_home, h2h_away = _parse_score(m)
        hg = hg or 0
        ag = ag or 0
        # Si el "home" en el H2H es el equipo local actual
        if h2h_home and current_home and h2h_home.lower() == current_home.lower():
            home_g += hg
            away_g += ag
            if hg > ag:
                home_w += 1
            elif ag > hg:
                away_w += 1
            else:
                draws += 1
        elif h2h_away and current_home and h2h_away.lower() == current_home.lower():
            home_g += ag
            away_g += hg
            if ag > hg:
                home_w += 1
            elif hg > ag:
                away_w += 1
            else:
                draws += 1
        else:
            # No podemos determinar, asumir que home en H2H es local
            home_g += hg
            away_g += ag
            if hg > ag:
                home_w += 1
            elif ag > hg:
                away_w += 1
            else:
                draws += 1

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

    # Liga (objeto con ID)
    league_obj = evento.get("league") or {}
    resumen["league_id"] = league_obj.get("id")

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
    h2h["_home_team"] = evento.get("home_team", "")
    h2h_filtrado = _filtrar_h2h_reciente(h2h, anos_max=3)
    resumen["h2h"] = h2h_filtrado

    # Entrenadores
    home_coach = evento.get("home_coach") or {}
    away_coach = evento.get("away_coach") or {}
    resumen["home_coach_id"] = home_coach.get("id")
    resumen["away_coach_id"] = away_coach.get("id")
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
        resumen["referee_id"] = referee.get("id")
        resumen["arbitro"] = {
            "nombre": referee.get("name"),
            "nacionalidad": referee.get("country"),
            "id": referee.get("id"),
            "amarillas_carrera": referee.get("yellowCards"),
            "rojas_carrera": referee.get("redCards"),
            "partidos_carrera": referee.get("career_games"),
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

    # Bajas de BSD (para deteccion de conflictos con SofaScore)
    bajas_raw = evento.get("unavailable_players") or {}
    resumen["bajas_bsd"] = {
        "local": _clasificar_bajas(bajas_raw.get("home", [])),
        "visitante": _clasificar_bajas(bajas_raw.get("away", [])),
    }

    return resumen


def resumir_prediccion(prediccion: dict) -> dict:
    """
    Resume una prediccion ML de BSD para enviar a la IA.
    Soporta formato v1 (flat) y v2 (markets anidados).

    Args:
        prediccion: Diccionario de prediccion desde BSD (v1 o v2).

    Returns:
        Diccionario resumido con solo los campos relevantes.
    """
    if not prediccion:
        return {}

    # Detectar formato v2: tiene clave 'markets'
    if "markets" in prediccion:
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

    # Formato v1 (flat)
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
