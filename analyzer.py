"""
Analizador de Value Betting usando DeepSeek via OpenRouter.

Toma los datos de un partido (desde BSD + SofaScore) y las predicciones ML,
los envía a DeepSeek para obtener un análisis experto de value betting.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from sofascore_client import (
    _formatear_alineaciones_para_prompt,
    _formatear_h2h_sofascore_para_prompt,
    _formatear_detalle_evento_para_prompt,
    _formatear_form_performance_para_prompt,
    _formatear_standings_para_prompt,
    _formatear_players_stats_para_prompt,
    _formatear_shotmap_para_prompt,
    _formatear_momentum_para_prompt,
    _formatear_avg_positions_para_prompt,
    _formatear_incidents_para_prompt,
    _formatear_match_statistics_para_prompt,
    _formatear_team_season_stats_para_prompt,
)
from flashscore_client import (
    _formatear_flashscore_para_prompt,
    _formatear_standings_flashscore_para_prompt,
)
from betsafe_client import _formatear_cuotas_betsafe_para_prompt
from valuestats_client import formatear_arbitro_valuestats_para_prompt
from whoscored_client import formatear_arbitro_whoscored_para_prompt
from bsd_client_v2 import (
    resumir_stats_v2_para_prompt,
    resumir_metadata_v2_para_prompt,
    resumir_player_stats_v2_para_prompt,
    resumir_standings_v2_para_prompt,
    resumir_squads_v2_para_prompt,
    resumir_manager_v2_para_prompt,
    resumir_arbitro_v2_para_prompt,
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME = "deepseek/deepseek-v4-pro"
MAX_TOKENS = 100000  # Holgado: incluye reasoning interno + respuesta visible
TEMPERATURE = 0.3  # Baja temperatura para análisis más consistente

SYSTEM_PROMPT = """Eres un experto analista de Value Betting en futbol, con mentalidad critica y esceptica ante datos imperfectos.

PROCESO DE ANALISIS:
1. Estima probabilidad real (%) de cada outcome para 1X2, BTTS, Over/Under 2.5
   usando forma reciente, xG, estilos tacticos y contexto ACTUAL de la temporada.
2. Como CONTEXTO ADICIONAL, analiza tambien:
   - Tiros totales y tiros al arco (proyecta si el partido sera de muchos/pocos disparos)
   - Tarjetas amarillas esperadas (considera faltas promedio, estilo arbitral, rivalidad)
   - Corners esperados (infiere del estilo de juego si no hay datos duros)
3. Calcula probabilidad implicita: (1 / cuota) * 100
4. Edge (%) = Probabilidad real - Probabilidad implicita
5. Edge > 0 -> valor positivo. Escala: 0-3% BAJO, 3-7% MEDIO, >7% ALTO.

REGLAS DE PONDERACION OBLIGATORIAS:
- METRICAS CUANTITATIVAS (xG, forma ultimos 5-10 partidos de ESTA temporada) pesan MAS que H2H historico.
- El H2H es solo una referencia secundaria. Si el H2H contradice la forma actual, ignoralo.
- Los partidos de H2H de mas de 3 anios atras son irrelevantes (plantillas y estilos cambiaron).
- Si el H2H proviene de Champions League pero los equipos nunca se enfrentaron con estas plantillas, dale peso BAJO.
- LESIONES/SUSPENSIONES (BSD): Los datos de bajas vienen de BSD. Usalos como REFERENCIA pero con PRECAUCION: BSD no siempre es preciso. Si un jugador aparece como lesionado en BSD pero SofaScore lo pone titular, SofaScore MANDA -> el jugador JUEGA.
- ARBITRO: Los datos del arbitro incluyen promedios por partido (amarillas, faltas, goles) via BSD v2. Son la referencia principal para estimar tarjetas. Si no estan disponibles, usa los promedios historicos de carrera como referencia secundaria.

REGLAS DE REMATES (TIROS):
- La fuente principal son los promedios de BSD en la seccion FORMA (remates_promedio, remates_arco_promedio).
- Si hay datos BSD v2 de stats (total_shots, attack, dangerous_attack, pass_accuracy_pct), incorporalos al analisis de intensidad ofensiva.
- SI HAY DATOS DE SHOTMAP (xG por disparo) desde BSD v2 o SofaScore, usalos para evaluar la calidad de las ocasiones.
- SI HAY DATOS DE JUGADORES con tiros y xG individual (BSD v2 player-stats), considera quien genera y quien esta en racha.
- El xg_per_minute de BSD v2 muestra en que momentos del partido los equipos generan peligro. Usalo para patrones de gol.
- Equipos con alto xG + alto volumen de remates al arco -> partido intenso ofensivamente.
- Si ambos equipos promedian >10 remates y >4 al arco -> Over 2.5 gana peso adicional.
- Si ambos equipos generan pocos remates al arco (<3) -> Under 2.5 gana peso.

REGLAS DE MOMENTUM:
- Si esta disponible el grafico de momentum de SofaScore, usalo para entender quien domino el partido.
- El momentum mide dominio acumulado minuto a minuto (no solo posesion, sino presion ofensiva).

REGLAS DE TARJETAS Y ARBITRO:
- La fuente PRINCIPAL para datos de arbitro es WhoScored (si esta disponible): desglose por competicion, YC/partido, RC/partido, faltas/partido.
- Si WhoScored NO esta disponible, usa ValueStats como alternativa.
- Si ninguno esta disponible, usa BSD v2 (avg_yellow_per_match, avg_fouls_per_match) como referencia.
- SI HAY DATOS FLASHSCORE del arbitro, cruzalos con los datos de WhoScored/ValueStats.
- IMPORTANTE: Si WhoScored muestra YC/partido distintos en liga local vs Champions League, usa el promedio de la competicion del partido actual, no el total.
- Derbis y partidos de alta rivalidad -> mas tarjetas esperadas.
- Si el arbitro NO esta asignado aun, NO inventes sustitutos.

REGLAS DE ALINEACIONES / DISPONIBILIDAD DE JUGADORES:
- LA ALINEACION DE SOFASCORE ES LA FUENTE UNICA Y DEFINITIVA de que jugadores juegan.
- Si SofaScore tiene alineacion confirmada: esos son EXACTAMENTE los jugadores que jugaran.
- Las BAJAS BSD son referencia SECUNDARIA. SofaScore MANDA.
- Si hay PLANTILLA BSD v2 (squad), usala para validar que jugadores son titulares habituales vs suplentes.
- SI HAY ESTADISTICAS POR JUGADOR (xG individual, rating, pases, duelos) desde BSD v2 player-stats, analiza que jugadores estan en mejor momento y como afectan al partido.
- SI HAY DATOS DE DT BSD v2 (win_pct, avg_goals, clean_sheet_pct, btts_pct, over_25_pct), son metricas cuantitativas valiosas para predecir el estilo del partido. Pesan mas que la descripcion cualitativa del perfil.

REGLAS DE OVERCONFIDENCE:
- Edges >10% son extremadamente raros. Verifica respaldo multiple.
- Un edge REALISTA en futbol de elite: 3-8%.

REGLAS DE OVER/UNDER y BTTS:
- Para Over/Under y BTTS, la forma goleadora de ESTA TEMPORADA pesa mas.
- Si ambos equipos tienen xG alto esta temporada (>1.5), el Over 2.5 tiene mas peso.
- No uses el argumento "H2H historico under" si los equipos actuales juegan distinto.
- COMPETICION: Si la forma reciente del equipo proviene de su liga local contra rivales inferiores (ej: PSG vs Le Havre), penaliza esas metricas al proyectar contra un rival de elite en Champions. Las estadisticas infladas por goleadas a equipos debiles NO se transfieren a partidos de maxima exigencia.
- VARIANZA: No te fies solo del promedio. Mira el desglose por partido. Si un equipo metio 17 goles en 5 partidos pero 9 fueron en un solo partido contra un rival debil, su promedio real de goles es ~2.0, no 3.4. Penaliza los outliers.

REGLAS DE CUOTAS:
- Las cuotas de BETSAFE son las OFICIALES. Usalas para calcular valor.
- Ignora cualquier otra cuota (BSD) que pueda aparecer. Betsafe manda.

SE CONSERVADOR: Prefiere quedarte corto en edges a inflarlos artificialmente.

DATOS CONTEXTUALES NUEVOS (BSD v2):
- Datos pre-partido (funfacts): hechos narrativos como "X no ha perdido en N partidos". Son contexto util pero no reemplazan metrica cuantitativa.
- Tabla de posiciones con xG: si BSD v2 provee standings con xGF/xGA/xGD, usalos para comparar rendimiento real vs esperado de cada equipo en la temporada.
- Precisión de pases y ball-tracking: si BSD v2 da pass_accuracy_pct, attack, dangerous_attack, incorporalos al analisis de dominio. Equipos con alta precision de pases + muchos dangerous_attack generan mas ocasiones claras.
- Travel distance y derby: si BSD v2 indica is_local_derby=true o travel_distance_km alto, ajusta expectativas (derby = mas tarjetas, viaje largo = posible fatiga visitante).

FORMATO DE RESPUESTA (se conciso, no repitas datos):

[PROBABILIDADES REALES]
1X2: L=X% / E=X% / V=X% | BTTS: Si=X% / No=X% | O2.5: Over=X% / Under=X%
(2-3 frases de justificacion por mercado)

[TIROS Y REMATES]
Estimacion de tiros totales y al arco (promedio esperado para el partido).
Justificacion: basada en estilo de juego y promedios de cada equipo.
Si hay datos de shotmap/xG individual, incorporalos en la justificacion.

[TARJETAS]
Amarillas esperadas (rango bajo/medio/alto). Menciona al arbitro si esta asignado.
Usa datos Flashscore del arbitro si estan disponibles.

[CORNERS]
Estimacion de corners totales. Si no hay datos, menciona e infiere del estilo.

[EDGE]
Tabla: Mercado | Seleccion | Cuota | Prob.Real | Prob.Implicita | Edge | Confianza
(Solo filas con edge > 0. Si no hay, indicalo)

[RECOMENDACION]
Mejor apuesta con valor (1-2 lineas). Si no hay valor, dilo.
Si hay contexto favorable para tiros/amarillas/corners, menciona como nota adicional."""



def _crear_prompt_usuario(datos_resumidos: dict, prediccion_resumida: dict) -> str:
    """
    Construye el prompt de usuario con los datos del partido formateados.

    Args:
        datos_resumidos: Datos resumidos del partido desde bsd_client + sofascore_client.
        prediccion_resumida: Predicción ML resumida desde bsd_client.

    Returns:
        String con el prompt formateado.
    """
    prompt = f"""
## DATOS DEL PARTIDO

**{datos_resumidos.get('partido', 'Desconocido')}**
Liga: {datos_resumidos.get('liga', 'Desconocida')}
Fecha: {datos_resumidos.get('fecha', 'Desconocida')}
{f"Estadio: {datos_resumidos.get('estadio', {}).get('nombre', '?')} ({datos_resumidos.get('estadio', {}).get('ciudad', '?')}, cap: {datos_resumidos.get('estadio', {}).get('capacidad', '?')})" if datos_resumidos.get('estadio') else ""}

### ÁRBITRO
{f"Nombre: {datos_resumidos['arbitro'].get('nombre', 'Desconocido')} ({datos_resumidos['arbitro'].get('nacionalidad', '?')})" if datos_resumidos.get('arbitro') else "ARBITRO NO ASIGNADO. No asumas nada sobre su estilo; simplemente omite el factor arbitral."}
{f"\n- Amarillas/partido: {datos_resumidos['arbitro'].get('avg_yellow_per_match', '?')}" if datos_resumidos.get('arbitro') and datos_resumidos['arbitro'].get('avg_yellow_per_match') is not None else ""}
{f"\n- Faltas/partido: {datos_resumidos['arbitro'].get('avg_fouls_per_match', '?')}" if datos_resumidos.get('arbitro') and datos_resumidos['arbitro'].get('avg_fouls_per_match') is not None else ""}
{f"\n- Goles/partido: {datos_resumidos['arbitro'].get('avg_goals_per_match', '?')}" if datos_resumidos.get('arbitro') and datos_resumidos['arbitro'].get('avg_goals_per_match') is not None else ""}

### CUOTAS DEL BOOKMAKER
{_formatear_cuotas_betsafe_para_prompt(datos_resumidos.get('_cuotas', {})) if datos_resumidos.get('_cuotas') else f'''- Local: {datos_resumidos['cuotas'].get('local', 'N/D')}
- Empate: {datos_resumidos['cuotas'].get('empate', 'N/D')}
- Visitante: {datos_resumidos['cuotas'].get('visitante', 'N/D')}
- Over 2.5: {datos_resumidos['cuotas'].get('over_25', 'N/D')}
- Under 2.5: {datos_resumidos['cuotas'].get('under_25', 'N/D')}
- BTTS Si: {datos_resumidos['cuotas'].get('btts_si', 'N/D')}
- BTTS No: {datos_resumidos['cuotas'].get('btts_no', 'N/D')}'''}

### FORMA LOCAL ({datos_resumidos['partido'].split(' vs ')[0]})
- Últimos partidos: {json.dumps(datos_resumidos.get('forma_local', {}), indent=2, ensure_ascii=False)}

### FORMA VISITANTE ({datos_resumidos['partido'].split(' vs ')[1] if ' vs ' in datos_resumidos.get('partido', '') else '?'})
- Últimos partidos: {json.dumps(datos_resumidos.get('forma_visitante', {}), indent=2, ensure_ascii=False)}

### HEAD TO HEAD (últimos 3 años)
- NOTA: Solo se muestran enfrentamientos recientes. El H2H antiguo (>3 años) fue excluido porque las plantillas y estilos cambiaron.
{f"- {json.dumps(datos_resumidos.get('h2h', {}), indent=2, ensure_ascii=False)}" if datos_resumidos.get('h2h') and datos_resumidos['h2h'].get('total_partidos') else "- No hay enfrentamientos previos registrados entre estos equipos. Ignora el factor H2H para este analisis."}

### NOTA SOBRE DATOS DISPONIBLES
- REMATES Y TIROS: Usa los promedios de BSD en la seccion FORMA (remates_promedio, remates_arco_promedio).
- AMARILLAS: Usa amarillas_promedio y faltas_promedio de BSD. Cruza esto con la info del arbitro.
- CORNERS: No hay datos de corners por equipo en las APIs disponibles. Haz una inferencia cualitativa basada en estilo de juego (equipos de posesion alta y muchos remates generan mas corners).
- POSESION: Infiere del estilo de juego y perfil de los entrenadores.
- FATIGA: No hay datos exactos de fechas de ultimos partidos. Usa el PPG y la forma reciente como proxy de fatiga/ritmo. Equipos con alta carga de partidos (Champions + Liga) suelen rotar mas.
- LESIONES: SofaScore YA NO proporciona datos de lesiones via API. Usa BAJAS BSD como referencia secundaria con PRECAUCION. La alineacion de SofaScore es quien define quien JUEGA.

### ALINEACION DEL PARTIDO (SofaScore)
- La alineacion de SofaScore es la unica fuente de disponibilidad de jugadores.
- Si hay CONFLICTOS BSD vs SofaScore, SofaScore MANDA (el jugador JUEGA).
{f"### ALINEACION CONFIRMADA (SofaScore)\n- {json.dumps(datos_resumidos.get('_sofascore', {}).get('alineaciones', {}), indent=2, ensure_ascii=False)}" if datos_resumidos.get("_sofascore", {}).get("alineaciones", {}).get("local", {}).get("confirmada") else ""}
{f"### ALINEACION PRELIMINAR (NO CONFIRMADA)\n- {json.dumps(datos_resumidos.get('_sofascore', {}).get('alineaciones', {}), indent=2, ensure_ascii=False)}" if datos_resumidos.get("_sofascore", {}).get("alineaciones") and not datos_resumidos.get("_sofascore", {}).get("alineaciones", {}).get("local", {}).get("confirmada") else ""}
{f"### SIN ALINEACION DISPONIBLE\n- SofaScore no tiene alineacion todavia (suele salir ~1h antes del partido). Asume plantilla tipo con los jugadores habituales." if not datos_resumidos.get("_sofascore", {}).get("alineaciones") else ""}

### BAJAS BSD (lesionados/suspendidos - referencia secundaria)
{json.dumps(datos_resumidos.get('bajas_bsd', {}), indent=2, ensure_ascii=False)}
- NOTA: Estas bajas son de BSD, NO de SofaScore. Usalas con precaucion.
- Si un jugador esta aqui Y NO en la alineacion de SofaScore → probable baja real.
- Si un jugador esta aqui Y SI en la alineacion de SofaScore → JUEGA (conflicto resuelto a favor de SofaScore).

### ESTILOS DE ENTRENADORES
- Local: {json.dumps(datos_resumidos.get('entrenador_local', {}), indent=2, ensure_ascii=False)}
- Visitante: {json.dumps(datos_resumidos.get('entrenador_visitante', {}), indent=2, ensure_ascii=False)}

### PREDICCIÓN ML (BSD CatBoost)
- Probabilidades 1X2: L={prediccion_resumida.get('prob_local', 'N/D')}% / E={prediccion_resumida.get('prob_empate', 'N/D')}% / V={prediccion_resumida.get('prob_visitante', 'N/D')}%
- xG esperado: Local {prediccion_resumida.get('xG_local', 'N/D')} - Visitante {prediccion_resumida.get('xG_visitante', 'N/D')}
- Prob. Over 2.5: {prediccion_resumida.get('prob_over_25', 'N/D')}%
- Prob. BTTS: {prediccion_resumida.get('prob_btts', 'N/D')}%
- Marcador más probable: {prediccion_resumida.get('marcador_probable', 'N/D')}
- Confianza del modelo: {prediccion_resumida.get('confianza_modelo', 'N/D')}

---
Analiza los datos anteriores y proporciona tu evaluación de value betting siguiendo el formato establecido.

    {_formatear_standings_para_prompt(datos_resumidos)}
    {_formatear_standings_flashscore_para_prompt(datos_resumidos)}
    {_formatear_detalle_evento_para_prompt(datos_resumidos)}
    {_formatear_h2h_sofascore_para_prompt(datos_resumidos)}
    {_formatear_alineaciones_para_prompt(datos_resumidos)}
    {_formatear_players_stats_para_prompt(datos_resumidos)}
    {_formatear_form_performance_para_prompt(datos_resumidos)}
    {_formatear_shotmap_para_prompt(datos_resumidos)}
    {_formatear_momentum_para_prompt(datos_resumidos)}
    {_formatear_avg_positions_para_prompt(datos_resumidos)}
    {_formatear_incidents_para_prompt(datos_resumidos)}
    {_formatear_match_statistics_para_prompt(datos_resumidos)}
    {_formatear_team_season_stats_para_prompt(datos_resumidos)}
    {_formatear_flashscore_para_prompt(datos_resumidos)}

---
## DATOS ADICIONALES BSD v2
{_v2_sections(datos_resumidos)}
"""
    return prompt


def _v2_sections(datos_resumidos: dict) -> str:
    """Construye las secciones de datos enriquecidos via BSD v2."""
    v2 = datos_resumidos.get("_bsd_v2", {})
    if not v2:
        return "(No se obtuvieron datos adicionales de BSD v2 para este partido)"

    partes = []

    # Detalle v2 (weather, derby, travel, etc.)
    detail = datos_resumidos.get("_v2_detail", {})
    if detail:
        has_any = any(detail.get(k) for k in ["is_local_derby", "is_neutral_ground"])
        has_any = has_any or detail.get("travel_distance_km") is not None
        has_any = has_any or (detail.get("weather") and detail["weather"].get("description"))
        if has_any:
            ln = ["\n### CONTEXTO DEL PARTIDO (BSD v2)"]
            if detail.get("is_local_derby"):
                ln.append("- DERBY LOCAL")
            if detail.get("is_neutral_ground"):
                ln.append("- Cancha neutral")
            if detail.get("travel_distance_km") is not None:
                ln.append(f"- Distancia de viaje visitante: {detail['travel_distance_km']} km")
            if detail.get("weather") and detail["weather"].get("description"):
                w = detail["weather"]
                ln.append(f"- Clima: {w['description']} (codigo {w.get('code')})")
            partes.append("\n".join(ln))

    # Stats v2
    partes.append(resumir_stats_v2_para_prompt(v2))

    # ValueStats arbitro
    partes.append(formatear_arbitro_valuestats_para_prompt(datos_resumidos))

    # WhoScored arbitro
    partes.append(formatear_arbitro_whoscored_para_prompt(datos_resumidos))

    # Player stats v2
    partes.append(resumir_player_stats_v2_para_prompt(v2))

    # Metadata v2
    partes.append(resumir_metadata_v2_para_prompt(v2))

    # Standings v2
    home_id = datos_resumidos.get("home_team_id")
    away_id = datos_resumidos.get("away_team_id")
    partes.append(resumir_standings_v2_para_prompt(v2, home_id, away_id))

    # Squads v2
    partes.append(resumir_squads_v2_para_prompt(v2))

    return "\n".join(p for p in partes if p.strip())


def analizar_partido(datos_resumidos: dict, prediccion_resumida: dict) -> str:
    """
    Envía los datos del partido a DeepSeek via OpenRouter para análisis.

    Args:
        datos_resumidos: Datos resumidos del partido.
        prediccion_resumida: Predicción ML resumida.

    Returns:
        Respuesta de texto del análisis.

    Raises:
        ValueError: Si la API key no está configurada.
        Exception: Si la llamada a la API falla.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY no configurada. Agrégala en el archivo .env"
        )

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    prompt_usuario = _crear_prompt_usuario(datos_resumidos, prediccion_resumida)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_usuario},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        extra_headers={
            "HTTP-Referer": "https://github.com/betting-ai",
            "X-Title": "Betting AI - Value Betting Analyzer",
        },
        extra_body={
            "include_reasoning": True,
        },
    )

    finish_reason = response.choices[0].finish_reason
    if finish_reason == "length":
        raise RuntimeError(
            f"Respuesta truncada (finish_reason='length'). "
            f"Aumenta MAX_TOKENS (actual: {MAX_TOKENS})"
        )

    razonamiento = getattr(response.choices[0].message, "reasoning_content", None) or ""
    razonamiento = getattr(response.choices[0].message, "reasoning_content", None) or ""
    contenido = response.choices[0].message.content
    if contenido is None:
        raise RuntimeError(
            "El modelo no devolvió contenido visible. "
            "Probablemente agotó los tokens en razonamiento interno. "
            f"Tokens de razonamiento usados: ~{len(razonamiento) if razonamiento else 'N/D'}"
        )

    if razonamiento:
        print(f"\n  [DeepSeek razonó {len(razonamiento)} chars internamente]")

    return contenido


if __name__ == "__main__":
    print("=== Probando Analyzer ===\n")

    # Datos de ejemplo para prueba (sin consumir APIs reales)
    datos_ejemplo = {
        "partido": "Barcelona vs Real Madrid",
        "liga": "La Liga",
        "fecha": "2026-04-30T21:00:00+02:00",
        "cuotas": {
            "local": 2.10, "empate": 3.60, "visitante": 3.20,
            "over_25": 1.65, "under_25": 2.20,
            "btts_si": 1.80, "btts_no": 1.95,
        },
        "forma_local": {
            "forma_string": "WWDLW", "victorias": 3, "empates": 1, "derrotas": 1,
            "goles_favor_ultimos_n": 11, "goles_contra_ultimos_n": 4,
            "xG_promedio": 1.8, "xG_contra_promedio": 0.9,
        },
        "forma_visitante": {
            "forma_string": "WDLWW", "victorias": 3, "empates": 1, "derrotas": 1,
            "goles_favor_ultimos_n": 9, "goles_contra_ultimos_n": 5,
            "xG_promedio": 1.5, "xG_contra_promedio": 1.1,
        },
        "h2h": {
            "total_partidos": 10, "victorias_local": 4, "empates": 2,
            "victorias_visitante": 4, "goles_local": 15, "goles_visitante": 14,
            "promedio_goles": 2.9,
        },
        "entrenador_local": {"nombre": "Flick", "formacion": "4-2-3-1", "perfil": "attacking"},
        "entrenador_visitante": {"nombre": "Ancelotti", "formacion": "4-3-3", "perfil": "balanced"},
    }

    prediccion_ejemplo = {
        "prob_local": 45.0, "prob_empate": 25.0, "prob_visitante": 30.0,
        "xG_local": 1.7, "xG_visitante": 1.2,
        "prob_over_25": 65.0, "prob_btts": 60.0,
        "marcador_probable": "2-1",
        "confianza_modelo": 0.72,
    }

    try:
        resultado = analizar_partido(datos_ejemplo, prediccion_ejemplo)
        print(resultado)
    except Exception as e:
        print(f"Error: {e}")
        print("(Esperado si OPENROUTER_API_KEY no está configurada)")
