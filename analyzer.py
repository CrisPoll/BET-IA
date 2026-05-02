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
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME = "deepseek/deepseek-v4-pro"
MAX_TOKENS = 100000  # Holgado: incluye reasoning interno + respuesta visible
TEMPERATURE = 0.3  # Baja temperatura para análisis más consistente

SYSTEM_PROMPT = """Eres un experto analista de Value Betting en fútbol, con mentalidad crítica y escéptica ante datos imperfectos.

PROCESO DE ANÁLISIS:
1. Estima probabilidad real (%) de cada outcome para 1X2, BTTS, Over/Under 2.5
   usando forma reciente, xG, estilos tácticos y contexto ACTUAL de la temporada.
2. Como CONTEXTO ADICIONAL, analiza también:
   - Tiros totales y tiros al arco (proyecta si el partido sera de muchos/pocos disparos)
   - Tarjetas amarillas esperadas (considera faltas promedio, estilo arbitral, rivalidad)
   - Corners esperados (infiere del estilo de juego si no hay datos duros)
3. Calcula probabilidad implícita: (1 / cuota) * 100
4. Edge (%) = Probabilidad real - Probabilidad implícita
5. Edge > 0 → valor positivo. Escala: 0-3% BAJO, 3-7% MEDIO, >7% ALTO.

REGLAS DE PONDERACIÓN OBLIGATORIAS:
- MÉTRICAS CUANTITATIVAS (xG, forma últimos 5-10 partidos de ESTA temporada) pesan MÁS que H2H histórico.
- El H2H es solo una referencia secundaria. Si el H2H contradice la forma actual, ignóralo.
- Los partidos de H2H de más de 3 años atrás son irrelevantes (plantillas y estilos cambiaron).
- Si el H2H proviene de Champions League pero los equipos nunca se enfrentaron con estas plantillas, dale peso BAJO.
- LESIONES/SUSPENSIONES (BSD): Los datos de bajas vienen de BSD. Usalos como REFERENCIA pero con PRECAUCION: BSD no siempre es preciso. Si un jugador aparece como lesionado en BSD pero SofaScore lo pone titular, SofaScore MANDA → el jugador JUEGA.
- ARBITRO (SofaScore): Si SofaScore trae arbitro, usalo como referencia principal (reemplaza al de BSD).

REGLAS DE REMATES (TIROS):
- La fuente principal son los promedios de BSD en la seccion FORMA (remates_promedio, remates_arco_promedio).
- Equipos con alto xG + alto volumen de remates al arco → partido intenso ofensivamente.
- Si ambos equipos promedian >10 remates y >4 al arco → Over 2.5 gana peso adicional.
- Si ambos equipos generan pocos remates al arco (<3) → Under 2.5 gana peso.

REGLAS DE TARJETAS Y ÁRBITRO:
- La fuente principal para amarillas son los promedios de BSD (amarillas_promedio, faltas_promedio).
- Si el árbitro tiene datos de carrera (amarillas_carrera, rojas_carrera), usalo para evaluar su tendencia. Un promedio alto de amarillas por partido indica arbitro "tarjetero".
- Derbis y partidos de alta rivalidad → mas tarjetas esperadas.
- Partidos con poco en juego (mitad de tabla, sin descenso) → menos tarjetas.
- Si el árbitro NO esta asignado aun, NO inventes sustitutos. Simplemente indica que falta ese dato.

REGLAS DE ALINEACIONES / DISPONIBILIDAD DE JUGADORES:
- LA ALINEACION DE SOFASCORE ES LA FUENTE UNICA Y DEFINITIVA de que jugadores juegan.
- Si SofaScore tiene alineacion confirmada: esos son EXACTAMENTE los jugadores que jugaran. No asumas ausencias adicionales.
- Si SofaScore NO tiene alineacion (suele salir ~1h antes del partido): asume la plantilla tipo con los jugadores habituales disponibles.
- Las BAJAS BSD (lesionados/suspendidos) son una referencia SECUNDARIA. Si BSD dice que X esta lesionado pero SofaScore lo pone titular, SofaScore MANDA: X JUEGA.
- Si un jugador NO aparece en la alineacion de SofaScore y BSD lo reporta como lesionado, probablemente sea baja real.
- Si no hay informacion de bajas/ausencias confiable, NO inventes debilidad ofensiva para justificar unders o BTTS No.

REGLAS DE OVERCONFIDENCE:
- En fútbol, edges >10% son extremadamente raros y casi siempre indican un error en los datos de entrada.
- Si calculas un edge >10%, verifica que esté respaldado por MÚLTIPLES factores independientes (no solo uno).
- Si el edge se basa principalmente en suposiciones debiles o H2H antiguo, REDÚCELO significativamente.
- Un edge REALISTA en fútbol de elite suele estar en el rango 3-8%. Edges >12% son sospechosos.

REGLAS DE OVER/UNDER y BTTS:
- Para Over/Under y BTTS, la forma goleadora de ESTA TEMPORADA pesa más que el H2H histórico.
- Si ambos equipos tienen xG alto esta temporada (>1.5 cada uno), el Over 2.5 tiene más peso.
- Si el H2H histórico muestra pocos goles pero la forma actual muestra muchos, la forma actual MANDA.
- No uses el argumento "H2H histórico under" si los equipos actuales juegan un fútbol radicalmente distinto.

SÉ CONSERVADOR: Prefiere quedarte corto en edges a inflarlos artificialmente. Un falso positivo (recomendar algo sin valor real) es peor que un falso negativo (no detectar una oportunidad real).

FORMATO DE RESPUESTA (sé conciso, no repitas datos):

[PROBABILIDADES REALES]
1X2: L=X% / E=X% / V=X% | BTTS: Sí=X% / No=X% | O2.5: Over=X% / Under=X%
(2-3 frases de justificación por mercado)

[TIROS Y REMATES]
Estimación de tiros totales y al arco (promedio esperado para el partido).
Justificación: basada en estilo de juego y promedios de cada equipo.

[TARJETAS]
Amarillas esperadas (rango bajo/medio/alto). Menciona al árbitro si está asignado.

[CORNERS]
Estimación de corners totales. Si no hay datos, menciónalo y haz una inferencia cualitativa.

[EDGE]
Tabla: Mercado | Selección | Cuota | Prob.Real | Prob.Implícita | Edge | Confianza
(Solo filas con edge > 0. Si no hay, indícalo)

[RECOMENDACIÓN]
Mejor apuesta con valor (1-2 líneas). Si no hay valor, dilo.
Si hay contexto favorable para tiros/amarillas/corners, menciónalo como nota adicional."""



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
{f"\n- Amarillas en carrera: {datos_resumidos['arbitro'].get('amarillas_carrera', '?')}" if datos_resumidos.get('arbitro') and datos_resumidos['arbitro'].get('amarillas_carrera') else ""}
{f"\n- Rojas en carrera: {datos_resumidos['arbitro'].get('rojas_carrera', '?')}" if datos_resumidos.get('arbitro') and datos_resumidos['arbitro'].get('rojas_carrera') else ""}
{f"\n- Partidos dirigidos: {datos_resumidos['arbitro'].get('partidos_carrera', '?')}" if datos_resumidos.get('arbitro') and datos_resumidos['arbitro'].get('partidos_carrera') else ""}

### CUOTAS DEL BOOKMAKER
- Local: {datos_resumidos['cuotas'].get('local', 'N/D')}
- Empate: {datos_resumidos['cuotas'].get('empate', 'N/D')}
- Visitante: {datos_resumidos['cuotas'].get('visitante', 'N/D')}
- Over 2.5: {datos_resumidos['cuotas'].get('over_25', 'N/D')}
- Under 2.5: {datos_resumidos['cuotas'].get('under_25', 'N/D')}
- BTTS Sí: {datos_resumidos['cuotas'].get('btts_si', 'N/D')}
- BTTS No: {datos_resumidos['cuotas'].get('btts_no', 'N/D')}

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
{_formatear_detalle_evento_para_prompt(datos_resumidos)}
{_formatear_h2h_sofascore_para_prompt(datos_resumidos)}
{_formatear_alineaciones_para_prompt(datos_resumidos)}
{_formatear_form_performance_para_prompt(datos_resumidos)}
"""
    return prompt


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
