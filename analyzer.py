"""
analyzer.py — Analizador de Mercados Estadísticos usando DeepSeek via OpenRouter.

VERSIÓN 2.0: Integra modelo cuantitativo + pipeline de agentes + persistencia DB + Kelly stakes.
Mantiene compatibilidad con llamadas anteriores (analizar_partido).
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
from betsafe_client import _formatear_cuotas_betsafe_para_prompt
from bsd_client_v2 import (
    resumir_stats_v2_para_prompt,
    resumir_metadata_v2_para_prompt,
    resumir_player_stats_v2_para_prompt,
    resumir_standings_v2_para_prompt,
    resumir_squads_v2_para_prompt,
    resumir_manager_v2_para_prompt,
    resumir_arbitro_v2_para_prompt,
)

# Nuevos módulos
from quant_model import run_full_projection
import prediction_db as db
from bankroll import (
    evaluate_stat_market,
    evaluate_1x2,
    format_recommendations,
    calibrate_probability,
    calculate_kelly_stake,
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME = "deepseek/deepseek-v4-pro"
MAX_TOKENS = 100000
TEMPERATURE = 0.3

# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT MEJORADO CON FEW-SHOT Y AWARENESS DEL MODELO CUANTITATIVO
# ═══════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = """
═══════════════════════════════════════════════════════════
EJEMPLOS DE ANÁLISIS (Anclajes de calidad)
═══════════════════════════════════════════════════════════

Ejemplo A: Proyección de tiros correcta
Partido: Man City vs Luton | Contexto: City favorito amplio, Luton encerrado.
Datos: City 18.2 tiros/prom local, Luton 8.1 tiros/prom visita.
Proyección correcta: Local 19-21, Visitante 5-7, Total 25-27.
Justificación: volumen local alto Luton no saldría del área. Rango aceptable 24-28.

Ejemplo B: Corrección por contexto (no seguir promedios ciegos)
Partido: Boca vs River (Libertadores octavos, vuelta, 1-1 en ida).
Datos: Boca 14.5 tiros/prom, River 13.2 tiros/prom.
Proyección correcta: Local 12-14, Visitante 9-11, Total 22-25.
Justificación: Derby cerrado + resultado ajustado = menos espacios. NO se proyectó 28+ tiros.

Ejemplo C: Tarjetas ajustadas por árbitro + contexto
Árbitro: 3.8 YC/partido en Libertadores. Derby. Descenso indirecto.
Proyección correcta: Total 5.0-6.0 amarillas (más del promedio de 3.5 por contexto).
Ajuste: árbitro estricto + tensión derby = +1.5 sobre la media conjunta.

Ejemplo D: Cuota baja = mercado already priced in
Cuota Over 2.5 goles @ 1.32. El mercado descuenta alto volumen.
Decisión correcta: NO recomendar Under sin evidencia abrumadora. Omitir este mercado.
"""

SYSTEM_PROMPT = f"""Eres un experto analista de MERCADOS ESTADÍSTICOS en fútbol. Tu especialidad es proyectar con precisión tiros, tarjetas, corners, goles por mitad, faltas, y demás estadísticas de partido basándote en CONTEXTO, perfiles de equipo, y datos duros. El value betting en mercados de resultado (1X2) es secundario.

INFORMACIÓN IMPORTANTE: Antes de cada análisis, recibirás una PROYECCIÓN CUANTITATIVA BASE generada por un modelo estadístico propio. Esta base usa xG, forma reciente ponderada, posición en tabla, distancia de viaje y regresión a la media. TU trabajo es VALIDARLA, AJUSTARLA según el contexto cualitativo, y COMUNICAR un rango final con justificación. Nunca ignores la base sin explicar por qué.

{FEW_SHOT_EXAMPLES}

═══════════════════════════════════════
PROCESO DE ANÁLISIS (EN ORDEN)
═══════════════════════════════════════

PASO 1 - CONTEXTO DEL PARTIDO (LO MÁS IMPORTANTE):
Antes de proyectar cualquier estadística, analiza el contexto. Sin contexto, los números no valen nada.

a) ¿QUÉ SE JUEGA? (MOTIVACIÓN):
   - ATENCION: La Ronda/Fase en BSD v2 indica si es final, semifinal, grupos o liga. Si es FINAL: partido unico, no hay tabla, motivacion maxima.
   - Mira la tabla de posiciones (si aplica). ¿Cuántos partidos quedan en la temporada? (total equipos - 1 - PJ jugados)
   - ¿Equipo peleando título, clasificación a copa, descenso, o sin nada en juego?
   - ¿Es eliminatoria (ida/vuelta)? ¿El resultado de ida condiciona el planteamiento?
   - Equipos sin nada que jugar tienden a partidos más abiertos o más apáticos. Evalúa cuál aplica según perfil del DT.
   - Equipos en descenso directo: desesperación = mas faltas, más tarjetas, más tiros apurados.
   - Equipo que con empate clasifica: planteamiento conservador, pocos tiros, muchas faltas tácticas.
   - Finales en cancha neutral: sin ventaja de localia, presion maxima, inicio tactico, mas tarjetas en 2T.

b) PERFIL DE CADA EQUIPO (ESTILO DE JUEGO):
   - Ofensivo/defensivo: posesión alta/baja, presión alta/baja, transiciones rápidas/lentas.
   - ¿Juega con laterales profundos? -> más corners y centros.
   - ¿Equipo de mucho disparo exterior? -> más tiros totales pero menos al arco.
   - ¿Equipo que fuerza faltas tácticas? -> más tarjetas rivales.
   - ¿Juega con línea alta? -> más offsides, más espacios a la espalda.
   - Perfil del DT: mira los datos de BSD v2 (win_pct, avg_goals, clean_sheet_pct, btts_pct, over_25_pct).

c) DINÁMICA DEL PARTIDO ESPERADA:
   - ¿Local fuerte vs visitante débil? -> local propone, visitante se encierra.
   - ¿Dos equipos ofensivos? -> ida y vuelta, muchos tiros de ambos, Over en todo.
   - ¿Dos equipos defensivos/calculadores? -> pocos tiros, pocos corners, muchas faltas.
   - ¿Derby o clásico? (BSD v2: is_local_derby=true) -> MÁS tarjetas, MÁS faltas, partido cortado.
   - ¿Viaje largo del visitante? (travel_distance_km alto) -> posible fatiga en 2do tiempo.
   - ¿Clima extremo? -> menos ritmo, menos tiros.
   - ¿Cancha neutral? -> quita ventaja de localía.

d) ¿CUÁNTOS PARTIDOS QUEDAN?:
   - Pocos partidos + necesidad urgente = equipos que arriesgan más.
   - Muchos partidos + posición cómoda = ritmo relajado.
   - Si un equipo ya está salvado/eliminado, sus estadísticas pueden empeorar frente a uno que se juega la vida.

PASO 2 - VALIDACIÓN DE PROYECCIÓN CUANTITATIVA BASE:
Recibirás números base del modelo. Tu tarea:
1. Si están razonables dado el contexto, confirma y refina a rangos.
2. Si discrepan significativamente, explica POR QUÉ y ajusta.
   Ejemplos de discrepancia válida:
   - Modelo proyecta 3.2 goles pero es derby cerrado + árbitro estricto. Ajustar a 2.2-2.5.
   - Modelo proyecta 22 tiros total pero visitante con 5 defensas y DT ultra-defensivo. Ajustar a 16-18.
   - Modelo proyecta 3.0 tarjetas pero árbitro histórico 4.5 YC/part en esta competición. Ajustar a 4.5-5.5.

PASO 3 - PROYECCIÓN DE ESTADÍSTICAS (POR MITADES 1T/2T):
Proyecta cada métrica con desglose. Fundamenta con datos de FORMA RECIENTE (desglose por partido), ESTADÍSTICAS DE TEMPORADA, PERFIL, y CONTEXTO.

═══════════════════════════════════════
REGLAS POR MERCADO ESTADÍSTICO
═══════════════════════════════════════

── TIROS TOTALES Y AL ARCO ──
- Fuente primaria: desglose POR PARTIDO en FORMA RECIENTE (SofaScore). NO promedies ciegamente.
- CONSISTENCIA: un equipo que hace 18-20-22-19-21 tiros es MUY distinto a uno que hace 8-25-6-22-9.
- Tiros 1T vs 2T: equipos que empiezan fuertes (presión alta primeros 20 min) generan más en 1T.
- Si el local es amplio favorito y el visitante se encierra: el local tira MUCHO (20+), el visitante tira POCO (<6).
- Si hay shotmap con xG/xGOT: evalúa calidad de ocasion, no solo volumen.
- Ajusta por COMPETICIÓN: tiros en liga contra rivales débiles NO se transfieren 1:1 a copa contra rivales de élite.
- Cuota Betsafe <1.35 en Over tiros: el mercado ya descuenta volumen alto. Solo recomienda Under con evidencia ABRUMADORA.

── GOLES POR MITAD (1T / 2T) ──
- Analiza el ritmo de inicio: ¿qué equipo suele marcar temprano?
- ¿Hay dato de goles 1T/2T en forma reciente? Usalo.
- Equipos que presionan alto al inicio: más goles en 1T.
- Fatiga del visitante (viaje largo, poco descanso) -> más goles en contra en 2T.
- Si el partido es de vuelta con resultado ajustado: primer tiempo trabado, goles en 2T cuando se abren.

── TARJETAS AMARILLAS ──
- ARBITRO: fuente principal. Siempre usa el promedio de la COMPETICIÓN ACTUAL del partido.
- Fuentes de árbitro en orden: SofaScore > ValueStats > Flashscore > WhoScored > Transfermarkt > BSD v2.
- Derby/rivalidad = MÁS tarjetas. Equipo sin nada que jugar = MENOS tarjetas. Descenso = MÁS tarjetas.
- TARJETAS POR MITAD:
  - 1T: árbitros suelen ser más permisivos al inicio. Pero si hay derby o mucha tensión, pueden salir temprano.
  - 2T: más tarjetas en promedio (fatiga, nervios, resultado apretado, pérdidas de tiempo). Ajusta al alza.

── CORNERS ──
- Fuente primaria: corners por partido en FORMA RECIENTE + promedio temporada.
- Equipos con laterales ofensivos + centros frecuentes = corners altos.
- Local dominante vs visitante encerrado = muchos corners locales (>7).
- Partido cerrado y táctico = pocos corners (<8 total).
- Por mitad: 2T suele tener más corners (equipos se vuelcan, más urgencia).

── FALTAS ──
- Fuente: promedio de faltas BSD + faltas por partido en stats SofaScore.
- Derby/descenso = más faltas. Partido amistoso/sin nada en juego = menos faltas.
- Árbitros rigurosos pitan más faltas. Árbitros permisivos dejan jugar.

═══════════════════════════════════════
REGLAS GENERALES DE PONDERACIÓN
═══════════════════════════════════════

- MÉTRICAS CUANTITATIVAS (xG, forma últimos 5-10 partidos de ESTA temporada) PESAN MÁS que H2H histórico.
- H2H >3 años es IRRELEVANTE. Plantillas y estilos cambiaron.
- VARIANZA: no te fíes del promedio. Mira el desglose por partido. Penaliza outliers.
- COMPETICIÓN: estadísticas de liga contra rivales débiles NO se transfieren a copa contra rivales de élite.
- LESIONES/SUSPENSIONES: SofaScore es la única fuente de quién juega. BSD es referencia secundaria.
- SI HAY ESTADÍSTICAS POR JUGADOR: un delantero con alto xG reciente indica que el equipo genera para él.
- No inventes datos. Si proyectas un número, fundamentalo con 2-3 razones claras.
- SE CONSERVADOR: prefiere quedarte corto en proyecciones a inflarlas. Un error de +3 tiros proyectados es peor que -3.
- IMPORTANTE: Incluye una nota de CONFIANZA para cada proyección principal (Alta/Media/Baja) según la calidad de los datos disponibles.

═══════════════════════════════════════
CUOTAS DE BETSAFE
═══════════════════════════════════════
- Betsafe es la fuente OFICIAL de cuotas. Ignora BSD.
- Mercados estadísticos disponibles (varian por partido): Total de Tiros (TSTOUM), Tiros al Arco (TOSG), Total Corners (TOCO), Total Amarillas (TOYC), Goles 1T (1HTG), Goles 2T (2HTG), BTTS 1T/2T.
- Para calcular edge en mercados estadísticos: compara tu proyección con la línea de Betsafe.
- Si cuota Over en algún mercado es <1.35, el mercado ya descuenta volumen alto. Solo recomienda Under con evidencia abrumadora.

FORMATO DE RESPUESTA (sé conciso, no repitas datos ya mostrados en el prompt):

[CONTEXTO DEL PARTIDO]
- ¿Qué se juega? (título/descenso/clasificación/nada). Partidos restantes.
- Perfil de cada equipo (estilo de juego, posesión, presión, laterales, disparo).
- Dinámica esperada (local propone/visitante encierra, ida y vuelta, cerrado/táctico, derby).
- Factores de motivación y urgencia.
(4-6 líneas máximo)

[VALIDACIÓN MODELO CUANTITATIVO]
- Base del modelo: goles X.X-Y.Y, tiros X-Y, corners X-Y, tarjetas X-Y.
- ¿Coincide o discrepa con tu análisis cualitativo? Justifica ajustes.

[PROYECCIÓN DE TIROS]
Totales: Local X-Y Visitante (Total: Z) | Al arco: Local X-Y Visitante (Total: Z)
1T: Loc X-Y Vis (Arco: X-Y) | 2T: Loc X-Y Vis (Arco: X-Y)
Justificación (2-3 líneas, menciona consistencia y contexto)

[PROYECCIÓN DE GOLES POR MITAD]
1T: Over/Under X.X goles esperados | 2T: Over/Under X.X goles esperados
Razonamiento (ritmo esperado, quién marca primero, fatiga, urgencia)

[PROYECCIÓN DE AMARILLAS]
Local: X | Visitante: Y | Total: Z | 1T: ~X | 2T: ~Y
Árbitro: [nombre] - [estilo: permisivo/moderado/estricto] - media YC en esta competición: X.X
Justificación (rivalidad, perfil de faltas de cada equipo, contexto)

[PROYECCIÓN DE CORNERS]
1T: X-Y | 2T: X-Y | Total: Z
Justificación (estilo ofensivo, laterales, centros, urgencia en 2T)

[PROYECCIÓN DE FALTAS]
Total: ~Z faltas
Justificación (1 línea)

[SAQUES DE BANDA] (omitir si no hay suficiente info para inferir)
Estimación cualitativa: rango bajo/medio/alto. Explicación breve.

[MERCADOS ESTADÍSTICOS CON VALOR]
Mercado | Selección | Línea/Cuota Betsafe | Proyección | Edge estimado | Confianza
(Solo filas con edge positivo en mercados estadísticos. Si no hay cuotas comparables de Betsafe para un mercado, no lo incluyas.)

[KELLY STAKES RECOMENDADOS]
(Opcional, si hay picks claros: cuánto apostar según Kelly Criterion fraccional)

[RESUMEN 1X2/OVER/BTTS] (solo si hay datos suficientes, prioridad baja)
1X2: L=X% / E=X% / V=X% | BTTS: Si=X% / No=X% | O2.5: Over=X% / Under=X%
(1 línea de justificación por mercado)

[RECOMENDACIÓN FINAL]
Mejor mercado estadístico con valor y por qué. Si no hay valor claro en estadísticas, dilo.
Menciona el factor contextual MÁS RELEVANTE que define el partido."""


def _formatear_arbitro_header(datos: dict) -> str:
    """Árbitro inline en el header del prompt."""
    arb = datos.get("arbitro")
    if not arb:
        return "ARBITRO NO ASIGNADO. No asumas nada sobre su estilo; omite el factor arbitral."

    nombre = arb.get("nombre", "Desconocido")
    pais = arb.get("nacionalidad", arb.get("pais", ""))
    header = f"Nombre: {nombre}"
    if pais:
        header += f" ({pais})"

    # Buscar la mejor fuente: SofaScore > WhoScored > Transfermarkt > BSD v2
    sf = arb.get("_sofascore_ref", {})
    ws = arb.get("_whoscored", {})
    tm = arb.get("_transfermarkt", {})

    ref_data = sf or ws or tm
    if ref_data:
        partes = [header]
        if ref_data.get("total_partidos"):
            partes.append(f"- Partidos dirigidos: {ref_data['total_partidos']}")
        if ref_data.get("yc_pp") is not None:
            partes.append(f"- Amarillas/partido: {ref_data['yc_pp']}")
        if ref_data.get("rc_pp") is not None:
            partes.append(f"- Rojas/partido: {ref_data['rc_pp']}")
        if ref_data.get("faltas_pp") is not None:
            partes.append(f"- Faltas/partido: {ref_data['faltas_pp']}")

        torneos = ref_data.get("torneos", ref_data.get("competiciones", []))
        if torneos:
            partes.append("- Promedios por competición:")
            for t in torneos[:8]:
                tn = t.get("nombre", "?")
                yc = t.get("yc_pp", "?")
                rc = t.get("rc_pp", "?")
                partes.append(f"    {tn}: {yc} YC/part, {rc} RC/part")
        return "\n".join(partes)

    # Fallback: BSD v2
    yc = arb.get("avg_yellow_per_match")
    faltas = arb.get("avg_fouls_per_match")
    if yc is not None or faltas is not None:
        partes = [header]
        if yc is not None:
            partes.append(f"- Amarillas/partido: {yc}")
        if faltas is not None:
            partes.append(f"- Faltas/partido: {faltas}")
        return "\n".join(partes)

    return header


def _formatear_forma_bsd_compact(datos: dict) -> str:
    """Formato compacto de la forma BSD."""
    partes = []
    for lado, key in [("Local", "forma_local"), ("Visitante", "forma_visitante")]:
        f = datos.get(key, {})
        if not f:
            continue
        linea = f"{lado}: {f.get('forma_string', '?')} | xG: {f.get('xG_promedio', '?')} | xGc: {f.get('xG_contra_promedio', '?')} | "
        linea += f"Tiros: {f.get('remates_promedio', '?')}/part | Arco: {f.get('remates_arco_promedio', '?')}/part | "
        linea += f"YC: {f.get('amarillas_promedio', '?')}/part | Faltas: {f.get('faltas_promedio', '?')}/part"
        partes.append(linea)
    return "\n".join(partes)


def _formatear_proyeccion_cuantitativa(quant: dict) -> str:
    """Crea una sección legible con la proyección base del modelo cuantitativo."""
    if not quant:
        return "(No se generó proyección cuantitativa base)"
    lines = [
        "PROYECCION CUANTITATIVA BASE (modelo estadístico):",
        f"  Goles: Local {quant.get('goals_local', '?')} - Visitante {quant.get('goals_visitor', '?')} (Total: {quant.get('total_goals', '?')})",
        f"  Tiros: Local {quant.get('tiros_local', '?')} - Visitante {quant.get('tiros_visitor', '?')} (Total: {quant.get('tiros_total', '?')})",
        f"  Tiros al Arco: L {quant.get('tiros_arco_local', '?')} - V {quant.get('tiros_arco_visitor', '?')}",
        f"  Corners: ~{quant.get('corners_total', '?')} total",
        f"  Amarillas: ~{quant.get('yc_total', '?')} total",
        f"  Faltas: ~{quant.get('fouls', '?')}",
        f"  Prob BTTS: {quant.get('btts_yes', '?')} | Prob Over 2.5: {quant.get('over25_yes', '?')}",
        "---",
        "Valida estos números con el contexto. Si discrepan, ajusta y explica por qué.",
    ]
    return "\n".join(lines)


def _crear_prompt_usuario(datos_resumidos: dict, prediccion_resumida: dict, quant_projections: dict) -> str:
    """Construye el prompt de usuario con los datos del partido formateados."""
    prompt = f"""
## DATOS DEL PARTIDO

**{datos_resumidos.get('partido', 'Desconocido')}**
Liga: {datos_resumidos.get('liga', 'Desconocida')}
Fecha: {datos_resumidos.get('fecha', 'Desconocida')}
{f"Estadio: {datos_resumidos.get('estadio', {}).get('nombre', '?')} ({datos_resumidos.get('estadio', {}).get('ciudad', '?')}, cap: {datos_resumidos.get('estadio', {}).get('capacidad', '?')})" if datos_resumidos.get('estadio') else ""}

PRIORIDAD: Proyecta estadísticas del partido (tiros, goles por mitad, amarillas, corners, faltas). El análisis 1X2/BTTS/Over es SECUNDARIO. Enfócate en el CONTEXTO.

### PROYECCIÓN CUANTITATIVA BASE
{_formatear_proyeccion_cuantitativa(quant_projections)}

### ARBITRO
{_formatear_arbitro_header(datos_resumidos)}

### CUOTAS DEL BOOKMAKER (Betsafe)
{_formatear_cuotas_betsafe_para_prompt(datos_resumidos.get('_cuotas', {})) if datos_resumidos.get('_cuotas') else f'''- Local: {datos_resumidos['cuotas'].get('local', 'N/D')}
- Empate: {datos_resumidos['cuotas'].get('empate', 'N/D')}
- Visitante: {datos_resumidos['cuotas'].get('visitante', 'N/D')}
- Over 2.5: {datos_resumidos['cuotas'].get('over_25', 'N/D')}
- Under 2.5: {datos_resumidos['cuotas'].get('under_25', 'N/D')}
- BTTS Si: {datos_resumidos['cuotas'].get('btts_si', 'N/D')}
- BTTS No: {datos_resumidos['cuotas'].get('btts_no', 'N/D')}'''}

### FORMA BSD (agregados)
{_formatear_forma_bsd_compact(datos_resumidos)}

### TABLA DE POSICIONES
{_format_standings_section(datos_resumidos)}
IMPORTANTE: Usa la tabla para determinar CUANTOS PARTIDOS QUEDAN (total equipos - 1 = partidos en la temporada, resta los PJ de cada equipo). Si un equipo ya completó su cupo o le quedan pocos partidos, eso define la urgencia y motivación.

### ESTILOS DE ENTRENADORES (BSD v2)
{resumir_manager_v2_para_prompt(datos_resumidos)}

### HEAD TO HEAD (últimos 3 años)
- NOTA: Solo se muestran enfrentamientos recientes. El H2H antiguo (>3 años) fue excluido.
{f"- {json.dumps(datos_resumidos.get('h2h', {}), indent=2, ensure_ascii=False)}" if datos_resumidos.get('h2h') and datos_resumidos['h2h'].get('total_partidos') else "- No hay enfrentamientos previos registrados. Ignora el factor H2H."}

### PREDICCION ML (BSD CatBoost) - Referencia secundaria
- Probabilidades 1X2: L={prediccion_resumida.get('prob_local', 'N/D')}% / E={prediccion_resumida.get('prob_empate', 'N/D')}% / V={prediccion_resumida.get('prob_visitante', 'N/D')}%"
- xG esperado: Local {prediccion_resumida.get('xG_local', 'N/D')} - Visitante {prediccion_resumida.get('xG_visitante', 'N/D')}
- Prob. Over 2.5: {prediccion_resumida.get('prob_over_25', 'N/D')}%
- Prob. BTTS: {prediccion_resumida.get('prob_btts', 'N/D')}%
- Marcador más probable: {prediccion_resumida.get('marcador_probable', 'N/D')}
- Confianza del modelo: {prediccion_resumida.get('confianza_modelo', 'N/D')}

### ALINEACIÓN DEL PARTIDO (SofaScore)
- La alineación de SofaScore es la única fuente de disponibilidad de jugadores.
- Si hay CONFLICTOS BSD vs SofaScore, SofaScore MANDA (el jugador JUEGA).
{f"### ALINEACIÓN CONFIRMADA (SofaScore)\\n- {json.dumps(datos_resumidos.get('_sofascore', {}).get('alineaciones', {}), indent=2, ensure_ascii=False)}" if datos_resumidos.get("_sofascore", {}).get("alineaciones", {}).get("local", {}).get("confirmada") else ""}
{f"### ALINEACIÓN PRELIMINAR (NO CONFIRMADA)\\n- {json.dumps(datos_resumidos.get('_sofascore', {}).get('alineaciones', {}), indent=2, ensure_ascii=False)}" if datos_resumidos.get("_sofascore", {}).get("alineaciones") and not datos_resumidos.get("_sofascore", {}).get("alineaciones", {}).get("local", {}).get("confirmada") else ""}
{f"### SIN ALINEACIÓN DISPONIBLE\\n- SofaScore no tiene alineación todavía (suele salir ~1h antes del partido). Asume plantilla tipo." if not datos_resumidos.get("_sofascore", {}).get("alineaciones") else ""}

### BAJAS BSD (lesionados/suspendidos - referencia secundaria)
{json.dumps(datos_resumidos.get('bajas_bsd', {}), indent=2, ensure_ascii=False)}
- NOTA: Si un jugador está aquí Y NO en SofaScore -> probable baja real.
- Si un jugador está aquí Y SÍ en SofaScore -> JUEGA.

---
## DATOS ESTADÍSTICOS DETALLADOS (fuente principal para proyecciones)

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

---
## DATOS ADICIONALES BSD v2
{_v2_sections(datos_resumidos)}

---
Analiza los datos anteriores. PRIORIZA las proyecciones estadísticas (tiros, goles por mitad, amarillas, corners, faltas). Usa el CONTEXSO como base de todas tus proyecciones. El análisis 1X2/BTTS/Over 2.5 es SECUNDARIO y debe ir al final.
"""
    return prompt


def _format_standings_section(datos_resumidos: dict) -> str:
    """Muestra standings. Omite para copas donde la tabla es combinada y confunde."""
    rn = (datos_resumidos.get("_v2_detail") or {}).get("round_number")
    liga = datos_resumidos.get("liga", "")
    if rn and _es_fase_eliminatoria_final(liga, rn):
        fase = _nombre_fase(liga, rn) if rn else "Copa"
        return f"({fase} - tabla general no aplica. Mira la Ronda/Fase en CONTEXTO y la forma reciente.)"

    partes = [
        _formatear_standings_para_prompt(datos_resumidos),
        _v2_standings_section(datos_resumidos),
    ]
    content = "\n".join(p for p in partes if p.strip())
    return content if content.strip() else "(No hay tabla de posiciones disponible para este partido)"


def _v2_standings_section(datos_resumidos: dict) -> str:
    v2 = datos_resumidos.get("_bsd_v2", {})
    if not v2:
        return ""
    home_id = datos_resumidos.get("home_team_id")
    away_id = datos_resumidos.get("away_team_id")
    return resumir_standings_v2_para_prompt(v2, home_id, away_id)


def _nombre_fase(liga: str, round_number: int) -> str:
    """Convierte round_number de BSD v2 a etiqueta legible para el LLM."""
    if "Champions" in liga or "Europa" in liga:
        fases = {29: "FINAL", 28: "Semifinal", 27: "Cuartos de Final",
                 25: "Octavos de Final", 17: "Ronda de Playoff",
                 1: "Fase Liga", 2: "Fase Liga", 3: "Fase Liga",
                 4: "Fase Liga", 5: "Fase Liga", 6: "Fase Liga",
                 7: "Fase Liga", 8: "Fase Liga"}
        return fases.get(round_number, f"Ronda {round_number}")
    if "Libertadores" in liga or "Sudamericana" in liga:
        if round_number <= 8:
            return f"Fase de grupos (fecha {round_number})"
        if round_number <= 16:
            return "Octavos de Final"
        if round_number <= 24:
            return "Cuartos de Final"
        if round_number <= 28:
            return "Semifinal"
        return "FINAL"
    return f"Jornada {round_number}"


def _es_fase_eliminatoria_final(liga: str, round_number: int) -> bool:
    """True si es una fase donde la tabla general no aplica.
    Para copas: la tabla viene combinada (todos los grupos juntos) y confunde al LLM."""
    copas = ["Champions", "Europa", "Libertadores", "Sudamericana"]
    if any(c in liga for c in copas):
        return True  # Toda copa usa formato distinto a liga
    return False


def _v2_sections(datos_resumidos: dict) -> str:
    v2 = datos_resumidos.get("_bsd_v2", {})
    if not v2:
        return "(No se obtuvieron datos adicionales de BSD v2)"

    partes = []
    detail = datos_resumidos.get("_v2_detail", {})
    if detail:
        has_any = any(detail.get(k) for k in ["is_local_derby", "is_neutral_ground", "round_number"])
        has_any = has_any or detail.get("travel_distance_km") is not None
        has_any = has_any or (detail.get("weather") and detail["weather"].get("description"))
        if has_any:
            ln = ["\n### CONTEXTO DEL PARTIDO (BSD v2)"]
            if detail.get("is_local_derby"):
                ln.append("- DERBY LOCAL")
            if detail.get("is_neutral_ground"):
                ln.append("- Cancha neutral")
            if detail.get("round_number") is not None:
                rn = detail["round_number"]
                fase = _nombre_fase(datos_resumidos.get("liga", ""), rn)
                ln.append(f"- Ronda/Fase: {fase}")
            if detail.get("travel_distance_km") is not None:
                ln.append(f"- Distancia de viaje visitante: {detail['travel_distance_km']} km")
            if detail.get("weather") and detail["weather"].get("description"):
                w = detail["weather"]
                ln.append(f"- Clima: {w['description']} (código {w.get('code')})")
            partes.append("\n".join(ln))

    partes.append(resumir_stats_v2_para_prompt(v2))
    partes.append(resumir_player_stats_v2_para_prompt(v2))
    partes.append(resumir_metadata_v2_para_prompt(v2))
    partes.append(resumir_squads_v2_para_prompt(v2))
    partes.append(resumir_arbitro_v2_para_prompt(datos_resumidos))

    return "\n".join(p for p in partes if p.strip())


def _call_deepseek(client, prompt_usuario: str, include_reasoning: bool = True) -> tuple:
    """Llama a DeepSeek y retorna (contenido, razonamiento)."""
    extra_body = {}
    if include_reasoning:
        extra_body["include_reasoning"] = True

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
        extra_body=extra_body,
    )

    finish_reason = response.choices[0].finish_reason
    razonamiento = getattr(response.choices[0].message, "reasoning_content", None) or ""
    contenido = response.choices[0].message.content

    return contenido, razonamiento, finish_reason


# ═══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL MEJORADA
# ═══════════════════════════════════════════════════════════════

def analizar_partido(datos_resumidos: dict, prediccion_resumida: dict) -> str:
    """
    Análisis completo con modelo cuantitativo + LLM + persistencia.
    Mantiene la firma original para compatibilidad.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY no configurada. Agrégala en el archivo .env")

    # 1. Inicializar DB
    db.init_db()

    # 2. Ejecutar modelo cuantitativo
    print("  [QuantModel] Calculando proyecciones base...")
    quant_projections, features = run_full_projection(datos_resumidos, prediccion_resumida)
    print(f"  [QuantModel] Goles: {quant_projections.get('goals_local')}-{quant_projections.get('goals_visitor')} | "
          f"Tiros: {quant_projections.get('tiros_total')} | Corners: {quant_projections.get('corners_total')} | "
          f"YC: {quant_projections.get('yc_total')}")

    # 3. Análisis LLM (monolítico, 1 llamada)
    print("  [Analyzer] Llamando a DeepSeek...")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    prompt_usuario = _crear_prompt_usuario(datos_resumidos, prediccion_resumida, quant_projections)

    contenido, razonamiento, finish_reason = _call_deepseek(client, prompt_usuario, include_reasoning=True)
    if contenido is None:
        print("  [DeepSeek] Agotó tokens en reasoning, reintentando sin reasoning...")
        contenido, razonamiento, finish_reason = _call_deepseek(client, prompt_usuario, include_reasoning=False)
    if contenido is None:
        raise RuntimeError("El modelo no devolvió contenido visible.")

    if finish_reason == "length":
        print(f"  [ADVERTENCIA] Respuesta truncada. Considera aumentar MAX_TOKENS.")
    if razonamiento:
        print(f"  [DeepSeek] Razonó {len(razonamiento)} chars internamente")

    analysis_text = contenido
    llm_projections = _extract_llm_projections(analysis_text)

    # 4. Evaluar Kelly stakes y agregar al output
    kelly_section = _build_kelly_section(datos_resumidos, quant_projections)
    if kelly_section:
        analysis_text += f"\n\n═══════════════════════════════════════════\nRECOMENDACIONES CUANTITATIVAS (Kelly Criterion)\n═══════════════════════════════════════════\n{kelly_section}"

    # 5. Guardar en base de datos
    match_id = str(datos_resumidos.get("id", datos_resumidos.get("match_id", "unknown")))
    match_name = datos_resumidos.get("partido", "Desconocido")
    league = datos_resumidos.get("liga", "Desconocida")
    match_date = datos_resumidos.get("fecha", "")
    betsafe_odds = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))

    pred_id = db.save_prediction(
        match_id=match_id,
        match_name=match_name,
        league=league,
        match_date=match_date,
        quant_projections=quant_projections,
        llm_projections=llm_projections,
        betsafe_odds=betsafe_odds,
        bsd_pred=prediccion_resumida,
        features=features,
    )
    print(f"  [DB] Predicción guardada con ID={pred_id}")

    # 6. Guardar bets recomendadas en DB
    _save_recommended_bets(pred_id, match_id, datos_resumidos, quant_projections)

    return analysis_text


def _extract_llm_projections(text: str) -> dict:
    """Extrae proyecciones del texto del LLM de forma heurística para guardar en DB."""
    import re
    proj = {}
    # Goles
    m = re.search(r"(?:goles|Goles)\s*:?\s*Local\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*Visitante", text, re.IGNORECASE)
    if m:
        proj["goals_local"] = float(m.group(1))
        proj["goals_visitor"] = float(m.group(2))

    # Tiros total
    m = re.search(r"(?:Tiros totales|Total.*tiros)\s*:?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        proj["tiros_total"] = float(m.group(1))

    # Corners total
    m = re.search(r"(?:Corners total|Total.*corners)\s*:?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        proj["corners_total"] = float(m.group(1))

    # Amarillas total
    m = re.search(r"(?:Amarillas total|Total.*amarillas|YC total)\s*:?\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        proj["yc_total"] = float(m.group(1))

    # Recommendation
    m = re.search(r"\[RECOMENDACI[ÓO]N FINAL\](.+?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if m:
        proj["recommendation"] = m.group(1).strip()[:500]

    # Confidence
    m = re.search(r"confianza\s*:?\s*(Alta|Media|Baja)", text, re.IGNORECASE)
    if m:
        proj["confidence"] = m.group(1).capitalize()

    return proj


def _build_kelly_section(datos_resumidos: dict, quant_projections: dict) -> str:
    """Construye la sección de Kelly stakes para mercados clave."""
    betsafe = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    markets = betsafe.get("markets", betsafe)
    if not markets or not isinstance(markets, dict):
        return ""

    lines = []

    # Evaluar mercados estadísticos si hay líneas
    # Nota: Las líneas exactas de Betsafe para stats suelen estar en markets como TSTOUM, TOCO, TOYC
    # Extraemos las líneas de forma heurística si existen
    for market_key in ["TSTOUM", "TOCO", "TOYC", "TOSG"]:
        market = markets.get(market_key)
        if not market:
            continue
        # Normalmente Betsafe devuelve dict con 'over' y 'under' o 'line' + 'odds'
        over_odds = None
        under_odds = None
        line = None
        if isinstance(market, dict):
            over_odds = market.get("over", market.get("Over"))
            under_odds = market.get("under", market.get("Under"))
            line = market.get("line", market.get("linea"))
            if line is None and isinstance(over_odds, dict):
                line = over_odds.get("line")
                over_odds = over_odds.get("odds", over_odds.get("cuota"))
            if line is None and isinstance(under_odds, dict):
                line = under_odds.get("line")
                under_odds = under_odds.get("odds", under_odds.get("cuota"))

        if line is None or over_odds is None or under_odds is None:
            continue

        metric_map = {"TSTOUM": "tiros", "TOCO": "corners", "TOYC": "tarjetas", "TOSG": "tiros al arco"}
        proj_total = None
        if market_key == "TSTOUM":
            proj_total = quant_projections.get("tiros_total")
        elif market_key == "TOCO":
            proj_total = quant_projections.get("corners_total")
        elif market_key == "TOYC":
            proj_total = quant_projections.get("yc_total")

        if proj_total is None:
            continue

        recs = evaluate_stat_market(
            mercado=metric_map.get(market_key, market_key),
            linea=float(line),
            cuota_over=float(over_odds),
            cuota_under=float(under_odds),
            proj_total=proj_total,
            proj_std=2.0,
        )
        if recs:
            formatted = format_recommendations(recs)
            if formatted and "Ninguna" not in formatted:
                lines.append(f"  {market_key} (línea {line}):")
                lines.append(formatted)

    # Evaluar 1X2 si hay cuotas
    cl = markets.get("1X2", {}).get("local") if isinstance(markets.get("1X2"), dict) else betsafe.get("local")
    cd = markets.get("1X2", {}).get("empate") if isinstance(markets.get("1X2"), dict) else betsafe.get("empate")
    cv = markets.get("1X2", {}).get("visitante") if isinstance(markets.get("1X2"), dict) else betsafe.get("visitante")

    if cl and cd and cv:
        # Calibrar probs usando quant_model
        pl = calibrate_probability(quant_projections.get("btts_yes") or 0.5)  # fallback
        # mejor: inferir de goles proyectados via Poisson aproximado
        import math
        gl = quant_projections.get("goals_local", 1.2)
        gv = quant_projections.get("goals_visitor", 1.0)
        # Aproximación muy básica para 1X2 desde goles esperados
        lambda_sum = gl + gv
        # Esto es solo un placeholder; el agente de valor del pipeline hace esto mejor
        # Por ahora, omitimos 1X2 Kelly en el output directo si no hay probs claras
        pass

    return "\n".join(lines) if lines else ""


def _save_recommended_bets(pred_id: int, match_id: str, datos_resumidos: dict, quant_projections: dict):
    """Registra en DB las apuestas recomendadas por Kelly."""
    betsafe = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    markets = betsafe.get("markets", betsafe)
    if not markets or not isinstance(markets, dict):
        return

    for market_key in ["TSTOUM", "TOCO", "TOYC"]:
        market = markets.get(market_key)
        if not market or not isinstance(market, dict):
            continue
        over_odds = market.get("over", market.get("Over"))
        under_odds = market.get("under", market.get("Under"))
        line = market.get("line", market.get("linea"))
        if isinstance(over_odds, dict):
            line = line or over_odds.get("line")
            over_odds = over_odds.get("odds", over_odds.get("cuota"))
        if isinstance(under_odds, dict):
            line = line or under_odds.get("line")
            under_odds = under_odds.get("odds", under_odds.get("cuota"))
        if line is None or over_odds is None or under_odds is None:
            continue

        metric_map = {"TSTOUM": "tiros", "TOCO": "corners", "TOYC": "tarjetas"}
        proj_total = None
        if market_key == "TSTOUM":
            proj_total = quant_projections.get("tiros_total")
        elif market_key == "TOCO":
            proj_total = quant_projections.get("corners_total")
        elif market_key == "TOYC":
            proj_total = quant_projections.get("yc_total")

        if proj_total is None:
            continue

        recs = evaluate_stat_market(
            mercado=metric_map.get(market_key, market_key),
            linea=float(line),
            cuota_over=float(over_odds),
            cuota_under=float(under_odds),
            proj_total=proj_total,
            proj_std=2.0,
        )
        for r in recs:
            if r.recomendado:
                db.save_bet(
                    prediction_id=pred_id,
                    match_id=match_id,
                    mercado=r.mercado,
                    seleccion=r.seleccion,
                    cuota=r.cuota,
                    stake=r.stake,
                    unidades=r.unidades,
                    stake_pct=r.stake_pct,
                    kelly_edge=r.kelly_edge,
                    kelly_fraction=r.kelly_fraction,
                )


if __name__ == "__main__":
    print("=== Probando Analyzer v2 ===\n")
    datos_ejemplo = {
        "id": "test-match-001",
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
            "remates_promedio": 15, "remates_arco_promedio": 5.2,
            "amarillas_promedio": 1.5, "faltas_promedio": 10,
        },
        "forma_visitante": {
            "forma_string": "WDLWW", "victorias": 3, "empates": 1, "derrotas": 1,
            "goles_favor_ultimos_n": 9, "goles_contra_ultimos_n": 5,
            "xG_promedio": 1.5, "xG_contra_promedio": 1.1,
            "remates_promedio": 13, "remates_arco_promedio": 4.1,
            "amarillas_promedio": 1.8, "faltas_promedio": 12,
        },
        "h2h": {
            "total_partidos": 10, "victorias_local": 4, "empates": 2,
            "victorias_visitante": 4, "goles_local": 15, "goles_visitante": 14,
            "promedio_goles": 2.9,
        },
        "entrenador_local": {"nombre": "Flick", "formacion": "4-2-3-1", "perfil": "attacking"},
        "entrenador_visitante": {"nombre": "Ancelotti", "formacion": "4-3-3", "perfil": "balanced"},
        "_sofascore": {"alineaciones": {"local": {"confirmada": True}}},
        "home_team": "Barcelona",
        "away_team": "Real Madrid",
    }
    pred_ejemplo = {
        "prob_local": 45.0, "prob_empate": 25.0, "prob_visitante": 30.0,
        "xG_local": 1.7, "xG_visitante": 1.2,
        "prob_over_25": 65.0, "prob_btts": 60.0,
        "marcador_probable": "2-1",
        "confianza_modelo": 0.72,
    }

    try:
        resultado = analizar_partido(datos_ejemplo, pred_ejemplo)
        print(resultado)
    except Exception as e:
        print(f"Error: {e}")
        print("(Esperado si OPENROUTER_API_KEY no está configurada)")
