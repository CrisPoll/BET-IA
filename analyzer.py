"""
analyzer.py — Analizador de Mercados Estadísticos usando DeepSeek via OpenRouter.

VERSIÓN 2.0: Integra modelo cuantitativo + pipeline de agentes + persistencia DB + Kelly stakes.
Mantiene compatibilidad con llamadas anteriores (analizar_partido).
"""

import os
import json
import re
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
   - Mira la tabla de posiciones (si aplica). ¿Cuántos partidos quedan según el formato real? En grupos CONMEBOL suelen ser 6 PJ; en ligas ida/vuelta suele ser (total equipos - 1) * 2. Si el formato no está claro, declara incertidumbre y no inventes escenarios.
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
- Mercados estadísticos disponibles (varian por partido): Total de Tiros (TSTOUM), Total de tiros por equipo ("Equipo - Total de tiros"), Tiros al Arco (TOSG), Total Corners (TOCO), corners por equipo ("Equipo - Total de tiros de esquina"), Total Amarillas (TOYC), Goles 1T (1HTG), Goles 2T (2HTG), BTTS 1T/2T.
- Para calcular edge en mercados estadísticos: compara tu proyección con la línea de Betsafe.
- Compara mercados de total del partido contra la proyección total del partido.
- En mercados "Equipo - Total de tiros" y "Equipo - Total de tiros de esquina", compara la línea contra la proyección Local/Visitante correspondiente.
- No incluyas picks que contradigan tu propia proyección: Over solo si la mediana/rango central queda por encima de la línea; Under solo si queda por debajo. Si el rango cruza la línea o el edge es muy leve, omite el mercado.
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
Si el árbitro NO está asignado, escribe exactamente: "Árbitro: no asignado; factor arbitral omitido." No infieras estilo ni media arbitral.
Justificación (rivalidad, perfil de faltas de cada equipo, contexto)

[PROYECCIÓN DE CORNERS]
Local: X-Y | Visitante: X-Y | Total: Z
1T: X-Y | 2T: X-Y
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
        comp_ref = arb.get("_sofascore_competicion") if arb.get("_fuente_yc") == "SofaScore" else None
        if comp_ref:
            partes.append(
                f"- Media en competición actual ({comp_ref.get('nombre', 'torneo actual')}): "
                f"{comp_ref.get('yc_pp', '?')} YC/part en {comp_ref.get('partidos', '?')} partidos"
            )
            if comp_ref.get("rc_pp") is not None:
                partes.append(f"- Rojas/partido en competición actual: {comp_ref.get('rc_pp')}")
        if ref_data.get("total_partidos"):
            partes.append(f"- Partidos dirigidos: {ref_data['total_partidos']}")
        if ref_data.get("yc_pp") is not None:
            label = "Amarillas/partido global" if comp_ref else "Amarillas/partido"
            partes.append(f"- {label}: {ref_data['yc_pp']}")
        if ref_data.get("rc_pp") is not None:
            partes.append(f"- Rojas/partido: {ref_data['rc_pp']}")
        if ref_data.get("faltas_pp") is not None:
            partes.append(f"- Faltas/partido: {ref_data['faltas_pp']}")

        torneos = ref_data.get("torneos", ref_data.get("competiciones", []))
        if torneos:
            partes.append("- Promedios por competición:")
            for t in torneos[:8]:
                tn = t.get("nombre", "?")
                apps = t.get("partidos")
                yc = t.get("yc_pp", "?")
                rc = t.get("rc_pp", "?")
                pen = t.get("penaltis")
                detalle = f"{yc} YC/part, {rc} RC/part"
                if apps is not None:
                    detalle = f"{apps} part, {detalle}"
                if pen is not None:
                    detalle += f", {pen} pen"
                partes.append(f"    {tn}: {detalle}")
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
        f"  Corners: Local {quant.get('corners_local', '?')} - Visitante {quant.get('corners_visitor', '?')} (Total: {quant.get('corners_total', '?')})",
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
IMPORTANTE: Usa la tabla para determinar PTS, PJ, V/E/D, GF, GC, DG, posición, partidos restantes y escenarios de clasificación/descenso. Si no hay tabla disponible, dilo explícitamente y no inventes puntos ni escenarios.

### CONTEXTO COMPETITIVO DERIVADO
{_format_competitive_context(datos_resumidos)}

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
- La alineación y `missingPlayers` de SofaScore son la fuente principal de disponibilidad de jugadores.
- Si la alineación está CONFIRMADA, SofaScore MANDA. Si está PRELIMINAR/POSIBLE, úsala como probable XI.
- En bajas/dudas, `missingPlayers` de SofaScore manda sobre BSD.
{f"### SIN ALINEACIÓN DISPONIBLE\n- SofaScore no tiene alineación todavía (suele salir ~1h antes del partido). Asume plantilla tipo." if not datos_resumidos.get("_sofascore", {}).get("alineaciones") else ""}

{_format_bajas_bsd_section(datos_resumidos)}

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
Analiza los datos anteriores. PRIORIZA las proyecciones estadísticas (tiros, goles por mitad, amarillas, corners, faltas). Usa el CONTEXTO como base de todas tus proyecciones. El análisis 1X2/BTTS/Over 2.5 es SECUNDARIO y debe ir al final.
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
    if content.strip():
        return content
    if rn and _es_copa_con_grupos(liga, rn):
        return "(Tabla de grupo no disponible en fuente confiable. No uses tablas globales de copa ni inventes puntos/escenarios.)"
    return "(No hay tabla de posiciones disponible para este partido)"


def _format_competitive_context(datos_resumidos: dict) -> str:
    """Deriva motivación desde la tabla real del grupo cuando SofaScore la provee."""
    standings = (datos_resumidos.get("_sofascore") or {}).get("standings") or {}
    if not standings:
        return "(Sin tabla SofaScore confiable: no afirmes escenarios de clasificación específicos.)"

    local_team, away_team = _match_team_names(datos_resumidos)
    local_key, local_info = _find_standing_team(local_team, standings)
    away_key, away_info = _find_standing_team(away_team, standings)
    rows = sorted(
        [(k, v) for k, v in standings.items() if not str(k).startswith("_liga")],
        key=lambda x: x[1].get("posicion") or 999,
    )
    if not local_info or not away_info or not rows:
        return "(Tabla disponible, pero no se pudo vincular ambos equipos del partido.)"

    total_equipos = (standings.get("_liga_info") or {}).get("total_equipos")
    expected_matches = 6 if total_equipos == 4 else None
    local_rest = _remaining_matches(local_info, expected_matches)
    away_rest = _remaining_matches(away_info, expected_matches)

    lines = []
    if expected_matches:
        lines.append(
            f"- Formato detectado: grupo de {total_equipos} equipos, {expected_matches} PJ por equipo; "
            f"restan {local_rest} para {local_key} y {away_rest} para {away_key}."
        )

    lines.append(
        f"- {local_key}: #{local_info.get('posicion')} con {local_info.get('puntos')} pts, "
        f"DG {local_info.get('diferencia')}, zona actual: {_zone(local_info)}."
    )
    lines.append(
        f"- {away_key}: #{away_info.get('posicion')} con {away_info.get('puntos')} pts, "
        f"DG {away_info.get('diferencia')}, zona actual: {_zone(away_info)}."
    )

    local_pos = local_info.get("posicion")
    away_pos = away_info.get("posicion")
    if total_equipos == 4 and expected_matches:
        third = next((info for _, info in rows if info.get("posicion") == 3), None)
        fourth = next((info for _, info in rows if info.get("posicion") == 4), None)
        second = next((info for _, info in rows if info.get("posicion") == 2), None)

        if away_pos == 1 and second:
            second_max = (second.get("puntos") or 0) + 3 * _remaining_matches(second, expected_matches)
            if (away_info.get("puntos") or 0) + 1 > second_max:
                lines.append(f"- {away_key}: ya está en zona Playoffs/octavos; un empate asegura el 1er puesto del grupo.")
            else:
                lines.append(f"- {away_key}: ya está en zona Playoffs/octavos y pelea el 1er puesto del grupo.")

        if local_pos == 3 and fourth:
            local_draw_pts = (local_info.get("puntos") or 0) + 1
            fourth_max = (fourth.get("puntos") or 0) + 3 * _remaining_matches(fourth, expected_matches)
            if local_draw_pts > fourth_max:
                lines.append(
                    f"- {local_key}: lectura principal = asegurar 3er puesto/Copa Sudamericana; "
                    "con empate queda fuera del alcance del 4º."
                )
            else:
                lines.append(f"- {local_key}: pelea principalmente el 3er puesto/Copa Sudamericana.")

        if local_pos and local_pos > 2 and third and second:
            lines.append(
                f"- No describas a {local_key} como obligado a ganar para clasificar a octavos salvo que el desempate lo permita; "
                "desde la tabla está fuera de zona Playoffs y el objetivo directo es la zona indicada por SofaScore."
            )

    return "\n".join(lines)


def _format_bajas_bsd_section(datos_resumidos: dict) -> str:
    """Muestra bajas BSD solo si SofaScore no trajo missingPlayers."""
    if _has_sofascore_missing_players(datos_resumidos):
        return ""
    bajas = datos_resumidos.get("bajas_bsd", {})
    if not bajas:
        return ""
    return "\n".join([
        "### BAJAS BSD (respaldo secundario)",
        json.dumps(bajas, indent=2, ensure_ascii=False),
        "- NOTA: Usa BSD solo porque SofaScore no listó bajas/dudas en lineups.",
        "- Si un jugador está en BAJAS BSD pero aparece como titular/suplente en SofaScore -> JUEGA.",
    ])


def _has_sofascore_missing_players(datos_resumidos: dict) -> bool:
    alin = ((datos_resumidos.get("_sofascore") or {}).get("alineaciones") or {})
    for side in ("local", "visitante"):
        bajas = (alin.get(side, {}) or {}).get("bajas", {})
        if bajas.get("confirmadas") or bajas.get("dudas"):
            return True
    return False


def _match_team_names(datos_resumidos: dict) -> tuple[str, str]:
    partido = datos_resumidos.get("partido", "")
    if " vs " not in partido:
        return "Local", "Visitante"
    return partido.split(" vs ", 1)


def _find_standing_team(name: str, standings: dict) -> tuple[str | None, dict | None]:
    if name in standings:
        return name, standings[name]
    name_low = name.lower()
    for key, value in standings.items():
        if str(key).startswith("_liga"):
            continue
        key_low = str(key).lower()
        if name_low in key_low or key_low in name_low:
            return key, value
    return None, None


def _remaining_matches(info: dict, expected_matches: int | None) -> int:
    if expected_matches is None:
        return 0
    played = info.get("partidos_jugados") or 0
    return max(expected_matches - played, 0)


def _zone(info: dict) -> str:
    return info.get("zona") or info.get("descripcion") or "sin zona marcada"


def _v2_standings_section(datos_resumidos: dict) -> str:
    v2 = datos_resumidos.get("_bsd_v2", {})
    if not v2:
        return ""
    rn = (datos_resumidos.get("_v2_detail") or {}).get("round_number")
    liga = datos_resumidos.get("liga", "")
    standings = v2.get("standings", {}) if isinstance(v2, dict) else {}
    rows = standings.get("standings", []) if isinstance(standings, dict) else []
    if rn and _es_copa_con_grupos(liga, rn) and len(rows) > 8:
        return ""
    home_id = datos_resumidos.get("home_team_id")
    away_id = datos_resumidos.get("away_team_id")
    return resumir_standings_v2_para_prompt(v2, home_id, away_id)


def _es_copa_con_grupos(liga: str, round_number: int) -> bool:
    copas = ["Champions", "Europa", "Libertadores", "Sudamericana"]
    return any(c in liga for c in copas) and round_number <= 8


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
    En fase de grupos/fase liga, la tabla sí define motivación y urgencia."""
    copas = ["Champions", "Europa", "Libertadores", "Sudamericana"]
    if any(c in liga for c in copas):
        return round_number > 8
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
    if not (datos_resumidos.get("_sofascore") or {}).get("alineaciones"):
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

    # 4. No anexar Kelly cuantitativo crudo al texto final.
    # El LLM ya recibe la base cuantitativa y devuelve picks/stakes ajustados
    # por contexto. Anexar Kelly desde quant_projections aquí puede contradecir
    # esas proyecciones finales, porque usa la media base previa al ajuste
    # cualitativo.

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


def _to_float(value):
    """Convierte cuotas/lineas de Betsafe a float tolerando coma decimal."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = value.strip().replace(",", ".")
        match = re.search(r"\d+(?:\.\d+)?", clean)
        if match:
            return float(match.group(0))
    return None


def _market_kind_from_name(name: str) -> str:
    """Clasifica mercados estadísticos de Betsafe por nombre visible."""
    n = (name or "").lower()
    if "tiros al arco" in n or "shots on goal" in n or "on target" in n:
        return "tiros_arco"
    if "tiros de esquina" in n or "corners" in n or "corner" in n:
        return "corners"
    if "tarjetas" in n or "yellow cards" in n or "amarillas" in n:
        return "tarjetas"
    if "total de tiros" in n or "total shots" in n:
        return "tiros"
    return ""


def _is_full_match_stat_total(name: str) -> bool:
    """Evita mercados por equipo/mitad que no son comparables con la proyección total."""
    n = (name or "").lower()
    excluded = [
        "1er tiempo", "1º tiempo", "primer tiempo", "1st half",
        "2º tiempo", "2do tiempo", "segundo tiempo", "2nd half",
        "mitad", "half",
    ]
    if any(token in n for token in excluded):
        return False
    if " - " in (name or ""):
        return False
    return True


def _selection_side(label: str) -> str:
    label = (label or "").lower()
    if any(token in label for token in ("menos de", "under")):
        return "under"
    if any(token in label for token in ("mas de", "más de", "over")):
        return "over"
    return ""


def _extract_line_from_text(*parts) -> float:
    joined = " ".join(str(p) for p in parts if p)
    matches = re.findall(r"\((\d+(?:[.,]\d+)?)\)|(?:mas|más|menos|over|under)\s*(?:de)?\s*(\d+(?:[.,]\d+)?)", joined, flags=re.IGNORECASE)
    for grouped, direct in matches:
        value = _to_float(grouped or direct)
        if value is not None:
            return value
    return None


def _iter_stat_markets(betsafe: dict):
    """Normaliza mercados estadísticos aunque Betsafe use ids dinámicos."""
    markets = betsafe.get("markets", betsafe) if isinstance(betsafe, dict) else {}
    if not isinstance(markets, dict):
        return []

    normalized = []
    template_map = {
        "TSTOUM": "tiros",
        "TOCO": "corners",
        "TOYC": "tarjetas",
        "TOSG": "tiros_arco",
    }

    for key, market in markets.items():
        if not isinstance(market, dict):
            continue
        name = market.get("name") or market.get("label") or str(key)
        if not _is_full_match_stat_total(name):
            continue
        template = str(market.get("marketTemplateId") or market.get("templateId") or key).upper()
        kind = _market_kind_from_name(name)
        if not kind:
            for marker, mapped in template_map.items():
                if marker in template:
                    kind = mapped
                    break
        if not kind:
            continue

        line = _to_float(market.get("line", market.get("linea", market.get("lineValue"))))
        line = line if line is not None else _extract_line_from_text(name)
        over_odds = market.get("over", market.get("Over"))
        under_odds = market.get("under", market.get("Under"))

        if isinstance(over_odds, dict):
            line = line if line is not None else _to_float(over_odds.get("line", over_odds.get("linea")))
            over_odds = over_odds.get("odds", over_odds.get("cuota", over_odds.get("odd")))
        if isinstance(under_odds, dict):
            line = line if line is not None else _to_float(under_odds.get("line", under_odds.get("linea")))
            under_odds = under_odds.get("odds", under_odds.get("cuota", under_odds.get("odd")))

        for sel in market.get("selections", []) or []:
            label = sel.get("label") or sel.get("name") or ""
            side = _selection_side(label)
            if not side:
                continue
            odd = sel.get("odd", sel.get("odds", sel.get("cuota")))
            if side == "over":
                over_odds = odd
            elif side == "under":
                under_odds = odd
            line = line if line is not None else _extract_line_from_text(label)

        over_odds = _to_float(over_odds)
        under_odds = _to_float(under_odds)
        if line is None or over_odds is None or under_odds is None:
            continue

        normalized.append({
            "kind": kind,
            "name": name,
            "line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
        })

    return normalized


def _projection_for_market(kind: str, quant_projections: dict):
    if kind == "tiros":
        return quant_projections.get("tiros_total")
    if kind == "corners":
        return quant_projections.get("corners_total")
    if kind == "tarjetas":
        return quant_projections.get("yc_total")
    if kind == "tiros_arco":
        local = quant_projections.get("tiros_arco_local")
        visitor = quant_projections.get("tiros_arco_visitor")
        if local is not None and visitor is not None:
            return local + visitor
    return None


def _collect_best_stat_recommendations(betsafe: dict, quant_projections: dict):
    metric_labels = {"tiros": "tiros", "corners": "corners", "tarjetas": "tarjetas", "tiros_arco": "tiros al arco"}
    best_by_kind = {}

    for market in _iter_stat_markets(betsafe):
        proj_total = _projection_for_market(market["kind"], quant_projections)
        if proj_total is None:
            continue
        recs = evaluate_stat_market(
            mercado=metric_labels.get(market["kind"], market["kind"]),
            linea=market["line"],
            cuota_over=market["over_odds"],
            cuota_under=market["under_odds"],
            proj_total=proj_total,
            proj_std=2.0,
        )
        for rec in recs:
            if not rec.recomendado:
                continue
            current = best_by_kind.get(market["kind"])
            if current is None or rec.kelly_edge > current[1].kelly_edge:
                best_by_kind[market["kind"]] = (market, rec)

    return [best_by_kind[k] for k in sorted(best_by_kind)]


def _build_kelly_section(datos_resumidos: dict, quant_projections: dict) -> str:
    """Construye la sección de Kelly stakes para mercados clave."""
    betsafe = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    lines = []
    for market, rec in _collect_best_stat_recommendations(betsafe, quant_projections):
        lines.append(
            f"  {market['name']} (línea {market['line']}):\n"
            f"  • {rec.mercado} | {rec.seleccion} @ {rec.cuota:.2f} | "
            f"Prob: {rec.prob_estimada:.1%} | Edge: {rec.kelly_edge:.1%} | "
            f"Stake: {rec.stake:.2f}u ({rec.stake_pct:.2f}%)"
        )

    return "\n".join(lines) if lines else ""


def _save_recommended_bets(pred_id: int, match_id: str, datos_resumidos: dict, quant_projections: dict):
    """Registra en DB las apuestas recomendadas por Kelly."""
    betsafe = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    for _, r in _collect_best_stat_recommendations(betsafe, quant_projections):
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
