"""
analyzer.py — Analizador de Mercados Estadísticos usando OpenRouter.

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
    _formatear_xi_impact_sofascore_para_prompt,
    _formatear_player_avgs_sofascore_para_prompt,
)
from betano_client import _formatear_cuotas_betano_para_prompt
from competition_config import get_competition_flags
from bsd_client_v2 import (
    resumir_stats_v2_para_prompt,
    resumir_metadata_v2_para_prompt,
    resumir_player_stats_v2_para_prompt,
    resumir_standings_v2_para_prompt,
    resumir_squads_v2_para_prompt,
    resumir_manager_v2_para_prompt,
    resumir_arbitro_v2_para_prompt,
    resumir_lineups_v2_para_prompt,
    resumir_player_impact_v2_para_prompt,
    resumir_player_avgs_v2_para_prompt,
    resumir_odds_v2_para_prompt,
    resumir_motivacion_v2_para_prompt,
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

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.8")
INCLUDE_REASONING = MODEL_NAME.startswith("deepseek/")
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
Decisión correcta: NO recomendar el Over como value por cuota muy baja; tampoco recomendar Under sin evidencia abrumadora. Omitir este mercado.

Ejemplo E: No forzar overs de un equipo sin urgencia y con ataque debilitado
Partido: Equipo local defensivo vs visitante eliminado/sin urgencia, visitante sin 2 atacantes titulares.
Datos: visitante promedia 15 tiros y 5 corners en liga/torneo.
Proyección correcta: Visitante 7-10 tiros, 1-3 corners. NO recomendar Over tiros/corners visitante salvo línea extremadamente baja y XI confirmado favorable.
Justificación: el "ADN de volumen" no pesa más que contexto competitivo + bajas ofensivas + game state probable.

Ejemplo F: Dominio no siempre significa corners
Partido: Favorito amplio en casa, cuota 1.25, ataque eficiente por dentro.
Datos: favorito genera muchas big chances y pocos tiros bloqueados/centros; rival puede soltarse si recibe gol temprano.
Proyección correcta: favorito puede ganar 3-0/4-1 sin superar corners de equipo. Evitar Over corners del favorito si depende solo de "dominio".
Justificación: goles tempranos y ocasiones limpias reducen necesidad de centros, rechaces y corners.

Ejemplo G: Pick con plan en vivo/cash out
Pick: Over 9.5 corners. Para que salga: ritmo por bandas, 4+ corners al descanso o 6+ al minuto 60, equipos buscando área y tiros bloqueados. Peligra si: 0-1 corners al 30', favorito gana temprano y baja ritmo, ataques por dentro sin centros. Cash out: considerar parcial si al 55'-60' hay 3 o menos corners y el partido se enfría; mantener si hay dominio territorial, centros y corners recientes aunque el conteo vaya justo.

Ejemplo H: Tiros totales altos NO implican tiros al arco altos
Partido: Equipo local obligado domina, 70% posesión, rival en bloque bajo; local sin su 9 titular.
Datos: local promedia 21 tiros, pero muchos remates bloqueados/fuera y xG por tiro bajo.
Proyección correcta: Local 18-22 tiros, pero solo 4-6 al arco. NO recomendar Over tiros al arco alto si falta calidad/definidor.
Justificación: volumen territorial puede producir corners/tiros, pero no precisión. Distingue cantidad de calidad.

Ejemplo I: Árbitro alto no basta para Over tarjetas
Árbitro: 5.6 YC/part en Libertadores. Partido con un equipo ya clasificado, marcador controlado y pocas faltas tácticas.
Proyección correcta: 4-5 amarillas, no 6+ automático. Over tarjetas solo fuerte si árbitro + tensión + faltas + duelos coinciden.
Justificación: el árbitro sube el piso, pero el guion del partido define si se llega al techo.

Ejemplo J: Una buena proyección NO siempre es una apuesta fuerte
Proyección: Total tarjetas 4.5-5.5. Línea: Over 4.5 @2.00.
Decisión correcta: puede ser una lectura razonable, pero NO es stake 10/15 si el rango empieza justo en la línea y depende de un solo factor (árbitro/final). Puede ser stake 5 solo con apoyos adicionales claros; si no, omitir o bajar a observación en vivo.
Justificación: "Rango toca línea" no equivale a value fuerte.

Ejemplo K: No duplicar exposición en picks correlacionados
Mercados: Over 3.5 tarjetas y Over 4.5 tarjetas en el mismo partido.
Decisión correcta: NO recomendar ambos. Elegir una sola línea y asignar stake 5/10/15 según edge real; si el edge es leve, omitir.
Justificación: si falla la hipótesis del árbitro/ritmo, pierden ambos. El portfolio debe evitar duplicar el mismo guion.

Ejemplo L: Under con rango cruzado NO es pick
Pick tentador: Fluminense Under 3.5 tiros al arco @1.72.
Proyección propia: 2-4 tiros al arco, mediana 3.0.
Decisión correcta: NO recomendar prepartido, porque el rango ya incluye 4 y cruza la línea. Puede quedar como observación en vivo si el guion arranca frío.
Justificación: la mediana no basta; en mercados volátiles como tiros al arco, el rango completo debe quedar del lado ganador o tener margen extraordinario.
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
   - Si un equipo está eliminado/sin urgencia real Y además tiene bajas ofensivas importantes, penaliza sus tiros, tiros al arco y corners. No dejes que el promedio histórico o el "ADN" del equipo anule ese contexto.
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
   - GAME STATE: si un favorito puede marcar temprano o ganar con alta eficacia, puede bajar su volumen de tiros/corners tras ponerse en ventaja. El rival puede subir tiros/corners en 2T si el partido se abre.
   - ¿Derby o clásico? (BSD v2: is_local_derby=true) -> MÁS tarjetas, MÁS faltas, partido cortado.
   - ¿Viaje largo del visitante? (travel_distance_km alto) -> posible fatiga en 2do tiempo.
   - ¿Clima extremo? -> menos ritmo, menos tiros.
   - ¿Cancha neutral? -> quita ventaja de localía.

d) ¿CUÁNTOS PARTIDOS QUEDAN?:
   - Pocos partidos + necesidad urgente = equipos que arriesgan más.
   - Muchos partidos + posición cómoda = ritmo relajado.
   - Si un equipo ya está salvado/eliminado, sus estadísticas pueden empeorar frente a uno que se juega la vida.

e) AJUSTE POR FORMATO COMPETITIVO:
   - FINAL / partido único: baja ritmo de 1T, baja tiros/corners del equipo que pueda aceptar bloque bajo, y no transformes "motivación máxima" en Overs automáticos. En finales, un equipo puede ganar/competir con 25-35% posesión, pocos tiros y pocos corners.
   - FINAL con favorito dominador vs rival reactivo: permite asedio unilateral. El total de corners/tiros puede salir alto por un solo equipo, mientras el rival queda muy bajo. No repartas corners/tiros de forma simétrica por promedios históricos.
   - LIGA regular: los promedios recientes y splits casa/fuera pesan más; ajusta por tabla, calendario y necesidad real, pero evita dramatizar cada partido como final.
   - ELIMINATORIA ida/vuelta: el global manda. Si la ida fue 0-0 o el global está empatado, 1T más táctico y 2T más abierto. Si un equipo está obligado, suben tiros/corners/faltas de ese equipo.
   - DESCENSO / promoción: tensión alta, más duelos y faltas; pero si ambos temen perder, puede bajar precisión, tiros al arco y goles en 1T.

f) SELECCIONES VS CLUBES:
   - No uses forma de clubes como sustituto directo de forma de selección. En selecciones pesan más: convocados, XI confirmado, rol internacional, continuidad del DT, viaje/sede y fase del torneo.
   - World Cup 2026: torneo de máxima presión. Evalúa fase real: grupos, eliminatoria o final. Si hay tabla/grupo, úsala; si es eliminatoria, considera prórroga/penales y 1T más conservador.
   - Mundial en sede neutral: baja ventaja de localía salvo local geográfico/crowd evidente. La presión puede subir faltas/tarjetas solo si hay tensión real, rivalidad, marcador apretado o duelos físicos.
   - International Friendly Games: rotación alta, cambios masivos, menor fricción y menor fiabilidad de datos. Baja confianza global prepartido y prioriza esperar XI/en vivo si no hay alineaciones confirmadas.
   - En amistosos, tarjetas/faltas rara vez pasan de stake 5: exige árbitro + rivalidad + XI competitivo + línea clara. Si falta uno de esos apoyos, omite.
   - En selecciones, una baja clave puede pesar más por falta de reemplazo natural, pero debes justificarla con rol, minutos, rating/valor, titularidad, balón parado o impacto estadístico; no basta el nombre.

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

PASO 4 - PLAN EN VIVO PARA PICKS:
Para cada pick recomendado, explica cómo debe ir el partido para que la apuesta esté viva y qué señales la ponen en peligro.
Incluye disparadores prácticos de cash out parcial/total, especialmente para usuarios que juegan stake fijo y gestionan en vivo.
No prometas cash out rentable: describe señales objetivas para proteger exposición cuando el guion invalida el edge inicial.

PASO 5 - FILTRO DE APUESTA CON STAKES 5 / 10 / 15:
El usuario solo juega stakes fijos de 5, 10 o 15 unidades. NO uses 1/10, 2/10, porcentajes, Kelly, bankroll ni stakes intermedios.
Tu trabajo es decidir si un mercado se omite o si merece Stake: 5, Stake: 10 o Stake: 15.
- Stake 5: edge positivo pero moderado, mercado volátil, XI preliminar, una o dos dudas razonables, o pick que requiere gestión en vivo cuidadosa.
- Stake 10: edge claro, rango bien ubicado del lado ganador, cuota razonable y al menos 2-3 apoyos independientes de datos/contexto.
- Stake 15: edge muy fuerte y raro; línea claramente mal puesta, rango amplio del lado ganador, datos de alta calidad, Betano desalineado, XI/fuentes confiables y baja dependencia de un solo guion. No uses stake 15 en amistosos, XI preliminar, mercados muy volátiles sin margen extra, picks correlacionados o edges basados en un solo factor.
- Si el edge es leve, el rango cruza la línea o el pick depende de demasiada varianza, NO lo recomiendes.
- Si recomiendas un pick, escribe exactamente uno de estos formatos: "Stake: 5", "Stake: 10" o "Stake: 15".
- Máximo 2 picks prepartido salvo que haya edges independientes y muy claros. Calidad > cantidad.
- En International Friendly Games, máximo 1 pick prepartido. Si el XI no está confirmado, lo normal es escribir "esperar XI/en vivo" y no forzar pick.
- No recomiendes dos picks que dependan del mismo guion. Ejemplo prohibido: Over 3.5 tarjetas + Over 4.5 tarjetas; Over tiros total + Over tiros del equipo dominante si ambos dependen del mismo asedio.

═══════════════════════════════════════
REGLAS POR MERCADO ESTADÍSTICO
═══════════════════════════════════════

── TIROS TOTALES Y AL ARCO ──
- Fuente primaria: desglose POR PARTIDO en FORMA RECIENTE (SofaScore). NO promedies ciegamente.
- Usa los SPLITS ESTADÍSTICOS DERIVADOS: para local prioriza torneo+casa/casa; para visitante prioriza torneo+fuera/fuera. Cruza producción propia con lo que concede el rival.
- CONSISTENCIA: un equipo que hace 18-20-22-19-21 tiros es MUY distinto a uno que hace 8-25-6-22-9.
- Tiros 1T vs 2T: equipos que empiezan fuertes (presión alta primeros 20 min) generan más en 1T.
- Si el local es amplio favorito y el visitante se encierra: el local tira MUCHO (20+), el visitante tira POCO (<6).
- Pero si el favorito es muy eficiente o puede resolver temprano, no infles automáticamente tiros/corners del favorito: ganar claro no siempre implica volumen extremo.
- Para recomendar Over tiros de un equipo visitante/underdog, exige urgencia real, XI ofensivo/creadores disponibles y línea suficientemente baja. Si está eliminado/sin urgencia o con bajas ofensivas, reduce la confianza u omite.
- En finales o partidos únicos, si el underdog/equipo reactivo puede competir aceptando bloque bajo, reduce sus tiros esperados 20-35% y sus tiros al arco si no tiene que perseguir marcador. No uses promedios de liga para inflar su volumen.
- Si hay shotmap con xG/xGOT: evalúa calidad de ocasion, no solo volumen.
- TIROS AL ARCO son mercado de CALIDAD, no solo de volumen. Para Over tiros al arco exige: delanteros/creadores disponibles, xG razonable, big chances, bajo porcentaje de tiros bloqueados/fuera y rival que permita remates limpios.
- Si un equipo tiene muchas bajas ofensivas, juega contra bloque bajo, remata mucho de fuera o suele tener muchos tiros bloqueados, proyecta tiros totales/corners altos pero tiros al arco moderados.
- No recomiendes Over tiros al arco si tu argumento principal es "habrá muchos tiros". Debe haber evidencia de calidad/precisión.
- Para Under tiros al arco, NO basta con bajas ofensivas o split bajo. Revisa si el XI restante tiene atacantes directos/rápidos, si el rival puede dejar transiciones al perseguir marcador y si el rango superior cruza la línea.
- En tiros al arco por equipo, si proyectas X-Y y la línea queda dentro del rango, omite. Ejemplo: 2-4 vs Under 3.5 = NO pick; 1-3 vs Under 3.5 sí puede ser candidato.
- Ajusta por COMPETICIÓN: tiros en liga contra rivales débiles NO se transfieren 1:1 a copa contra rivales de élite.
- Cuota Betano <1.35 en Over tiros: el mercado ya descuenta volumen alto. Solo recomienda Under con evidencia ABRUMADORA.

── GOLES POR MITAD (1T / 2T) ──
- Analiza el ritmo de inicio: ¿qué equipo suele marcar temprano?
- ¿Hay dato de goles 1T/2T en forma reciente? Usalo.
- Equipos que presionan alto al inicio: más goles en 1T.
- Fatiga del visitante (viaje largo, poco descanso) -> más goles en contra en 2T.
- Si el partido es de vuelta con resultado ajustado: primer tiempo trabado, goles en 2T cuando se abren.

── TARJETAS AMARILLAS ──
- ARBITRO: fuente principal. Siempre usa el promedio de la COMPETICIÓN ACTUAL del partido.
- Fuentes de árbitro en orden: SofaScore > ValueStats > Flashscore > WhoScored > Transfermarkt > BSD v2.
- En splits, YC P/Prov significa tarjetas propias/provocadas. Para tarjetas de un equipo, combina disciplina propia con tarjetas que suele provocar el rival.
- No proyectes Over tarjetas alto SOLO por árbitro. Exige al menos 2-3 apoyos: tensión real, necesidad competitiva, faltas tácticas, duelo físico, rivalidad, marcador apretado o equipos con YC/faltas altas.
- Para recomendar Over tarjetas 4.5 o superior, exige normalmente faltas proyectadas >=24-25 o equipos claramente tarjeteros/duelos de alta fricción. Si las faltas proyectadas quedan en 20-23, el Over 4.5 debe omitirse salvo evidencia excepcional.
- Si el rango de tarjetas empieza justo en la línea (ej. proyección 4.5-5.5 vs Over 4.5), no lo trates como edge fuerte. Puede ser stake 5 solo con varios apoyos independientes y cuota claramente mal puesta; si no, omite.
- Si un equipo ya está clasificado/sin urgencia y el partido puede ser controlado, baja 0.5-1.0 amarillas aunque el árbitro sea alto.
- Derby/rivalidad = MÁS tarjetas. Equipo sin nada que jugar = MENOS tarjetas. Descenso = MÁS tarjetas.
- TARJETAS POR MITAD:
  - 1T: árbitros suelen ser más permisivos al inicio. Pero si hay derby o mucha tensión, pueden salir temprano.
  - 2T: más tarjetas en promedio (fatiga, nervios, resultado apretado, pérdidas de tiempo). Ajusta al alza.

── CORNERS ──
- Fuente primaria: corners por partido en FORMA RECIENTE + promedio temporada.
- Cruza corners a favor del equipo con corners concedidos por el rival, usando casa/fuera y competición actual cuando estén disponibles.
- Equipos con laterales ofensivos + centros frecuentes = corners altos.
- Local dominante vs visitante encerrado = muchos corners locales (>7).
- Dominio territorial NO equivale automáticamente a corners. Big chances limpias, goles tempranos, tiros poco bloqueados o ataques por dentro pueden producir marcador amplio con pocos corners.
- Si un equipo puede dominar 65%+ posesión contra bloque bajo, proyecta posible concentración de corners: el favorito puede llevarse 65-85% de los corners totales y el rival quedarse en 0-2. No repartas 3-5 corners al rival solo por promedio histórico.
- En finales, un rival reactivo que marca primero puede reducir aún más sus tiros/corners propios y defender área. Ajusta por game state probable.
- Corners por equipo son de alta varianza. Para recomendarlos exige evidencia múltiple: corners propios recientes, corners concedidos por rival, volumen de centros/bloqueos y game state favorable.
- No recomiendes Over corners de un equipo eliminado/sin urgencia con bajas ofensivas salvo línea muy baja y XI confirmado favorable.
- Partido cerrado y táctico = pocos corners (<8 total).
- Por mitad: 2T suele tener más corners (equipos se vuelcan, más urgencia).

── FALTAS ──
- Fuente: promedio de faltas BSD + faltas por partido en stats SofaScore.
- Cruza faltas cometidas por el equipo con faltas recibidas/provocadas por el rival.
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
- IMPORTANCIA DE BAJAS: no llames "importante", "clave" o "sensible" a una baja solo por nombre o reputación. Exige evidencia explícita: titularidad/minutos recientes, impacto por jugador, rating/valor alto, rol táctico claro (9, creador, lateral profundo, central líder, mediocentro de corte, portero), capitán, balón parado o producción estadística.
- Si SofaScore incluye `impacto ALTA/MEDIA/BAJA/DESCONOCIDA` junto a una baja, úsalo como resumen de evidencia, no como verdad absoluta. ALTA/MEDIA puede justificar ajuste; BAJA/DESCONOCIDA no debe mover mucho la proyección.
- Si aparece IMPACTO XI SOFASCORE, úsalo para evaluar continuidad/peso interno de quienes sí juegan: portero suplente, XI alternativo, falta de creadores, delanteros con bajo uso/producción, o titulares habituales. No confundas impacto de XI con impacto de bajas. El score del XI NO mide calidad absoluta entre equipos; compáralo siempre contra tabla, nivel de liga, cuotas, valor de plantilla y contexto.
- Si aparece MEDIAS POR JUGADOR, úsalo para ajustar mercados de equipo y jugadores: tiros/90 + SOT/90 + xG/90 para amenaza, xA/key passes/90/centros para creacion y corners, YC/90/faltas/90/acciones defensivas para tarjetas. No infieras rendimiento alto por nombre si las medias no lo respaldan.
- Si solo tienes nombre/posición/estado, trátala como "baja de disponibilidad", no como baja de alto impacto. Puedes ajustar levemente por posición, pero baja la confianza y no construyas picks sobre esa ausencia.
- Si las bajas probadas afectan delanteros, extremos, laterales profundos o creadores principales, baja tiros al arco, corners y goles esperados del equipo afectado. Si la alineación es preliminar, baja la confianza en mercados dependientes de jugadores.
- SI HAY ESTADÍSTICAS POR JUGADOR: un delantero con alto xG reciente indica que el equipo genera para él.
- CONTRADICCIÓN INTERNA: si escribes que un equipo no tiene urgencia, tiene bajas ofensivas o probablemente será apático, no recomiendes overs de tiros/corners de ese equipo salvo que expliques evidencia muy fuerte y línea baja.
- No inventes datos. Si proyectas un número, fundamentalo con 2-3 razones claras.
- SE CONSERVADOR: prefiere quedarte corto en proyecciones a inflarlas. Un error de +3 tiros proyectados es peor que -3.
- IMPORTANTE: Incluye una nota de CONFIANZA para cada proyección principal (Alta/Media/Baja) según la calidad de los datos disponibles.

═══════════════════════════════════════
CUOTAS DE BETANO
═══════════════════════════════════════
- Betano es la única línea oficial para cuotas y EV. BSD/Polymarket solo sirve como comparador de sesgo o desalineación, nunca como fuente principal.
- Mercados Betano permitidos para picks prepartido: 1X2, goles principales, BTTS principal, corners, tarjetas, tiros/remates, remates de jugador y faltas.
- No uses hándicap, doble oportunidad ni combinadas como picks prepartido.
- Para calcular edge en mercados estadísticos: compara tu proyección con la línea de Betano.
- Compara mercados de total del partido contra la proyección total del partido.
- En mercados Betano por equipo como "Cruzeiro Remates totales", "Cruzeiro Tiros al Arco", "Cruzeiro Total de Faltas Cometidas" o "Cruzeiro Córners", compara la línea contra la proyección Local/Visitante correspondiente.
- En mercados Betano de jugador como "Jugador - remates" o "Jugador - tiros al arco", compara contra MEDIAS POR JUGADOR: tiros/90, SOT/90, xG/90, rol, minutos, titularidad y rival. Solo recomendar si el jugador aparece titular/probable o hay evidencia fuerte de minutos; si está en bajas/dudas o no aparece en XI probable, omite.
- Para remates de jugador, Stake 15 solo si XI confirmado, media individual muy por encima de la línea, rol ofensivo claro, rival concede volumen y cuota desalineada. Con XI preliminar o suplente probable, máximo Stake 5 o esperar en vivo.
- No incluyas picks que contradigan tu propia proyección: Over solo si el rango queda claramente por encima de la línea; Under solo si el rango queda claramente por debajo. Si el rango cruza la línea o el edge es muy leve, omite el mercado.
- Requisitos mínimos de edge: tiros total/equipo necesita al menos ~1.5 tiros de margen sobre la línea; corners total/equipo ~1.0 corner; tarjetas ~0.7 amarillas; goles ~0.25 xG. Si no alcanza, omite.
- Usa estos márgenes como piso, no como garantía. Stake 5 puede aceptar edge positivo moderado; stake 10 exige margen claro; stake 15 exige margen amplio, datos robustos y baja varianza relativa. Si el mercado es volátil (tarjetas/corners por equipo/tiros al arco), exige margen extra o evidencia cualitativa fuerte.
- Para tiros al arco exige margen mínimo ~1.0-1.5 SOT y evidencia de calidad. Si el equipo tiene muchas bajas ofensivas o remata mucho bloqueado/fuera, sube el umbral o omite.
- Para Under tiros al arco por equipo: el techo de tu rango debe quedar por debajo de la línea. Si proyectas 2-4 contra Under 3.5, está prohibido recomendarlo como pick de valor.
- Para tarjetas exige edge y contexto. Árbitro alto sin tensión/faltas suficientes = pick de confianza Media como máximo, o se omite si la línea es exigente.
- Si cuota Over en algún mercado es <1.35, el mercado ya descuenta volumen alto. No lo incluyas como pick de valor; solo recomienda Under con evidencia abrumadora.
- En el bloque de valor NO incluyas mercados descartados, borderline, filas con ❌, ni picks "solo si aparece línea". Los descartes pueden ir brevemente en RECOMENDACIÓN FINAL como "evitar".
- Una proyección razonable NO obliga a recomendar pick. Antes de recomendar, confirma que la apuesta merece al menos Stake: 5. Si no, omite.
- Si el partido es amistoso internacional, solo recomienda pick si existe Betano real con línea comparable, XI confirmado o mercado poco dependiente de titulares, y edge robusto. Sin Betano real o con XI preliminar, no hay pick estadístico fuerte prepartido.

CONTROL FINAL ANTES DE RECOMENDAR PICKS:
1. ¿El pick contradice motivación, bajas ofensivas, XI preliminar o game state probable? Si sí, omítelo.
2. ¿Es Over tiros al arco basado solo en volumen de tiros, sin calidad/definidores/xG? Omítelo.
3. ¿El rango cruza la línea o el margen no supera el mínimo de edge? Omítelo. No uses solo la mediana para justificarlo.
4. ¿Es un Over de cuota <1.35? Omítelo como pick.
5. ¿Es corner por equipo? Exige más evidencia y usa confianza más baja salvo datos muy robustos.
6. ¿Hay otro pick recomendado que depende del mismo guion? Si sí, elige solo el mejor y elimina el otro.
7. ¿El pick depende de un solo factor dominante (solo árbitro, solo final, solo posesión, solo promedio)? Si sí, omítelo o déjalo como observación en vivo.
8. ¿El pick depende de bajas? Si no puedes demostrar la importancia de esas bajas con datos de rol/minutos/impacto/valor/rating, baja confianza u omítelo.
9. ¿Lo jugarías al menos con Stake: 5? Si no, NO lo incluyas en Picks Con Valor. Si sí, clasifícalo en 5/10/15 según fuerza del edge.
10. ¿Es un pick marcado como NO, omitido, descartado, borderline o "solo si"? Entonces NO puede aparecer en Picks Con Valor.
11. ¿Es amistoso internacional y ya hay un pick recomendado? Entonces no agregues más picks prepartido.

REGLAS DE GESTIÓN EN VIVO / CASH OUT:
- Para cada pick claro, da un guion esperado: qué debe verse en 1T, al descanso y entre 55'-65'.
- Define señales de peligro por mercado:
  * Over tiros/corners: ritmo bajo, pocos ataques al área, pocos centros/bloqueos, favorito cómodo, conteo muy por debajo del ritmo necesario.
  * Over tiros al arco: muchos tiros fuera/bloqueados, bajo xG, porteros sin trabajo, ataques desde lejos, falta de remates claros en área.
  * Under tiros/corners: partido roto, gol temprano que obliga a perseguir, ida y vuelta, muchas transiciones, conteo alto antes del 30'.
  * Over tarjetas/faltas: árbitro permisivo, pocas faltas tácticas, marcador sin tensión, jugadores evitando duelos.
  * Under goles/BTTS No: gol temprano, xG alto, big chances repetidas, defensas abiertas o roja temprana.
  * Over goles/BTTS Sí: pocos tiros al arco, xG bajo, ritmo lento, favorito controla sin arriesgar.
- Cash out parcial: usar cuando el pick sigue vivo pero el guion se deteriora o el conteo va justo.
- Cash out total: usar cuando el guion principal se rompe claramente y la apuesta depende de varianza extrema.
- Mantener: si el conteo va justo pero los indicadores subyacentes son buenos (ritmo, ataques, centros, faltas, tensión, xG, tiros bloqueados).

FORMATO DE RESPUESTA DISCORD-FRIENDLY:
Objetivo: que se vea claro en Discord móvil/escritorio. NO uses tablas Markdown porque Discord no las renderiza bien. NO escribas paredes de texto. Usa bloques cortos, separadores y negritas.

Reglas visuales:
- Título inicial en una línea: **ANÁLISIS ESTADÍSTICO: Local vs Visitante**
- Usa separadores `---` entre bloques principales.
- Usa encabezados con `**1. Contexto**`, `**2. Validación Base**`, etc.
- Usa bullets cortos. Máximo 2 líneas por bullet.
- Usa `>` para destacar una conclusión o alerta.
- Los picks van como tarjetas numeradas, no como tabla.
- Evita tablas con `|`.
- Evita repetir datos enormes del prompt.

Estructura obligatoria:

**ANÁLISIS ESTADÍSTICO: Local vs Visitante**
Liga - fase/fecha si está disponible

---
**1. Contexto**
- Qué se juega: ...
- Local: perfil + urgencia.
- Visitante: perfil + urgencia.
- Guion probable: ...
> Factor clave: ...

---
**2. Validación Base**
- Base: Goles X.X | Tiros X | Corners X | Amarillas X | Faltas X.
- Ajuste principal: sube/baja/confirmo por ...
- Riesgo de lectura: ...

---
**3. Proyecciones**
**Tiros**
- Total: Local X-Y | Visitante X-Y | Partido X-Y.
- Al arco: Local X-Y | Visitante X-Y | Partido X-Y.
- 1T: ... | 2T: ...
- Confianza: Alta/Media/Baja. Razón breve.

**Goles por mitad**
- 1T: ~X.X goles esperados.
- 2T: ~X.X goles esperados.
- Lectura: ...

**Amarillas**
- Local X-Y | Visitante X-Y | Total X-Y.
- 1T: ~X | 2T: ~Y.
- Árbitro: [nombre] - [estilo] - media YC en esta competición: X.X.
- Si el árbitro NO está asignado, escribe exactamente: "Árbitro: no asignado; factor arbitral omitido."
- Confianza: ...

**Corners**
- Local X-Y | Visitante X-Y | Total X-Y.
- 1T: X-Y | 2T: X-Y.
- Confianza: ...

**Faltas**
- Local X-Y | Visitante X-Y | Total ~X-Y.
- Lectura: ...

**Saques de banda** (omitir si no hay suficiente info)
- Rango: bajo/medio/alto. Razón breve.

---
**4. Picks Con Valor**
Incluye SOLO picks recomendados. No incluyas descartes, borderline, ni picks sin cuota comparable.
El usuario solo juega stakes fijos 5, 10 o 15. Incluye solo mercados aptos para una de esas tres unidades.
Guía rápida: Stake 5 = edge positivo/moderado o más varianza; Stake 10 = edge claro y robusto; Stake 15 = edge excepcional, muy bien respaldado y baja dependencia de un solo guion.
Prohibido incluir aquí tarjetas con "NO", "omitido", "descartar", "borderline" o "solo si aparece línea". Esos mercados van únicamente en Recomendación Final / Evitar.

**Pick 1 - Mercado**
- Selección: ...
- Línea/cuota Betano: ...
- Proyección: ...
- Edge: ...
- Confianza: Alta/Media/Baja.
- Riesgo principal: ...
- Qué tendría que salir mal: ...
- Stake: [elige exactamente uno: 5, 10 o 15].
- Razón: 1-2 líneas.

**Pick 2 - Mercado** ...

Si no hay value claro:
> No hay pick estadístico fuerte apto para Stake 5/10/15 prepartido. Mejor esperar alineaciones/en vivo.

---
**5. Plan En Vivo / Cash Out**
Para cada pick recomendado, usa este formato de tarjeta. No uses tabla.

**Pick 1 - [mercado]**
- Para que salga: señales concretas por minuto/conteo.
- Peligra si: señales de guion contrario.
- Cash out parcial: cuándo proteger parte.
- Cash out total: cuándo el guion se rompió.
- Mantener si: indicadores subyacentes siguen bien aunque el conteo vaya justo.

---
**6. Resumen 1X2 / Goles** (prioridad baja)
- 1X2: L X% | E X% | V X%. Lectura breve.
- BTTS: Sí X% | No X%. Lectura breve.
- Over 2.5: Over X% | Under X%. Lectura breve.

---
**7. Recomendación Final**
> Mejor pick: [mercado + selección + cuota mínima].

- Motivo principal: ...
- Factor que puede invalidarlo: ...
- Evitar: mercados que parecen tentadores pero no tienen edge."""


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
        f"  Faltas: Local {quant.get('fouls_local', '?')} - Visitante {quant.get('fouls_visitor', '?')} (Total: {quant.get('fouls', '?')})",
        f"  Prob BTTS: {quant.get('btts_yes', '?')} | Prob Over 2.5: {quant.get('over25_yes', '?')}",
        "---",
        "Valida estos números con el contexto. Si discrepan, ajusta y explica por qué.",
    ]
    return "\n".join(lines)


def _fmt_num(value) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else "-"


def _split_metric_pair(split: dict, metric: str) -> str:
    data = split.get(metric, {}) if isinstance(split, dict) else {}
    return f"{_fmt_num(data.get('for'))}/{_fmt_num(data.get('against'))}"


def _formatear_splits_cuantitativos(datos_resumidos: dict, features: dict | None) -> str:
    """Muestra splits derivados de SofaScore para mercados estadísticos."""
    splits = (features or {}).get("market_splits", {})
    if not splits:
        return "(No hay splits cuantitativos suficientes de SofaScore para este partido.)"

    local_team = datos_resumidos.get("partido", "").split(" vs ")[0] if " vs " in datos_resumidos.get("partido", "") else "Local"
    away_team = datos_resumidos.get("partido", "").split(" vs ")[1] if " vs " in datos_resumidos.get("partido", "") else "Visitante"

    rows = [
        "F/C = a favor/concedidos. YC P/Prov = tarjetas propias/provocadas. Faltas Com/Rec = cometidas/recibidas.",
        "Equipo | Split | n | Tiros F/C | Arco F/C | Corners F/C | YC P/Prov | Faltas Com/Rec",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    split_order = {
        "local": [("current_home", "torneo+casa"), ("home", "casa"), ("current", "torneo"), ("recent", "reciente")],
        "visitor": [("current_away", "torneo+fuera"), ("away", "fuera"), ("current", "torneo"), ("recent", "reciente")],
    }

    for side, team_name in [("local", local_team), ("visitor", away_team)]:
        team_splits = splits.get(side, {})
        shown = 0
        for split_key, label in split_order[side]:
            split = team_splits.get(split_key)
            if not split:
                continue
            rows.append(
                f"{team_name} | {label} | {split.get('n', '-')} | "
                f"{_split_metric_pair(split, 'shots')} | "
                f"{_split_metric_pair(split, 'sot')} | "
                f"{_split_metric_pair(split, 'corners')} | "
                f"{_split_metric_pair(split, 'yc')} | "
                f"{_split_metric_pair(split, 'fouls')}"
            )
            shown += 1
            if shown >= 3:
                break

    return "\n".join(rows)


def _formatear_cuotas_prepartido_para_prompt(datos_resumidos: dict) -> str:
    """Formatea cuotas reales de Betano o deja claro que solo hay fallback básico."""
    cuotas_betano = datos_resumidos.get("_cuotas")
    if cuotas_betano:
        return _formatear_cuotas_betano_para_prompt(cuotas_betano)

    cuotas = datos_resumidos.get("cuotas", {}) or {}
    if not cuotas:
        return "### CUOTAS\n(No hay cuotas prepartido disponibles.)"

    return "\n".join([
        "### CUOTAS BASE / FALLBACK (sin Betano real)",
        "No se pegó URL completa de Betano. Estas cuotas son solo referencia básica; no hay mercados estadísticos reales para picks.",
        f"- Local: {cuotas.get('local', 'N/D')}",
        f"- Empate: {cuotas.get('empate', 'N/D')}",
        f"- Visitante: {cuotas.get('visitante', 'N/D')}",
        f"- Over 2.5: {cuotas.get('over_25', 'N/D')}",
        f"- Under 2.5: {cuotas.get('under_25', 'N/D')}",
        f"- BTTS Si: {cuotas.get('btts_si', 'N/D')}",
        f"- BTTS No: {cuotas.get('btts_no', 'N/D')}",
    ])


def _crear_prompt_usuario(datos_resumidos: dict, prediccion_resumida: dict, quant_projections: dict, quant_features: dict | None = None) -> str:
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

{_formatear_cuotas_prepartido_para_prompt(datos_resumidos)}

{resumir_odds_v2_para_prompt(datos_resumidos.get("_bsd_v2", {}))}

### FORMA BSD (agregados)
{_formatear_forma_bsd_compact(datos_resumidos)}

### SPLITS ESTADÍSTICOS DERIVADOS (SofaScore → modelo cuantitativo)
{_formatear_splits_cuantitativos(datos_resumidos, quant_features)}

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
- Probabilidades 1X2: L={prediccion_resumida.get('prob_local', 'N/D')}% / E={prediccion_resumida.get('prob_empate', 'N/D')}% / V={prediccion_resumida.get('prob_visitante', 'N/D')}%
- xG esperado: Local {prediccion_resumida.get('xG_local', 'N/D')} - Visitante {prediccion_resumida.get('xG_visitante', 'N/D')}
- Prob. Over 2.5: {prediccion_resumida.get('prob_over_25', 'N/D')}%
- Prob. BTTS: {prediccion_resumida.get('prob_btts', 'N/D')}%
- Marcador más probable: {prediccion_resumida.get('marcador_probable', 'N/D')}
- Confianza del modelo: {prediccion_resumida.get('confianza_modelo', 'N/D')}

### ALINEACIÓN DEL PARTIDO (SofaScore)
- La alineación y `missingPlayers` de SofaScore son la fuente principal de disponibilidad de jugadores.
- Si la alineación está CONFIRMADA, SofaScore MANDA. Si está PRELIMINAR/POSIBLE, úsala como probable XI.
- En bajas/dudas, `missingPlayers` de SofaScore manda sobre BSD.
{_sofascore_lineup_status_note(datos_resumidos)}
{f"### SIN ALINEACIÓN DISPONIBLE\n- SofaScore no tiene alineación todavía (suele salir ~1h antes del partido). Asume plantilla tipo." if not datos_resumidos.get("_sofascore", {}).get("alineaciones") else ""}

{resumir_lineups_v2_para_prompt(datos_resumidos)}

{_format_bajas_bsd_section(datos_resumidos)}

---
## DATOS ESTADÍSTICOS DETALLADOS (fuente principal para proyecciones)

    {_formatear_detalle_evento_para_prompt(datos_resumidos)}
    {_formatear_h2h_sofascore_para_prompt(datos_resumidos)}
    {_formatear_alineaciones_para_prompt(datos_resumidos)}
    {_formatear_xi_impact_sofascore_para_prompt(datos_resumidos)}
    {_formatear_player_avgs_sofascore_para_prompt(datos_resumidos)}
    {_formatear_players_stats_para_prompt(datos_resumidos)}
    {resumir_player_avgs_v2_para_prompt(datos_resumidos)}
    {resumir_player_impact_v2_para_prompt(datos_resumidos)}
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


def _sofascore_lineup_status_note(datos_resumidos: dict) -> str:
    """Nota visible para que el modelo no trate XI preliminares como confirmados."""
    alin = ((datos_resumidos.get("_sofascore") or {}).get("alineaciones") or {})
    if not alin:
        return ""
    sides = [alin.get("local") or {}, alin.get("visitante") or {}]
    if sides and all(side.get("confirmada") for side in sides if side):
        return "- Estado SofaScore: ALINEACIÓN CONFIRMADA."
    return (
        "- Estado SofaScore: ALINEACIÓN PRELIMINAR/POSIBLE, NO confirmada. "
        "No trates el XI como definitivo; úsalo solo para perfil probable. "
        "Las bajas/dudas de `missingPlayers` siguen siendo la referencia principal de disponibilidad."
    )


def _format_standings_section(datos_resumidos: dict) -> str:
    """Muestra standings. Omite para copas donde la tabla es combinada y confunde."""
    flags = _competition_flags_for_match(datos_resumidos)
    if flags.get("is_friendly"):
        return "(International Friendly Games: sin tabla competitiva; motivación y rotación inciertas.)"

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
    international_context = _format_international_context(datos_resumidos)
    knockout_context = _format_knockout_playoff_context(datos_resumidos)
    bsd_context = resumir_motivacion_v2_para_prompt(datos_resumidos)
    standings = (datos_resumidos.get("_sofascore") or {}).get("standings") or {}
    if not standings:
        return "\n".join(
            p for p in [
                international_context,
                knockout_context,
                bsd_context,
                "(Sin tabla SofaScore/BSD confiable: no afirmes escenarios de clasificación específicos.)",
            ] if p
        )

    local_team, away_team = _match_team_names(datos_resumidos)
    local_key, local_info = _find_standing_team(local_team, standings)
    away_key, away_info = _find_standing_team(away_team, standings)
    rows = sorted(
        [(k, v) for k, v in standings.items() if not str(k).startswith("_liga")],
        key=lambda x: x[1].get("posicion") or 999,
    )
    if not local_info or not away_info or not rows:
        fallback = "(Tabla disponible, pero no se pudo vincular ambos equipos del partido.)"
        if knockout_context:
            return "\n".join(p for p in [international_context, knockout_context, bsd_context] if p)
        return "\n".join(p for p in [international_context, bsd_context, fallback] if p)

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

    content = "\n".join(p for p in [international_context, knockout_context, "\n".join(lines)] if p)
    if bsd_context:
        content += "\n" + bsd_context
    return content


def _league_id_for_match(datos_resumidos: dict):
    league_id = datos_resumidos.get("league_id")
    if league_id:
        return league_id
    league_obj = datos_resumidos.get("league") or {}
    return league_obj.get("id")


def _competition_flags_for_match(datos_resumidos: dict) -> dict:
    return get_competition_flags(
        _league_id_for_match(datos_resumidos),
        datos_resumidos.get("liga"),
    )


def _format_international_context(datos_resumidos: dict) -> str:
    """Reglas visibles para selecciones, Mundial y amistosos."""
    flags = _competition_flags_for_match(datos_resumidos)
    if not flags.get("is_international"):
        return ""

    lines = []
    if flags.get("is_world_cup"):
        lines.append(
            "- Contexto internacional: World Cup 2026. Prioriza fase real, tabla/grupo si existe, sede neutral y presión competitiva."
        )
        lines.append(
            "- Implicación estadística: 1T puede ser más táctico; en eliminatoria considera prórroga/penales y no infles tarjetas/faltas sin tensión real."
        )
    elif flags.get("is_friendly"):
        lines.append(
            "- Contexto internacional: amistoso. Sin tabla competitiva; alta rotación, cambios masivos y motivación incierta."
        )
        lines.append(
            "- Implicación estadística: baja confianza prepartido; máximo 1 pick y preferir esperar XI/en vivo si alineaciones no están confirmadas."
        )
    else:
        lines.append(
            "- Contexto internacional: no trates forma de clubes como forma directa de selección; valida convocados, XI y rol internacional."
        )
    return "\n".join(lines)


def _format_knockout_playoff_context(datos_resumidos: dict) -> str:
    """Detecta eliminatorias/playoffs aunque la tabla de liga no incluya a ambos equipos."""
    ss = datos_resumidos.get("_sofascore") or {}
    torneo = ss.get("torneo", "") or ""
    liga = datos_resumidos.get("liga", "") or ""
    v2_detail = datos_resumidos.get("_v2_detail") or {}
    fase = _nombre_fase(liga, v2_detail.get("round_number")) if v2_detail.get("round_number") is not None else ""
    has_classification_market = _has_bookmaker_classification_market(datos_resumidos)

    context_text = " ".join([torneo, liga, fase]).lower()
    is_promotion_playoff = any(
        token in context_text
        for token in ("relegation/promotion", "relegation playoff", "promotion playoff", "descenso", "promoción", "promocion")
    )
    is_knockout = has_classification_market or is_promotion_playoff
    if not is_knockout:
        return ""

    lines = []
    if is_promotion_playoff:
        lines.append(
            "- Contexto especial: eliminatoria/playoff de promoción-descenso. La tabla de liga regular no vincula necesariamente a ambos equipos; trátalo como mata-mata."
        )
    elif has_classification_market:
        lines.append(
            "- Contexto especial: Betano ofrece mercado 'Avanza/Método de clasificación'; trátalo como eliminatoria, no como liga regular."
        )

    h2h_note = _recent_h2h_score_note(datos_resumidos)
    if h2h_note:
        lines.append(h2h_note)

    if lines:
        lines.append("- Implicación estadística: presión máxima, primer tramo táctico, más fricción/faltas; el 2T puede abrirse si el global sigue empatado o hay gol.")
    return "\n".join(lines)


def _has_bookmaker_classification_market(datos_resumidos: dict) -> bool:
    cuotas = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    markets = cuotas.get("markets", {}) if isinstance(cuotas, dict) else {}
    for market in markets.values() if isinstance(markets, dict) else []:
        name = (market.get("name") or market.get("label") or "").lower() if isinstance(market, dict) else ""
        if "avanza" in name or "clasificaci" in name or "classification" in name:
            return True
    return False


def _recent_h2h_score_note(datos_resumidos: dict) -> str:
    h2h = datos_resumidos.get("h2h") or {}
    ultimos = h2h.get("ultimos_enfrentamientos") or []
    if not ultimos:
        return ""
    last = ultimos[0]
    score = str(last.get("score") or "").strip()
    date = str(last.get("date") or "")[:10]
    if score:
        return f"- Antecedente inmediato de la llave/H2H reciente: {score}" + (f" el {date}." if date else ".")
    return ""


def _format_bajas_bsd_section(datos_resumidos: dict) -> str:
    """Muestra bajas BSD solo si SofaScore no trajo missingPlayers."""
    if _has_sofascore_missing_players(datos_resumidos):
        return ""
    bajas = datos_resumidos.get("bajas_bsd", {})
    if not bajas:
        return ""
    has_any_baja = False
    for side in ("local", "visitante"):
        side_bajas = bajas.get(side, {}) if isinstance(bajas, dict) else {}
        if side_bajas.get("confirmadas") or side_bajas.get("dudas") or side_bajas.get("desconocido"):
            has_any_baja = True
            break
    if not has_any_baja:
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
    groups = standings.get("groups") if isinstance(standings, dict) else None
    if rn and _es_copa_con_grupos(liga, rn):
        if isinstance(groups, dict) and len(groups) > 1:
            return ""
        if isinstance(groups, list) and len(groups) > 1:
            return ""
        if isinstance(rows, dict) and len(rows) > 1:
            return ""
        if isinstance(rows, list) and len(rows) > 8:
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


def _call_model(client, prompt_usuario: str, include_reasoning: bool = False) -> tuple:
    """Llama al modelo configurado y retorna (contenido, razonamiento)."""
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
    print(f"  [Analyzer] Llamando a {MODEL_NAME}...")
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    prompt_usuario = _crear_prompt_usuario(datos_resumidos, prediccion_resumida, quant_projections, features)

    contenido, razonamiento, finish_reason = _call_model(client, prompt_usuario, include_reasoning=INCLUDE_REASONING)
    if contenido is None:
        print("  [Modelo] Agotó tokens en reasoning, reintentando sin reasoning...")
        contenido, razonamiento, finish_reason = _call_model(client, prompt_usuario, include_reasoning=False)
    if contenido is None:
        raise RuntimeError("El modelo no devolvió contenido visible.")

    if finish_reason == "length":
        print(f"  [ADVERTENCIA] Respuesta truncada. Considera aumentar MAX_TOKENS.")
    if razonamiento:
        print(f"  [Modelo] Razonó {len(razonamiento)} chars internamente")

    analysis_text = contenido
    llm_projections = _extract_llm_projections(analysis_text)

    # 4. No anexar Kelly cuantitativo crudo al texto final.
    # El LLM ya recibe la base cuantitativa y devuelve picks ajustados
    # por contexto. Anexar Kelly desde quant_projections aquí puede contradecir
    # esas proyecciones finales, porque usa la media base previa al ajuste
    # cualitativo.

    # 5. Guardar en base de datos
    match_id = str(datos_resumidos.get("id", datos_resumidos.get("match_id", "unknown")))
    match_name = datos_resumidos.get("partido", "Desconocido")
    league = datos_resumidos.get("liga", "Desconocida")
    match_date = datos_resumidos.get("fecha", "")
    bookmaker_odds = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))

    pred_id = db.save_prediction(
        match_id=match_id,
        match_name=match_name,
        league=league,
        match_date=match_date,
        quant_projections=quant_projections,
        llm_projections=llm_projections,
        bookmaker_odds=bookmaker_odds,
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
    """Convierte cuotas/lineas del bookmaker a float tolerando coma decimal."""
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
    """Clasifica mercados estadísticos del bookmaker por nombre visible."""
    n = (name or "").lower()
    if "faltas" in n or "fouls" in n:
        return "faltas"
    if "tiros al arco" in n or "shots on goal" in n or "on target" in n:
        return "tiros_arco"
    if "tiros de esquina" in n or "córner" in n or "córners" in n or "corners" in n or "corner" in n:
        return "corners"
    if "tarjetas" in n or "yellow cards" in n or "amarillas" in n:
        return "tarjetas"
    if "total de tiros" in n or "remates" in n or "total shots" in n:
        return "tiros"
    return ""


def _is_full_match_stat_total(name: str) -> bool:
    """Evita mercados por equipo/mitad que no son comparables con la proyección total."""
    n = (name or "").lower()
    excluded = [
        "1er tiempo", "1º tiempo", "primer tiempo", "1st half",
        "2º tiempo", "2do tiempo", "segundo tiempo", "2nd half",
        "mitad", "half",
        "jugador", "player",
    ]
    if any(token in n for token in excluded):
        return False
    if " - " in (name or "") or "|" in (name or ""):
        return False
    return True


def _selection_side(label: str) -> str:
    label = (label or "").lower()
    if re.search(r"\b(menos|under)\b", label):
        return "under"
    if re.search(r"\b(mas|más|over)\b", label):
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


def _iter_stat_markets(bookmaker: dict):
    """Normaliza mercados estadísticos aunque el bookmaker use ids dinámicos."""
    markets = bookmaker.get("markets", bookmaker) if isinstance(bookmaker, dict) else {}
    if not isinstance(markets, dict):
        return []

    normalized = []
    template_map = {
        "TSTOUM": "tiros",
        "TOCO": "corners",
        "TOYC": "tarjetas",
        "TOSG": "tiros_arco",
        "TOFOUL": "faltas",
        "TOFLS": "faltas",
    }

    for key, market in markets.items():
        if not isinstance(market, dict):
            continue
        if market.get("category") in {"Team Match Stats", "Team Corners", "Team Fouls", "Player Shots", "Player Shots On Target"}:
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
    if kind == "faltas":
        return quant_projections.get("fouls")
    if kind == "tiros_arco":
        local = quant_projections.get("tiros_arco_local")
        visitor = quant_projections.get("tiros_arco_visitor")
        if local is not None and visitor is not None:
            return local + visitor
    return None


def _collect_best_stat_recommendations(bookmaker: dict, quant_projections: dict):
    metric_labels = {"tiros": "tiros", "corners": "corners", "tarjetas": "tarjetas", "tiros_arco": "tiros al arco", "faltas": "faltas"}
    metric_std = {"tiros": 3.0, "corners": 2.0, "tarjetas": 1.6, "tiros_arco": 1.8, "faltas": 3.5}
    best_by_kind = {}

    for market in _iter_stat_markets(bookmaker):
        proj_total = _projection_for_market(market["kind"], quant_projections)
        if proj_total is None:
            continue
        recs = evaluate_stat_market(
            mercado=metric_labels.get(market["kind"], market["kind"]),
            linea=market["line"],
            cuota_over=market["over_odds"],
            cuota_under=market["under_odds"],
            proj_total=proj_total,
            proj_std=metric_std.get(market["kind"], 2.0),
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
    bookmaker = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    lines = []
    for market, rec in _collect_best_stat_recommendations(bookmaker, quant_projections):
        lines.append(
            f"  {market['name']} (línea {market['line']}):\n"
            f"  • {rec.mercado} | {rec.seleccion} @ {rec.cuota:.2f} | "
            f"Prob: {rec.prob_estimada:.1%} | Edge: {rec.kelly_edge:.1%} | "
            f"Stake: {rec.stake:.2f}u ({rec.stake_pct:.2f}%)"
        )

    return "\n".join(lines) if lines else ""


def _save_recommended_bets(pred_id: int, match_id: str, datos_resumidos: dict, quant_projections: dict):
    """Registra en DB las apuestas recomendadas por Kelly."""
    bookmaker = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    for _, r in _collect_best_stat_recommendations(bookmaker, quant_projections):
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
