"""
agents_pipeline.py — Pipeline de agentes estructurado en 3 pasos.

1. ContextAgent: Extrae y resume el contexto cualitativo (motivación, estilo, dinámica).
2. ProjectionAgent: Recibe contexto + datos duros y emite proyecciones numéricas.
3. ValueAgent: Cruza proyecciones con cuotas Betsafe y emite picks con Kelly stakes.

Este enfoque reduce la carga cognitiva del LLM y mejora la precisión numérica.
También facilita few-shot prompting por etapa.
"""

import os
import json
from typing import Dict, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-v4-pro"
TEMPERATURE = 0.3
MAX_TOKENS_STEP = 8000  # Cada paso usa menos tokens

# ═══════════════════════════════════════════════════════════════
# FEW-SHOT EJEMPLOS (anclajes para reducir alucinaciones)
# ═══════════════════════════════════════════════════════════════

FEWSHOT_CONTEXT = """
Ejemplo 1 (correcto):
Partido: Manchester City vs Luton Town (Premier League, Abril 2025)
- City pelea título, 6 partidos restantes, puntero con 1 punto de ventaja.
- Luton en zona de descenso directo, necesita mínimo empate para no hundirse.
- Contexto: City propone con posesión alta, Luton se encierra y contraataca.
- Factor clave: City juega cada 3 días (Champions + Premier). Posible rotación.

Ejemplo 2 (correcto):
Partido: Boca vs River (Copa Libertadores, octavos de final, vuelta)
- Ida 1-1. Boca local en La Bombonera. River viajó 450km.
- Boca necesita ganar o empatar 0-0 (penales). River con ventaja del gol de visitante.
- Contexto: clásico muy cerrado, primer tiempo táctico, River espera contraataque.
- Factor clave: árbitro histórico 4.2 YC/partido en Libertadores. Partido cortado = menos tiros fluidos.
"""

FEWSHOT_PROJECTION = """
Ejemplo 1:
Contexto: City favorito amplio, Luton encerrado.
Datos: City promedia 18.2 tiros en casa, Luton permite 14.5 tiros de visitante.
Proyección Tiros: Local 19-21, Visitante 5-7, Total 25-27.
Justificación: City domina posesión, Luton no sale. Volume alto para local, bajo para visitante.
Rango aceptable: 24-28 tiros totales. Menos de 22 sería anómalo.

Ejemplo 2:
Contexto: Derby cerrado, árbitro estricto.
Datos: Boca promedia 13 tiros en Libertadores, River 11. Ambos con defensas sólidas.
Proyección Tiros: Local 12-14, Visitante 9-11, Total 22-24.
Justificación: Clásico + necesidad de no perder = ritmo cortado. No esperar 25+ tiros.
Rango aceptable: 20-25 totales.
"""

FEWSHOT_VALUE = """
Ejemplo 1:
Proyección tiros total: 25.5. Cuota Betsafe Over 23.5 @ 1.80. Prob estimada Over: 72%.
Edge: 72% - 55.5% = 16.5%. Kelly 1/4: stake 2.3% del bankroll.
Recomendación: APOSTAR Over 23.5 tiros @ 1.80, 2.3% stake.

Ejemplo 2:
Proyección tarjetas total: 4.5. Cuota Betsafe Over 4.5 @ 1.70. Prob estimada: 52%.
Edge: 52% - 58.8% = -6.8%. 
Recomendación: NO APOSTAR. Sin edge positivo.
"""


def _client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY no configurada")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def _call_llm(system: str, user: str, max_tokens: int = MAX_TOKENS_STEP) -> str:
    client = _client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        extra_headers={
            "HTTP-Referer": "https://github.com/betting-ai",
            "X-Title": "Betting AI - Agent Pipeline",
        },
    )
    return response.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════
# PASO 1: AGENTE DE CONTEXTO
# ═══════════════════════════════════════════════════════════════

SYSTEM_CONTEXT = f"""Eres un analista táctico de fútbol. Tu trabajo es EXTRAER el contexto cualitativo de un partido.
NO hagas proyecciones numéricas aquí. Solo describe:
- Motivación y urgencia de cada equipo
- Perfiles de estilo (posesión, presión, laterales, transiciones)
- Dinámica esperada del partido
- Factores externos (derby, clima, viaje, rotaciones)

Sé conciso: 5-8 viñetas máximo. Prioriza los 2-3 factores MÁS determinantes.

{FEWSHOT_CONTEXT}"""


def step_context(datos_resumidos: dict, prediccion_resumida: dict) -> str:
    """Paso 1: Extrae contexto cualitativo."""
    # Construir input ligero (solo lo necesario para contexto)
    home = datos_resumidos.get("home_team", datos_resumidos.get("local", "Local"))
    away = datos_resumidos.get("away_team", datos_resumidos.get("visitante", "Visitante"))
    liga = datos_resumidos.get("liga", "Desconocida")

    # Standings
    standings_lines = []
    st = datos_resumidos.get("standings", datos_resumidos.get("_standings", {}))
    if st and isinstance(st, dict):
        for entry in st.get("table", st.get("standings", [])):
            if isinstance(entry, dict):
                standings_lines.append(
                    f"  {entry.get('position', '?')}. {entry.get('team', entry.get('name', '?'))} "
                    f"(PJ:{entry.get('played', '?')} PTS:{entry.get('points', '?')})"
                )

    # Forma
    fl = datos_resumidos.get("forma_local", {})
    fv = datos_resumidos.get("forma_visitante", {})
    form_lines = [
        f"Local: {fl.get('forma_string', '?')} | xG:{fl.get('xG_promedio', '?')} | xGc:{fl.get('xG_contra_promedio', '?')}",
        f"Visitante: {fv.get('forma_string', '?')} | xG:{fv.get('xG_promedio', '?')} | xGc:{fv.get('xG_contra_promedio', '?')}",
    ]

    # Contexto especial
    v2d = datos_resumidos.get("_v2_detail", {})
    extra = []
    if v2d.get("is_local_derby"):
        extra.append("ES DERBY LOCAL")
    if v2d.get("is_neutral_ground"):
        extra.append("Cancha neutral")
    if v2d.get("travel_distance_km"):
        extra.append(f"Viaje visitante: {v2d['travel_distance_km']} km")

    user_prompt = f"""Partido: {home} vs {away} ({liga})

Tabla de posiciones:
{chr(10).join(standings_lines) if standings_lines else '(no disponible)'}

Forma reciente:
{chr(10).join(form_lines)}

Factores adicionales: {', '.join(extra) if extra else 'Ninguno destacado'}

Extrae el contexto cualitativo en viñetas."""

    return _call_llm(SYSTEM_CONTEXT, user_prompt, max_tokens=2000)


# ═══════════════════════════════════════════════════════════════
# PASO 2: AGENTE DE PROYECCIÓN NUMÉRICA
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROJECTION = f"""Eres un proyector estadístico de fútbol. Recibes CONTEXTO cualitativo + DATOS DUROS
y debes emitir proyecciones numéricas concisas para cada métrica.

REGLAS ESTRICTAS:
- Usa los datos de FORMA RECIENTE como base. No promedies ciegamente, evalúa consistencia.
- Aplica el contexto cualitativo como AJUSTE, no como base.
- Proyecta con RANGOS (ej: 18-20 tiros), no números exactos.
- Para tiros 1T/2T: equipos que presionan alto → más en 1T. Equipos que crecen → más en 2T.
- Para tarjetas: 2T suele tener más (fatiga + nervios).
- Para corners: 2T suele tener más (equipos se vuelcan).
- GOLES: si el contexto dice "partido cerrado", proyecta total bajo incluso si los promedios son altos.

FORMATO OBLIGATORIO (responde SOLO con esto, sin texto adicional):

TIROS_TOTALES: Local X-Y | Visitante X-Y | Total Z-W
TIROS_ARCO: Local X-Y | Visitante X-W | Total Z-W
GOLES: 1T X-Y | 2T Z-W | Total A-B
AMARILLAS: Local X | Visitante Y | Total Z | 1T ~A | 2T ~B
CORNERS: 1T X-Y | 2T Z-W | Total A-B
FALTAS: ~X
CONTEXTUAL_ADJUSTMENT: [ajuste principal aplicado en una frase]

{FEWSHOT_PROJECTION}"""


def step_projection(context_summary: str, datos_resumidos: dict, quant_projections: dict) -> str:
    """Paso 2: Proyecta números usando contexto + datos duros + base cuantitativa."""
    home = datos_resumidos.get("home_team", datos_resumidos.get("local", "Local"))
    away = datos_resumidos.get("away_team", datos_resumidos.get("visitante", "Visitante"))

    # Datos duros brutos (formas)
    fl = datos_resumidos.get("forma_local", {})
    fv = datos_resumidos.get("forma_visitante", {})

    # Base cuantitativa
    quant_lines = []
    if quant_projections:
        quant_lines = [
            f"Base cuantitativa del modelo:",
            f"  Goles esperados: Local {quant_projections.get('goals_local', '?')} - Visitante {quant_projections.get('goals_visitor', '?')}",
            f"  Tiros totales: ~{quant_projections.get('tiros_total', '?')}",
            f"  Corners totales: ~{quant_projections.get('corners_total', '?')}",
            f"  Amarillas totales: ~{quant_projections.get('yc_total', '?')}",
            f"  Faltas: ~{quant_projections.get('fouls', '?')}",
        ]

    # H2H
    h2h = datos_resumidos.get("h2h", {})
    h2h_line = ""
    if h2h and h2h.get("total_partidos"):
        h2h_line = f"H2H últimos 3 años: {h2h.get('total_partidos')} partidos. GOLES promedio: {h2h.get('promedio_goles', '?')}"

    user_prompt = f"""Partido: {home} vs {away}

CONTEXTO DEL PARTIDO (del analista):
{context_summary}

{chr(10).join(quant_lines)}

FORMA LOCAL (últimos datos):
  xG: {fl.get('xG_promedio', '?')}, xGc: {fl.get('xG_contra_promedio', '?')}
  Tiros/prom: {fl.get('remates_promedio', '?')}, Arco/prom: {fl.get('remates_arco_promedio', '?')}
  Amarillas/prom: {fl.get('amarillas_promedio', '?')}, Faltas/prom: {fl.get('faltas_promedio', '?')}

FORMA VISITANTE:
  xG: {fv.get('xG_promedio', '?')}, xGc: {fv.get('xG_contra_promedio', '?')}
  Tiros/prom: {fv.get('remates_promedio', '?')}, Arco/prom: {fv.get('remates_arco_promedio', '?')}
  Amarillas/prom: {fv.get('amarillas_promedio', '?')}, Faltas/prom: {fv.get('faltas_promedio', '?')}

{h2h_line}

EMITE las proyecciones en el formato requerido."""

    return _call_llm(SYSTEM_PROJECTION, user_prompt, max_tokens=4000)


# ═══════════════════════════════════════════════════════════════
# PASO 3: AGENTE DE VALOR
# ═══════════════════════════════════════════════════════════════

SYSTEM_VALUE = f"""Eres un especialista en detección de valor (edge) en cuotas de apuestas.
Recibes PROYECCIONES NUMÉRICAS + CUOTAS BETSAFE y decides si hay apuestas con valor positivo.

REGLAS:
- Edge = Probabilidad estimada - Probabilidad implícita en la cuota (1/cuota).
- Solo recomienda si edge >= 2% y stake Kelly ≥ 0.1% del bankroll.
- Si cuota < 1.35, el mercado descuenta mucho. Requiere evidencia ABWUMADORA para ir en contra.
- Sé SELECTIVO: mejor no apostar que forzar un pick mediocre.
- Mercados estadísticos (tiros, corners, tarjetas) tienen PRIORIDAD sobre 1X2.

FORMATO:

[RANGO DE PROBABILIDADES]
BTTS Si: X% | Over 2.5: X% | 1X2: L=X% D=X% V=X%

[MERCADOS CON VALOR]
Mercado | Selección | Cuota | Proyección | Edge estimado | Kelly Stake
(Solo filas con edge claro; si ninguno, decir "Sin valor detectado")

[RECOMENDACIÓN FINAL]
Mejor apuesta: [mercado+selección] @ [cuota] con stake de X% del bankroll.
Justificación en 1-2 líneas.

{FEWSHOT_VALUE}"""


def step_value(projection_text: str, betsafe_odds: dict, quant_probs: dict) -> str:
    """Paso 3: Evalúa valor contra cuotas Betsafe."""
    # Parsear cuotas clave
    odds_lines = []
    if betsafe_odds:
        markets = betsafe_odds.get("markets", betsafe_odds)
        for k, v in markets.items():
            if isinstance(v, dict):
                odds_lines.append(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
            else:
                odds_lines.append(f"  {k}: {v}")
    else:
        odds_lines.append("(No se proporcionaron cuotas Betsafe)")

    # Probabilidades base del modelo cuantitativo
    prob_lines = []
    if quant_probs:
        prob_lines = [
            f"Probabilidades base modelo cuantitativo:",
            f"  BTTS: {quant_probs.get('btts_yes', '?')}",
            f"  Over 2.5: {quant_probs.get('over25_yes', '?')}",
        ]

    user_prompt = f"""Proyecciones del partido:
{projection_text}

CUOTAS BETSAFE:
{chr(10).join(odds_lines)}

{chr(10).join(prob_lines)}

Evalúa valor y emite recomendaciones con stakes."""

    return _call_llm(SYSTEM_VALUE, user_prompt, max_tokens=4000)


# ═══════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    datos_resumidos: dict,
    prediccion_resumida: dict,
    quant_projections: dict,
    quant_probs: dict,
) -> Tuple[str, str, str]:
    """
    Ejecuta los 3 agentes en secuencia.
    Retorna (contexto, proyección, valor) como strings.
    """
    print("  [Pipeline] Paso 1/3: Extrayendo contexto...")
    context = step_context(datos_resumidos, prediccion_resumida)

    print("  [Pipeline] Paso 2/3: Proyectando estadísticas...")
    projection = step_projection(context, datos_resumidos, quant_projections)

    print("  [Pipeline] Paso 3/3: Evaluando valor...")
    betsafe_odds = datos_resumidos.get("_cuotas", datos_resumidos.get("cuotas", {}))
    value = step_value(projection, betsafe_odds, quant_probs)

    return context, projection, value


def format_full_output(context: str, projection: str, value: str) -> str:
    """Une los 3 pasos en un análisis final legible."""
    return f"""
═══════════════════════════════════════════
CONTEXTO DEL PARTIDO
═══════════════════════════════════════════
{context}

═══════════════════════════════════════════
PROYECCIONES ESTADÍSTICAS
═══════════════════════════════════════════
{projection}

═══════════════════════════════════════════
ANÁLISIS DE VALOR
═══════════════════════════════════════════
{value}
"""


if __name__ == "__main__":
    # Test con datos dummy
    dummy_datos = {
        "home_team": "Manchester City",
        "away_team": "Luton Town",
        "liga": "Premier League",
        "forma_local": {"forma_string": "WWDLW", "xG_promedio": 2.1, "xG_contra_promedio": 0.8, "remates_promedio": 18, "remates_arco_promedio": 6.5, "amarillas_promedio": 1.2, "faltas_promedio": 9},
        "forma_visitante": {"forma_string": "LDLWD", "xG_promedio": 0.9, "xG_contra_promedio": 1.8, "remates_promedio": 8, "remates_arco_promedio": 2.5, "amarillas_promedio": 2.0, "faltas_promedio": 14},
        "standings": {"table": [
            {"position": 1, "team": "Man City", "played": 32, "points": 76},
            {"position": 18, "team": "Luton", "played": 32, "points": 25},
        ]},
        "h2h": {"total_partidos": 3, "promedio_goles": 3.0},
        "_v2_detail": {},
        "_cuotas": {"over_25": 1.65, "btts_si": 1.85, "local": 1.15, "empate": 7.50, "visitante": 18.0},
    }
    dummy_pred = {"xG_local": 2.0, "xG_visitante": 0.8, "prob_over_25": 0.70, "prob_btts": 0.55}
    dummy_quant = {"goals_local": 2.1, "goals_visitor": 0.7, "tiros_total": 24.0, "corners_total": 10.5, "yc_total": 3.2, "fouls": 22.0}
    dummy_probs = {"btts_yes": 0.52, "over25_yes": 0.72}

    ctx, proj, val = run_pipeline(dummy_datos, dummy_pred, dummy_quant, dummy_probs)
    print(format_full_output(ctx, proj, val))
