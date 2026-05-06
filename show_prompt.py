"""
Muestra el prompt completo que se envía a DeepSeek via OpenRouter.
Útil para depuración, auditoría y entender qué datos llegan al modelo.

Uso:
    python show_prompt.py              → interactivo: elegir partido y ver prompt
    python show_prompt.py --save=prompt.txt  → igual pero guarda a archivo
    python show_prompt.py --no-color   → output sin códigos ANSI
"""

import sys
import logging
from analyzer import SYSTEM_PROMPT, _crear_prompt_usuario

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _separador(char="=", width=80):
    print(f"\n{char * width}\n")


def _mostrar_partidos(partidos: list) -> bool:
    if not partidos:
        print("\n  No hay próximos partidos disponibles en este momento.")
        return False

    print(f"\n  Próximos partidos ({len(partidos)} encontrados):\n")
    for i, p in enumerate(partidos, 1):
        local = p.get("home_team", "?")
        visitante = p.get("away_team", "?")
        liga = p.get("_league_name", p.get("league", {}).get("name", "?"))
        fecha = p.get("event_date", "?")
        if len(fecha) > 16:
            fecha = fecha[:16].replace("T", " ")
        print(f"  {i:>3}. [{liga}] {local} vs {visitante}")
        print(f"       {fecha}")
    return True


def _seleccionar_partido(partidos: list) -> dict:
    while True:
        try:
            entrada = input("\n  Seleccioná un partido (número) o 'q' para salir: ").strip()
            if entrada.lower() == "q":
                return None
            indice = int(entrada) - 1
            if 0 <= indice < len(partidos):
                return partidos[indice]
            print(f"  Número fuera de rango. Elegí entre 1 y {len(partidos)}.")
        except ValueError:
            print("  Por favor, ingresá un número válido.")


def _cargar_datos_partido(partido: dict):
    """
    Misma lógica de carga que main._cargar_datos_partido pero sin prints verbose.
    """
    from bsd_client import (
        obtener_detalle_partido,
        obtener_predicciones,
        resumir_datos_partido,
        resumir_prediccion,
    )
    from sofascore_client import enriquecer_datos_partido, obtener_datos_completos_sofascore
    from flashscore_client import enriquecer_datos_partido as enriquecer_flashscore
    from betsafe_client import obtener_cuotas_betsafe_desde_url

    match_id = partido.get("id")
    local = partido.get("home_team", "?")
    visitante = partido.get("away_team", "?")
    es_sofascore_only = partido.get("_source") == "sofascore_only"

    print(f"\n  Cargando datos de: {local} vs {visitante}...")

    if es_sofascore_only:
        try:
            datos_resumidos = obtener_datos_completos_sofascore(partido)
        except Exception as e:
            print(f"  Error SofaScore: {e}")
            return None, None

        ss = datos_resumidos.get("_sofascore", {})
        if not ss.get("disponible"):
            print(f"  SofaScore no disponible: {ss.get('error', '?')}")
            return None, None

        print("  [OK] SofaScore (datos completos)")

        try:
            datos_resumidos = enriquecer_flashscore(datos_resumidos)
        except Exception:
            pass

        prediccion_resumida = {}
        return datos_resumidos, prediccion_resumida

    # ── Flujo BSD + SofaScore ──
    try:
        detalle = obtener_detalle_partido(match_id)
        print("  [OK] BSD")
    except Exception as e:
        print(f"  Error BSD: {e}")
        return None, None

    try:
        predicciones = obtener_predicciones(match_id=match_id)
        if predicciones:
            prediccion = predicciones[0]
        else:
            league_id = partido.get("league", {}).get("id")
            predicciones = obtener_predicciones(league_id=league_id)
            prediccion = predicciones[0] if predicciones else {}
        print("  [OK] Predicción ML (CatBoost)")
    except Exception as e:
        print(f"  [WARN] Predicciones ML: {e}")
        prediccion = {}

    datos_resumidos = resumir_datos_partido(detalle)
    datos_resumidos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"
    prediccion_resumida = resumir_prediccion(prediccion)

    # Enriquecimiento BSD v2
    try:
        from bsd_client_v2 import enriquecer_con_v2
        datos_resumidos = enriquecer_con_v2(datos_resumidos, match_id)
        prediccion_v2 = datos_resumidos.pop("_v2_prediction_raw", {})
        if prediccion_v2 and not prediccion_resumida:
            prediccion_resumida = resumir_prediccion(prediccion_v2)
        v2 = datos_resumidos.get("_bsd_v2", {})
        v2_parts = []
        if v2.get("stats") and "_error" not in v2["stats"]:
            v2_parts.append("stats (shotmap,momentum,xg_per_minute)")
        if v2.get("metadata") and "_error" not in v2["metadata"]:
            v2_parts.append("metadata (facts,AI preview)")
        if v2.get("player_stats") and "_error" not in v2["player_stats"]:
            v2_parts.append("player-stats")
        if v2.get("standings") and "_error" not in v2["standings"]:
            v2_parts.append("standings (xG)")
        if v2_parts:
            print(f"  [OK] BSD v2 ({', '.join(v2_parts)})")
    except Exception as e:
        print(f"  [WARN] BSD v2: {e}")

    try:
        datos_resumidos = enriquecer_datos_partido(datos_resumidos)
        ss_info = datos_resumidos.get("_sofascore", {})
        if ss_info.get("disponible"):
            partes = [k for k in ["alineaciones", "h2h", "detalle_evento", "form_performance"] if ss_info.get(k)]
            print(f"  [OK] SofaScore ({', '.join(partes) if partes else 'basico'})")
    except Exception as e:
        print(f"  [WARN] SofaScore: {e}")

    try:
        datos_resumidos = enriquecer_flashscore(datos_resumidos)
        if datos_resumidos.get("_flashscore", {}).get("disponible"):
            print("  [OK] Flashscore")
    except Exception:
        pass

    try:
        print("  Pegá la URL de Betsafe o el eventId (Enter para omitir):")
        url = input("  > ").strip()
        if url:
            cuotas = obtener_cuotas_betsafe_desde_url(url)
            if cuotas and "error" not in cuotas and cuotas.get("markets"):
                datos_resumidos["_cuotas"] = cuotas
                print("  [OK] Betsafe")
            else:
                print(f"  Error: {cuotas.get('error', '?')}")
    except Exception:
        pass

    # WhoScored arbitro
    try:
        from whoscored_client import enriquecer_arbitro_whoscored
        print("  Pegá la URL de WhoScored del árbitro (Enter para omitir):")
        url_ws = input("  > ").strip()
        if url_ws:
            datos_resumidos = enriquecer_arbitro_whoscored(datos_resumidos, url_ws)
            ws = (datos_resumidos.get("arbitro") or {}).get("_whoscored", {})
            if ws:
                print(f"  [OK] WhoScored arbitro: {ws.get('yc_pp', '?')} YC/part, {ws.get('total_partidos', '?')} partidos")
            else:
                print("  [WARN] WhoScored no devolvio datos")
    except Exception as e:
        print(f"  [WARN] WhoScored: {e}")

    return datos_resumidos, prediccion_resumida


def show_prompt(datos, prediccion, save_to=None):
    """
    Construye y muestra el prompt completo que se enviaría a DeepSeek.
    """
    user_prompt = _crear_prompt_usuario(datos, prediccion)
    total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    total_tokens_approx = total_chars // 4

    header = f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                     PROMPT COMPLETO → DEEPSEEK                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  Modelo      : deepseek/deepseek-v4-pro (via OpenRouter)                     │
│  Endpoint    : https://openrouter.ai/api/v1                                  │
│  max_tokens  : 100000                                                        │
│  temperature : 0.3                                                           │
│  reasoning   : True (include_reasoning)                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  System prompt : {len(SYSTEM_PROMPT):>7,} chars ({len(SYSTEM_PROMPT)//4:,} ~tokens)                 │
│  User prompt   : {len(user_prompt):>7,} chars ({len(user_prompt)//4:,} ~tokens)                 │
│  TOTAL         : {total_chars:>7,} chars ({total_tokens_approx:,} ~tokens)                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    print(header)

    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PROMPT COMPLETO ENVIADO A DEEPSEEK\n")
            f.write(f"Modelo: deepseek/deepseek-v4-pro\n")
            f.write(f"Temperature: 0.3 | max_tokens: 100000 | include_reasoning: True\n")
            f.write(f"System: {len(SYSTEM_PROMPT)} chars | User: {len(user_prompt)} chars\n")
            f.write("=" * 80 + "\n\n")
            f.write("=== SYSTEM PROMPT ===\n\n")
            f.write(SYSTEM_PROMPT)
            f.write("\n\n=== USER PROMPT ===\n\n")
            f.write(user_prompt)
        print(f"  ↳ Guardado en: {save_to}\n")

    _separador("#")
    print("SYSTEM PROMPT")
    _separador("#")
    print(SYSTEM_PROMPT)

    _separador("#")
    print("USER PROMPT")
    _separador("#")
    print(user_prompt)

    _separador("─")
    print(f"Fin del prompt. Total: {total_chars:,} caracteres.")
    _separador("─")


def main():
    save_to = None
    for arg in sys.argv[1:]:
        if arg.startswith("--save="):
            save_to = arg.split("=", 1)[1]
        elif arg == "--no-color":
            pass
        elif arg in ("--help", "-h"):
            print(__doc__)
            return

    from bsd_client import obtener_proximos_partidos
    from sofascore_client import (
        verificar_salud_sofascore,
        obtener_partidos_sofascore_only,
    )

    print("\n  Cargando próximos partidos...")
    verificar_salud_sofascore(detallado=False)

    partidos = obtener_proximos_partidos()
    partidos_ss = obtener_partidos_sofascore_only()
    partidos.extend(partidos_ss)
    partidos.sort(
        key=lambda p: p.get("event_date", "")
        if isinstance(p.get("event_date"), str)
        else ""
    )

    if not _mostrar_partidos(partidos):
        return

    partido = _seleccionar_partido(partidos)
    if partido is None:
        print("\n  Cancelado.")
        return

    datos, prediccion = _cargar_datos_partido(partido)
    if datos is None:
        print("\n  No se pudieron obtener datos suficientes.")
        return

    show_prompt(datos, prediccion, save_to=save_to)


if __name__ == "__main__":
    main()
