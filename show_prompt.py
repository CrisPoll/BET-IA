"""
Muestra el prompt completo que se envía al modelo via OpenRouter.
Útil para depuración, auditoría y entender qué datos llegan al modelo.

Uso:
    python show_prompt.py              → interactivo: elegir partido y ver prompt
    python show_prompt.py --save=prompt.txt  → igual pero guarda a archivo
    python show_prompt.py --no-color   → output sin códigos ANSI

Incluye el mismo enriquecimiento que el flujo principal:
    BSD v2, SofaScore, Betano opcional, proyeccion cuantitativa y splits.
"""

import sys
import logging
from analyzer import (
    INCLUDE_REASONING,
    MAX_TOKENS,
    MODEL_NAME,
    OPENROUTER_BASE_URL,
    SYSTEM_PROMPT,
    TEMPERATURE,
    _crear_prompt_usuario,
)

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


def _resumir_capas_datos(datos: dict) -> list[str]:
    """Resumen corto de las capas que efectivamente llegan al USER_PROMPT."""
    capas = []
    v2 = datos.get("_bsd_v2", {}) if isinstance(datos, dict) else {}
    ss = datos.get("_sofascore", {}) if isinstance(datos, dict) else {}

    if v2:
        stats = v2.get("stats", {})
        if _has_v2_stats_data(stats):
            capas.append("BSD v2 stats")
        if v2.get("metadata") and "_error" not in v2["metadata"]:
            capas.append("BSD v2 metadata")
        lineups_v2 = v2.get("lineups", {})
        if lineups_v2 and "_error" not in lineups_v2:
            if _has_bsd_lineup_payload(lineups_v2):
                capas.append(f"BSD v2 lineups {_lineup_status_label(lineups_v2)}")
            elif lineups_v2.get("lineup_status"):
                capas.append(f"BSD v2 lineup status: {lineups_v2.get('lineup_status')}")
        if _has_player_impact_data(v2.get("player_impact")):
            capas.append("BSD v2 impacto jugadores")
        if v2.get("odds") and "_error" not in v2["odds"] and v2["odds"].get("odds"):
            capas.append("BSD v2 odds comparador")
        if v2.get("motivation"):
            capas.append("BSD v2 motivacion/fixtures")
        if _has_v2_standings_for_prompt(datos, v2.get("standings")):
            capas.append("BSD v2 standings")

    if ss.get("disponible"):
        ss_parts = []
        if ss.get("alineaciones"):
            ss_parts.append(f"alineaciones {_sofascore_lineup_label(ss.get('alineaciones'))}")
            if _has_sofascore_missing_impact(ss.get("alineaciones")):
                ss_parts.append("impacto bajas")
            if _has_sofascore_xi_impact(ss.get("alineaciones")):
                ss_parts.append("impacto XI")
        ss_parts.extend(
            name for name in ["h2h", "detalle_evento", "form_performance", "standings"]
            if ss.get(name)
        )
        capas.append(f"SofaScore {', '.join(ss_parts) if ss_parts else 'basico'}")

    if datos.get("_cuotas"):
        capas.append("Betano mercados reales")

    return capas


def _has_v2_stats_data(stats: dict) -> bool:
    if not isinstance(stats, dict) or "_error" in stats:
        return False
    per_team = stats.get("stats", {})
    home = per_team.get("home", {}) if isinstance(per_team, dict) else {}
    away = per_team.get("away", {}) if isinstance(per_team, dict) else {}
    return bool(
        home.get("total_shots") is not None
        or away.get("total_shots") is not None
        or stats.get("shotmap")
        or stats.get("xg_per_minute")
        or stats.get("momentum")
    )


def _has_bsd_lineup_payload(lineups: dict) -> bool:
    return bool(lineups.get("lineups") or lineups.get("unavailable_players"))


def _has_sofascore_missing_impact(alineaciones: dict) -> bool:
    if not isinstance(alineaciones, dict):
        return False
    for side in ("local", "visitante"):
        bajas = ((alineaciones.get(side) or {}).get("bajas") or {})
        for bucket in ("confirmadas", "dudas"):
            for baja in bajas.get(bucket, []) or []:
                if baja.get("impacto_baja"):
                    return True
    return False


def _has_sofascore_xi_impact(alineaciones: dict) -> bool:
    if not isinstance(alineaciones, dict):
        return False
    for side in ("local", "visitante"):
        for player in (alineaciones.get(side) or {}).get("titulares", []) or []:
            if player.get("impacto_jugador"):
                return True
    return False


def _has_player_impact_data(impact: dict) -> bool:
    if not isinstance(impact, dict) or "_error" in impact:
        return False
    for player in impact.get("players", []) or []:
        stats = player.get("stats") or {}
        if not isinstance(stats, dict) or "_error" in stats:
            continue
        for key in ("xg_p90", "shots_p90", "key_passes_p90", "yellow_p90", "saves_p90", "goals", "assists"):
            value = stats.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return True
    return False


def _has_v2_standings_for_prompt(datos: dict, standings: dict) -> bool:
    if not isinstance(standings, dict) or "_error" in standings:
        return False
    liga = datos.get("liga", "")
    rn = (datos.get("_v2_detail") or {}).get("round_number")
    is_cup_group = rn and any(c in liga for c in ["Champions", "Europa", "Libertadores", "Sudamericana"]) and rn <= 8
    rows = standings.get("standings")
    groups = standings.get("groups")
    if is_cup_group and isinstance(rows, dict) and len(rows) > 1:
        return False
    if is_cup_group and isinstance(rows, list) and len(rows) > 8:
        return False
    if is_cup_group and isinstance(groups, dict) and len(groups) > 1:
        return False
    if is_cup_group and isinstance(groups, list) and len(groups) > 1:
        return False
    return bool(rows or groups)


def _lineup_status_label(lineups: dict) -> str:
    if lineups.get("confirmed") or lineups.get("is_confirmed"):
        return "confirmadas"
    status = lineups.get("lineup_status")
    if status:
        return f"({status})"
    return "preliminares"


def _sofascore_lineup_label(alineaciones: dict) -> str:
    sides = [alineaciones.get("local") or {}, alineaciones.get("visitante") or {}]
    if sides and all(side.get("confirmada") for side in sides if side):
        return "confirmadas"
    return "preliminares"


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
    from betano_client import obtener_cuotas_betano_desde_url

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
        lineups_v2 = v2.get("lineups", {})
        if lineups_v2 and "_error" not in lineups_v2:
            if lineups_v2.get("lineups") or lineups_v2.get("unavailable_players"):
                v2_parts.append(f"lineups BSD {_lineup_status_label(lineups_v2)}")
            elif lineups_v2.get("lineup_status"):
                v2_parts.append(f"lineup status BSD: {lineups_v2.get('lineup_status')}")
        if v2.get("player_stats") and "_error" not in v2["player_stats"]:
            v2_parts.append("player-stats")
        if v2.get("player_impact") and "_error" not in v2["player_impact"]:
            v2_parts.append("impacto jugadores")
        if v2.get("odds") and "_error" not in v2["odds"]:
            v2_parts.append("odds consenso")
        if v2.get("motivation"):
            v2_parts.append("motivacion/fixtures")
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
            partes = [k for k in ["alineaciones", "h2h", "detalle_evento", "form_performance", "standings"] if ss_info.get(k)]
            print(f"  [OK] SofaScore ({', '.join(partes) if partes else 'basico'})")
    except Exception as e:
        print(f"  [WARN] SofaScore: {e}")

    try:
        print("  Pegá la URL completa de Betano (Enter para omitir):")
        url = input("  > ").strip()
        if url:
            cuotas = obtener_cuotas_betano_desde_url(url)
            if cuotas and "error" not in cuotas and cuotas.get("markets"):
                datos_resumidos["_cuotas"] = cuotas
                print("  [OK] Betano")
            else:
                print(f"  Error: {cuotas.get('error', '?')}")
    except Exception:
        pass

    return datos_resumidos, prediccion_resumida


def show_prompt(datos, prediccion, save_to=None):
    """
    Construye y muestra el prompt completo que se enviaria al modelo configurado.
    """
    from quant_model import run_full_projection
    quant_projections, features = run_full_projection(datos, prediccion)
    user_prompt = _crear_prompt_usuario(datos, prediccion, quant_projections, features)
    total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
    total_tokens_approx = total_chars // 4
    capas = _resumir_capas_datos(datos)

    header = f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│                     PROMPT COMPLETO → MODELO                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│  Modelo      : {MODEL_NAME:<61}│
│  Endpoint    : {OPENROUTER_BASE_URL:<61}│
│  max_tokens  : {str(MAX_TOKENS):<61}│
│  temperature : {str(TEMPERATURE):<61}│
│  reasoning   : {str(INCLUDE_REASONING):<61}│
├──────────────────────────────────────────────────────────────────────────────┤
│  System prompt : {len(SYSTEM_PROMPT):>7,} chars ({len(SYSTEM_PROMPT)//4:,} ~tokens)                 │
│  User prompt   : {len(user_prompt):>7,} chars ({len(user_prompt)//4:,} ~tokens)                 │
│  TOTAL         : {total_chars:>7,} chars ({total_tokens_approx:,} ~tokens)                 │
└──────────────────────────────────────────────────────────────────────────────┘
"""
    print(header)
    if capas:
        print("  Capas de datos detectadas:")
        for capa in capas:
            print(f"  - {capa}")
        print()

    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("PROMPT COMPLETO ENVIADO AL MODELO\n")
            f.write(f"Modelo: {MODEL_NAME}\n")
            f.write(f"Temperature: {TEMPERATURE} | max_tokens: {MAX_TOKENS} | include_reasoning: {INCLUDE_REASONING}\n")
            f.write(f"System: {len(SYSTEM_PROMPT)} chars | User: {len(user_prompt)} chars\n")
            if capas:
                f.write("Capas de datos: " + ", ".join(capas) + "\n")
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
