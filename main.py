"""
Punto de entrada interactivo de betting-ai v2.

Menú en consola para:
   1. Ver próximos partidos
   2. Seleccionar partido y analizar (con modelo cuantitativo + LLM + persistencia)
   3. Ver performance del sistema (MAE, ROI, calibración)
   4. Registrar resultado post-partido
   5. Ver resumen de predicciones recientes
"""

import logging
import sys
from bsd_client import (
    obtener_proximos_partidos,
    obtener_detalle_partido,
    obtener_predicciones,
    resumir_datos_partido,
    resumir_prediccion,
    depurar_partido,
)
from bsd_client_v2 import enriquecer_con_v2
from sofascore_client import enriquecer_datos_partido, verificar_salud_sofascore, obtener_partidos_sofascore_only, obtener_datos_completos_sofascore
from flashscore_client import enriquecer_datos_partido as enriquecer_flashscore
from betsafe_client import obtener_cuotas_betsafe_desde_url
from analyzer import analizar_partido
import prediction_db as db
from quant_model import run_full_projection
from bankroll import get_stats_summary as get_bankroll_summary

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Inicializar DB al arrancar
db.init_db()


def _separador():
    print("\n" + "=" * 70 + "\n")


def _mostrar_partidos(partidos: list):
    if not partidos:
        print("\n  No hay próximos partidos disponibles en este momento.")
        print("  (Puede que no haya partidos programados en los próximos 7 días)")
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
            entrada = input("\n  Selecciona un partido (número) o 'q' para volver: ").strip()
            if entrada.lower() == "q":
                return None
            indice = int(entrada) - 1
            if 0 <= indice < len(partidos):
                return partidos[indice]
            print(f"  Número fuera de rango. Elige entre 1 y {len(partidos)}.")
        except ValueError:
            print("  Por favor, ingresa un número válido.")


def _cargar_datos_partido(partido: dict, verbose: bool = True):
    match_id = partido.get("id")
    local = partido.get("home_team", "?")
    visitante = partido.get("away_team", "?")
    es_sofascore_only = partido.get("_source") == "sofascore_only"

    print(f"\n  Obteniendo datos de: {local} vs {visitante}...")

    if es_sofascore_only:
        try:
            datos_resumidos = obtener_datos_completos_sofascore(partido)
        except Exception as e:
            print(f"  ✗ Error al obtener datos de SofaScore: {e}")
            return None, None

        ss = datos_resumidos.get("_sofascore", {})
        if not ss.get("disponible"):
            print(f"  ✗ SofaScore no disponible: {ss.get('error', '?')}")
            return None, None

        print("  ✓ SofaScore (datos completos)")

        try:
            datos_resumidos = enriquecer_flashscore(datos_resumidos)
        except Exception:
            pass

        prediccion_resumida = {}

        if verbose:
            print(f"\n  SOFASCORE-ONLY (sin BSD): {local} vs {visitante}")
            parts = []
            if ss.get("alineaciones"):
                parts.append("alineaciones")
            if ss.get("h2h"):
                parts.append("H2H")
            if ss.get("detalle_evento"):
                parts.append("arbitro/managers")
            if ss.get("form_performance"):
                parts.append("form performance")
            print(f"  Datos disponibles: {', '.join(parts) if parts else 'basico'}")
        return datos_resumidos, prediccion_resumida

    # ── Flujo BSD + SofaScore (normal) ──
    try:
        detalle = obtener_detalle_partido(match_id)
        print("  ✓ Datos BSD del partido obtenidos")
    except Exception as e:
        print(f"  ✗ Error al obtener detalle del partido: {e}")
        return None, None

    try:
        predicciones = obtener_predicciones(match_id=match_id)
        if predicciones:
            prediccion = predicciones[0]
            print("  ✓ Predicción ML (CatBoost) obtenida")
        else:
            league_id = partido.get("league", {}).get("id")
            predicciones = obtener_predicciones(league_id=league_id)
            prediccion = predicciones[0] if predicciones else {}
            if prediccion:
                print("  ✓ Predicción ML obtenida (por liga)")
            else:
                print("  ⚠ No se encontraron predicciones ML para este partido")
                prediccion = {}
    except Exception as e:
        print(f"  ⚠ Error al obtener predicciones: {e}")
        prediccion = {}

    datos_resumidos = resumir_datos_partido(detalle)
    datos_resumidos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"

    prediccion_resumida = resumir_prediccion(prediccion)

    try:
        datos_resumidos = enriquecer_con_v2(datos_resumidos, match_id)
        v2 = datos_resumidos.get("_bsd_v2", {})
        v2_parts = []
        if v2.get("stats") and "_error" not in v2["stats"]:
            v2_parts.append("stats (shotmap, momentum, xg_per_minute)")
        if v2.get("metadata") and "_error" not in v2["metadata"]:
            v2_parts.append("metadata (facts, AI preview)")
        if v2.get("player_stats") and "_error" not in v2["player_stats"]:
            v2_parts.append("player-stats")
        if v2.get("standings") and "_error" not in v2["standings"]:
            v2_parts.append("standings (xG)")
        if v2_parts:
            print(f"  ✓ BSD v2: {', '.join(v2_parts)}")
        prediccion_v2 = datos_resumidos.pop("_v2_prediction_raw", {})
        if prediccion_v2 and not prediccion_resumida:
            prediccion_resumida = resumir_prediccion(prediccion_v2)
            print("  ✓ Prediccion ML via BSD v2")
    except Exception as e:
        print(f"  ⚠ BSD v2 enrichment: {e}")

    try:
        datos_resumidos = enriquecer_datos_partido(datos_resumidos)
        if datos_resumidos.get("_sofascore", {}).get("disponible"):
            ss_info = datos_resumidos["_sofascore"]
            partes = []
            if ss_info.get("alineaciones"):
                partes.append("alineaciones")
            if ss_info.get("h2h"):
                partes.append("H2H detallado")
            if ss_info.get("detalle_evento"):
                partes.append("arbitro + managers + lesiones")
            if ss_info.get("form_performance"):
                partes.append("form (performance)")
            print(f"  ✓ SofaScore: {', '.join(partes) if partes else 'encontrado (sin alineaciones aun)'}")
    except Exception as e:
        print(f"  ⚠ SofaScore no disponible: {e}")

    try:
        datos_resumidos = enriquecer_flashscore(datos_resumidos)
        if datos_resumidos.get("_flashscore", {}).get("disponible"):
            fs = datos_resumidos["_flashscore"]
            partes = []
            if fs.get("arbitro_stats"):
                partes.append(f"arbitro ({fs['arbitro_stats'].get('promedio', '?')} YC/partido)")
            if fs.get("equipo_stats"):
                partes.append("tendencias equipos")
            if partes:
                print(f"  ✓ Flashscore: {', '.join(partes)}")
    except Exception:
        pass

    try:
        print(f"     Pega URL del arbitro (WhoScored/Transfermarkt/SofaScore) o Enter para omitir:")
        url_arb = input("     > ").strip()
        if url_arb:
            if "whoscored.com" in url_arb:
                from whoscored_client import enriquecer_arbitro_whoscored
                datos_resumidos = enriquecer_arbitro_whoscored(datos_resumidos, url_arb)
                ws = (datos_resumidos.get("arbitro") or {}).get("_whoscored", {})
                if ws:
                    print(f"     ✓ WhoScored: {ws.get('yc_pp', '?')} YC/part, {ws.get('total_partidos', '?')} partidos")
            elif "transfermarkt" in url_arb:
                from transfermarkt_client import enriquecer_arbitro_transfermarkt
                datos_resumidos = enriquecer_arbitro_transfermarkt(datos_resumidos, url_arb)
                tm = (datos_resumidos.get("arbitro") or {}).get("_transfermarkt", {})
                if tm:
                    print(f"     ✓ Transfermarkt: {tm.get('yc_pp', '?')} YC/part, {tm.get('total_partidos', '?')} partidos")
            elif "sofascore.com" in url_arb:
                from sofascore_client import enriquecer_arbitro_sofascore
                datos_resumidos = enriquecer_arbitro_sofascore(datos_resumidos, url_arb)
                sf = (datos_resumidos.get("arbitro") or {}).get("_sofascore_ref", {})
                if sf:
                    print(f"     ✓ SofaScore: {sf.get('yc_pp', '?')} YC/part, {sf.get('total_partidos', '?')} partidos")
    except Exception:
        pass

    try:
        print(f"     Pega la URL de Betsafe (Enter para omitir):")
        url = input("     > ").strip()
        if url:
            cuotas = obtener_cuotas_betsafe_desde_url(url)
            if cuotas and "error" not in cuotas and cuotas.get("markets"):
                datos_resumidos["_cuotas"] = cuotas
                print(f"     ✓ Betsafe: {len(cuotas.get('markets', {}))} mercados en tiempo real")
            else:
                print(f"     ✗ Error: {cuotas.get('error', '?')}")
    except Exception:
        pass

    if verbose:
        depurar_partido(datos_resumidos, prediccion_resumida)

    return datos_resumidos, prediccion_resumida


def _menu_performance():
    _separador()
    print("  ESTADÍSTICAS DE PERFORMANCE (últimos 30 días)")
    summary = db.get_stats_summary(days=30)
    print(f"    Predicciones registradas: {summary['predictions']}")
    print(f"    Resultados reales cargados: {summary['results']}")
    print(f"    Apuestas realizadas: {summary['bets']}")
    if summary['roi_pct'] is not None:
        print(f"    P/L: {summary['profit']:.2f} | ROI: {summary['roi_pct']:.2f}%")
    else:
        print(f"    P/L: {summary['profit']:.2f} | ROI: N/D (sin stakes)")

    print("\n  MAE por métrica (últimos 90 días):")
    mae_rows = db.get_mae_by_metric(days=90)
    if not mae_rows:
        print("    (Sin datos suficientes aún. Registra resultados con post_match.py)")
    for row in mae_rows:
        print(f"    {row['metric']}: MAE={row['mae']:.2f} (n={row['n']})")

    print("\n  ROI por mercado (últimos 90 días):")
    roi_rows = db.get_roi(days=90)
    if not roi_rows:
        print("    (Sin apuestas evaluadas)")
    for row in roi_rows:
        print(f"    {row['mercado']}: ROI={row['roi_pct']:.2f}% | Profit={row['total_profit']:.2f} | Bets={row['total_bets']}")

    print("\n  Calibración de probabilidades (últimos 90 días):")
    cal = db.get_calibration(days=90)
    if not cal:
        print("    (Sin datos suficientes)")
    for c in cal:
        print(f"    {c['metric']} bin={c['bin']}: pred={c['avg_prediction']} vs real={c['actual_rate']} | error={c['calibration_error']} (n={c['n']})")

    input("\n  Presiona Enter para volver...")


def _menu_predicciones_recientes():
    _separador()
    preds = db.get_recent_predictions(limit=20, days=30)
    if not preds:
        print("  No hay predicciones recientes en la base de datos.")
        input("\n  Presiona Enter para volver...")
        return

    print(f"  Últimas predicciones (mostrando {len(preds)}):\n")
    for p in preds:
        resultado = ""
        if p.get("score_local") is not None:
            resultado = f" | Resultado: {p['score_local']}-{p['score_visitor']}"
        print(f"  {p['match_name']} ({p['league']}) [{p['match_date'][:10]}]{resultado}")
        print(f"     Quant: G={p['proj_total_goals']} T={p['proj_tiros_total']} C={p['proj_corners_total']} YC={p['proj_yc_total']}")

    input("\n  Presiona Enter para volver...")


def main():
    print("\n" + "╔" + "═" * 48 + "╗")
    print("║" + "     🎯 BETTING AI - Value Betting Analyzer v2     ".center(48) + "║")
    print("║" + "     Cuantitativo + LLM + Feedback Loop           ".center(48) + "║")
    print("╚" + "═" * 48 + "╝")

    partidos_cache = []

    while True:
        _separador()
        print("  MENÚ PRINCIPAL")
        print("  " + "-" * 30)
        print("  1. Ver próximos partidos")
        print("  2. Seleccionar partido y analizar")
        if partidos_cache:
            print(f"  3. Actualizar lista de partidos ({len(partidos_cache)} en caché)")
        else:
            print("  3. Salir")
        print("  4. Ver performance del sistema (MAE / ROI / Calibración)")
        print("  5. Ver predicciones recientes")
        print("  0. Salir")

        opcion = input("\n  Opción: ").strip()

        if opcion == "1":
            print("\n  Cargando próximos partidos...")
            try:
                verificar_salud_sofascore(detallado=True)
                partidos_cache = obtener_proximos_partidos()
                partidos_ss = obtener_partidos_sofascore_only()
                partidos_cache.extend(partidos_ss)
                partidos_cache.sort(key=lambda p: p.get("event_date", "") if isinstance(p.get("event_date"), str) else "")
                _mostrar_partidos(partidos_cache)
            except Exception as e:
                print(f"\n  ✗ Error al obtener partidos: {e}")
                print("  Verifica que BSD_API_KEY esté configurada en el archivo .env")

        elif opcion == "2":
            if not partidos_cache:
                print("\n  Primero carga los partidos (opción 1).")
                continue

            if not _mostrar_partidos(partidos_cache):
                continue

            partido = _seleccionar_partido(partidos_cache)
            if partido is None:
                continue

            _separador()
            datos, prediccion = _cargar_datos_partido(partido)

            if datos is None:
                print("\n  No se pudieron obtener los datos suficientes para el análisis.")
                continue

            if "--show-prompt" in sys.argv:
                _separador()
                from analyzer import SYSTEM_PROMPT, _crear_prompt_usuario
                from quant_model import run_full_projection
                qp, _ = run_full_projection(datos, prediccion)
                user_prompt = _crear_prompt_usuario(datos, prediccion, qp)
                print(f"=== SYSTEM PROMPT ({len(SYSTEM_PROMPT)} chars) ===")
                print(SYSTEM_PROMPT)
                print(f"\n=== USER PROMPT ({len(user_prompt)} chars) ===")
                print(user_prompt)
                print(f"\n=== TOTAL: {len(SYSTEM_PROMPT) + len(user_prompt)} chars ===")
                _separador()
                input("\n  Presiona Enter para continuar...")
                continue

            print("\n  Consultando modelo cuantitativo + DeepSeek...")
            try:
                analisis = analizar_partido(datos, prediccion)
                _separador()
                print(analisis)
                _separador()
            except Exception as e:
                print(f"\n  ✗ Error al analizar: {e}")
                print("  Verifica que OPENROUTER_API_KEY esté configurada en el archivo .env")

        elif opcion == "3":
            if partidos_cache:
                print("\n  Actualizando lista de partidos...")
                try:
                    partidos_cache = obtener_proximos_partidos()
                    _mostrar_partidos(partidos_cache)
                except Exception as e:
                    print(f"\n  ✗ Error al actualizar: {e}")
            else:
                print("\n  ¡Hasta luego!")
                sys.exit(0)

        elif opcion == "4":
            _menu_performance()

        elif opcion == "5":
            _menu_predicciones_recientes()

        elif opcion == "0":
            print("\n  ¡Hasta luego!")
            sys.exit(0)

        else:
            print("\n  Opción no válida.")


if __name__ == "__main__":
    main()
