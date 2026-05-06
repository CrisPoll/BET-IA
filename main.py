"""
Punto de entrada interactivo de betting-ai.

Menú en consola para:
   1. Ver próximos partidos de ligas europeas y torneos internacionales
   2. Seleccionar un partido por número
   3. Obtener todos los datos automáticamente (BSD + SofaScore)
   4. Mostrar el análisis completo de DeepSeek
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
from valuestats_client import enriquecer_arbitro_valuestats
from betsafe_client import obtener_cuotas_betsafe, obtener_cuotas_betsafe_desde_url
from analyzer import analizar_partido

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        # Liga solo en SofaScore, sin BSD
        try:
            datos_resumidos = obtener_datos_completos_sofascore(partido)
        except Exception as e:
            print(f"  \u2717 Error al obtener datos de SofaScore: {e}")
            return None, None

        ss = datos_resumidos.get("_sofascore", {})
        if not ss.get("disponible"):
            print(f"  \u2717 SofaScore no disponible: {ss.get('error', '?')}")
            return None, None

        print("  \u2713 SofaScore (datos completos)")

        # Flashscore enrichment
        try:
            datos_resumidos = enriquecer_flashscore(datos_resumidos)
        except Exception:
            pass

        # No hay prediccion BSD, construir una vacia
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
        print("  \u2713 Datos BSD del partido obtenidos")
    except Exception as e:
        print(f"  \u2717 Error al obtener detalle del partido: {e}")
        return None, None

    # 2. Obtener predicciones ML (BSD)
    try:
        predicciones = obtener_predicciones(match_id=match_id)
        if predicciones:
            prediccion = predicciones[0]
            print("  \u2713 Predicción ML (CatBoost) obtenida")
        else:
            league_id = partido.get("league", {}).get("id")
            predicciones = obtener_predicciones(league_id=league_id)
            prediccion = predicciones[0] if predicciones else {}
            if prediccion:
                print("  \u2713 Predicción ML obtenida (por liga)")
            else:
                print("  \u26a0 No se encontraron predicciones ML para este partido")
                prediccion = {}
    except Exception as e:
        print(f"  \u26a0 Error al obtener predicciones: {e}")
        prediccion = {}

    # 3. Resumir datos
    datos_resumidos = resumir_datos_partido(detalle)
    datos_resumidos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"

    prediccion_resumida = resumir_prediccion(prediccion)

    # 3.5. Enriquecer con BSD v2 (managers, referee, stats, metadata, player-stats, standings, squads)
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
            print(f"  \u2713 BSD v2: {', '.join(v2_parts)}")
        # Actualizar prediccion si v2 devolvio una
        prediccion_v2 = datos_resumidos.pop("_v2_prediction_raw", {})
        if prediccion_v2 and not prediccion_resumida:
            prediccion_resumida = resumir_prediccion(prediccion_v2)
            print("  \u2713 Prediccion ML via BSD v2")
    except Exception as e:
        print(f"  \u26a0 BSD v2 enrichment: {e}")

    # 4. Enriquecer con SofaScore (alineaciones, arbitro, managers, lesiones, H2H, form)
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
            print(f"  \u2713 SofaScore: {', '.join(partes) if partes else 'encontrado (sin alineaciones aun)'}")
        else:
            pass
    except Exception as e:
        print(f"  \u26a0 SofaScore no disponible: {e}")

    # 5. Enriquecer con Flashscore (tarjetas/arbitros)
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
                print(f"  \u2713 Flashscore: {', '.join(partes)}")
    except Exception:
        pass

    # 5.5. Enriquecer arbitro con ValueStats (datos superiores a BSD v2 para tarjetas)
    try:
        datos_resumidos = enriquecer_arbitro_valuestats(datos_resumidos)
        vs = (datos_resumidos.get("arbitro") or {}).get("_valuestats", {})
        if vs:
            print(f"  \u2713 ValueStats arbitro: {vs.get('yc_promedio', '?')} YC/part, {vs.get('total_partidos', '?')} partidos")
    except Exception:
        pass

    # 6. Obtener cuotas Betsafe en tiempo real
    try:
        partes = datos_resumidos.get("partido", "").split(" vs ")
        home = partes[0].strip() if len(partes) > 0 else ""
        away = partes[1].strip() if len(partes) > 1 else ""
        liga = datos_resumidos.get("liga", "")
        if home and away:
            cuotas = obtener_cuotas_betsafe(home, away, liga)
            if cuotas and "error" not in cuotas and cuotas.get("markets"):
                datos_resumidos["_cuotas"] = cuotas
                print(f"  \u2713 Betsafe: {len(cuotas.get('markets', {}))} mercados en tiempo real")
            else:
                print(f"  \u26a0 Betsafe no encontro el partido automaticamente")
                print(f"     Si tenes la URL de Betsafe, pega el link o eventId (o Enter para omitir):")
                url = input("     > ").strip()
                if url:
                    cuotas = obtener_cuotas_betsafe_desde_url(url)
                    if cuotas and "error" not in cuotas and cuotas.get("markets"):
                        datos_resumidos["_cuotas"] = cuotas
                        print(f"     \u2713 Betsafe: {len(cuotas.get('markets', {}))} mercados en tiempo real")
                    else:
                        print(f"     \u2717 Error: {cuotas.get('error', '?')}")
    except Exception:
        pass

    if verbose:
        depurar_partido(datos_resumidos, prediccion_resumida)

    return datos_resumidos, prediccion_resumida


def main():
    print("\n" + "\u2554" + "\u2550" * 48 + "\u2557")
    print("\u2551" + "     \U0001F3AF BETTING AI - Value Betting Analyzer     ".center(48) + "\u2551")
    print("\u2551" + "     Análisis de apuestas con valor esperado    ".center(48) + "\u2551")
    print("\u255a" + "\u2550" * 48 + "\u255d")

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
        print("  0. Salir" if partidos_cache else "")

        opcion = input("\n  Opción: ").strip()

        if opcion == "1":
            print("\n  Cargando próximos partidos de ligas europeas, torneos internacionales y sudamericanos...")
            print("  (Brasileirao, Premier League, La Liga, Bundesliga + UCL, Libertadores, Sudamericana + Liga 1 Peru)")
            try:
                verificar_salud_sofascore(detallado=True)
                partidos_cache = obtener_proximos_partidos()
                partidos_ss = obtener_partidos_sofascore_only()
                partidos_cache.extend(partidos_ss)
                partidos_cache.sort(key=lambda p: p.get("event_date", "") if isinstance(p.get("event_date"), str) else "")
                _mostrar_partidos(partidos_cache)
            except Exception as e:
                print(f"\n  \u2717 Error al obtener partidos: {e}")
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

            # Si se pasa --show-prompt, muestra el prompt en vez de llamar a DeepSeek
            if "--show-prompt" in sys.argv:
                _separador()
                from analyzer import SYSTEM_PROMPT, _crear_prompt_usuario
                user_prompt = _crear_prompt_usuario(datos, prediccion)
                print(f"=== SYSTEM PROMPT ({len(SYSTEM_PROMPT)} chars) ===")
                print(SYSTEM_PROMPT)
                print(f"\n=== USER PROMPT ({len(user_prompt)} chars) ===")
                print(user_prompt)
                print(f"\n=== TOTAL: {len(SYSTEM_PROMPT) + len(user_prompt)} chars enviados a DeepSeek ===")
                _separador()
                input("\n  Presioná Enter para continuar...")
                continue

            print("\n  Consultando a DeepSeek para análisis de value betting...")
            try:
                analisis = analizar_partido(datos, prediccion)
                _separador()
                print(analisis)
                _separador()
            except Exception as e:
                print(f"\n  \u2717 Error al analizar con IA: {e}")
                print("  Verifica que OPENROUTER_API_KEY esté configurada en el archivo .env")

        elif opcion == "3":
            if partidos_cache:
                print("\n  Actualizando lista de partidos...")
                try:
                    partidos_cache = obtener_proximos_partidos()
                    _mostrar_partidos(partidos_cache)
                except Exception as e:
                    print(f"\n  \u2717 Error al actualizar: {e}")
            else:
                print("\n  ¡Hasta luego!")
                sys.exit(0)

        elif opcion == "0":
            print("\n  ¡Hasta luego!")
            sys.exit(0)

        else:
            print("\n  Opción no válida.")


if __name__ == "__main__":
    main()
