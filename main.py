"""
Punto de entrada interactivo de betting-ai.
 
Menú en consola para:
   1. Ver próximos partidos de ligas europeas y torneos internacionales
   2. Seleccionar un partido por número
   3. Obtener todos los datos automáticamente
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
from sofascore_client import enriquecer_datos_partido
from odds_client import enriquecer_cuotas
from score365_client import enriquecer_datos_partido as enriquecer_365
from analyzer import analizar_partido

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _separador():
    """Imprime una línea separadora."""
    print("\n" + "=" * 70 + "\n")


def _mostrar_partidos(partidos: list):
    """
    Muestra la lista numerada de partidos disponibles.

    Args:
        partidos: Lista de partidos desde la BSD API.
    """
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
        # Truncar la fecha para mostrarla más limpia
        if len(fecha) > 16:
            fecha = fecha[:16].replace("T", " ")
        print(f"  {i:>3}. [{liga}] {local} vs {visitante}")
        print(f"       {fecha}")
    return True


def _seleccionar_partido(partidos: list) -> dict:
    """Pide al usuario que seleccione un partido por número."""
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
    """
    Carga todos los datos necesarios de un partido desde BSD.

    Args:
        partido: Diccionario del partido (de la lista).
        verbose: Si True, muestra dump completo de datos recibidos.

    Returns:
        Tupla (datos_resumidos, prediccion_resumida) o (None, None) si falla.
    """
    match_id = partido.get("id")
    local = partido.get("home_team", "?")
    visitante = partido.get("away_team", "?")

    print(f"\n  Obteniendo datos de: {local} vs {visitante}...")

    # 1. Obtener detalle completo del partido
    try:
        detalle = obtener_detalle_partido(match_id)
        print("  ✓ Datos del partido obtenidos")
    except Exception as e:
        print(f"  ✗ Error al obtener detalle del partido: {e}")
        return None, None

    # 2. Obtener predicciones ML para este partido
    try:
        predicciones = obtener_predicciones(match_id=match_id)
        if predicciones:
            prediccion = predicciones[0]
            print("  ✓ Predicción ML obtenida")
        else:
            # Intentar por liga si no hay predicción específica
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

    # 3. Resumir datos para la IA
    datos_resumidos = resumir_datos_partido(detalle)
    datos_resumidos["_league_name"] = partido.get("_league_name",
        partido.get("league", {}).get("name", "?"))
    # Asegurar que el nombre del partido sea correcto
    datos_resumidos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"

    prediccion_resumida = resumir_prediccion(prediccion)

    # 3.5 Enriquecer con fuentes externas (SofaScore + 365Score + Odds API)
    # SofaScore: alineaciones, stats equipo, H2H alternativo
    try:
        datos_resumidos = enriquecer_datos_partido(datos_resumidos)
        if datos_resumidos.get("_sofascore", {}).get("disponible"):
            ss_info = datos_resumidos["_sofascore"]
            partes = []
            if ss_info.get("alineaciones"):
                partes.append("alineaciones")
            if ss_info.get("stats_equipo_local"):
                partes.append("stats equipo (remates/tarjetas/corners)")
            if ss_info.get("standings"):
                partes.append("standings")
            if ss_info.get("h2h"):
                partes.append("H2H detallado")
            if ss_info.get("top_jugadores_local"):
                partes.append("top jugadores + ratings")
            if ss_info.get("detalle_evento"):
                partes.append("arbitro + managers + lesiones")
            print(f"  ✓ SofaScore: {', '.join(partes) if partes else 'encontrado (sin alineaciones aun)'}")
        else:
            pass  # Silencio si no esta en SofaScore
    except Exception as e:
        print(f"  ⚠ SofaScore no disponible: {e}")

    # 365Score: fuente adicional de stats
    try:
        datos_resumidos = enriquecer_365(datos_resumidos)
        if datos_resumidos.get("_365score", {}).get("disponible"):
            t365 = datos_resumidos["_365score"]
            if t365.get("stats_equipo_local_365") or t365.get("stats_equipo_visitante_365"):
                print("  ✓ 365Score: stats adicionales obtenidos")
            else:
                print("  ✓ 365Score: encontrado (sin stats detalladas)")
    except Exception as e:
        print(f"  ⚠ 365Score no disponible: {e}")

    # The Odds API: cuotas frescas multi-bookmaker
    try:
        datos_resumidos = enriquecer_cuotas(datos_resumidos)
        if datos_resumidos.get("_odds_api", {}).get("disponible"):
            print("  ✓ Cuotas actualizadas via The Odds API")
    except Exception as e:
        print(f"  ⚠ Odds API no disponible: {e}")

    # Mostrar resumen de lo recibido
    if verbose:
        print(f"\n  ┌─ RESUMEN DE DATOS RECIBIDOS ─────────────────────────────")
        print(f"  │ BSD API:           ✓ datos base + H2H + forma + cuotas + prediccion")
        print(f"  │ BSD CatBoost ML:   ✓ prediccion (1X2, xG, O2.5, BTTS)")
        ss_info = datos_resumidos.get("_sofascore", {})
        if ss_info.get("disponible"):
            extras = []
            if ss_info.get("alineaciones"):
                extras.append("alineacion titular (fuente autoritativa)")
            if ss_info.get("stats_equipo_local"):
                extras.append("stats equipo (remates, tarjetas, corners)")
            if ss_info.get("standings"):
                extras.append("standings (tabla de posiciones)")
            if ss_info.get("h2h"):
                extras.append("H2H detallado")
            if ss_info.get("top_jugadores_local"):
                extras.append("top jugadores + ratings")
            if ss_info.get("detalle_evento"):
                extras.append("arbitro/managers/lesiones")
            if ss_info.get("estadisticas_disponibles"):
                extras.append("stats post-partido")
            print(f"  │ SofaScore:         ✓ ({', '.join(extras)})" if extras else f"  │ SofaScore:         ✓ (encontrado)")
        else:
            print(f"  │ SofaScore:         ✗ no disponible")
        t365 = datos_resumidos.get("_365score", {})
        if t365.get("disponible"):
            print(f"  │ 365Score:          ✓ stats adicionales")
        else:
            print(f"  │ 365Score:          ✗ no disponible")
        odds_info = datos_resumidos.get("_odds_api", {})
        if odds_info.get("disponible"):
            print(f"  │ The Odds API:      ✓ cuotas frescas ({odds_info.get('bookmakers', '?')})")
        else:
            print(f"  │ The Odds API:      ✗ no disponible")
        print(f"  └──────────────────────────────────────────────────────────\n")

        depurar_partido(datos_resumidos, prediccion_resumida)

    return datos_resumidos, prediccion_resumida


def main():
    """Función principal del menú interactivo."""
    print("\n" + "╔" + "═" * 48 + "╗")
    print("║" + "     🎯 BETTING AI - Value Betting Analyzer     ".center(48) + "║")
    print("║" + "     Análisis de apuestas con valor esperado    ".center(48) + "║")
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
        print("  0. Salir" if partidos_cache else "")

        opcion = input("\n  Opción: ").strip()

        if opcion == "1":
            print("\n  Cargando próximos partidos de ligas europeas y torneos internacionales...")
            print("  (Premier, La Liga, Serie A, Bundesliga, Ligue 1, Champions, Europa League, Libertadores, Sudamericana)")
            try:
                partidos_cache = obtener_proximos_partidos()
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

            # Cargar y analizar
            _separador()
            datos, prediccion = _cargar_datos_partido(partido)

            if datos is None:
                print("\n  No se pudieron obtener los datos suficientes para el análisis.")
                continue

            # Llamar a la IA
            print("  Consultando a DeepSeek para análisis de value betting...")
            try:
                analisis = analizar_partido(datos, prediccion)
                _separador()
                print(analisis)
                _separador()
            except Exception as e:
                print(f"\n  ✗ Error al analizar con IA: {e}")
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

        elif opcion == "0":
            print("\n  ¡Hasta luego!")
            sys.exit(0)

        else:
            print("\n  Opción no válida.")


if __name__ == "__main__":
    main()
