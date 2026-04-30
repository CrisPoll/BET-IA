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
from sofascore_client import enriquecer_datos_partido, verificar_salud_sofascore
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

    print(f"\n  Obteniendo datos de: {local} vs {visitante}...")

    # 1. Obtener detalle completo del partido (BSD)
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
            print("\n  Cargando próximos partidos de ligas europeas y torneos internacionales...")
            print("  (Premier, La Liga, Serie A, Bundesliga, Ligue 1, Champions, Europa League, Libertadores, Sudamericana)")
            try:
                verificar_salud_sofascore(detallado=True)
                partidos_cache = obtener_proximos_partidos()
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
