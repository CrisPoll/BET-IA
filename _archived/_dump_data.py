"""
Script de diagnÃ³stico: vuelca TODOS los datos recibidos de cada fuente
para un partido especÃ­fico, sin pasar por DeepSeek.

Uso: python _dump_data.py [nÃºmero_de_partido]
"""
import json
import logging
import sys

from bsd_client import (
    obtener_proximos_partidos,
    obtener_detalle_partido,
    obtener_predicciones,
    resumir_datos_partido,
    resumir_prediccion,
)
from sofascore_client import enriquecer_datos_partido as enriquecer_sofascore

logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def dump_source(name, data, prefix="  "):
    print(f"\n{'='*70}")
    print(f"{prefix}ð¦ {name}")
    print(f"{'='*70}")
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "1"

    print("Cargando prÃ³ximos partidos...")
    try:
        partidos = obtener_proximos_partidos()
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Encontrados: {len(partidos)}")
    if not partidos:
        print("No hay partidos disponibles.")
        return

    for i, p in enumerate(partidos, 1):
        local = p.get("home_team", "?")
        visitante = p.get("away_team", "?")
        liga = p.get("_league_name", "?")
        fecha = p.get("event_date", "?")[:16].replace("T", " ")
        print(f"  {i}. [{liga}] {local} vs {visitante}  ({fecha})")

    try:
        idx = int(which) - 1
        if idx < 0 or idx >= len(partidos):
            idx = 0
    except ValueError:
        idx = 0

    partido = partidos[idx]
    match_id = partido.get("id")
    local = partido.get("home_team", "?")
    visitante = partido.get("away_team", "?")
    print(f"\n>>> Analizando: {local} vs {visitante} (id={match_id})")

    # PASO 1: BSD
    print("\n--- PASO 1/2: BSD API ---")
    try:
        detalle = obtener_detalle_partido(match_id)
        print(f"  â Detalle obtenido ({len(json.dumps(detalle))} bytes)")
    except Exception as e:
        print(f"  â Error: {e}")
        return

    try:
        predicciones = obtener_predicciones(match_id=match_id)
        if predicciones:
            pred = predicciones[0]
        else:
            league_id = partido.get("league", {}).get("id")
            predicciones = obtener_predicciones(league_id=league_id)
            pred = predicciones[0] if predicciones else {}
        print(f"  â PredicciÃ³n ML obtenida")
    except Exception as e:
        print(f"  â  Error predicciones: {e}")
        pred = {}

    datos = resumir_datos_partido(detalle)
    datos["_league_name"] = partido.get("_league_name", partido.get("league", {}).get("name", "?"))
    datos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"
    pred_resumida = resumir_prediccion(pred)

    dump_source("BSD PREDICCIÃN RESUMIDA", pred_resumida)

    # PASO 2: SofaScore
    print("\n--- PASO 2/2: SofaScore ---")
    try:
        datos = enriquecer_sofascore(datos)
        ss = datos.get("_sofascore", {})
        if ss.get("disponible"):
            print(f"  â SofaScore disponible (event_id={ss.get('event_id')})")
            dump_source("SOFASCORE ENRIQUECIDO", datos["_sofascore"])
        else:
            print(f"  â No disponible: {ss.get('error', '?')}")
    except Exception as e:
        print(f"  â Error: {e}")

    # RESUMEN FINAL
    print(f"\n{'='*70}")
    print("  ð RESUMEN DE FUENTES DISPONIBLES")
    print(f"{'='*70}")
    print(f"  BSD API:           â")
    print(f"  BSD CatBoost ML:   â")
    ss = datos.get("_sofascore", {})
    if ss.get("disponible"):
        extras = []
        if ss.get("alineaciones"):
            extras.append("alineacion")
        if ss.get("h2h"):
            extras.append("H2H")
        if ss.get("detalle_evento"):
            extras.append("arbitro/managers/lesiones")
        if ss.get("form_performance"):
            extras.append("form reciente")
        print(f"  SofaScore:         â ({', '.join(extras)})")
    else:
        print(f"  SofaScore:         â {ss.get('error', '?')}")

    # Vuelco final completo del resumen
    print(f"\n{'='*70}")
    print("  ð¦ DATOS_RESUMIDOS COMPLETO (dict final)")
    print(f"{'='*70}")
    print(json.dumps(datos, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
