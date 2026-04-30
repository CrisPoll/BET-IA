"""
Dump completo de TODOS los datos recibidos de BSD + SofaScore para debug.
Correr: python _debug_dump.py [numero_de_partido]
"""
import json
import sys

from bsd_client import (
    obtener_proximos_partidos,
    obtener_detalle_partido,
    obtener_predicciones,
    resumir_datos_partido,
    resumir_prediccion,
)
from sofascore_client import enriquecer_datos_partido


def dump(title, data, depth=2):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(json.dumps(data, indent=depth, ensure_ascii=False, default=str))


def main():
    # 1. Cargar partidos
    print("Cargando próximos partidos via BSD...")
    try:
        partidos = obtener_proximos_partidos()
    except Exception as e:
        print(f"ERROR BSD: {e}")
        return

    if not partidos:
        print("No hay partidos.")
        return

    # Mostrar lista
    print(f"\n{len(partidos)} partidos encontrados:\n")
    for i, p in enumerate(partidos, 1):
        local = p.get("home_team", "?")
        visitante = p.get("away_team", "?")
        liga = p.get("_league_name", "?")
        fecha = p.get("event_date", "")[:16].replace("T", " ")
        print(f"  {i:>2}. [{liga}] {local} vs {visitante}  ({fecha})")

    # Seleccionar
    which = sys.argv[1] if len(sys.argv) > 1 else input("\nNúmero de partido: ").strip()
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
    liga = partido.get("_league_name", partido.get("league", {}).get("name", "?"))

    print(f"\n>>> Analizando: [{liga}] {local} vs {visitante} (id={match_id})")

    # 2. Dump RAW del partido BSD (lista)
    print(f"\n{'#'*70}")
    print(f"  RAW PARTIDO (BSD list)")
    print(f"{'#'*70}")
    print(json.dumps(partido, indent=2, ensure_ascii=False, default=str))

    # 3. BSD: detalle completo
    print(f"\n{'#'*70}")
    print(f"  BSD: DETALLE COMPLETO (/events/{match_id}/)")
    print(f"{'#'*70}")
    try:
        detalle = obtener_detalle_partido(match_id)
        print(json.dumps(detalle, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # 4. BSD: predicciones
    print(f"\n{'#'*70}")
    print(f"  BSD: PREDICCIONES ML")
    print(f"{'#'*70}")
    try:
        predicciones = obtener_predicciones(match_id=match_id)
        if predicciones:
            print(json.dumps(predicciones[0], indent=2, ensure_ascii=False, default=str))
        else:
            league_id = partido.get("league", {}).get("id")
            predicciones = obtener_predicciones(league_id=league_id)
            if predicciones:
                print(f"(por liga {league_id})")
                print(json.dumps(predicciones[0], indent=2, ensure_ascii=False, default=str))
            else:
                print("NO HAY PREDICCIONES DISPONIBLES")
    except Exception as e:
        print(f"ERROR: {e}")

    # 5. BSD: datos resumidos (lo que ve la IA)
    datos = resumir_datos_partido(detalle)
    datos["_league_name"] = liga
    datos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"
    pred = resumir_prediccion(predicciones[0] if predicciones else {})

    dump("BSD: DATOS RESUMIDOS (input a IA)", datos)

    # 6. SofaScore: enriquecimiento
    print(f"\n{'#'*70}")
    print(f"  SOFASCORE: ENRIQUECIMIENTO")
    print(f"{'#'*70}")
    try:
        datos = enriquecer_datos_partido(datos)
        ss = datos.get("_sofascore", {})
        if ss.get("disponible"):
            print(f"  ✓ Event ID: {ss.get('event_id')}")
            print(f"  ✓ Torneo: {ss.get('torneo')}")
            print(f"  ✓ Status: {ss.get('status')}")
        else:
            print(f"  ✗ {ss.get('error', 'no disponible')}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # 7. Dump COMPLETO final
    dump("DATOS FINALES COMPLETOS (BSD + SofaScore)", datos)


if __name__ == "__main__":
    main()
