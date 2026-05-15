"""
post_match.py — CLI para registrar resultados reales de partidos y evaluar performance.

Uso: python post_match.py
       -> Menú interactivo para buscar predicción y cargar resultado.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import prediction_db as db
from bankroll import update_bet_result


def _separador():
    print("\n" + "=" * 70 + "\n")


def _escanear_input(prompt: str, tipo=int, default=None):
    entrada = input(prompt).strip()
    if entrada == "" and default is not None:
        return default
    try:
        return tipo(entrada)
    except ValueError:
        return default


def menu_principal():
    db.init_db()
    while True:
        _separador()
        print("  POST-MATCH: REGISTRO DE RESULTADOS")
        print("  " + "-" * 30)
        print("  1. Registrar resultado de un partido")
        print("  2. Registrar resultado de apuestas (ganada/perdida/push)")
        print("  3. Ver predicciones sin resultado")
        print("  0. Salir")

        opcion = input("\n  Opción: ").strip()

        if opcion == "1":
            registrar_resultado()
        elif opcion == "2":
            registrar_apuestas()
        elif opcion == "3":
            ver_pendientes()
        elif opcion == "0":
            print("\n  Saliendo...")
            break
        else:
            print("  Opción no válida.")


def registrar_resultado():
    _separador()
    print("  Buscar predicción por nombre del partido (o parte de él):")
    q = input("  > ").strip()
    if not q:
        return

    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM predictions WHERE match_name LIKE ? ORDER BY created_at DESC LIMIT 20",
            (f"%{q}%",)
        )
        rows = cursor.fetchall()

    if not rows:
        print("  No se encontraron predicciones con ese criterio.")
        return

    print(f"\n  Predicciones encontradas:")
    for i, r in enumerate(rows, 1):
        print(f"  {i}. {r['match_name']} ({r['league']}) [{r['match_date'][:10]}]")

    idx = _escanear_input("\n  Selecciona número (0 para cancelar): ", int, 0)
    if idx <= 0 or idx > len(rows):
        return

    sel = rows[idx - 1]
    match_id = sel["match_id"]
    match_name = sel["match_name"]

    print(f"\n  Registrando resultado para: {match_name}")
    print("  Deja en blanco si no tienes el dato.\n")

    result = {}
    result["score_local"] = _escanear_input("  Goles LOCAL: ", int, None)
    result["score_visitor"] = _escanear_input("  Goles VISITANTE: ", int, None)
    result["tiros_local"] = _escanear_input("  Tiros LOCAL: ", int, None)
    result["tiros_visitor"] = _escanear_input("  Tiros VISITANTE: ", int, None)
    result["tiros_arco_local"] = _escanear_input("  Tiros al arco LOCAL: ", int, None)
    result["tiros_arco_visitor"] = _escanear_input("  Tiros al arco VISITANTE: ", int, None)
    result["corners_local"] = _escanear_input("  Corners LOCAL: ", int, None)
    result["corners_visitor"] = _escanear_input("  Corners VISITANTE: ", int, None)
    result["yc_local"] = _escanear_input("  Amarillas LOCAL: ", int, None)
    result["yc_visitor"] = _escanear_input("  Amarillas VISITANTE: ", int, None)
    result["rc_local"] = _escanear_input("  Rojas LOCAL: ", int, 0)
    result["rc_visitor"] = _escanear_input("  Rojas VISITANTE: ", int, 0)
    result["fouls_local"] = _escanear_input("  Faltas LOCAL: ", int, None)
    result["fouls_visitor"] = _escanear_input("  Faltas VISITANTE: ", int, None)

    sl = result.get("score_local")
    sv = result.get("score_visitor")
    if sl is not None and sv is not None:
        result["btts"] = (sl > 0 and sv > 0)
        result["over25"] = (sl + sv > 2.5)
        if sl > sv:
            result["result_1x2"] = "local"
        elif sl == sv:
            result["result_1x2"] = "draw"
        else:
            result["result_1x2"] = "visitor"

    rid = db.save_result(match_id, result)
    if rid == -1:
        print(f"\n  ⚠ No se encontró predicción previa para {match_name}.")
    else:
        print(f"\n  ✓ Resultado registrado. Se calcularon errores automáticamente.")
        print(f"    (Revisa performance en main.py -> opción 4)")


def registrar_apuestas():
    _separador()
    print("  Buscar apuestas por match_id o nombre:")
    q = input("  > ").strip()
    if not q:
        return

    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT b.*, p.match_name FROM bets b JOIN predictions p ON p.id = b.prediction_id "
            "WHERE b.resultado = 'pending' AND (b.match_id LIKE ? OR p.match_name LIKE ?) "
            "ORDER BY b.placed_at DESC LIMIT 20",
            (f"%{q}%", f"%{q}%")
        )
        rows = cursor.fetchall()

    if not rows:
        print("  No hay apuestas pendientes con ese criterio.")
        return

    print(f"\n  Apuestas pendientes:")
    for i, r in enumerate(rows, 1):
        print(f"  {i}. {r['match_name']} | {r['mercado']} {r['seleccion']} @ {r['cuota']} | Stake: {r['stake']:.2f}")

    idx = _escanear_input("\n  Selecciona número (0 para cancelar): ", int, 0)
    if idx <= 0 or idx > len(rows):
        return

    sel = rows[idx - 1]
    bet_id = sel["id"]

    print(f"\n  Resultado de la apuesta '{sel['mercado']} {sel['seleccion']} @ {sel['cuota']}':")
    print("  1. Ganada  2. Perdida  3. Push (devolución)  0. Cancelar")
    res = input("  > ").strip()

    profit = 0.0
    if res == "1":
        resultado = "won"
        profit = sel["stake"] * (sel["cuota"] - 1.0)
    elif res == "2":
        resultado = "lost"
        profit = -sel["stake"]
    elif res == "3":
        resultado = "push"
        profit = 0.0
    else:
        return

    update_bet_result(bet_id, resultado, profit)
    print(f"\n  ✓ Apuesta actualizada: {resultado.upper()} | Profit: {profit:.2f}")


def ver_pendientes():
    _separador()
    with db._conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT p.* FROM predictions p "
            "LEFT JOIN results r ON r.match_id = p.match_id "
            "WHERE r.id IS NULL AND p.created_at >= datetime('now', '-15 days') "
            "ORDER BY p.match_date DESC LIMIT 30"
        )
        rows = cursor.fetchall()

    if not rows:
        print("  No hay predicciones pendientes de resultado.")
        return

    print(f"  Predicciones sin resultado (últimos 15 días):\n")
    for r in rows:
        print(f"  {r['match_name']} ({r['league']}) [{r['match_date'][:10]}] ID={r['match_id']}")
        print(f"     Quant: G={r['proj_total_goals']} T={r['proj_tiros_total']} C={r['proj_corners_total']} YC={r['proj_yc_total']}")

    print("\n  Usa 'Registrar resultado' para cargar los datos.")
    input("  Presiona Enter para volver...")


if __name__ == "__main__":
    menu_principal()
