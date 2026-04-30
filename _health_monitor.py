r"""
Health monitor para endpoints de SofaScore.

Ejecuta verificar_salud_sofascore() y:
  1. Loggea el resultado en _health_log.jsonl con timestamp
  2. Si DISCORD_HEALTH_WEBHOOK esta en .env, envia alerta a Discord SOLO si algo fallo
  3. Persiste un fingerprint del schema de cada endpoint en _schema_fingerprint.json
     Si algun campo esperado desaparece entre ejecuciones, lo detecta.

Uso:
  python _health_monitor.py            → ejecuta, loggea, alerta si falla
  python _health_monitor.py --quiet    → sin output a consola (para Task Scheduler)

Windows Task Scheduler (cada 30 min):
  Trigger: Daily, repeat every 30 minutes
  Action: python _health_monitor.py --quiet
  Start in: D:\Projects\NuevoAgenIa\betting-ai
"""

import json
import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

LOG_FILE = "_health_log.jsonl"
FINGERPRINT_FILE = "_schema_fingerprint.json"
WEBHOOK_URL = os.getenv("DISCORD_HEALTH_WEBHOOK", "")
QUIET = "--quiet" in sys.argv

# Campos esperados por endpoint (lo que el health-check valida internamente)
EXPECTED_SCHEMA = {
    "scheduled_events": ["events", "events[].homeTeam.name", "events[].awayTeam.name", "events[].id"],
    "event_detail": ["event.homeTeam", "event.referee"],
    "lineups": ["home.formation", "home.players[].player.name"],
    "h2h": ["teamDuel.homeWins", "teamDuel.awayWins", "teamDuel.draws"],
    "performance": ["events[].winnerCode", "events[].homeScore.current", "events[].awayScore.current"],
}


def _log(result: dict):
    entry = {"ts": datetime.now().isoformat(), **result}
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_fingerprint() -> dict | None:
    try:
        with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_fingerprint(result: dict):
    """Guarda el schema actual como fingerprint para futuras comparaciones."""
    fp = {}
    for name, ep in result.get("endpoints", {}).items():
        if ep.get("ok") and EXPECTED_SCHEMA.get(name):
            fp[name] = {
                "saved_at": datetime.now().isoformat(),
                "expected_fields": EXPECTED_SCHEMA[name],
            }
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=2, ensure_ascii=False)


def _check_fingerprint_drift(result: dict) -> list:
    """
    Compara el resultado actual contra el fingerprint guardado.
    Si los campos que antes estaban bien ahora fallan → drift detectado.
    """
    prev = _load_fingerprint()
    if not prev:
        return []

    drifts = []
    for name, old in prev.items():
        ep = result.get("endpoints", {}).get(name, {})
        if ep.get("ok") and not ep.get("campos_ok"):
            drifts.append(f"{name}: schema cambio detectado (antes OK, ahora campos faltantes)")
        elif not ep.get("ok") and old:
            drifts.append(f"{name}: antes OK, ahora FAIL ({ep.get('error', '?')})")

    return drifts


def _alert_discord(result: dict):
    """Envia un embed de Discord solo si algo fallo o hay schema drift."""
    if not WEBHOOK_URL:
        return

    failed = [(name, ep) for name, ep in result.get("endpoints", {}).items() if not ep["ok"]]
    drifts = _check_fingerprint_drift(result)

    if not failed and not drifts:
        return

    fields = []

    if failed:
        for name, ep in failed:
            motivo = ep.get("error", "error desconocido")
            schema = " [schema CAMBIO]" if not ep.get("campos_ok") else ""
            fields.append({"name": f"FAIL: {name}", "value": f"{motivo}{schema}", "inline": False})

    if drifts:
        for d in drifts:
            fields.append({"name": "DRIFT DETECTADO", "value": d, "inline": False})

    total_issues = len(failed) + len(drifts)
    embed = {
        "title": f"SofaScore Health: {total_issues} problema(s)",
        "description": f"Detectado a las {datetime.now().strftime('%H:%M')}",
        "color": 0xff0000,
        "fields": fields,
        "footer": {"text": "Betting AI - Health Monitor"},
        "timestamp": datetime.now().isoformat(),
    }

    try:
        requests.post(WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
    except Exception as e:
        if not QUIET:
            print(f"  [WARN] No se pudo enviar alerta a Discord: {e}")


def main():
    from sofascore_client import verificar_salud_sofascore

    if not QUIET:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] SofaScore Health Monitor")

    result = verificar_salud_sofascore(detallado=not QUIET)
    _log(result)

    # Guardar fingerprint (solo si todo OK, para tener baseline limpio)
    if result["ok"]:
        _save_fingerprint(result)

    # Drift detection
    drifts = _check_fingerprint_drift(result)
    if drifts and not QUIET:
        print(f"  Schema drift detectado:")
        for d in drifts:
            print(f"    {d}")

    _alert_discord(result)

    if not QUIET:
        status = "OK" if result["ok"] else "FAIL"
        print(f"  Status: {status} | Log: {LOG_FILE} | Fingerprint: {FINGERPRINT_FILE}")
        if result["ok"] and not drifts:
            print("  Todo en orden.")
        elif drifts:
            print("  Alertado via Discord." if WEBHOOK_URL else "  (Discord webhook no configurado)")


if __name__ == "__main__":
    main()
