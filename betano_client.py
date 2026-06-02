"""
Betano client - cuotas prepartido via JSON directo.

Betano renderiza la pagina del partido con un endpoint JSON equivalente:
    /api{path_publico}?bt=<tab>

No usamos Playwright: la URL publica completa trae el path necesario y el
endpoint responde con los mercados de cada tab usando el Referer correcto.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse, urlunparse

from curl_cffi import requests


BETANO_TABS = ("0", "1", "4", "5", "6", "7", "8", "10", "11", "13")
BETANO_TIMEOUT = 30

BETANO_COMPETITION_ALIASES = {
    "World Cup 2026": [
        "world cup 2026",
        "world cup",
        "fifa world cup",
        "mundial",
    ],
    "International Friendly Games": [
        "international friendly games",
        "international friendly",
        "international",
        "amistosos internacionales",
        "amistoso internacional",
        "friendlies",
    ],
}


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _norm(value: str) -> str:
    value = _strip_accents(value or "").lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def betano_competition_search_terms(competition_name: str) -> list[str]:
    """Alias para futuras busquedas/filtros de competiciones en Betano."""
    needle = _norm(competition_name)
    for canonical, aliases in BETANO_COMPETITION_ALIASES.items():
        terms = [canonical, *aliases]
        if any(_norm(term) in needle or needle in _norm(term) for term in terms):
            return terms
    return [competition_name] if competition_name else []


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:[.,]\d+)?", str(value))
    if match:
        return float(match.group(0).replace(",", "."))
    return None


def _format_line(value: float | int | str | None) -> str:
    line = _to_float(value)
    if line is None:
        return ""
    return str(int(line)) if line.is_integer() else str(line).rstrip("0").rstrip(".")


def _extraer_info_url_betano(url: str) -> dict:
    """Valida una URL publica de Betano y extrae host, path y event_id."""
    raw = (url or "").strip()
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return {"error": "Betano requiere URL completa del partido, no solo event id."}
    if "betano" not in parsed.netloc.lower():
        return {"error": f"La URL no parece ser de Betano: {url}"}

    path = parsed.path or ""
    if path.startswith("/api/"):
        path = path[4:]
    if not path.endswith("/"):
        path += "/"

    event_match = re.search(r"/(\d+)/$", path)
    if not event_match:
        return {"error": f"No se pudo extraer event_id de Betano desde: {url}"}

    public_url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return {
        "base": f"{parsed.scheme}://{parsed.netloc}",
        "path": path,
        "event_id": event_match.group(1),
        "public_url": public_url,
    }


def _selection_side(label: str) -> str:
    n = _norm(label)
    if re.search(r"\b(menos|under)\b", n):
        return "under"
    if re.search(r"\b(mas|over)\b", n):
        return "over"
    return ""


def _team_prefix(name: str, home_team: str = "", away_team: str = "") -> bool:
    n = _norm(name)
    for team in (home_team, away_team):
        t = _norm(team)
        if t and (n.startswith(t + " ") or n.startswith(t + " - ")):
            return True
    return False


def _is_player_prop_market(name: str, home_team: str = "", away_team: str = "") -> bool:
    """Betano mezcla props de jugador en tabs estadísticos; no sirven para picks generales/equipo."""
    raw = name or ""
    n = _norm(raw)
    if "[" in raw and "]" in raw:
        return True
    if _team_prefix(raw, home_team, away_team):
        return False

    full_match_names = {
        "remates totales",
        "tiros al arco",
        "total de faltas cometidas",
        "total de faltas",
        "faltas totales",
        "mas/menos corners",
        "corners en primer tiempo mas/menos",
        "corners en segundo tiempo mas/menos",
        "tarjetas totales mas/menos",
        "total de tarjetas rojas",
    }
    if n in full_match_names:
        return False

    player_stat_markers = [
        " remates totales",
        " tiros al arco",
        " atajadas",
        " asistencias",
        " fueras de juego",
        " faltas cometidas",
        " recibir una tarjeta",
    ]
    return any(marker in n for marker in player_stat_markers)


def _categorize_betano_market(name: str, home_team: str = "", away_team: str = "") -> str:
    n = _norm(name)
    team_prefixed = _team_prefix(name, home_team, away_team)

    if n == "resultado del partido":
        return "1X2"
    if "doble oportunidad" in n:
        return "Double Chance"
    if "ambos equipos anotan" in n:
        return "BTTS"
    if "handicap asiatico" in n:
        return "Asian Handicap"
    if "handicap" in n:
        return "Handicap"
    if "corner" in n or "corners" in n or "corneres" in n:
        return "Team Corners" if team_prefixed else "Corners"
    if "tarjeta" in n:
        return "Cards"
    if "falta" in n:
        return "Team Fouls" if team_prefixed else "Fouls"
    if "tiros al arco" in n or "remates" in n or "tiros" in n:
        return "Team Match Stats" if team_prefixed else "Match Stats"
    if "goles" in n and ("mas/menos" in n or "totales" in n):
        return "Over/Under"
    if "primer tiempo" in n or "segundo tiempo" in n or "tiempo completo" in n:
        return "Halves"
    return "Other"


def _normalizar_selection(sel: dict) -> dict | None:
    odd = sel.get("price", sel.get("odd", sel.get("odds")))
    if odd is None:
        return None
    label = sel.get("name") or sel.get("fullName") or sel.get("label") or ""
    line = _to_float(sel.get("handicap"))
    return {
        "id": str(sel.get("id", "")),
        "label": label,
        "name": label,
        "fullName": sel.get("fullName"),
        "odd": odd,
        "handicap": line,
        "betRef": sel.get("betRef"),
    }


def _append_line(name: str, line) -> str:
    line_text = _format_line(line)
    if not line_text:
        return name
    if re.search(rf"\({re.escape(line_text)}\)\s*$", name):
        return name
    return f"{name} ({line_text})"


def _should_split_market(category: str, selections: list[dict]) -> bool:
    lines = {
        s.get("handicap")
        for s in selections
        if s.get("handicap") not in (None, 0, 0.0)
    }
    if len(lines) <= 1:
        return False
    if category in {
        "Over/Under",
        "Corners",
        "Team Corners",
        "Cards",
        "Fouls",
        "Team Fouls",
        "Match Stats",
        "Team Match Stats",
    }:
        return True
    return any(_selection_side(s.get("label", "")) for s in selections)


def _normalizar_market(raw: dict, home_team: str = "", away_team: str = "") -> list[dict]:
    if raw.get("tableLayout"):
        return []

    base_name = raw.get("name") or raw.get("label") or raw.get("type") or str(raw.get("id", ""))
    if _is_player_prop_market(base_name, home_team, away_team):
        return []

    selections = [
        s for s in (_normalizar_selection(sel) for sel in raw.get("selections", []) or [])
        if s
    ]
    if not selections:
        return []

    category = _categorize_betano_market(base_name, home_team, away_team)
    raw_id = str(raw.get("id") or raw.get("uniqueId") or base_name)
    common = {
        "type": raw.get("type"),
        "typeId": raw.get("typeId"),
        "marketTemplateId": raw.get("type"),
        "category": category,
    }

    if not _should_split_market(category, selections):
        line = _to_float(raw.get("handicap"))
        if line in (None, 0, 0.0) and category in {
            "Over/Under",
            "Corners",
            "Team Corners",
            "Cards",
            "Fouls",
            "Team Fouls",
            "Match Stats",
            "Team Match Stats",
        }:
            selection_lines = {
                s.get("handicap")
                for s in selections
                if s.get("handicap") not in (None, 0, 0.0)
            }
            if len(selection_lines) == 1:
                line = float(next(iter(selection_lines)))
        return [{
            **common,
            "id": raw_id,
            "name": _append_line(base_name, line) if line not in (None, 0, 0.0) else base_name,
            "line": line,
            "selections": selections,
        }]

    markets = []
    grouped: dict[float, list[dict]] = {}
    for sel in selections:
        line = sel.get("handicap")
        if line in (None, 0, 0.0):
            continue
        grouped.setdefault(float(line), []).append(sel)

    for line in sorted(grouped):
        markets.append({
            **common,
            "id": f"{raw_id}:{_format_line(line)}",
            "name": _append_line(base_name, line),
            "line": line,
            "selections": grouped[line],
        })
    return markets


def _extract_teams(event: dict) -> tuple[str, str]:
    participants = event.get("participants") or []
    if len(participants) >= 2:
        home = participants[0].get("name") or participants[0].get("label") or ""
        away = participants[1].get("name") or participants[1].get("label") or ""
        if home and away:
            return home, away

    name = event.get("name") or event.get("shortName") or ""
    if " - " in name:
        home, away = name.split(" - ", 1)
        return home.strip(), away.strip()
    return "", ""


def _set_base_odds(result: dict) -> None:
    for market in result["markets"].values():
        name = _norm(market.get("name", ""))
        selections = market.get("selections", [])
        if name == "resultado del partido":
            for sel in selections:
                label = _norm(sel.get("label", ""))
                if label == "1":
                    result["local"] = sel.get("odd")
                elif label == "x":
                    result["empate"] = sel.get("odd")
                elif label == "2":
                    result["visitante"] = sel.get("odd")
        elif name == "goles totales mas/menos (2.5)":
            for sel in selections:
                side = _selection_side(sel.get("label", ""))
                if side == "over":
                    result["over_25"] = sel.get("odd")
                elif side == "under":
                    result["under_25"] = sel.get("odd")
        elif name == "ambos equipos anotan":
            for sel in selections:
                label = _norm(sel.get("label", ""))
                if label in {"si", "yes"}:
                    result["btts_si"] = sel.get("odd")
                elif label in {"no"}:
                    result["btts_no"] = sel.get("odd")


def _normalizar_betano_payloads(payloads: list[dict], source_url: str = "") -> dict:
    valid_payloads = [p for p in payloads if isinstance(p, dict) and p.get("data", {}).get("event")]
    if not valid_payloads:
        return {"error": "Betano no devolvio datos de evento validos."}

    event = valid_payloads[0]["data"]["event"]
    home_team, away_team = _extract_teams(event)
    markets: dict[str, dict] = {}

    for payload in valid_payloads:
        ev = payload.get("data", {}).get("event", {})
        for raw_market in ev.get("markets", []) or []:
            for market in _normalizar_market(raw_market, home_team, away_team):
                key = f"{market['id']}:{market['name']}"
                markets[key] = market

    if not markets:
        return {"error": "No se encontraron mercados con cuotas en Betano."}

    result = {
        "event_id": str(event.get("id", "")),
        "home_team": home_team,
        "away_team": away_team,
        "competition": event.get("leagueName") or event.get("leagueDescription") or "",
        "source": "betano",
        "bookmaker": "Betano",
        "url": source_url,
        "markets": markets,
        "total_markets": len(markets),
    }
    _set_base_odds(result)
    return result


def _fetch_betano_tab(info: dict, tab: str) -> tuple[dict | None, str | None]:
    api_url = f"{info['base']}/api{info['path']}?bt={tab}"
    headers = {
        "accept": "application/json, text/plain, */*",
        "referer": info["public_url"],
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    try:
        response = requests.get(
            api_url,
            headers=headers,
            impersonate="chrome120",
            timeout=BETANO_TIMEOUT,
        )
    except Exception as exc:
        return None, f"bt={tab}: {exc}"

    if response.status_code != 200:
        return None, f"bt={tab}: HTTP {response.status_code}"

    try:
        payload = response.json()
    except Exception as exc:
        return None, f"bt={tab}: JSON invalido ({exc})"

    if payload.get("errorCode"):
        return None, f"bt={tab}: Betano errorCode={payload.get('errorCode')}"
    if not payload.get("data", {}).get("event"):
        return None, f"bt={tab}: sin data.event"
    return payload, None


def obtener_cuotas_betano_desde_url(url: str) -> dict:
    """Obtiene cuotas Betano desde la URL publica completa del partido."""
    info = _extraer_info_url_betano(url)
    if info.get("error"):
        return {"error": info["error"]}

    payloads = []
    errors = []
    for tab in BETANO_TABS:
        payload, error = _fetch_betano_tab(info, tab)
        if payload:
            payloads.append(payload)
        elif error:
            errors.append(error)

    result = _normalizar_betano_payloads(payloads, info["public_url"])
    if "error" in result and errors:
        return {"error": f"{result['error']} Detalle: {'; '.join(errors[:4])}"}
    if errors:
        result["_warnings"] = errors[:5]
    return result


def _market_line(market: dict):
    line = _to_float(market.get("line"))
    if line is not None:
        return line
    match = re.search(r"\((\d+(?:[.,]\d+)?)\)\s*$", market.get("name", ""))
    return _to_float(match.group(1)) if match else None


def _line_in(market: dict, allowed: set[float]) -> bool:
    line = _market_line(market)
    return line is None or any(abs(line - value) < 0.001 for value in allowed)


def _is_relevant_halves_market(market: dict) -> bool:
    n = _norm(market.get("name", ""))
    if "penal" in n or "apuesta sin empate" in n:
        return False
    return any(token in n for token in [
        "resultado primer tiempo",
        "resultado segundo tiempo",
        "mas/menos goles en primer tiempo",
        "mas/menos goles en segundo tiempo",
        "primer tiempo/tiempo completo",
        "ambos equipos anotan primer tiempo",
        "ambos equipos anotan segundo tiempo",
    ])


def _is_primary_goals_market(market: dict) -> bool:
    n = _norm(market.get("name", ""))
    if market.get("category") != "Over/Under" or not _line_in(market, {1.5, 2.5, 3.5}):
        return False
    if "primer tiempo" in n or "segundo tiempo" in n or "tiempo completo" in n:
        return False
    return (
        n.startswith("goles totales mas/menos")
        or n.startswith("total de goles mas/menos")
        or n.startswith("mas/menos goles")
    )


def _is_primary_btts_market(market: dict) -> bool:
    return market.get("category") == "BTTS" and _norm(market.get("name", "")) == "ambos equipos anotan"


def _format_market(market: dict) -> str:
    selections = []
    for sel in (market.get("selections") or [])[:8]:
        label = sel.get("label") or sel.get("name") or "?"
        selections.append(f"{label}: @{sel.get('odd', '?')}")
    return f"  {market.get('name', '?')}: {' | '.join(selections)}"


def _formatear_cuotas_betano_para_prompt(result: dict) -> str:
    """Formatea cuotas Betano para el prompt con foco en mercados estadisticos."""
    if not result or "error" in result or not result.get("markets"):
        return ""

    parts = ["\n### CUOTAS BETANO (tiempo real)"]
    comp = result.get("competition", "")
    if comp:
        parts.append(f"Competicion: {comp}")
    parts.append("")

    markets = [
        m for m in result.get("markets", {}).values()
        if not _is_player_prop_market(m.get("name", ""), result.get("home_team", ""), result.get("away_team", ""))
    ]
    sections = [
        ("GANADOR", lambda m: m.get("category") == "1X2"),
        ("GOLES", _is_primary_goals_market),
        ("BTTS", _is_primary_btts_market),
        ("CORNERS", lambda m: m.get("category") in {"Corners", "Team Corners"} and _line_in(m, {8.5, 9.5, 10.5, 11.5})),
        ("TARJETAS", lambda m: m.get("category") == "Cards" and _line_in(m, {3.5, 4.5, 5.5})),
        ("TIROS", lambda m: m.get("category") in {"Match Stats", "Team Match Stats"}),
        ("FALTAS", lambda m: m.get("category") in {"Fouls", "Team Fouls"}),
    ]

    for label, predicate in sections:
        shown = [m for m in markets if predicate(m)]
        if not shown:
            continue
        shown.sort(key=lambda m: (m.get("category", ""), m.get("name", ""), _market_line(m) or 999))
        parts.append(f"**{label}**")
        for market in shown[:24]:
            parts.append(_format_market(market))
        parts.append("")

    parts.append("IMPORTANTE: Cuotas oficiales de Betano en tiempo real.")
    return "\n".join(parts)
