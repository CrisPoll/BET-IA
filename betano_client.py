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


FULL_MATCH_STAT_MARKET_NAMES = {
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

PLAYER_SHOT_MARKERS = (
    " remates totales",
    " tiros al arco",
    " remates al arco",
    " tiros a puerta",
    " shots on target",
    " total shots",
)


def _is_player_shots_market(name: str, home_team: str = "", away_team: str = "") -> bool:
    """Props de jugador que si queremos conservar: remates y tiros al arco."""
    raw = name or ""
    n = _norm(raw)
    if _team_prefix(raw, home_team, away_team) or n in FULL_MATCH_STAT_MARKET_NAMES:
        return False

    bracketed = "[" in raw and "]" in raw
    if bracketed and any(token in n for token in ("remates", "tiros al arco", "shots")):
        return True
    return any(marker in n for marker in PLAYER_SHOT_MARKERS)


def _is_player_shots_table_market(name: str) -> bool:
    """Mercados tabla donde cada fila es un jugador y el nombre base es la estadistica."""
    n = _norm(name)
    if any(token in n for token in ("falta", "tarjeta", "asistencia", "fuera de juego")):
        return False
    return any(token in n for token in ("remates", "tiros al arco", "remates al arco", "tiros a puerta", "shots"))


def _is_player_prop_market(name: str, home_team: str = "", away_team: str = "") -> bool:
    """Detecta props de jugador mezclados en tabs estadísticos."""
    raw = name or ""
    n = _norm(raw)
    if "[" in raw and "]" in raw:
        return True
    if _team_prefix(raw, home_team, away_team):
        return False

    if n in FULL_MATCH_STAT_MARKET_NAMES:
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


def _is_unsupported_player_prop_market(name: str, home_team: str = "", away_team: str = "") -> bool:
    """Filtra props de jugador que no queremos mandar al prompt por ahora."""
    return (
        _is_player_prop_market(name, home_team, away_team)
        and not _is_player_shots_market(name, home_team, away_team)
    )


def _player_shots_stat_type(name: str) -> str:
    n = _norm(name)
    if "tiros al arco" in n or "remates al arco" in n or "tiros a puerta" in n or "on target" in n:
        return "player_shots_on_target"
    return "player_shots"


def _extract_player_name_from_shots_market(name: str) -> str:
    raw = name or ""
    bracket = re.search(r"\[([^\]]+)\]", raw)
    if bracket:
        return bracket.group(1).strip()

    cleaned = re.sub(
        r"\b(remates totales|tiros al arco|remates al arco|tiros a puerta|total shots|shots on target)\b.*$",
        "",
        raw,
        flags=re.IGNORECASE,
    ).strip(" -:()")
    return cleaned or raw


def _categorize_betano_market(name: str, home_team: str = "", away_team: str = "") -> str:
    n = _norm(name)
    team_prefixed = _team_prefix(name, home_team, away_team)

    if _is_player_shots_market(name, home_team, away_team):
        return "Player Shots On Target" if _player_shots_stat_type(name) == "player_shots_on_target" else "Player Shots"
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
        "player_name": sel.get("_player_context"),
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
        "Player Shots",
        "Player Shots On Target",
    }:
        return True
    return any(_selection_side(s.get("label", "")) for s in selections)


def _looks_like_selection_label(value) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 80:
        return False
    n = _norm(text)
    return bool(
        re.search(r"\b\d+(\.\d+)?\+?$", n)
        or any(token in n for token in ("mas", "menos", "over", "under"))
        or re.search(r"\b(si|no)\b", n)
    )


def _context_from_node(node: dict, base_name: str, current: str = "") -> str:
    context = current or node.get("_player_context") or ""
    if context:
        return context

    base_norm = _norm(base_name)
    for key in ("playerName", "player_name", "participantName", "participant", "competitorName"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            nested = value.get("name") or value.get("label") or value.get("fullName")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

    for key in ("name", "label", "title", "fullName"):
        value = node.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        value_norm = _norm(value)
        if value_norm == base_norm:
            continue
        if _looks_like_selection_label(value):
            continue
        return value.strip()

    return ""


def _extract_raw_selections(raw: dict, base_name: str) -> list[dict]:
    selections = []

    def walk(node, context: str = ""):
        if isinstance(node, dict):
            node_context = _context_from_node(node, base_name, context)
            if any(node.get(key) is not None for key in ("price", "odd", "odds")):
                sel = dict(node)
                if node_context and not sel.get("_player_context"):
                    sel["_player_context"] = node_context
                selections.append(sel)
                return
            for key, value in node.items():
                if key in {"event", "events", "market", "markets"}:
                    continue
                walk(value, node_context)
        elif isinstance(node, list):
            for item in node:
                walk(item, context)

    for sel in raw.get("selections", []) or []:
        walk(sel)

    if raw.get("tableLayout"):
        for key, value in raw.items():
            if key in {"selections", "event", "events", "market", "markets"}:
                continue
            walk(value)

    return selections


def _market_common_without_line(common: dict) -> dict:
    return {key: value for key, value in common.items() if key != "raw_line"}


def _build_markets_from_selections(base_name: str, raw_id: str, common: dict, selections: list[dict]) -> list[dict]:
    category = common.get("category", "Other")
    output_common = _market_common_without_line(common)

    if not _should_split_market(category, selections):
        line = _to_float(common.get("raw_line"))
        if line in (None, 0, 0.0) and category in {
            "Over/Under",
            "Corners",
            "Team Corners",
            "Cards",
            "Fouls",
            "Team Fouls",
            "Match Stats",
            "Team Match Stats",
            "Player Shots",
            "Player Shots On Target",
        }:
            selection_lines = {
                s.get("handicap")
                for s in selections
                if s.get("handicap") not in (None, 0, 0.0)
            }
            if len(selection_lines) == 1:
                line = float(next(iter(selection_lines)))
        return [{
            **output_common,
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
            **output_common,
            "id": f"{raw_id}:{_format_line(line)}",
            "name": _append_line(base_name, line),
            "line": line,
            "selections": grouped[line],
        })
    return markets


def _normalizar_market(raw: dict, home_team: str = "", away_team: str = "") -> list[dict]:
    base_name = raw.get("name") or raw.get("label") or raw.get("type") or str(raw.get("id", ""))
    raw_selections = _extract_raw_selections(raw, base_name)
    has_player_context = any(sel.get("_player_context") for sel in raw_selections)
    is_player_table = bool(
        raw.get("tableLayout")
        and has_player_context
        and _is_player_shots_table_market(base_name)
        and not _team_prefix(base_name, home_team, away_team)
    )

    if raw.get("tableLayout") and not (_is_player_shots_market(base_name, home_team, away_team) or is_player_table):
        return []

    selections = [
        s for s in (_normalizar_selection(sel) for sel in raw_selections)
        if s
    ]
    if not selections:
        return []

    if _is_unsupported_player_prop_market(base_name, home_team, away_team) and not is_player_table:
        return []

    category = (
        "Player Shots On Target" if _player_shots_stat_type(base_name) == "player_shots_on_target" else "Player Shots"
    ) if is_player_table else _categorize_betano_market(base_name, home_team, away_team)
    raw_id = str(raw.get("id") or raw.get("uniqueId") or base_name)
    common = {
        "type": raw.get("type"),
        "typeId": raw.get("typeId"),
        "marketTemplateId": raw.get("type"),
        "category": category,
        "raw_line": raw.get("handicap"),
    }
    if is_player_table:
        markets = []
        grouped_by_player: dict[str, list[dict]] = {}
        for sel in selections:
            player = sel.get("player_name")
            if not player:
                continue
            grouped_by_player.setdefault(player, []).append(sel)
        for player, player_selections in grouped_by_player.items():
            player_common = {
                **common,
                "player_name": player,
                "stat_type": _player_shots_stat_type(base_name),
            }
            player_base_name = f"{player} {base_name}".strip()
            markets.extend(_build_markets_from_selections(
                player_base_name,
                f"{raw_id}:{_norm(player)}",
                player_common,
                player_selections,
            ))
        return markets

    if category in {"Player Shots", "Player Shots On Target"}:
        common["player_name"] = _extract_player_name_from_shots_market(base_name)
        common["stat_type"] = _player_shots_stat_type(base_name)

    return _build_markets_from_selections(base_name, raw_id, common, selections)


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


def _format_player_shots_market(market: dict) -> str:
    player = market.get("player_name") or _extract_player_name_from_shots_market(market.get("name", ""))
    stat_label = "tiros al arco" if market.get("stat_type") == "player_shots_on_target" else "remates"
    selections = []
    for sel in (market.get("selections") or [])[:10]:
        label = sel.get("label") or sel.get("name") or "?"
        selections.append(f"{label}: @{sel.get('odd', '?')}")
    return f"  {player} - {stat_label}: {' | '.join(selections)}"


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
        if not _is_unsupported_player_prop_market(m.get("name", ""), result.get("home_team", ""), result.get("away_team", ""))
    ]
    sections = [
        ("GANADOR", lambda m: m.get("category") == "1X2"),
        ("GOLES", _is_primary_goals_market),
        ("BTTS", _is_primary_btts_market),
        ("CORNERS", lambda m: m.get("category") in {"Corners", "Team Corners"} and _line_in(m, {8.5, 9.5, 10.5, 11.5})),
        ("TARJETAS", lambda m: m.get("category") == "Cards" and _line_in(m, {3.5, 4.5, 5.5})),
        ("TIROS", lambda m: m.get("category") in {"Match Stats", "Team Match Stats"}),
        ("REMATES JUGADORES", lambda m: m.get("category") in {"Player Shots", "Player Shots On Target"}),
        ("FALTAS", lambda m: m.get("category") in {"Fouls", "Team Fouls"}),
    ]

    for label, predicate in sections:
        shown = [m for m in markets if predicate(m)]
        if not shown:
            continue
        shown.sort(key=lambda m: (m.get("category", ""), m.get("name", ""), _market_line(m) or 999))
        parts.append(f"**{label}**")
        for market in shown[:24]:
            if market.get("category") in {"Player Shots", "Player Shots On Target"}:
                parts.append(_format_player_shots_market(market))
            else:
                parts.append(_format_market(market))
        parts.append("")

    parts.append("IMPORTANTE: Cuotas oficiales de Betano en tiempo real.")
    parts.append("Para REMATES JUGADORES, comparar contra MEDIAS POR JUGADOR y confirmar que el jugador sea titular/probable antes de recomendar.")
    return "\n".join(parts)
