"""
quant_model.py — Modelo predictivo cuantitativo ligero para betting-ai.

No entrena XGBoost desde cero (necesitaría muchos datos históricos).
En cambio, implementa un modelo de proyección basado en regresión
ponderada + features de forma reciente, estilo y contexto.

La idea: dar una base numérica sólida antes de que el LLM aplique
ajustes cualitativos.
"""

import json
import math
from typing import Dict, Optional, List

from competition_config import get_competition_flags


_MARKET_STAT_KEYS = {
    "shots": "tiros_total",
    "sot": "tiros_arco",
    "corners": "corners",
    "yc": "amarillas",
    "fouls": "faltas",
}

_FORWARD_POSITIONS = {"f", "fw", "forward", "striker"}


def _weighted_avg(values: List[float], weights: Optional[List[float]] = None) -> Optional[float]:
    """Promedio ponderado; por defecto pondera más lo reciente."""
    if not values:
        return None
    if weights is None:
        # Peso exponencial decreciente: reciente vale más
        n = len(values)
        weights = [0.35, 0.25, 0.20, 0.12, 0.08]
        # Normalizar si hay menos de 5
        weights = weights[:n]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

    return round(sum(v * w for v, w in zip(values, weights)), 2)


def _std(values: List[float]) -> float:
    """Desviación estándar."""
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _regress_to_mean(value: float, league_avg: float, n_matches: int, prior_matches: int = 10) -> float:
    """
    Regresión a la media: ajusta un promedio hacia la liga según
    el tamaño de muestra. Muestra pequeña = más regresión.
    """
    if n_matches <= 0:
        return league_avg
    # Formula bayesiana simple
    weight = n_matches / (n_matches + prior_matches)
    return round(value * weight + league_avg * (1 - weight), 2)


def _extract_last_n(values: List[float], n: int = 5) -> List[float]:
    """Extrae los últimos N valores no-nulos."""
    clean = [v for v in values if v is not None]
    return clean[-n:] if len(clean) >= n else clean


def _parse_form_string(form_str: str) -> int:
    """Devuelve puntos esperados base según forma (W=3, D=1, L=0 promedio)."""
    if not form_str:
        return 1.5
    puntos = 0
    for r in form_str.upper():
        if r == "W":
            puntos += 3
        elif r == "D":
            puntos += 1
        elif r == "L":
            puntos += 0
    return round(puntos / len(form_str), 2)


def _first_number(*values):
    for value in values:
        parsed = _maybe_float(value)
        if parsed is not None:
            return parsed
    return None


def _maybe_float(value) -> Optional[float]:
    """Convierte números aunque SofaScore/BSD los entregue como string."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = value.strip().replace(",", ".")
        try:
            return float(clean)
        except ValueError:
            return None
    return None


def _num(value, default: float) -> float:
    """Devuelve value si es numérico; si no, default."""
    parsed = _maybe_float(value)
    return parsed if parsed is not None else default


def _season_per_match(stats: dict, key: str) -> Optional[float]:
    value = _maybe_float(stats.get(key))
    matches = _maybe_float(stats.get("matches_played"))
    if value is None or matches is None or matches <= 0:
        return None
    return round(value / matches, 2)


def _sofascore_recent_avg(ss: dict, perf_key: str, stat_key: str) -> Optional[float]:
    """Promedio reciente de SofaScore para un equipo, escogiendo home/away del partido histórico."""
    perf = ss.get(perf_key, {})
    detalle = perf.get("detalle", [])
    stats_list = ss.get(f"{perf_key}_stats_per_match", [])
    values = []
    for idx, match in enumerate(detalle):
        if idx >= len(stats_list) or not stats_list[idx]:
            continue
        stat = stats_list[idx].get(stat_key, {})
        side = "home" if match.get("local") else "away"
        value = stat.get(side)
        parsed = _maybe_float(value)
        if parsed is not None:
            values.append(parsed)
    return _weighted_avg(values[:5]) if values else None


def _norm_text(value: str) -> str:
    return (value or "").strip().casefold()


def _stat_value_for_team(stat: dict, stat_key: str, team_home: bool, against: bool = False) -> Optional[float]:
    values = stat.get(stat_key, {}) if isinstance(stat, dict) else {}
    if not isinstance(values, dict):
        return None
    side = "home" if team_home else "away"
    if against:
        side = "away" if team_home else "home"
    value = values.get(side)
    return _maybe_float(value)


def _split_stat_avg(
    ss: dict,
    perf_key: str,
    stat_key: str,
    venue: Optional[bool] = None,
    tournament: Optional[str] = None,
    against: bool = False,
) -> tuple[Optional[float], int]:
    """Promedio ponderado por split: reciente, casa/fuera y/o torneo."""
    perf = ss.get(perf_key, {})
    detalle = perf.get("detalle", [])
    stats_list = ss.get(f"{perf_key}_stats_per_match", [])
    target_tournament = _norm_text(tournament or "")
    values = []

    for idx, match in enumerate(detalle):
        if idx >= len(stats_list) or not stats_list[idx]:
            continue
        team_home = bool(match.get("local"))
        if venue is not None and team_home != venue:
            continue
        if target_tournament and _norm_text(match.get("torneo", "")) != target_tournament:
            continue
        value = _stat_value_for_team(stats_list[idx], stat_key, team_home, against=against)
        if value is not None:
            values.append(value)

    return (_weighted_avg(values[:5]) if values else None, len(values))


def _build_team_market_splits(ss: dict, perf_key: str, current_tournament: str) -> dict:
    """Crea splits a favor/concedidos para mercados estadísticos."""
    split_specs = {
        "recent": {},
        "home": {"venue": True},
        "away": {"venue": False},
    }
    if current_tournament:
        split_specs.update({
            "current": {"tournament": current_tournament},
            "current_home": {"tournament": current_tournament, "venue": True},
            "current_away": {"tournament": current_tournament, "venue": False},
        })

    result = {}
    for split_name, filters in split_specs.items():
        split = {}
        max_n = 0
        for metric, stat_key in _MARKET_STAT_KEYS.items():
            avg_for, n_for = _split_stat_avg(ss, perf_key, stat_key, against=False, **filters)
            avg_against, n_against = _split_stat_avg(ss, perf_key, stat_key, against=True, **filters)
            if avg_for is not None or avg_against is not None:
                split[metric] = {"for": avg_for, "against": avg_against}
                max_n = max(max_n, n_for, n_against)
        if split:
            split["n"] = max_n
            result[split_name] = split
    return result


def _apply_market_split_features(features: dict, ss: dict):
    current_tournament = ss.get("torneo", "")
    splits = {
        "local": _build_team_market_splits(ss, "form_performance_local", current_tournament),
        "visitor": _build_team_market_splits(ss, "form_performance_visitante", current_tournament),
    }
    if any(splits.values()):
        features["market_splits"] = splits


def _split_value(features: dict, side: str, split: str, metric: str, field: str) -> Optional[float]:
    value = (
        features.get("market_splits", {})
        .get(side, {})
        .get(split, {})
        .get(metric, {})
        .get(field)
    )
    return _maybe_float(value)


def _best_split_value(
    features: dict,
    side: str,
    metric: str,
    field: str,
    preferred_splits: list[str],
    fallback: Optional[float] = None,
) -> Optional[float]:
    for split in preferred_splits:
        value = _split_value(features, side, split, metric, field)
        if value is not None:
            return value
    return _maybe_float(fallback)


def _blend_attack_defense(attack: Optional[float], opponent_conceded: Optional[float], attack_weight: float = 0.62) -> Optional[float]:
    if attack is not None and opponent_conceded is not None:
        return attack * attack_weight + opponent_conceded * (1 - attack_weight)
    return attack if attack is not None else opponent_conceded


def _apply_sofascore_features(features: dict, ss: dict):
    """Usa SofaScore como fuente primaria cuando BSD no trae promedios suficientes."""
    if not ss or not ss.get("disponible"):
        return

    local_perf = ss.get("form_performance_local", {})
    visitor_perf = ss.get("form_performance_visitante", {})
    if local_perf.get("forma_string"):
        features["form_local_pts"] = _parse_form_string(local_perf.get("forma_string", ""))
    if visitor_perf.get("forma_string"):
        features["form_visitor_pts"] = _parse_form_string(visitor_perf.get("forma_string", ""))
    features["xG_local"] = _first_number(features.get("xG_local"), local_perf.get("promedio_goles_favor"))
    features["xG_visitor"] = _first_number(features.get("xG_visitor"), visitor_perf.get("promedio_goles_favor"))
    features["xGc_local"] = _first_number(features.get("xGc_local"), local_perf.get("promedio_goles_contra"))
    features["xGc_visitor"] = _first_number(features.get("xGc_visitor"), visitor_perf.get("promedio_goles_contra"))

    sf_map = [
        ("tiros_local_avg", "form_performance_local", "tiros_total"),
        ("tiros_visitor_avg", "form_performance_visitante", "tiros_total"),
        ("tiros_arco_local_avg", "form_performance_local", "tiros_arco"),
        ("tiros_arco_visitor_avg", "form_performance_visitante", "tiros_arco"),
        ("yc_local_avg", "form_performance_local", "amarillas"),
        ("yc_visitor_avg", "form_performance_visitante", "amarillas"),
        ("fouls_local_avg", "form_performance_local", "faltas"),
        ("fouls_visitor_avg", "form_performance_visitante", "faltas"),
    ]
    for target, perf_key, stat_key in sf_map:
        value = _sofascore_recent_avg(ss, perf_key, stat_key)
        if value is not None:
            features[target] = value

    features["corners_local_avg"] = _sofascore_recent_avg(ss, "form_performance_local", "corners")
    features["corners_visitor_avg"] = _sofascore_recent_avg(ss, "form_performance_visitante", "corners")
    _apply_market_split_features(features, ss)

    season_local = ss.get("team_stats_local", {})
    season_visitor = ss.get("team_stats_visitante", {})
    season_map = [
        ("tiros_local_avg", _season_per_match(season_local, "shots")),
        ("tiros_visitor_avg", _season_per_match(season_visitor, "shots")),
        ("tiros_arco_local_avg", _season_per_match(season_local, "shots_on_target")),
        ("tiros_arco_visitor_avg", _season_per_match(season_visitor, "shots_on_target")),
        ("yc_local_avg", _season_per_match(season_local, "yellow_cards")),
        ("yc_visitor_avg", _season_per_match(season_visitor, "yellow_cards")),
        ("fouls_local_avg", _season_per_match(season_local, "fouls")),
        ("fouls_visitor_avg", _season_per_match(season_visitor, "fouls")),
        ("corners_local_avg", _season_per_match(season_local, "corners")),
        ("corners_visitor_avg", _season_per_match(season_visitor, "corners")),
    ]
    for target, value in season_map:
        if features.get(target) is None and _maybe_float(value) is not None:
            features[target] = value


def _position_is_forward(position: str) -> bool:
    return (position or "").strip().casefold() in _FORWARD_POSITIONS


def _missing_forward_count(lineup_side: dict) -> int:
    bajas = (lineup_side or {}).get("bajas", {})
    count = 0
    for player in bajas.get("confirmadas", []) or []:
        if _position_is_forward(player.get("posicion")):
            count += 1
    return count


def _apply_lineup_absence_features(features: dict, ss: dict):
    """Extrae ausencias ofensivas simples para no inflar tiros al arco."""
    lineups = ss.get("alineaciones", {}) if isinstance(ss, dict) else {}
    if not isinstance(lineups, dict):
        return
    features["missing_forwards_local"] = _missing_forward_count(lineups.get("local", {}))
    features["missing_forwards_visitor"] = _missing_forward_count(lineups.get("visitante", {}))


def _sot_cap_ratio(features: dict, side: str) -> float:
    """Techo conservador para tiros al arco sobre tiros totales."""
    missing_key = "missing_forwards_local" if side == "local" else "missing_forwards_visitor"
    missing_forwards = int(features.get(missing_key) or 0)
    if missing_forwards >= 2:
        return 0.32
    if missing_forwards == 1:
        return 0.34
    return 0.38


def _cap_sot_projection(raw_sot: float, shots: float, features: dict, side: str) -> float:
    """Evita que un equipo con mucho volumen proyecte SOT irreal sin calidad ofensiva."""
    cap = max(1.0, shots * _sot_cap_ratio(features, side))
    return max(1.0, min(raw_sot, cap))


def build_features(datos_resumidos: dict, prediccion_resumida: dict) -> dict:
    """
    Extrae y normaliza features de los datos existentes para
    alimentar al modelo cuantitativo.
    """
    features = {}
    league_id = datos_resumidos.get("league_id")
    league_name = datos_resumidos.get("liga")
    features.update(get_competition_flags(league_id, league_name))

    # --- Datos BSD v1 ---
    forma_loc = datos_resumidos.get("forma_local", {})
    forma_vis = datos_resumidos.get("forma_visitante", {})

    features["form_local_pts"] = _parse_form_string(forma_loc.get("forma_string", ""))
    features["form_visitor_pts"] = _parse_form_string(forma_vis.get("forma_string", ""))

    features["xG_local"] = forma_loc.get("xG_promedio")
    features["xG_visitor"] = forma_vis.get("xG_promedio")
    features["xGc_local"] = forma_loc.get("xG_contra_promedio")
    features["xGc_visitor"] = forma_vis.get("xG_contra_promedio")

    features["tiros_local_avg"] = forma_loc.get("remates_promedio")
    features["tiros_visitor_avg"] = forma_vis.get("remates_promedio")
    features["tiros_arco_local_avg"] = forma_loc.get("remates_arco_promedio")
    features["tiros_arco_visitor_avg"] = forma_vis.get("remates_arco_promedio")

    features["yc_local_avg"] = forma_loc.get("amarillas_promedio")
    features["yc_visitor_avg"] = forma_vis.get("amarillas_promedio")
    features["fouls_local_avg"] = forma_loc.get("faltas_promedio")
    features["fouls_visitor_avg"] = forma_vis.get("faltas_promedio")

    # --- Datos BSD v2 (enriquecidos) ---
    v2 = datos_resumidos.get("_bsd_v2", {})
    if v2:
        stats = v2.get("stats", {})
        if stats and "_error" not in stats:
            per_team = stats.get("stats", {})
            home_st = per_team.get("home", {})
            away_st = per_team.get("away", {})
            features["xg_per_min_home"] = (home_st.get("xg") or {}).get("actual")
            features["xg_per_min_away"] = (away_st.get("xg") or {}).get("actual")

    v2_detail = datos_resumidos.get("_v2_detail", {})
    features["is_derby"] = 1 if v2_detail.get("is_local_derby") else 0
    features["is_neutral"] = 1 if v2_detail.get("is_neutral_ground") else 0
    features["travel_km"] = v2_detail.get("travel_distance_km", 0) or 0

    # --- Alineaciones (SofaScore) ---
    ss = datos_resumidos.get("_sofascore", {})
    _apply_sofascore_features(features, ss)
    _apply_lineup_absence_features(features, ss)
    lineups = ss.get("alineaciones", {})
    lineup_sides = [side for side in [lineups.get("local", {}), lineups.get("visitante", {})] if side]
    features["lineup_confirmed"] = 1 if lineup_sides and all(
        side.get("confirmada") for side in lineup_sides
    ) else 0

    # --- Tabla de posiciones / contexto ---
    standings = datos_resumidos.get("_standings", {})
    if not standings:
        standings = datos_resumidos.get("standings", {})
    features["home_position"] = None
    features["away_position"] = None
    if standings and isinstance(standings, dict):
        # Buscar posiciones de local y visitante
        home = datos_resumidos.get("home_team", datos_resumidos.get("local", ""))
        away = datos_resumidos.get("away_team", datos_resumidos.get("visitante", ""))
        # Normalizar para matching
        from utils import normalizar_nombre
        home_norm = normalizar_nombre(home)
        away_norm = normalizar_nombre(away)
        # Intentar extraer de diferentes formatos de standings
        for source in [standings]:
            for entry in source.get("table", source.get("standings", [])):
                if isinstance(entry, dict):
                    team_name = normalizar_nombre(entry.get("team", entry.get("name", "")))
                    pos = entry.get("position", entry.get("rank"))
                    if team_name == home_norm and pos is not None:
                        features["home_position"] = pos
                    if team_name == away_norm and pos is not None:
                        features["away_position"] = pos

    # --- BSD CatBoost preds (puede ser None) ---
    features["bsd_prob_local"] = prediccion_resumida.get("prob_local")
    features["bsd_prob_draw"] = prediccion_resumida.get("prob_empate")
    features["bsd_prob_visitor"] = prediccion_resumida.get("prob_visitante")
    features["bsd_xg_local"] = prediccion_resumida.get("xG_local")
    features["bsd_xg_visitor"] = prediccion_resumida.get("xG_visitante")
    features["bsd_over25"] = prediccion_resumida.get("prob_over_25")
    features["bsd_btts"] = prediccion_resumida.get("prob_btts")

    # --- Cuotas bookmaker ---
    cuotas = datos_resumidos.get("_cuotas", {})
    if not cuotas:
        cuotas = datos_resumidos.get("cuotas", {})
    features["odds_local"] = cuotas.get("local")
    features["odds_draw"] = cuotas.get("empate")
    features["odds_visitor"] = cuotas.get("visitante")
    features["odds_over25"] = cuotas.get("over_25")

    return features


def project_goals(features: dict, league_avg_goals: float = 2.65) -> Dict[str, float]:
    """
    Proyecta goles esperados combinando xG de BSD + CatBoost + forma.
    """
    # Base: xG o CatBoost
    base_local = features.get("bsd_xg_local") or features.get("xG_local") or 1.3
    base_visitor = features.get("bsd_xg_visitor") or features.get("xG_visitor") or 1.0

    # Ajuste por forma (W/D/L)
    form_loc = features.get("form_local_pts", 1.5)
    form_vis = features.get("form_visitor_pts", 1.5)
    form_adj = (form_loc - form_vis) * 0.1  # +/- según forma

    # Ajuste por posición en tabla
    pos_loc = features.get("home_position")
    pos_vis = features.get("away_position")
    pos_adj = 0.0
    if pos_loc and pos_vis:
        diff = pos_vis - pos_loc  # Si visitante está más abajo (número mayor), local tiene ventaja
        pos_adj = diff * 0.02

    # Ajuste derby/neutral
    derby_adj = -0.15 if features.get("is_derby") else 0.0
    neutral_adj = -0.1 if features.get("is_neutral") else 0.0
    travel_adj = -0.0005 * (features.get("travel_km") or 0)

    proj_local = max(0.3, _regress_to_mean(
        base_local + form_adj + pos_adj + derby_adj + neutral_adj + travel_adj,
        league_avg_goals / 2, n_matches=5
    ))
    proj_visitor = max(0.2, _regress_to_mean(
        base_visitor - form_adj - pos_adj + derby_adj + neutral_adj - travel_adj,
        league_avg_goals / 2, n_matches=5
    ))

    total = proj_local + proj_visitor

    return {
        "goals_local": round(proj_local, 2),
        "goals_visitor": round(proj_visitor, 2),
        "total_goals": round(total, 2),
    }


def project_tiros(features: dict) -> Dict[str, float]:
    """Proyecta tiros totales y al arco."""
    loc_attack = _best_split_value(
        features, "local", "shots", "for",
        ["current_home", "home", "current", "recent"],
        features.get("tiros_local_avg"),
    )
    vis_conceded = _best_split_value(
        features, "visitor", "shots", "against",
        ["current_away", "away", "current", "recent"],
    )
    vis_attack = _best_split_value(
        features, "visitor", "shots", "for",
        ["current_away", "away", "current", "recent"],
        features.get("tiros_visitor_avg"),
    )
    loc_conceded = _best_split_value(
        features, "local", "shots", "against",
        ["current_home", "home", "current", "recent"],
    )

    loc_sot_attack = _best_split_value(
        features, "local", "sot", "for",
        ["current_home", "home", "current", "recent"],
        features.get("tiros_arco_local_avg"),
    )
    vis_sot_conceded = _best_split_value(
        features, "visitor", "sot", "against",
        ["current_away", "away", "current", "recent"],
    )
    vis_sot_attack = _best_split_value(
        features, "visitor", "sot", "for",
        ["current_away", "away", "current", "recent"],
        features.get("tiros_arco_visitor_avg"),
    )
    loc_sot_conceded = _best_split_value(
        features, "local", "sot", "against",
        ["current_home", "home", "current", "recent"],
    )

    t_loc = _num(_blend_attack_defense(loc_attack, vis_conceded), 12.0)
    t_vis = _num(_blend_attack_defense(vis_attack, loc_conceded), 10.0)
    ta_loc = _num(_blend_attack_defense(loc_sot_attack, vis_sot_conceded), 4.0)
    ta_vis = _num(_blend_attack_defense(vis_sot_attack, loc_sot_conceded), 3.0)

    # Ajuste por posición y forma
    pos_adj = 0.0
    if features.get("home_position") and features.get("away_position"):
        diff = features["away_position"] - features["home_position"]
        pos_adj = diff * 0.15

    proj_loc = max(5, t_loc + pos_adj)
    proj_vis = max(4, t_vis - pos_adj)

    ta_loc_raw = max(1, ta_loc + pos_adj * 0.3)
    ta_vis_raw = max(1, ta_vis - pos_adj * 0.3)
    ta_loc_proj = _cap_sot_projection(ta_loc_raw, proj_loc, features, "local")
    ta_vis_proj = _cap_sot_projection(ta_vis_raw, proj_vis, features, "visitor")

    return {
        "tiros_local": round(proj_loc, 1),
        "tiros_visitor": round(proj_vis, 1),
        "tiros_total": round(proj_loc + proj_vis, 1),
        "tiros_arco_local": round(ta_loc_proj, 1),
        "tiros_arco_visitor": round(ta_vis_proj, 1),
    }


def project_corners(features: dict) -> Dict[str, float]:
    """Proyecta córners."""
    corners_loc = _blend_attack_defense(
        _best_split_value(
            features, "local", "corners", "for",
            ["current_home", "home", "current", "recent"],
            features.get("corners_local_avg"),
        ),
        _best_split_value(
            features, "visitor", "corners", "against",
            ["current_away", "away", "current", "recent"],
        ),
    )
    corners_vis = _blend_attack_defense(
        _best_split_value(
            features, "visitor", "corners", "for",
            ["current_away", "away", "current", "recent"],
            features.get("corners_visitor_avg"),
        ),
        _best_split_value(
            features, "local", "corners", "against",
            ["current_home", "home", "current", "recent"],
        ),
    )
    if corners_loc is not None and corners_vis is not None:
        base = corners_loc + corners_vis
    else:
        # Heurística: más tiros + laterales ofensivos = más corners
        t_total = (features.get("tiros_local_avg") or 12) + (features.get("tiros_visitor_avg") or 10)
        base = t_total * 0.45  # ~45% de tiros generan corners en promedio
    form_adj = (features.get("form_local_pts", 1.5) + features.get("form_visitor_pts", 1.5) - 3.0) * 0.5
    derby_adj = 1.0 if features.get("is_derby") else 0.0

    total = max(5, base + form_adj + derby_adj)
    if corners_loc is not None and corners_vis is not None and (corners_loc + corners_vis) > 0:
        loc_ratio = corners_loc / (corners_loc + corners_vis)
    else:
        loc_ratio = 0.55 if (features.get("home_position") or 10) < (features.get("away_position") or 10) else 0.5
    return {
        "corners_local": round(total * loc_ratio, 1),
        "corners_visitor": round(total * (1 - loc_ratio), 1),
        "corners_total": round(total, 1),
    }


def project_cards(features: dict) -> Dict[str, float]:
    """Proyecta tarjetas amarillas."""
    loc_yc_own = _best_split_value(
        features, "local", "yc", "for",
        ["current_home", "home", "current", "recent"],
        features.get("yc_local_avg"),
    )
    vis_yc_provoked = _best_split_value(
        features, "visitor", "yc", "against",
        ["current_away", "away", "current", "recent"],
    )
    vis_yc_own = _best_split_value(
        features, "visitor", "yc", "for",
        ["current_away", "away", "current", "recent"],
        features.get("yc_visitor_avg"),
    )
    loc_yc_provoked = _best_split_value(
        features, "local", "yc", "against",
        ["current_home", "home", "current", "recent"],
    )

    yc_loc = _num(_blend_attack_defense(loc_yc_own, vis_yc_provoked, attack_weight=0.75), 1.8)
    yc_vis = _num(_blend_attack_defense(vis_yc_own, loc_yc_provoked, attack_weight=0.75), 1.8)

    derby_adj = 1.2 if features.get("is_derby") else 0.0
    form_adj = abs(features.get("form_local_pts", 1.5) - features.get("form_visitor_pts", 1.5)) * 0.2
    travel_adj = 0.0001 * (features.get("travel_km") or 0)

    total = yc_loc + yc_vis + derby_adj + form_adj + travel_adj
    if features.get("is_friendly"):
        yc_loc *= 0.85
        yc_vis *= 0.85
        total *= 0.82
    min_total = 1.4 if features.get("is_friendly") else 2.0
    return {
        "yc_local": round(max(0, yc_loc + derby_adj / 2), 1),
        "yc_visitor": round(max(0, yc_vis + derby_adj / 2 + travel_adj), 1),
        "yc_total": round(max(min_total, total), 1),
    }


def project_fouls(features: dict) -> Dict[str, float]:
    """Proyecta faltas cometidas por equipo y totales."""
    loc_committed = _best_split_value(
        features, "local", "fouls", "for",
        ["current_home", "home", "current", "recent"],
        features.get("fouls_local_avg"),
    )
    vis_received = _best_split_value(
        features, "visitor", "fouls", "against",
        ["current_away", "away", "current", "recent"],
    )
    vis_committed = _best_split_value(
        features, "visitor", "fouls", "for",
        ["current_away", "away", "current", "recent"],
        features.get("fouls_visitor_avg"),
    )
    loc_received = _best_split_value(
        features, "local", "fouls", "against",
        ["current_home", "home", "current", "recent"],
    )

    f_loc = _num(_blend_attack_defense(loc_committed, vis_received, attack_weight=0.7), 12.0)
    f_vis = _num(_blend_attack_defense(vis_committed, loc_received, attack_weight=0.7), 12.0)
    derby_adj = 3.0 if features.get("is_derby") else 0.0
    f_loc_proj = max(4, f_loc + derby_adj / 2)
    f_vis_proj = max(4, f_vis + derby_adj / 2)
    if features.get("is_friendly"):
        f_loc_proj = max(4, f_loc_proj * 0.88)
        f_vis_proj = max(4, f_vis_proj * 0.88)
    return {
        "fouls_local": round(f_loc_proj, 1),
        "fouls_visitor": round(f_vis_proj, 1),
        "fouls": round(max(10, f_loc_proj + f_vis_proj), 1),
    }


def project_probabilities(features: dict, goal_proj: dict) -> Dict[str, float]:
    """
    Estima probabilidades para BTTS y Over 2.5 usando goles proyectados
    como input de una distribución de Poisson simplificada.
    """
    gl = goal_proj["goals_local"]
    gv = goal_proj["goals_visitor"]

    # P(BTTS) ≈ P(Local>0) * P(Visitante>0)
    p_loc_scores = 1 - math.exp(-gl)
    p_vis_scores = 1 - math.exp(-gv)
    btts = p_loc_scores * p_vis_scores

    # P(Over 2.5) ≈ 1 - P(0,1,2 goles). Aproximación por suma de lambdas
    total_lambda = gl + gv
    # P(0 goles)
    p0 = math.exp(-total_lambda)
    # P(1 gol)
    p1 = total_lambda * math.exp(-total_lambda)
    # P(2 goles)
    p2 = (total_lambda ** 2 / 2) * math.exp(-total_lambda)
    over25 = 1 - (p0 + p1 + p2)

    return {
        "btts_yes": round(min(0.95, max(0.05, btts)), 3),
        "over25_yes": round(min(0.95, max(0.05, over25)), 3),
    }


def run_full_projection(datos_resumidos: dict, prediccion_resumida: dict) -> Dict:
    """
    Pipeline completo: features → proyecciones cuantitativas.
    Retorna un dict listo para guardar en DB y para el prompt.
    """
    features = build_features(datos_resumidos, prediccion_resumida)
    goals = project_goals(features)
    tiros = project_tiros(features)
    corners = project_corners(features)
    cards = project_cards(features)
    fouls = project_fouls(features)
    probs = project_probabilities(features, goals)

    projection = {}
    projection.update(goals)
    projection.update(tiros)
    projection.update(corners)
    projection.update(cards)
    projection.update(fouls)
    projection.update(probs)
    return projection, features


if __name__ == "__main__":
    # Test básico
    test_data = {
        "forma_local": {"forma_string": "WWDLW", "xG_promedio": 1.8, "remates_promedio": 16, "remates_arco_promedio": 5.2, "amarillas_promedio": 1.5, "faltas_promedio": 11},
        "forma_visitante": {"forma_string": "LDWWD", "xG_promedio": 1.1, "remates_promedio": 10, "remates_arco_promedio": 3.0, "amarillas_promedio": 2.1, "faltas_promedio": 13},
        "_v2_detail": {"is_local_derby": True, "travel_distance_km": 450},
        "_sofascore": {},
        "home_team": "Boca Juniors",
        "away_team": "River Plate",
    }
    test_pred = {"xG_local": 1.6, "xG_visitante": 1.2}
    proj, feats = run_full_projection(test_data, test_pred)
    print("Proyección:", json.dumps(proj, indent=2, ensure_ascii=False))
    print("Features:", json.dumps(feats, indent=2, ensure_ascii=False))
