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
        if isinstance(value, (int, float)):
            return value
    return None


def _num(value, default: float) -> float:
    """Devuelve value si es numérico; si no, default."""
    return float(value) if isinstance(value, (int, float)) else default


def _season_per_match(stats: dict, key: str) -> Optional[float]:
    value = stats.get(key)
    matches = stats.get("matches_played")
    if not isinstance(value, (int, float)) or not isinstance(matches, (int, float)) or matches <= 0:
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
        if isinstance(value, (int, float)):
            values.append(value)
    return _weighted_avg(values[:5]) if values else None


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
        if features.get(target) is None and isinstance(value, (int, float)):
            features[target] = value


def build_features(datos_resumidos: dict, prediccion_resumida: dict) -> dict:
    """
    Extrae y normaliza features de los datos existentes para
    alimentar al modelo cuantitativo.
    """
    features = {}

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
    lineups = ss.get("alineaciones", {})
    features["lineup_confirmed"] = 1 if lineups.get("local", {}).get("confirmada") else 0

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

    # --- Cuotas Betsafe ---
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
    t_loc = _num(features.get("tiros_local_avg"), 12.0)
    t_vis = _num(features.get("tiros_visitor_avg"), 10.0)
    ta_loc = _num(features.get("tiros_arco_local_avg"), 4.0)
    ta_vis = _num(features.get("tiros_arco_visitor_avg"), 3.0)

    # Ajuste por posición y forma
    pos_adj = 0.0
    if features.get("home_position") and features.get("away_position"):
        diff = features["away_position"] - features["home_position"]
        pos_adj = diff * 0.15

    proj_loc = max(5, t_loc + pos_adj)
    proj_vis = max(4, t_vis - pos_adj)

    return {
        "tiros_local": round(proj_loc, 1),
        "tiros_visitor": round(proj_vis, 1),
        "tiros_total": round(proj_loc + proj_vis, 1),
        "tiros_arco_local": round(max(1, ta_loc + pos_adj * 0.3), 1),
        "tiros_arco_visitor": round(max(1, ta_vis - pos_adj * 0.3), 1),
    }


def project_corners(features: dict) -> Dict[str, float]:
    """Proyecta córners."""
    corners_loc = features.get("corners_local_avg")
    corners_vis = features.get("corners_visitor_avg")
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
    yc_loc = _num(features.get("yc_local_avg"), 1.8)
    yc_vis = _num(features.get("yc_visitor_avg"), 1.8)

    derby_adj = 1.2 if features.get("is_derby") else 0.0
    form_adj = abs(features.get("form_local_pts", 1.5) - features.get("form_visitor_pts", 1.5)) * 0.2
    travel_adj = 0.0001 * (features.get("travel_km") or 0)

    total = yc_loc + yc_vis + derby_adj + form_adj + travel_adj
    return {
        "yc_local": round(max(0, yc_loc + derby_adj / 2), 1),
        "yc_visitor": round(max(0, yc_vis + derby_adj / 2 + travel_adj), 1),
        "yc_total": round(max(2, total), 1),
    }


def project_fouls(features: dict) -> Dict[str, float]:
    """Proyecta faltas totales."""
    f_loc = _num(features.get("fouls_local_avg"), 12.0)
    f_vis = _num(features.get("fouls_visitor_avg"), 12.0)
    derby_adj = 3.0 if features.get("is_derby") else 0.0
    return {
        "fouls": round(max(10, f_loc + f_vis + derby_adj), 1),
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
