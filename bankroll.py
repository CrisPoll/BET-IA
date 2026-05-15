"""
bankroll.py — Gestión de bankroll: Kelly Criterion, calibración de probabilidades,
y registro de stakes por apuesta.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Parámetros configurables (podrían ir en .env)
DEFAULT_BANKROLL = 1000.0  # unidades
DEFAULT_KELLY_FRACTION = 0.25  # Half-Kelly conservador por defecto
MAX_STAKE_PCT = 0.05  # Nunca más del 5% del bankroll en una apuesta
MIN_EDGE_BPS = 0.02  # Mínimo edge de 2% para apostar


@dataclass
class BetRecommendation:
    mercado: str
    seleccion: str
    cuota: float
    prob_estimada: float
    stake: float
    unidades: float
    stake_pct: float
    kelly_edge: float
    kelly_fraction: float
    recomendado: bool
    razon_no_recomendado: str = ""


def calculate_kelly_stake(
    cuota: float,
    prob_estimada: float,
    bankroll: float = DEFAULT_BANKROLL,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> Tuple[float, float, float]:
    """
    Calcula el stake óptimo según Kelly Criterion fraccional.

    Returns:
        (stake en unidades, % del bankroll, edge)
    """
    if cuota <= 1.0:
        return 0.0, 0.0, -1.0

    # Edge implícito de la cuota
    prob_mercado = 1.0 / cuota
    edge = prob_estimada - prob_mercado

    if edge <= 0:
        return 0.0, 0.0, edge

    # Kelly full
    b = cuota - 1.0  # odds decimal - 1
    kelly_full = (b * prob_estimada - (1 - prob_estimada)) / b

    if kelly_full <= 0:
        return 0.0, 0.0, edge

    # Kelly fraccional
    stake_pct = kelly_full * kelly_fraction
    stake_pct = min(stake_pct, MAX_STAKE_PCT)
    stake = stake_pct * bankroll

    return round(stake, 2), round(stake_pct * 100, 2), round(edge, 4)


def calibrate_probability(
    raw_prob: float,
    metric: str = "btts",
    calibration_map: Optional[Dict] = None,
) -> float:
    """
    Ajusta una probabilidad cruda usando calibración histórica.
    Si no hay datos de calibración, aplica shrinkage bayesiano simple.
    """
    raw_prob = max(0.05, min(0.95, raw_prob))

    if calibration_map:
        # Buscar el bin más cercano y ajustar
        target_bin = round(raw_prob, 1)
        calibrated = calibration_map.get(metric, {}).get(target_bin)
        if calibrated is not None:
            # Mezclar cruda con calibrada (regularización)
            return round(0.3 * raw_prob + 0.7 * calibrated, 3)

    # Shrinkage bayesiano por defecto: regresar hacia 0.5 para probabilidades extremas
    # Esto reduce el overconfidence
    if raw_prob > 0.7:
        return round(raw_prob - (raw_prob - 0.7) * 0.15, 3)
    elif raw_prob < 0.3:
        return round(raw_prob + (0.3 - raw_prob) * 0.15, 3)
    return round(raw_prob, 3)


def evaluate_stat_market(
    mercado: str,
    linea: float,
    cuota_over: float,
    cuota_under: float,
    proj_total: float,
    proj_std: float = 2.0,  # desviación estándar estimada
    bankroll: float = DEFAULT_BANKROLL,
) -> List[BetRecommendation]:
    """
    Evalúa un mercado estadístico (Over/Under) usando distribución normal
    como aproximación para variables continuas (tiros, corners, tarjetas).
    """
    recs = []

    if proj_total is None or linea is None:
        return recs

    # P(Over) ≈ 1 - CDF(linea) con media=proj_total, std=proj_std
    z = (linea - proj_total) / max(0.5, proj_std)
    p_over = 1 - _approx_cdf(z)
    p_under = 1 - p_over

    # Calibrar
    p_over = calibrate_probability(p_over, metric=mercado)
    p_under = calibrate_probability(p_under, metric=mercado)

    for seleccion, cuota, prob in [("Over", cuota_over, p_over), ("Under", cuota_under, p_under)]:
        if cuota is None or cuota <= 1.0:
            continue
        stake, stake_pct, edge = calculate_kelly_stake(cuota, prob, bankroll)
        recomendado = edge >= MIN_EDGE_BPS
        razon = ""
        if not recomendado:
            razon = f"Edge {edge:.1%} < mínimo {MIN_EDGE_BPS:.1%}"
        elif stake_pct < 0.1:
            recomendado = False
            razon = f"Stake muy pequeño ({stake_pct:.2f}%)"

        recs.append(BetRecommendation(
            mercado=mercado,
            seleccion=seleccion,
            cuota=cuota,
            prob_estimada=prob,
            stake=stake,
            unidades=stake / (bankroll / 10),  # 1 unidad = 10% del bankroll
            stake_pct=stake_pct,
            kelly_edge=edge,
            kelly_fraction=DEFAULT_KELLY_FRACTION,
            recomendado=recomendado,
            razon_no_recomendado=razon,
        ))

    return recs


def evaluate_1x2(
    cuota_local: float,
    cuota_empate: float,
    cuota_visitante: float,
    prob_local: float,
    prob_draw: float,
    prob_visitor: float,
    bankroll: float = DEFAULT_BANKROLL,
) -> List[BetRecommendation]:
    """Evalúa mercado 1X2 básico."""
    recs = []
    probs = [("Local", cuota_local, prob_local), ("Empate", cuota_empate, prob_draw), ("Visitante", cuota_visitante, prob_visitor)]

    for seleccion, cuota, prob in probs:
        if cuota is None or prob is None:
            continue
        prob = calibrate_probability(prob, metric="1x2")
        stake, stake_pct, edge = calculate_kelly_stake(cuota, prob, bankroll)
        recomendado = edge >= MIN_EDGE_BPS
        razon = f"Edge {edge:.1%} < mínimo" if not recomendado else ""
        if recomendado and stake_pct < 0.1:
            recomendado = False
            razon = "Stake muy pequeño"

        recs.append(BetRecommendation(
            mercado="1X2",
            seleccion=seleccion,
            cuota=cuota,
            prob_estimada=prob,
            stake=stake,
            unidades=stake / (bankroll / 10),
            stake_pct=stake_pct,
            kelly_edge=edge,
            kelly_fraction=DEFAULT_KELLY_FRACTION,
            recomendado=recomendado,
            razon_no_recomendado=razon,
        ))

    return recs


def _approx_cdf(z: float) -> float:
    """Aproximación de la función de distribución acumulada normal (Abramowitz & Stegun)."""
    # Error function approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if z >= 0 else -1
    z = abs(z) / math.sqrt(2.0)

    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-z * z)

    return 0.5 * (1.0 + sign * y)


def format_recommendations(recs: List[BetRecommendation]) -> str:
    """Formatea recomendaciones para mostrar al usuario."""
    lines = []
    for r in recs:
        if not r.recomendado:
            continue
        lines.append(
            f"  • {r.mercado} | {r.seleccion} @ {r.cuota:.2f} | "
            f"Prob: {r.prob_estimada:.1%} | Edge: {r.kelly_edge:.1%} | "
            f"Stake: {r.stake:.2f}u ({r.stake_pct:.2f}%)"
        )
    if not lines:
        return "  (Ninguna apuesta con edge suficiente)"
    return "\n".join(lines)


if __name__ == "__main__":
    # Test
    recs = evaluate_stat_market("Total Tiros", 20.5, 1.85, 1.95, proj_total=22.0, proj_std=3.5)
    print("Evaluación Tiros:")
    for r in recs:
        print(f"  {r.seleccion}: stake={r.stake}u, edge={r.kelly_edge:.2%}, recomendado={r.recomendado}")

    recs_1x2 = evaluate_1x2(2.10, 3.40, 3.60, 0.48, 0.28, 0.24)
    print("\nEvaluación 1X2:")
    for r in recs_1x2:
        print(f"  {r.seleccion}: stake={r.stake}u, edge={r.kelly_edge:.2%}")
