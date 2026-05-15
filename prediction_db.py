"""
prediction_db.py — Capa de persistencia SQLite para betting-ai.

Registra predicciones, resultados reales, stakes y métricas.
Permite calcular MAE, log-loss, ROI y calibración.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "output", "predictions.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Crea las tablas si no existen."""
    with _conn() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                match_name TEXT,
                league TEXT,
                match_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                -- Proyecciones cuantitativas
                proj_goals_local REAL,
                proj_goals_visitor REAL,
                proj_total_goals REAL,
                proj_tiros_local REAL,
                proj_tiros_visitor REAL,
                proj_tiros_total REAL,
                proj_tiros_arco_local REAL,
                proj_tiros_arco_visitor REAL,
                proj_corners_local REAL,
                proj_corners_visitor REAL,
                proj_corners_total REAL,
                proj_yc_local REAL,
                proj_yc_visitor REAL,
                proj_yc_total REAL,
                proj_fouls REAL,
                proj_btts_yes REAL,
                proj_over25_yes REAL,

                -- Proyecciones LLM (ajustes cualitativos)
                llm_proj_goals_local REAL,
                llm_proj_goals_visitor REAL,
                llm_proj_tiros_total REAL,
                llm_proj_corners_total REAL,
                llm_proj_yc_total REAL,
                llm_recommendation TEXT,
                llm_confidence TEXT,

                -- Cuotas Betsafe registradas (JSON)
                betsafe_odds_json TEXT,

                -- Predicción BSD CatBoost (JSON)
                bsd_pred_json TEXT,

                -- Features usadas (JSON para reproducibilidad)
                features_json TEXT,

                UNIQUE(match_id, created_at)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
                match_id TEXT,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,

                -- Resultados reales del partido
                score_local INTEGER,
                score_visitor INTEGER,
                tiros_local INTEGER,
                tiros_visitor INTEGER,
                tiros_arco_local INTEGER,
                tiros_arco_visitor INTEGER,
                corners_local INTEGER,
                corners_visitor INTEGER,
                yc_local INTEGER,
                yc_visitor INTEGER,
                rc_local INTEGER,
                rc_visitor INTEGER,
                fouls_local INTEGER,
                fouls_visitor INTEGER,
                btts INTEGER,  -- 0 o 1
                over25 INTEGER,  -- 0 o 1

                -- Performance de mercado 1X2 (si aplica)
                result_1x2 TEXT,  -- 'local', 'draw', 'visitor'

                UNIQUE(match_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
                match_id TEXT,
                placed_at TEXT DEFAULT CURRENT_TIMESTAMP,

                mercado TEXT,
                seleccion TEXT,
                cuota REAL,
                stake REAL,
                unidades REAL,
                stake_pct REAL,  -- % de bankroll

                resultado TEXT DEFAULT 'pending',
                profit REAL,

                -- Kelly params
                kelly_edge REAL,
                kelly_fraction REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
                match_id TEXT,
                metric TEXT,
                predicted REAL,
                actual REAL,
                abs_error REAL,
                squared_error REAL,
                pct_error REAL,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        print(f"  [DB] Base de datos inicializada: {DB_PATH}")


def save_prediction(
    match_id: str,
    match_name: str,
    league: str,
    match_date: str,
    quant_projections: dict,
    llm_projections: dict,
    betsafe_odds: dict,
    bsd_pred: dict,
    features: dict,
) -> int:
    """Guarda una predicción. Retorna el ID insertado."""
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                match_id, match_name, league, match_date,
                proj_goals_local, proj_goals_visitor, proj_total_goals,
                proj_tiros_local, proj_tiros_visitor, proj_tiros_total,
                proj_tiros_arco_local, proj_tiros_arco_visitor,
                proj_corners_local, proj_corners_visitor, proj_corners_total,
                proj_yc_local, proj_yc_visitor, proj_yc_total,
                proj_fouls, proj_btts_yes, proj_over25_yes,
                llm_proj_goals_local, llm_proj_goals_visitor,
                llm_proj_tiros_total, llm_proj_corners_total, llm_proj_yc_total,
                llm_recommendation, llm_confidence,
                betsafe_odds_json, bsd_pred_json, features_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                match_id, match_name, league, match_date,
                quant_projections.get("goals_local"),
                quant_projections.get("goals_visitor"),
                quant_projections.get("total_goals"),
                quant_projections.get("tiros_local"),
                quant_projections.get("tiros_visitor"),
                quant_projections.get("tiros_total"),
                quant_projections.get("tiros_arco_local"),
                quant_projections.get("tiros_arco_visitor"),
                quant_projections.get("corners_local"),
                quant_projections.get("corners_visitor"),
                quant_projections.get("corners_total"),
                quant_projections.get("yc_local"),
                quant_projections.get("yc_visitor"),
                quant_projections.get("yc_total"),
                quant_projections.get("fouls"),
                quant_projections.get("btts_yes"),
                quant_projections.get("over25_yes"),
                llm_projections.get("goals_local"),
                llm_projections.get("goals_visitor"),
                llm_projections.get("tiros_total"),
                llm_projections.get("corners_total"),
                llm_projections.get("yc_total"),
                llm_projections.get("recommendation"),
                llm_projections.get("confidence"),
                json.dumps(betsafe_odds, ensure_ascii=False) if betsafe_odds else None,
                json.dumps(bsd_pred, ensure_ascii=False) if bsd_pred else None,
                json.dumps(features, ensure_ascii=False) if features else None,
            ),
        )
        return cursor.lastrowid


def save_result(match_id: str, result_data: dict) -> int:
    """
    Guarda el resultado real de un partido.
    Retorna -1 si no se encontró prediction_id correspondiente.
    """
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM predictions WHERE match_id = ? ORDER BY created_at DESC LIMIT 1",
            (match_id,),
        )
        row = cursor.fetchone()
        if not row:
            return -1
        pred_id = row["id"]

        cursor.execute(
            """
            INSERT INTO results (
                prediction_id, match_id,
                score_local, score_visitor,
                tiros_local, tiros_visitor,
                tiros_arco_local, tiros_arco_visitor,
                corners_local, corners_visitor,
                yc_local, yc_visitor, rc_local, rc_visitor,
                fouls_local, fouls_visitor,
                btts, over25, result_1x2
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_id) DO UPDATE SET
                score_local=excluded.score_local,
                score_visitor=excluded.score_visitor,
                tiros_local=excluded.tiros_local,
                tiros_visitor=excluded.tiros_visitor,
                corners_local=excluded.corners_local,
                corners_visitor=excluded.corners_visitor,
                yc_local=excluded.yc_local,
                yc_visitor=excluded.yc_visitor,
                btts=excluded.btts,
                over25=excluded.over25,
                result_1x2=excluded.result_1x2,
                recorded_at=CURRENT_TIMESTAMP
            """,
            (
                pred_id, match_id,
                result_data.get("score_local"),
                result_data.get("score_visitor"),
                result_data.get("tiros_local"),
                result_data.get("tiros_visitor"),
                result_data.get("tiros_arco_local"),
                result_data.get("tiros_arco_visitor"),
                result_data.get("corners_local"),
                result_data.get("corners_visitor"),
                result_data.get("yc_local"),
                result_data.get("yc_visitor"),
                result_data.get("rc_local"),
                result_data.get("rc_visitor"),
                result_data.get("fouls_local"),
                result_data.get("fouls_visitor"),
                1 if result_data.get("btts") else 0,
                1 if result_data.get("over25") else 0,
                result_data.get("result_1x2"),
            ),
        )

        # Calcular errores y guardarlos
        _record_errors(conn, pred_id, match_id, result_data)

        return cursor.lastrowid


def _record_errors(conn, pred_id: int, match_id: str, actual: dict):
    """Compara predicción con resultado real y guarda errores."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (pred_id,),
    )
    pred_row = cursor.fetchone()
    if not pred_row:
        return

    # Combinar cuantitativo + LLM (si existe LLM, pesa más)
    def blended(field_quant, field_llm):
        q = pred_row[field_quant]
        l = pred_row[field_llm]
        if q is not None and l is not None:
            return 0.5 * q + 0.5 * l
        return q if q is not None else l

    pairs = [
        ("goals_local", blended("proj_goals_local", "llm_proj_goals_local"), actual.get("score_local")),
        ("goals_visitor", blended("proj_goals_visitor", "llm_proj_goals_visitor"), actual.get("score_visitor")),
        ("tiros_total", blended("proj_tiros_total", "llm_proj_tiros_total"), (actual.get("tiros_local") or 0) + (actual.get("tiros_visitor") or 0)),
        ("corners_total", blended("proj_corners_total", "llm_proj_corners_total"), (actual.get("corners_local") or 0) + (actual.get("corners_visitor") or 0)),
        ("yc_total", blended("proj_yc_total", "llm_proj_yc_total"), (actual.get("yc_local") or 0) + (actual.get("yc_visitor") or 0)),
    ]

    for metric, predicted, real in pairs:
        if predicted is None or real is None:
            continue
        err = abs(predicted - real)
        pct_err = err / real if real != 0 else None
        cursor.execute(
            """
            INSERT INTO model_errors (prediction_id, match_id, metric, predicted, actual, abs_error, squared_error, pct_error)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT DO NOTHING
            """,
            (pred_id, match_id, metric, round(predicted, 2), real, round(err, 2), round(err ** 2, 2), round(pct_err, 4) if pct_err else None),
        )


def save_bet(prediction_id: int, match_id: str, mercado: str, seleccion: str,
             cuota: float, stake: float, unidades: float, stake_pct: float,
             kelly_edge: float, kelly_fraction: float) -> int:
    """Registra una apuesta."""
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO bets (prediction_id, match_id, mercado, seleccion, cuota, stake, unidades, stake_pct, kelly_edge, kelly_fraction)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (prediction_id, match_id, mercado, seleccion, cuota, stake, unidades, stake_pct, kelly_edge, kelly_fraction),
        )
        return cursor.lastrowid


def update_bet_result(bet_id: int, resultado: str, profit: float):
    """Actualiza el resultado de una apuesta."""
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE bets SET resultado=?, profit=? WHERE id=?",
            (resultado, profit, bet_id),
        )


def get_match_history(match_id: str) -> Optional[Dict]:
    """Obtiene predicción + resultado + bets para un match_id."""
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.*, r.* FROM predictions p
            LEFT JOIN results r ON r.match_id = p.match_id
            WHERE p.match_id = ?
            ORDER BY p.created_at DESC
            LIMIT 1
            """,
            (match_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def get_recent_predictions(limit: int = 50, days: int = 30) -> List[Dict]:
    """Lista predicciones recientes con resultados si existen."""
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.*, r.score_local, r.score_visitor, r.btts, r.over25
            FROM predictions p
            LEFT JOIN results r ON r.prediction_id = p.id
            WHERE p.created_at >= datetime('now', '-{} days')
            ORDER BY p.created_at DESC
            LIMIT {}
            """.format(days, limit)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_mae_by_metric(metric: str = None, days: int = 90) -> List[Dict]:
    """
    Calcula MAE (Mean Absolute Error) por métrica.
    """
    with _conn() as conn:
        cursor = conn.cursor()
        if metric:
            cursor.execute(
                """
                SELECT metric, AVG(abs_error) as mae, COUNT(*) as n
                FROM model_errors
                WHERE recorded_at >= datetime('now', '-{} days')
                AND metric = ?
                """.format(days),
                (metric,),
            )
        else:
            cursor.execute(
                """
                SELECT metric, AVG(abs_error) as mae, COUNT(*) as n
                FROM model_errors
                WHERE recorded_at >= datetime('now', '-{} days')
                GROUP BY metric
                ORDER BY mae ASC
                """.format(days)
            )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_roi(days: int = 90) -> List[Dict]:
    """ROI por mercado y global."""
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT mercado, SUM(profit) as total_profit, SUM(stake) as total_staked,
                   COUNT(*) as total_bets,
                   (SUM(profit) / NULLIF(SUM(stake), 0)) * 100 as roi_pct
            FROM bets
            WHERE placed_at >= datetime('now', '-{} days')
            AND resultado != 'pending'
            GROUP BY mercado
            """.format(days)
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_calibration(days: int = 90) -> List[Dict]:
    """
    Evalúa calibración de probabilidades para BTTS y Over 2.5.
    Compara predicción vs frecuencia real de ocurrencia.
    """
    with _conn() as conn:
        cursor = conn.cursor()
        output = []
        for metric, col_pred, col_res in [
            ("btts", "p.proj_btts_yes", "r.btts"),
            ("over25", "p.proj_over25_yes", "r.over25"),
        ]:
            cursor.execute(
                f"""
                SELECT {col_pred} as prob, {col_res} as result
                FROM predictions p
                JOIN results r ON r.prediction_id = p.id
                WHERE p.created_at >= datetime('now', '-{days} days')
                AND {col_pred} IS NOT NULL
                AND {col_res} IS NOT NULL
                """
            )
            rows = cursor.fetchall()
            if not rows:
                continue
            # Agrupar en bins de probabilidad
            bins = {}
            for r in rows:
                p = r["prob"]
                if p is None:
                    continue
                bin_key = round(p, 1)
                if bin_key not in bins:
                    bins[bin_key] = {"pred": 0, "actual": 0, "n": 0}
                bins[bin_key]["pred"] += p
                bins[bin_key]["actual"] += r["result"]
                bins[bin_key]["n"] += 1

            for bk, bv in bins.items():
                if bv["n"] < 5:
                    continue
                output.append({
                    "metric": metric,
                    "bin": bk,
                    "avg_prediction": round(bv["pred"] / bv["n"], 3),
                    "actual_rate": round(bv["actual"] / bv["n"], 3),
                    "n": bv["n"],
                    "calibration_error": abs(round(bv["pred"] / bv["n"], 3) - round(bv["actual"] / bv["n"], 3)),
                })
        return output


def get_stats_summary(days: int = 30) -> Dict:
    """Resumen rápido de métricas del sistema."""
    with _conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as total_preds FROM predictions WHERE created_at >= datetime('now', '-{} days')".format(days)
        )
        total_preds = cursor.fetchone()["total_preds"]

        cursor.execute(
            "SELECT COUNT(*) as total_results FROM results WHERE recorded_at >= datetime('now', '-{} days')".format(days)
        )
        total_results = cursor.fetchone()["total_results"]

        cursor.execute(
            """
            SELECT COUNT(*) as total_bets, SUM(profit) as total_profit, SUM(stake) as total_staked
            FROM bets WHERE placed_at >= datetime('now', '-{} days') AND resultado != 'pending'
            """.format(days)
        )
        bets_row = cursor.fetchone()

        roi = None
        if bets_row and bets_row["total_staked"]:
            roi = round((bets_row["total_profit"] or 0) / bets_row["total_staked"] * 100, 2)

        mae_rows = get_mae_by_metric(days=days)

        return {
            "days": days,
            "predictions": total_preds,
            "results": total_results,
            "bets": bets_row["total_bets"] if bets_row else 0,
            "profit": round(bets_row["total_profit"] or 0, 2) if bets_row else 0,
            "roi_pct": roi,
            "mae_by_metric": mae_rows,
        }


if __name__ == "__main__":
    init_db()
    print(get_stats_summary())
