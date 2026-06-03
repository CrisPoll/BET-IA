import json
import unittest
from pathlib import Path

from betano_client import (
    _extraer_info_url_betano,
    _formatear_cuotas_betano_para_prompt,
    _normalizar_betano_payloads,
)


FIXTURE = Path(__file__).parent / "fixtures" / "betano_event_sample.json"


class BetanoClientTest(unittest.TestCase):
    def test_requires_public_betano_url(self):
        info = _extraer_info_url_betano("85915325")
        self.assertIn("error", info)

        info = _extraer_info_url_betano(
            "https://www.betano.pe/cuotas-de-partido/palmeiras-sp-chapecoense-sc/85915325/"
        )
        self.assertEqual(info["event_id"], "85915325")
        self.assertEqual(info["path"], "/cuotas-de-partido/palmeiras-sp-chapecoense-sc/85915325/")

    def test_normalizes_core_markets(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        result = _normalizar_betano_payloads([payload], "https://www.betano.pe/cuotas-de-partido/sample/85915325/")

        self.assertEqual(result["source"], "betano")
        self.assertEqual(result["bookmaker"], "Betano")
        self.assertEqual(result["local"], 1.36)
        self.assertEqual(result["empate"], 4.9)
        self.assertEqual(result["visitante"], 9.0)
        self.assertEqual(result["over_25"], 1.67)
        self.assertEqual(result["under_25"], 2.2)
        self.assertEqual(result["btts_si"], 1.98)
        self.assertEqual(result["btts_no"], 1.78)

        names = {market["name"] for market in result["markets"].values()}
        self.assertIn("Goles totales Más/Menos (2.5)", names)
        self.assertIn("Más/Menos Córners (10.5)", names)
        self.assertIn("Tarjetas Totales Más/Menos (4.5)", names)
        self.assertIn("Remates totales (27.5)", names)
        self.assertIn("Tiros al Arco (9.5)", names)
        self.assertIn("Total de Faltas Cometidas (27.5)", names)
        self.assertIn("Alan Franco Remates totales", names)
        self.assertNotIn("Faltas cometidas [Alan Franco] (1.5)", names)

        player_market = next(m for m in result["markets"].values() if m["name"] == "Alan Franco Remates totales")
        self.assertEqual(player_market["category"], "Player Shots")
        self.assertEqual(player_market["player_name"], "Alan Franco")
        self.assertEqual(player_market["stat_type"], "player_shots")

    def test_prompt_formatter_mentions_betano(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        result = _normalizar_betano_payloads([payload], "https://www.betano.pe/cuotas-de-partido/sample/85915325/")
        text = _formatear_cuotas_betano_para_prompt(result)

        self.assertIn("CUOTAS BETANO", text)
        self.assertIn("Ambos equipos anotan", text)
        self.assertIn("Remates totales (27.5)", text)
        self.assertIn("REMATES JUGADORES", text)
        self.assertIn("Alan Franco - remates", text)
        self.assertNotIn("Faltas cometidas [Alan Franco]", text)
        self.assertNotIn("BETSAFE", text)

    def test_normalizes_player_shots_table_layout(self):
        payload = {
            "data": {
                "event": {
                    "id": "table-event",
                    "name": "Local FC - Away FC",
                    "participants": [{"name": "Local FC"}, {"name": "Away FC"}],
                    "markets": [
                        {
                            "id": "table-player-shots",
                            "name": "Remates totales",
                            "tableLayout": True,
                            "rows": [
                                {
                                    "name": "Alan Franco",
                                    "selections": [
                                        {"id": "ps1", "name": "1+", "price": 2.12, "handicap": 0.0},
                                        {"id": "ps2", "name": "2+", "price": 6.8, "handicap": 0.0},
                                    ],
                                },
                                {
                                    "name": "Bruno Henrique",
                                    "selections": [
                                        {"id": "ps3", "name": "1+", "price": 1.62, "handicap": 0.0},
                                    ],
                                },
                            ],
                        },
                        {
                            "id": "table-player-fouls",
                            "name": "Faltas cometidas",
                            "tableLayout": True,
                            "rows": [
                                {
                                    "name": "Alan Franco",
                                    "selections": [{"id": "pf1", "name": "Más 1.5", "price": 3.32, "handicap": 1.5}],
                                }
                            ],
                        },
                    ],
                }
            }
        }

        result = _normalizar_betano_payloads([payload], "https://www.betano.pe/cuotas-de-partido/sample/table-event/")
        names = {market["name"] for market in result["markets"].values()}
        self.assertIn("Alan Franco Remates totales", names)
        self.assertIn("Bruno Henrique Remates totales", names)
        self.assertNotIn("Alan Franco Faltas cometidas", names)

        market = next(m for m in result["markets"].values() if m["name"] == "Alan Franco Remates totales")
        self.assertEqual(market["category"], "Player Shots")
        self.assertEqual(market["player_name"], "Alan Franco")
        self.assertEqual(market["stat_type"], "player_shots")

        text = _formatear_cuotas_betano_para_prompt(result)
        self.assertIn("Alan Franco - remates: 1+: @2.12 | 2+: @6.8", text)
        self.assertIn("Bruno Henrique - remates: 1+: @1.62", text)
        self.assertNotIn("Faltas cometidas", text)

    def test_normalizes_player_shots_on_target_market(self):
        payload = {
            "data": {
                "event": {
                    "id": "sot-event",
                    "name": "Local FC - Away FC",
                    "participants": [{"name": "Local FC"}, {"name": "Away FC"}],
                    "markets": [
                        {
                            "id": "player-sot",
                            "name": "Tiros al Arco [Alan Franco]",
                            "type": "PSOT",
                            "typeId": 596,
                            "selections": [
                                {"id": "sot1", "name": "1+", "price": 4.5, "handicap": 0.0},
                            ],
                        }
                    ],
                }
            }
        }

        result = _normalizar_betano_payloads([payload], "https://www.betano.pe/cuotas-de-partido/sample/sot-event/")
        market = next(iter(result["markets"].values()))

        self.assertEqual(market["category"], "Player Shots On Target")
        self.assertEqual(market["player_name"], "Alan Franco")
        self.assertEqual(market["stat_type"], "player_shots_on_target")
        self.assertIn("Alan Franco - tiros al arco: 1+: @4.5", _formatear_cuotas_betano_para_prompt(result))

    def test_prompt_formatter_keeps_only_main_betting_markets(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        result = _normalizar_betano_payloads([payload], "https://www.betano.pe/cuotas-de-partido/sample/85915325/")
        result["markets"].update({
            "team-goals": {
                "name": "Palmeiras SP - Goles totales Más/Menos (1.5)",
                "category": "Over/Under",
                "line": 1.5,
                "selections": [{"label": "Más 1.5", "odd": 2.1}, {"label": "Menos 1.5", "odd": 1.7}],
            },
            "btts-combo": {
                "name": "Doble oportunidad / Ambos equipos anotan",
                "category": "BTTS",
                "selections": [{"label": "1X y Sí", "odd": 2.5}],
            },
            "double-chance": {
                "name": "Doble oportunidad",
                "category": "Double Chance",
                "selections": [{"label": "1X", "odd": 1.09}],
            },
            "asian-handicap": {
                "name": "Hándicap Asiático",
                "category": "Asian Handicap",
                "selections": [{"label": "Palmeiras -1.5", "odd": 1.9}],
            },
        })

        text = _formatear_cuotas_betano_para_prompt(result)

        self.assertNotIn("Palmeiras SP - Goles totales", text)
        self.assertNotIn("Doble oportunidad", text)
        self.assertNotIn("HANDICAP", text.upper())
        self.assertNotIn("Hándicap", text)


if __name__ == "__main__":
    unittest.main()
