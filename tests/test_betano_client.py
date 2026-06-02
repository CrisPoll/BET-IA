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
        self.assertNotIn("Alan Franco Remates totales", names)
        self.assertNotIn("Faltas cometidas [Alan Franco] (1.5)", names)

    def test_prompt_formatter_mentions_betano(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        result = _normalizar_betano_payloads([payload], "https://www.betano.pe/cuotas-de-partido/sample/85915325/")
        text = _formatear_cuotas_betano_para_prompt(result)

        self.assertIn("CUOTAS BETANO", text)
        self.assertIn("Ambos equipos anotan", text)
        self.assertIn("Remates totales (27.5)", text)
        self.assertNotIn("Alan Franco", text)
        self.assertNotIn("BETSAFE", text)

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
