import unittest

from bsd_client_v2 import _select_lineup_players_for_profile, resumir_player_avgs_v2_para_prompt
from sofascore_client import _formatear_player_avgs_sofascore_para_prompt


class PlayerAverageFormatterTest(unittest.TestCase):
    def test_formats_sofascore_player_averages(self):
        datos = {
            "partido": "Local FC vs Away FC",
            "_sofascore": {
                "disponible": True,
                "alineaciones": {
                    "local": {
                        "confirmada": True,
                        "titulares": [
                            {
                                "nombre": "Forward One",
                                "posicion": "F",
                                "impacto_jugador": {
                                    "stats_resumen": {
                                        "apps": 12,
                                        "minutes": 900,
                                        "avg_rating": 7.12,
                                        "goals": 5,
                                        "assists": 2,
                                        "shots_p90": 2.5,
                                        "shots_on_target_p90": 1.1,
                                        "xg_p90": 0.42,
                                        "key_passes_p90": 1.3,
                                    }
                                },
                            }
                        ],
                        "bajas": {
                            "confirmadas": [
                                {
                                    "nombre": "Missing Creator",
                                    "posicion": "M",
                                    "impacto_baja": {
                                        "stats_resumen": {
                                            "apps": 10,
                                            "minutes": 740,
                                            "xa_p90": 0.21,
                                            "key_passes_p90": 2.0,
                                        }
                                    },
                                }
                            ]
                        },
                    },
                    "visitante": {"confirmada": False, "titulares": []},
                },
            },
        }

        text = _formatear_player_avgs_sofascore_para_prompt(datos)

        self.assertIn("MEDIAS POR JUGADOR", text)
        self.assertIn("Forward One", text)
        self.assertIn("tiros 2.50/90", text)
        self.assertIn("SOT 1.10/90", text)
        self.assertIn("xG 0.42/90", text)
        self.assertIn("Missing Creator", text)
        self.assertIn("key 2/90", text)

    def test_formats_sofascore_missing_player_averages_without_starter_rows(self):
        datos = {
            "partido": "Local FC vs Away FC",
            "_sofascore": {
                "disponible": True,
                "alineaciones": {
                    "local": {
                        "confirmada": False,
                        "titulares": [],
                        "bajas": {
                            "confirmadas": [
                                {
                                    "nombre": "Missing Shooter",
                                    "posicion": "F",
                                    "impacto_baja": {
                                        "stats_resumen": {
                                            "apps": 9,
                                            "minutes": 620,
                                            "shots_p90": 2.84,
                                            "shots_on_target_p90": 1.16,
                                        }
                                    },
                                }
                            ]
                        },
                    },
                    "visitante": {"confirmada": False, "titulares": []},
                },
            },
        }

        text = _formatear_player_avgs_sofascore_para_prompt(datos)

        self.assertIn("MEDIAS POR JUGADOR", text)
        self.assertIn("Missing Shooter", text)
        self.assertIn("[baja]", text)
        self.assertIn("tiros 2.84/90", text)

    def test_formats_bsd_v2_player_averages(self):
        datos = {
            "partido": "Local FC vs Away FC",
            "_bsd_v2": {
                "player_impact": {
                    "season_id": 123,
                    "players": [
                        {
                            "name": "BSD Shooter",
                            "position": "F",
                            "side": "home",
                            "starter": True,
                            "stats": {
                                "apps": 8,
                                "minutes": 650,
                                "avg_rating": 6.91,
                                "goals": 3,
                                "assists": 1,
                                "shots_p90": 2.21,
                                "shots_on_target_p90": 0.83,
                                "xg_p90": 0.31,
                            },
                        }
                    ],
                }
            },
        }

        text = resumir_player_avgs_v2_para_prompt(datos)

        self.assertIn("MEDIAS POR JUGADOR", text)
        self.assertIn("BSD Shooter", text)
        self.assertIn("tiros 2.21/90", text)
        self.assertIn("SOT 0.83/90", text)
        self.assertIn("xG 0.31/90", text)

    def test_bsd_lineup_profile_keeps_all_eleven_starters(self):
        players = [{"id": "g1", "name": "GK", "position": "G", "starter": True}]
        players.extend({"id": f"d{i}", "name": f"D{i}", "position": "D", "starter": True} for i in range(1, 5))
        players.extend({"id": f"m{i}", "name": f"M{i}", "position": "M", "starter": True} for i in range(1, 4))
        players.extend({"id": f"f{i}", "name": f"F{i}", "position": "F", "starter": True} for i in range(1, 4))
        players.append({"id": "sub1", "name": "Sub One", "position": "F", "starter": False})

        selected = _select_lineup_players_for_profile(players)
        selected_ids = {p["id"] for p in selected}

        self.assertEqual(len(selected), 11)
        self.assertIn("d4", selected_ids)
        self.assertNotIn("sub1", selected_ids)


if __name__ == "__main__":
    unittest.main()
