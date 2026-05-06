"""
Flashscore client - referee & team card statistics + full match scraping.

Integra capacidades de scraping completo de Flashscore:
- CSV cache de arbitros/equipos (tarjetas historicas)
- Scraping on-demand de partido completo (goles, tarjetas, sustituciones, estadisticas)
- Lineups, H2H, form reciente, tabla de posiciones
- Parseo del protocolo binario df_sui_1 de Flashscore
"""

import re
import json
import asyncio
from pathlib import Path
from collections import defaultdict

from utils import normalizar_nombre

try:
    import pandas as pd
except ImportError:
    pd = None

OUTPUT_DIR = Path("output")
ARBITROS_CSV = OUTPUT_DIR / "arbitros_tarjetas.csv"
EQUIPOS_CSV = OUTPUT_DIR / "equipos_tarjetas.csv"
PARTIDOS_CSV = OUTPUT_DIR / "partidos_tarjetas.csv"

CONCURRENT = 4

BSD_TO_FLASHSCORE = {
    "athletic bilbao": "Ath Bilbao",
    "atletico madrid": "Atl. Madrid",
    "real betis": "Betis",
    "fc barcelona": "Barcelona",
    "real madrid cf": "Real Madrid",
    "real sociedad": "Real Sociedad",
    "villarreal cf": "Villarreal",
    "valencia cf": "Valencia",
    "sevilla fc": "Sevilla",
    "girona fc": "Girona",
    "osasuna": "Osasuna",
    "rc celta": "Celta Vigo",
    "rcd mallorca": "Mallorca",
    "deportivo alaves": "Alaves",
    "levante ud": "Levante",
    "elche cf": "Elche",
    "rcd espanyol": "Espanyol",
    "rayo vallecano": "Rayo Vallecano",
    "getafe cf": "Getafe",
    "real oviedo": "Oviedo",
}

STANDINGS_URLS = {
    "La Liga": "https://www.flashscore.com/football/spain/laliga/standings/",
    "Premier League": "https://www.flashscore.com/football/england/premier-league/standings/",
    "Bundesliga": "https://www.flashscore.com/football/germany/bundesliga/standings/",
    "Brasileirão Serie A": "https://www.flashscore.com/football/brazil/serie-a-betano/standings/",
    "Champions League": "https://www.flashscore.com/football/europe/champions-league/standings/",
    "Europa League": "https://www.flashscore.com/football/europe/europa-league/standings/",
    "Copa Libertadores": "https://www.flashscore.com/football/south-america/copa-libertadores/standings/",
    "Copa Sudamericana": "https://www.flashscore.com/football/south-america/copa-sudamericana/standings/",
}

RESULTS_URLS = {
    "La Liga": "https://www.flashscore.com/football/spain/laliga/results/",
    "Premier League": "https://www.flashscore.com/football/england/premier-league/results/",
    "Bundesliga": "https://www.flashscore.com/football/germany/bundesliga/results/",
    "Brasileirão Serie A": "https://www.flashscore.com/football/brazil/serie-a-betano/results/",
    "Champions League": "https://www.flashscore.com/football/europe/champions-league/results/",
    "Europa League": "https://www.flashscore.com/football/europe/europa-league/results/",
    "Copa Libertadores": "https://www.flashscore.com/football/south-america/copa-libertadores/results/",
    "Copa Sudamericana": "https://www.flashscore.com/football/south-america/copa-sudamericana/results/",
}


def _normalizar_para_flashscore(nombre_bsd: str) -> str:
    if not nombre_bsd:
        return ""
    norm = normalizar_nombre(nombre_bsd)
    if norm in BSD_TO_FLASHSCORE:
        return BSD_TO_FLASHSCORE[norm]
    return nombre_bsd


def _normalizar_referee(ref_name: str) -> str:
    if not ref_name:
        return ""
    ref = ref_name.strip()
    ref = re.sub(r"\s*\(Esp\)\s*", "", ref)
    return ref


def _cargar_cache_arbitros() -> dict[str, dict]:
    if not ARBITROS_CSV.exists() or pd is None:
        return {}
    df = pd.read_csv(ARBITROS_CSV, encoding="utf-8-sig")
    cache = {}
    for _, row in df.iterrows():
        ref = str(row.get("Árbitro", "")).strip()
        if not ref or ref == "Sin datos":
            continue
        cache[ref] = {
            "partidos": int(row.get("Partidos", 0)),
            "total_amarillas": int(row.get("Total Amarillas", 0)),
            "amarillas_local": int(row.get("Amarillas Local", 0)),
            "amarillas_visitante": int(row.get("Amarillas Visitante", 0)),
            "promedio": float(row.get("Promedio Amarillas/Partido", 0)),
        }
    return cache


def _cargar_cache_equipos() -> dict[str, dict]:
    if not EQUIPOS_CSV.exists() or pd is None:
        return {}
    df = pd.read_csv(EQUIPOS_CSV, encoding="utf-8-sig")
    cache = {}
    for _, row in df.iterrows():
        team = str(row.get("Equipo", "")).strip()
        if not team:
            continue
        cache[team.lower()] = {
            "partidos": int(row.get("Partidos", 0)),
            "total_amarillas": int(row.get("Total Amarillas", 0)),
            "promedio": float(row.get("Promedio Amarillas/Partido", 0)),
        }
    return cache


# ══════════════════════════════════════════════════════
# Parseo del protocolo binario df_sui_1 de Flashscore
# ══════════════════════════════════════════════════════

def parse_sui_response(sui_text: str) -> dict:
    if not sui_text:
        return {}

    data = {
        "goals": [], "cards": [], "substitutions": [],
        "ht_score": {"home": 0, "away": 0},
        "ft_score": {"home": 0, "away": 0},
        "referee": "", "stadium": "", "city": "",
        "attendance": "", "capacity": "",
        "home_team": "", "away_team": "",
    }

    segments = re.split(r"¬~", sui_text)
    total_home = 0
    total_away = 0

    for seg in segments:
        pairs = seg.split("¬")

        if "AC÷1st Half" in seg:
            for p in pairs:
                if p.startswith("IG÷"): data["ht_score"]["home"] = int(p.split("÷")[1])
                if p.startswith("IH÷"): data["ht_score"]["away"] = int(p.split("÷")[1])
            continue

        if "AC÷2nd Half" in seg:
            for p in pairs:
                if p.startswith("IG÷"):
                    total_home = data["ht_score"]["home"] + int(p.split("÷")[1])
                if p.startswith("IH÷"):
                    total_away = data["ht_score"]["away"] + int(p.split("÷")[1])
            continue

        if "III÷" in seg:
            ctx = {}
            ik_tags = []

            for p in pairs:
                if "÷" not in p: continue
                k, _, v = p.partition("÷")
                ctx[k] = v
                if k == "IK":
                    ik_tags.append((v, dict(ctx)))

            for ik_type, incident in ik_tags:
                ia = incident.get("IA", "")
                minute = incident.get("IB", "")
                player = incident.get("IF", "")
                player_id = incident.get("IM", "")
                side = "home" if ia == "1" else "away"

                if ik_type == "Goal":
                    data["goals"].append({
                        "minute": minute, "player": player, "player_id": player_id,
                        "side": side,
                        "score_home": incident.get("INX", ""),
                        "score_away": incident.get("IOX", ""),
                    })
                elif ik_type == "Assistance":
                    if data["goals"]:
                        data["goals"][-1]["assist"] = player
                        data["goals"][-1]["assist_id"] = player_id
                elif ik_type in ("Yellow Card", "2nd Yellow Card"):
                    card_type = "red" if "2nd" in ik_type.lower() or incident.get("ID") == "2" else "yellow"
                    data["cards"].append({
                        "minute": minute, "player": player, "player_id": player_id,
                        "side": side, "type": card_type,
                        "reason": incident.get("IL", ""),
                    })
                elif ik_type == "Substitution - In":
                    data["substitutions"].append({
                        "minute": minute, "player_in": player, "player_in_id": player_id,
                        "side": side,
                    })
                elif ik_type == "Substitution - Out":
                    if data["substitutions"]:
                        last_sub = data["substitutions"][-1]
                        if last_sub.get("side") == side and "player_out" not in last_sub:
                            last_sub["player_out"] = player
                            last_sub["player_out_id"] = player_id

        if "MIT÷" in seg:
            info_map = {}
            current_key = None
            for p in pairs:
                if p.startswith("MIT÷"):
                    current_key = p.split("÷")[1]
                elif p.startswith("MIV÷"):
                    info_map[current_key] = p.split("÷")[1] if "÷" in p else p

            data["referee"] = info_map.get("REF", "")
            data["stadium"] = info_map.get("VEN", "")
            data["city"] = info_map.get("TWN", "")
            data["attendance"] = info_map.get("ATT", "")
            data["capacity"] = info_map.get("CAP", "")

    data["ft_score"]["home"] = total_home
    data["ft_score"]["away"] = total_away

    data["yellow_cards"] = {
        "home": sum(1 for c in data["cards"] if c["side"] == "home" and c["type"] == "yellow"),
        "away": sum(1 for c in data["cards"] if c["side"] == "away" and c["type"] == "yellow"),
    }
    data["red_cards"] = {
        "home": sum(1 for c in data["cards"] if c["side"] == "home" and c["type"] == "red"),
        "away": sum(1 for c in data["cards"] if c["side"] == "away" and c["type"] == "red"),
    }

    return data


# ══════════════════════════════════════════════════════
# Scraping de pagina de partido individual
# ══════════════════════════════════════════════════════

async def _handle_cookie_consent(page):
    try:
        btn = page.locator("#onetrust-accept-btn-handler")
        await btn.wait_for(timeout=3000)
        await btn.click()
        await page.wait_for_timeout(500)
    except Exception:
        pass


async def _scrape_statistics(page) -> dict:
    stats = {}
    try:
        for sel in ['a[href*="match-statistics"]', 'text=STATS', '[href*="statistics"]']:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                await btn.click(timeout=3000)
                await page.wait_for_timeout(3000)
                break

        text = await page.evaluate("() => document.body.innerText")
        idx = text.find("TOP STATS")
        if idx < 0:
            return stats

        after = text[idx:]
        lines = [l.strip() for l in after.split("\n") if l.strip()]
        current_section = "top_stats"
        i = 1

        while i + 2 < len(lines):
            line = lines[i]

            if (line == line.upper() and len(line) > 2 and
                not line.startswith("(") and "%" not in line and
                not any(c.isdigit() for c in line.replace("%", "").replace(".", ""))):
                current_section = line.lower().replace(" ", "_")
                i += 1
                continue

            if any(x in line for x in ["MATCH", "ODDS", "LINEUPS", "COMMENTARY",
                                         "PLAYER STATS", "REPORT", "HIGHLIGHTS",
                                         "1ST HALF", "2ND HALF", "YouTube"]):
                break

            if line.startswith("(") and line.endswith(")"):
                i += 1
                continue

            home_val = lines[i]
            label = lines[i + 1]
            away_val = lines[i + 2]

            def is_numeric(v):
                if not v: return False
                cleaned = v.replace("%", "").replace(",", ".")
                try:
                    float(cleaned)
                    return True
                except ValueError:
                    return False

            label_ok = (len(label) > 2 and
                        not label.startswith("(") and
                        label != label.upper())

            if is_numeric(home_val) and is_numeric(away_val) and label_ok:
                key = f"{current_section}__{label}".lower().replace(" ", "_").replace("(", "").replace(")", "")
                stats[key] = {"home": home_val, "away": away_val}
                i += 3
                while i < len(lines) and lines[i].startswith("(") and lines[i].endswith(")"):
                    i += 1
            else:
                i += 1
    except Exception:
        pass
    return stats


async def _scrape_lineups(page) -> dict:
    lineups = {"home": {"formation": "", "starting": [], "subs": [], "coach": ""},
               "away": {"formation": "", "starting": [], "subs": [], "coach": ""}}

    try:
        lineups_btn = page.locator('a[href*="lineups"], button:has-text("Lineups"), a:has-text("Lineups")').first
        if await lineups_btn.count() > 0:
            await lineups_btn.click(timeout=3000)
            await page.wait_for_timeout(3000)

        data = await page.evaluate("""
            () => {
                const result = {
                    home: { formation: '', starting: [], subs: [], coach: '' },
                    away: { formation: '', starting: [], subs: [], coach: '' }
                };
                const formations = document.querySelectorAll('[data-testid="wcl-scores-overline-02"]');
                if (formations.length >= 3) {
                    result.home.formation = formations[0].textContent.trim();
                    result.away.formation = formations[2].textContent.trim();
                }
                const playerNames = document.querySelectorAll('.wcl-lineupsParticipantName_6G3NS, [data-testid="wcl-lineupsParticipantName"]');
                const allPlayers = [];
                playerNames.forEach(el => {
                    const numberEl = el.querySelector('.wcl-participantNumber_yH3F0, [class*="participantNumber"]');
                    const nameEl = el.querySelector('.wcl-participantName_HhMjB, button[class*="participantName"]');
                    const name = nameEl ? nameEl.textContent.trim() : '';
                    const number = numberEl ? numberEl.textContent.trim() : '';
                    if (name && name.length > 1) { allPlayers.push({ name, number }); }
                });
                if (allPlayers.length === 0) {
                    const images = document.querySelectorAll('[data-testid="wcl-lineupsParticipantImage"]');
                    images.forEach(img => {
                        const name = img.getAttribute('alt') || '';
                        let parent = img.parentElement;
                        let number = '';
                        if (parent) {
                            const numEl = parent.querySelector('[class*="participantNumber"], [class*="number"]');
                            if (numEl) number = numEl.textContent.trim();
                        }
                        if (name && name.length > 1) { allPlayers.push({ name, number }); }
                    });
                }
                const mid = Math.floor(allPlayers.length / 2);
                result.home.starting = allPlayers.slice(0, mid);
                result.away.starting = allPlayers.slice(mid);
                const coachEls = document.querySelectorAll('[class*="lf__coach"], [class*="coachName"], [data-testid*="coach"]');
                if (coachEls.length >= 2) {
                    result.home.coach = coachEls[0].textContent.trim().replace(/^Coach:?\\s*/i, '');
                    result.away.coach = coachEls[1].textContent.trim().replace(/^Coach:?\\s*/i, '');
                }
                return result;
            }
        """)
        lineups = data
    except Exception:
        pass
    return lineups


async def _scrape_h2h_and_form(page, match_id: str) -> dict:
    result = {"h2h": [], "home_form": [], "away_form": []}
    try:
        for sel in ['a[href*="h2h"], button:has-text("H2H")']:
            btn = page.locator(sel).first
            if await btn.count() > 0:
                await btn.click(timeout=3000)
                await page.wait_for_timeout(4000)
                break

        text = await page.evaluate("() => document.body.innerText")
        sections = {}
        home_name = away_name = ""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("LAST MATCHES:"):
                team = line.replace("LAST MATCHES:", "").strip()
                if "HOME_FORM" not in sections:
                    sections["HOME_FORM"] = team
                    home_name = team
                else:
                    sections["AWAY_FORM"] = team
                    away_name = team
            elif line.startswith("HEAD-TO-HEAD"):
                sections["H2H"] = "H2H"

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        def parse_matches_section(lines_list, start_idx):
            matches = []
            i = start_idx
            while i < len(lines_list) and len(matches) < 10:
                if re.match(r"\d{2}\.\d{2}\.\d{2}$", lines_list[i]):
                    date = lines_list[i]
                    comp = lines_list[i + 1] if i + 1 < len(lines_list) else ""
                    home = lines_list[i + 2] if i + 2 < len(lines_list) else ""
                    away = lines_list[i + 3] if i + 3 < len(lines_list) else ""
                    g1 = lines_list[i + 4] if i + 4 < len(lines_list) else ""
                    g2 = lines_list[i + 5] if i + 5 < len(lines_list) else ""
                    if g1.isdigit() and g2.isdigit():
                        matches.append({
                            "date": date, "competition": comp,
                            "home": home, "away": away,
                            "home_goals": int(g1), "away_goals": int(g2),
                        })
                        i += 7
                    else:
                        i += 1
                else:
                    i += 1
                    if lines_list[i - 1] in ("ODDS", "HEAD-TO-HEAD", "LAST MATCHES:") or \
                       "SHOW MORE" in lines_list[i - 1].upper():
                        break
            return matches

        home_start = away_start = h2h_start = -1
        for i, line in enumerate(lines):
            if "LAST MATCHES: " + home_name in line and home_start < 0:
                home_start = i + 1
            elif "LAST MATCHES:" in line and home_start >= 0 and away_start < 0:
                away_start = i + 1
            elif line == "HEAD-TO-HEAD MATCHES":
                h2h_start = i + 2

        if home_start > 0:
            result["home_form"] = parse_matches_section(lines, home_start)
        if away_start > 0:
            result["away_form"] = parse_matches_section(lines, away_start)
        if h2h_start > 0:
            result["h2h"] = parse_matches_section(lines, h2h_start)
    except Exception:
        pass
    return result


async def _scrape_standings_page(page, league_url: str) -> list[dict]:
    standings = []
    try:
        await page.goto(league_url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(5000)
        await _handle_cookie_consent(page)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)

        text = await page.evaluate("() => document.body.innerText")
        lines = text.split("\n")

        start = -1
        for i, line in enumerate(lines):
            if "GP" in line and "W" in line and "D" in line and "L" in line and "Pts" in line:
                start = i + 1
                break
        if start < 0:
            return standings

        for i in range(start, min(start + 30, len(lines))):
            line = lines[i].strip()
            m = re.search(
                r"(\d{1,2})\s+(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+):(\d+)\s+(-?\d+)\s+(\d+)",
                line
            )
            if m:
                standings.append({
                    "pos": int(m.group(1)),
                    "team": m.group(2).strip(),
                    "played": int(m.group(3)),
                    "wins": int(m.group(4)),
                    "draws": int(m.group(5)),
                    "losses": int(m.group(6)),
                    "gf": int(m.group(7)),
                    "ga": int(m.group(8)),
                    "gd": int(m.group(9)),
                    "pts": int(m.group(10)),
                })
            if len(standings) >= 20:
                break
    except Exception:
        pass
    return standings


async def _find_match_id(page, home_team: str, away_team: str, league: str) -> str | None:
    url = RESULTS_URLS.get(league)
    if not url:
        return None

    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    await page.wait_for_timeout(2000)
    await _handle_cookie_consent(page)

    last = 0
    for i in range(20):
        ids = await page.evaluate("""
            () => [...new Set([...document.querySelectorAll('[id^="g_1_"]')]
                .map(e => e.id.replace('g_1_', '')))]
        """)
        if len(ids) == last and i > 2:
            break
        last = len(ids)
        sh = page.locator("a.event__more--static")
        if await sh.count() > 0:
            try:
                await sh.first.click()
                await page.wait_for_timeout(1500)
            except Exception:
                pass
        else:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)

    ids = await page.evaluate("""
        () => [...new Set([...document.querySelectorAll('[id^="g_1_"]')]
            .map(e => e.id.replace('g_1_', '')))]
    """)

    home_lower = home_team.lower().strip()
    away_lower = away_team.lower().strip()

    for mid in ids:
        try:
            el = page.locator(f"#g_1_{mid}")
            text = (await el.inner_text()).lower()
            if home_lower in text and away_lower in text:
                return mid
        except Exception:
            continue
    return None


async def scrape_match_full(context, match_id: str, league: str = None) -> dict | None:
    page = await context.new_page()
    sui_text = None

    async def cap(response):
        nonlocal sui_text
        if f"df_sui_1_{match_id}" in response.url:
            try:
                sui_text = await response.text()
            except Exception:
                pass

    page.on("response", cap)

    try:
        await page.goto(
            f"https://www.flashscore.com/match/{match_id}/#/match-summary",
            wait_until="domcontentloaded", timeout=25000
        )
        await page.wait_for_timeout(3000)
    except Exception:
        await page.close()
        return None

    page.remove_listener("response", cap)
    await _handle_cookie_consent(page)

    sui_data = parse_sui_response(sui_text)
    text = await page.evaluate("() => document.body.innerText")

    home_name = sui_data.get("home_team", "")
    away_name = sui_data.get("away_team", "")

    if not home_name:
        m = re.search(r"(\S.+?)\s+(?:v|vs)\s+(\S.+?)\s+\((\d+/\d+/\d+)\)", text)
        if m:
            home_name = m.group(1)
            away_name = m.group(2)
    if not home_name:
        names = await page.evaluate("""
            () => {
                const h = document.querySelector('.duelParticipant__home .participant__participantName');
                const a = document.querySelector('.duelParticipant__away .participant__participantName');
                return [h ? h.textContent.trim() : '', a ? a.textContent.trim() : ''];
            }
        """)
        home_name, away_name = names[0], names[1]

    sui_data["home_team"] = home_name or sui_data.get("home_team", "")
    sui_data["away_team"] = away_name or sui_data.get("away_team", "")

    match_date = ""
    date_m = re.search(r"(\d{2}/\d{2}/\d{4}|\d{2}\.\d{2}\.\d{4})", text)
    if date_m:
        match_date = date_m.group(1)

    stats = await _scrape_statistics(page)
    lineups = await _scrape_lineups(page)
    h2h_form = await _scrape_h2h_and_form(page, match_id)

    await page.close()

    return {
        "match_id": match_id,
        "home_team": sui_data["home_team"],
        "away_team": sui_data["away_team"],
        "date": match_date,
        "ft_score": sui_data["ft_score"],
        "ht_score": sui_data["ht_score"],
        "goals": sui_data["goals"],
        "cards": sui_data["cards"],
        "yellow_cards": sui_data["yellow_cards"],
        "red_cards": sui_data["red_cards"],
        "substitutions": sui_data["substitutions"],
        "referee": sui_data["referee"],
        "stadium": sui_data["stadium"],
        "city": sui_data["city"],
        "attendance": sui_data["attendance"],
        "capacity": sui_data["capacity"],
        "statistics": stats,
        "lineups": lineups,
        "h2h": h2h_form.get("h2h", []),
        "home_form": h2h_form.get("home_form", []),
        "away_form": h2h_form.get("away_form", []),
    }


# ══════════════════════════════════════════════════════
# Funciones de cache (arbitros/equipos)
# ══════════════════════════════════════════════════════

def _buscar_stats_arbitro_cache(referee_name: str, cache_arbitros: dict | None = None) -> dict | None:
    if cache_arbitros is None:
        cache_arbitros = _cargar_cache_arbitros()
    if not cache_arbitros:
        return None
    ref_norm = _normalizar_referee(referee_name)
    for nombre, stats in cache_arbitros.items():
        if ref_norm.lower() == nombre.lower():
            return stats
        if ref_norm.lower() in nombre.lower() or nombre.lower() in ref_norm.lower():
            return stats
    return None


def _buscar_stats_equipo_cache(team_name: str, cache_equipos: dict | None = None) -> dict | None:
    if cache_equipos is None:
        cache_equipos = _cargar_cache_equipos()
    if not cache_equipos:
        return None
    fs_name = _normalizar_para_flashscore(team_name)
    key = fs_name.lower()
    if key in cache_equipos:
        return cache_equipos[key]
    norm = normalizar_nombre(team_name)
    if norm in cache_equipos:
        return cache_equipos[norm]
    for k, v in cache_equipos.items():
        if norm in k or k in norm:
            return v
    return None


# ══════════════════════════════════════════════════════
# Funcion principal de enriquecimiento (cache + opcional live)
# ══════════════════════════════════════════════════════

def enriquecer_datos_partido(datos_bsd: dict) -> dict:
    """
    Enrich BSD/SofaScore match data with Flashscore card statistics.

    Busca en cache CSV:
    - Estadisticas historicas del arbitro (tarjetas por partido, home/away bias)
    - Estadisticas de tarjetas de ambos equipos

    Si el cache no existe para esta liga, devuelve _flashscore: {disponible: False}.
    """
    resultado = {"disponible": False}

    cache_arbitros = _cargar_cache_arbitros()
    cache_equipos = _cargar_cache_equipos()

    referee_name = ""
    if datos_bsd.get("arbitro") and isinstance(datos_bsd["arbitro"], dict):
        referee_name = datos_bsd["arbitro"].get("nombre", "")
    if not referee_name:
        ss = datos_bsd.get("_sofascore", {})
        detalle = ss.get("detalle_evento", {})
        r = detalle.get("arbitro", {})
        referee_name = r.get("nombre", "") if isinstance(r, dict) else ""

    if referee_name:
        ref_stats = _buscar_stats_arbitro_cache(referee_name, cache_arbitros)
        if ref_stats:
            resultado["arbitro_stats"] = ref_stats
            resultado["disponible"] = True

    partes = datos_bsd.get("partido", "").split(" vs ")
    home_team = partes[0].strip() if len(partes) > 0 else ""
    away_team = partes[1].strip() if len(partes) > 1 else ""

    if home_team or away_team:
        equipo_stats = {}
        if home_team:
            ht = _buscar_stats_equipo_cache(home_team, cache_equipos)
            if ht:
                equipo_stats["local"] = ht
                resultado["disponible"] = True
        if away_team:
            at = _buscar_stats_equipo_cache(away_team, cache_equipos)
            if at:
                equipo_stats["visitante"] = at
                resultado["disponible"] = True
        if equipo_stats:
            resultado["equipo_stats"] = equipo_stats

    datos_bsd["_flashscore"] = resultado
    return datos_bsd


def enriquecer_datos_partido_async_sync(datos_bsd: dict, league: str | None = None) -> dict:
    """
    Wrapper sincrono que corre el enriquecimiento (cache + opcional live scraping).
    Para compatibilidad con el resto del codigo que es sincrono.
    """
    if not league:
        league = datos_bsd.get("liga", "")

    datos_bsd = enriquecer_datos_partido(datos_bsd)

    if not datos_bsd.get("_flashscore", {}).get("disponible"):
        partes = datos_bsd.get("partido", "").split(" vs ")
        home = partes[0] if len(partes) > 0 else ""
        away = partes[1] if len(partes) > 1 else ""
        if home and away and league in RESULTS_URLS:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(_scrape_and_enrich(datos_bsd, home, away, league))
                loop.close()
                if result:
                    datos_bsd = result
            except Exception:
                pass

    return datos_bsd


async def _scrape_and_enrich(datos_bsd: dict, home: str, away: str, league: str) -> dict | None:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return None

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        match_id = await _find_match_id(page, home, away, league)
        if not match_id:
            await browser.close()
            return None

        match_data = await scrape_match_full(context, match_id, league)
        await browser.close()

        if match_data:
            cache_arbitros = _cargar_cache_arbitros()
            ref_stats = _buscar_stats_arbitro_cache(match_data.get("referee", ""), cache_arbitros)
            cache_equipos = _cargar_cache_equipos()
            home_stats = _buscar_stats_equipo_cache(home, cache_equipos)
            away_stats = _buscar_stats_equipo_cache(away, cache_equipos)

            flashscore = {
                "disponible": True,
                "match_data": match_data,
            }
            if ref_stats:
                flashscore["arbitro_stats"] = ref_stats
            equipo_stats = {}
            if home_stats:
                equipo_stats["local"] = home_stats
            if away_stats:
                equipo_stats["visitante"] = away_stats
            if equipo_stats:
                flashscore["equipo_stats"] = equipo_stats

            datos_bsd["_flashscore"] = flashscore
            return datos_bsd

    return None


# ══════════════════════════════════════════════════════
# Scraping de tabla de posiciones
# ══════════════════════════════════════════════════════

async def _scrape_standings_async(league: str) -> list[dict]:
    url = STANDINGS_URLS.get(league)
    if not url:
        return []

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()
        standings = await _scrape_standings_page(page, url)
        await browser.close()
        return standings


def obtener_standings_flashscore(league: str) -> list[dict]:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_scrape_standings_async(league))
        loop.close()
        return result
    except Exception:
        return []


# ══════════════════════════════════════════════════════
# Formateo para prompts del analyzer
# ══════════════════════════════════════════════════════

def _formatear_flashscore_para_prompt(datos: dict) -> str:
    """
    Formatea los datos de Flashscore para incluir en el prompt del analyzer.

    Args:
        datos: Dict completo con clave _flashscore.

    Returns:
        String formateado para el prompt, o cadena vacia si no hay datos.
    """
    fs = datos.get("_flashscore", {})
    if not fs.get("disponible"):
        return ""

    partes = ["\n### DATOS FLASHSCORE (Tarjetas y Arbitros)"]

    ref_stats = fs.get("arbitro_stats")
    if ref_stats:
        avg = ref_stats.get("promedio", 0)
        partes.append(f"\n**Estadisticas del arbitro**")
        partes.append(f"- Partidos dirigidos esta temporada: {ref_stats.get('partidos', '?')}")
        partes.append(f"- Total amarillas: {ref_stats.get('total_amarillas', '?')}")
        partes.append(f"- Promedio amarillas/partido: {avg}")
        partes.append(f"- Amarillas Local: {ref_stats.get('amarillas_local', '?')}")
        partes.append(f"- Amarillas Visitante: {ref_stats.get('amarillas_visitante', '?')}")
        if avg >= 5:
            partes.append("- TENDENCIA: Arbitro MUY tarjetero (>5 YC/partido). Esperar muchas tarjetas.")
        elif avg >= 4:
            partes.append("- TENDENCIA: Arbitro tarjetero (>4 YC/partido). Esperar tarjetas por encima de la media.")
        elif avg >= 3:
            partes.append("- TENDENCIA: Arbitro moderado (3-4 YC/partido).")
        else:
            partes.append("- TENDENCIA: Arbitro permisivo (<3 YC/partido). Esperar pocas tarjetas.")

    eq_stats = fs.get("equipo_stats")
    if eq_stats:
        partes.append(f"\n**Tendencias de tarjetas por equipo**")
        for side, label in [("local", "LOCAL"), ("visitante", "VISITANTE")]:
            st = eq_stats.get(side)
            if st:
                partes.append(
                    f"- {label}: {st.get('promedio', '?')} YC/partido "
                    f"({st.get('partidos', '?')} partidos, {st.get('total_amarillas', '?')} total amarillas)"
                )

    match_data = fs.get("match_data")
    if match_data:
        partes.append(f"\n**Datos del partido en Flashscore**")

        if match_data.get("ft_score", {}).get("home", 0) or match_data.get("ft_score", {}).get("away", 0):
            partes.append(
                f"- Resultado: {match_data.get('home_team', '?')} {match_data.get('ft_score', {}).get('home', '?')} "
                f"- {match_data.get('ft_score', {}).get('away', '?')} {match_data.get('away_team', '?')}"
            )

        goals = match_data.get("goals", [])
        if goals:
            partes.append(f"\n**Goles ({len(goals)}):**")
            for g in goals:
                asist = f" (asist: {g['assist']})" if g.get("assist") else ""
                partes.append(f"  - {g['minute']}': {g['player']}{asist} [{g['side']}]")

        cards = match_data.get("cards", [])
        if cards:
            partes.append(f"\n**Tarjetas ({len(cards)}):**")
            yellow_cards = [c for c in cards if c.get("type") == "yellow"]
            red_cards = [c for c in cards if c.get("type") == "red"]
            if yellow_cards:
                partes.append(
                    f"  Amarillas: {match_data.get('yellow_cards', {}).get('home', 0)}L "
                    f"- {match_data.get('yellow_cards', {}).get('away', 0)}V"
                )
            if red_cards:
                partes.append(
                    f"  Rojas: {match_data.get('red_cards', {}).get('home', 0)}L "
                    f"- {match_data.get('red_cards', {}).get('away', 0)}V"
                )
            for c in cards[:10]:
                partes.append(f"  - {c['minute']}': {c['player']} ({c['type']}) [{c['side']}]")

        substitutions = match_data.get("substitutions", [])
        if substitutions:
            partes.append(f"\n**Sustituciones ({len(substitutions)}):**")
            for s in substitutions[:6]:
                partes.append(f"  - {s['minute']}': IN {s.get('player_in', '?')} OUT {s.get('player_out', '?')}")

        if match_data.get("referee"):
            partes.append(f"\n- Arbitro: {match_data['referee']}")
        if match_data.get("stadium"):
            partes.append(f"- Estadio: {match_data['stadium']} ({match_data.get('city', '')}, cap: {match_data.get('capacity', '?')})")
        if match_data.get("attendance"):
            partes.append(f"- Asistencia: {match_data['attendance']}")

        stats = match_data.get("statistics", {})
        if stats:
            partes.append(f"\n**Estadisticas del partido:**")
            key_labels = {
                "top_stats__ball_possession": "Posesion",
                "top_stats__expected_goals": "xG",
                "top_stats__total_shots": "Tiros totales",
                "top_stats__shots_on_target": "Tiros al arco",
                "top_stats__shots_off_target": "Tiros desviados",
                "top_stats__corners": "Corners",
                "top_stats__fouls": "Faltas",
                "top_stats__yellow_cards": "Amarillas",
                "top_stats__red_cards": "Rojas",
                "top_stats__offsides": "Offsides",
            }
            for key, label in key_labels.items():
                v = stats.get(key)
                if v:
                    partes.append(f"  - {label}: {v.get('home', '?')} - {v.get('away', '?')}")

        lineups_data = match_data.get("lineups", {})
        if lineups_data:
            for side, sname in [("home", match_data.get("home_team", "Local")), ("away", match_data.get("away_team", "Visitante"))]:
                ld = lineups_data.get(side, {})
                if ld.get("formation"):
                    partes.append(f"\n**Alineacion {sname} ({ld.get('formation', '?')})**")
                    starters = ld.get("starting", [])
                    if starters:
                        names = [f"#{p.get('number', '?')} {p.get('name', '?')}" for p in starters]
                        partes.append(f"  Titulares: {', '.join(names)}")
                    if ld.get("coach"):
                        partes.append(f"  DT: {ld['coach']}")

        h2h = match_data.get("h2h", [])
        if h2h:
            partes.append(f"\n**H2H historico (Flashscore):**")
            for m in h2h[:5]:
                partes.append(f"  - {m.get('date', '?')}: {m.get('home', '?')} {m.get('home_goals', '?')}-{m.get('away_goals', '?')} {m.get('away', '?')}")

        home_form = match_data.get("home_form", [])
        away_form = match_data.get("away_form", [])
        if home_form or away_form:
            partes.append(f"\n**Forma reciente (Flashscore):**")
            if home_form:
                form_str = " ".join(f"{m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}" for m in home_form[:5])
                partes.append(f"  {match_data.get('home_team', 'Local')}: {form_str}")
            if away_form:
                form_str = " ".join(f"{m['home']} {m['home_goals']}-{m['away_goals']} {m['away']}" for m in away_form[:5])
                partes.append(f"  {match_data.get('away_team', 'Visitante')}: {form_str}")

    partes.append("\nIMPORTANTE: Los datos de Flashscore son historicos de esta temporada. Usalos para evaluar tendencia de tarjetas del arbitro y equipos.")
    partes.append("Cruza esta info con el estilo arbitral y las faltas_promedio de BSD para proyectar tarjetas en este partido.")

    return "\n".join(partes)


def _formatear_standings_flashscore_para_prompt(datos: dict) -> str:
    fs = datos.get("_flashscore", {})
    standings = fs.get("standings", [])
    if not standings:
        return ""

    league = datos.get("liga", "")
    partes = [f"\n### TABLA DE POSICIONES FLASHSCORE ({league})"]

    equipo_local = datos.get("partido", "").split(" vs ")[0] if " vs " in datos.get("partido", "") else ""
    equipo_visitante = datos.get("partido", "").split(" vs ")[1] if " vs " in datos.get("partido", "") else ""

    for row in standings:
        marker = ""
        if equipo_local.lower() in row.get("team", "").lower():
            marker = ">> LOCAL"
        elif equipo_visitante.lower() in row.get("team", "").lower():
            marker = ">> VISITANTE"
        partes.append(
            f"  #{row.get('pos', '?')} {row.get('team', '?')} | "
            f"PJ:{row.get('played', '?')} V:{row.get('wins', '?')} E:{row.get('draws', '?')} D:{row.get('losses', '?')} | "
            f"GF:{row.get('gf', '?')} GC:{row.get('ga', '?')} DG:{row.get('gd', '?')} | "
            f"PTS:{row.get('pts', '?')} {marker}"
        )

    return "\n".join(partes)


# ══════════════════════════════════════════════════════
# Generacion de CSVs de arbitros/equipos desde scrapes
# ══════════════════════════════════════════════════════

def generar_csvs_desde_scrapes(matches: list) -> dict:
    """Genera CSVs de arbitros y equipos desde una lista de partidos scrapeados."""
    if pd is None:
        return {"error": "pandas no instalado"}

    OUTPUT_DIR.mkdir(exist_ok=True)

    match_rows = []
    card_rows = []
    goal_rows = []
    ref_d = defaultdict(lambda: {"matches": 0, "yc": 0, "yc_h": 0, "yc_a": 0, "rc": 0})

    for m in matches:
        match_rows.append({
            "match_id": m.get("match_id", ""),
            "date": m.get("date", ""),
            "home_team": m.get("home_team", ""),
            "away_team": m.get("away_team", ""),
            "ft_home": m.get("ft_score", {}).get("home", 0),
            "ft_away": m.get("ft_score", {}).get("away", 0),
            "yellow_home": m.get("yellow_cards", {}).get("home", 0),
            "yellow_away": m.get("yellow_cards", {}).get("away", 0),
            "red_home": m.get("red_cards", {}).get("home", 0),
            "red_away": m.get("red_cards", {}).get("away", 0),
            "referee": m.get("referee", ""),
            "stadium": m.get("stadium", ""),
        })

        for c in m.get("cards", []):
            card_rows.append({
                "match_id": m.get("match_id", ""),
                "home_team": m.get("home_team", ""),
                "away_team": m.get("away_team", ""),
                "minute": c.get("minute", ""),
                "player": c.get("player", ""),
                "side": c.get("side", ""),
                "type": c.get("type", ""),
                "reason": c.get("reason", ""),
                "referee": m.get("referee", ""),
            })

        for g in m.get("goals", []):
            goal_rows.append({
                "match_id": m.get("match_id", ""),
                "home_team": m.get("home_team", ""),
                "away_team": m.get("away_team", ""),
                "minute": g.get("minute", ""),
                "player": g.get("player", ""),
                "assist": g.get("assist", ""),
                "side": g.get("side", ""),
            })

        ref = m.get("referee") or "Unknown"
        ref_d[ref]["matches"] += 1
        ref_d[ref]["yc"] += m.get("yellow_cards", {}).get("home", 0) + m.get("yellow_cards", {}).get("away", 0)
        ref_d[ref]["yc_h"] += m.get("yellow_cards", {}).get("home", 0)
        ref_d[ref]["yc_a"] += m.get("yellow_cards", {}).get("away", 0)
        ref_d[ref]["rc"] += m.get("red_cards", {}).get("home", 0) + m.get("red_cards", {}).get("away", 0)

    ref_rows = []
    for ref, d in sorted(ref_d.items(), key=lambda x: x[1]["matches"], reverse=True):
        n = d["matches"]
        ref_rows.append({
            "Árbitro": ref, "Partidos": n,
            "Total Amarillas": d["yc"], "Amarillas Local": d["yc_h"], "Amarillas Visitante": d["yc_a"],
            "Total Rojas": d["rc"],
            "Promedio Amarillas/Partido": round(d["yc"] / n, 2) if n else 0,
        })

    pd.DataFrame(match_rows).to_csv(PARTIDOS_CSV, index=False, encoding="utf-8-sig")
    if card_rows:
        pd.DataFrame(card_rows).to_csv(OUTPUT_DIR / "tarjetas_detalle.csv", index=False, encoding="utf-8-sig")
    if goal_rows:
        pd.DataFrame(goal_rows).to_csv(OUTPUT_DIR / "goles_flashscore.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ref_rows).to_csv(ARBITROS_CSV, index=False, encoding="utf-8-sig")

    # Equipos CSV
    team_d = defaultdict(lambda: {"matches": 0, "yc": 0})
    for m in matches:
        ht = m.get("home_team", "")
        at = m.get("away_team", "")
        if ht:
            team_d[ht]["matches"] += 1
            team_d[ht]["yc"] += m.get("yellow_cards", {}).get("home", 0)
        if at:
            team_d[at]["matches"] += 1
            team_d[at]["yc"] += m.get("yellow_cards", {}).get("away", 0)

    team_rows = []
    for team, d in sorted(team_d.items()):
        n = d["matches"]
        team_rows.append({
            "Equipo": team, "Partidos": n,
            "Total Amarillas": d["yc"],
            "Promedio Amarillas/Partido": round(d["yc"] / n, 2) if n else 0,
        })
    pd.DataFrame(team_rows).to_csv(EQUIPOS_CSV, index=False, encoding="utf-8-sig")

    return {
        "partidos": len(matches),
        "arbitros": len(ref_rows),
        "equipos": len(team_rows),
        "archivos": {
            "partidos": str(PARTIDOS_CSV),
            "arbitros": str(ARBITROS_CSV),
            "equipos": str(EQUIPOS_CSV),
        },
    }
