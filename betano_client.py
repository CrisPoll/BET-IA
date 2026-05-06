"""
Betano client - extrae cuotas en tiempo real usando cookies de sesion.

Usa Playwright con cookies exportadas del navegador para acceder
a la pagina de cuotas de Betano y extraer TODOS los mercados disponibles.

Requiere:
    BETANO_COOKIES en .env (cookies copiadas del navegador)

Uso:
    cuotas = obtener_cuotas_betano("Palmeiras", "Santos", "Brasileirão Serie A")
"""

import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from utils import normalizar_nombre

load_dotenv()

BETANO_BASE = "https://www.betano.pe"
BETANO_COOKIES = os.getenv("BETANO_COOKIES", "")

# Mapeo liga BSD -> patron de busqueda en top-events-v2
LEAGUE_PATTERNS = {
    "Brasileirão Serie A": "brasileirao",
    "La Liga": "laliga",
    "Premier League": "premier",
    "Bundesliga": "bundesliga",
    "Liga 1 Peru": "liga 1",
}

MARKET_NAME_MAP = {
    "Resultado Final": "1X2",
    "Doble oportunidad": "Double Chance",
    "Ambos equipos anotan": "BTTS",
    "Total de goles": "Over/Under",
    "Total de goles local": "Over/Under Local",
    "Total de goles visitante": "Over/Under Visitante",
    "Handicap": "Handicap",
    "Handicap asiático": "Asian Handicap",
    "Marcador exacto": "Correct Score",
    "Resultado/Ambos anotan": "Result/BTTS",
    "Primer tiempo - Resultado": "1X2 1H",
    "Primer tiempo - Total de goles": "Over/Under 1H",
    "Primer tiempo - Ambos anotan": "BTTS 1H",
    "Segundo tiempo - Resultado": "1X2 2H",
    "Total de corners": "Corners",
    "Total de tarjetas": "Cards",
    "Total de tarjetas amarillas": "Yellow Cards",
    "Total de remates": "Shots",
    "Total de remates al arco": "Shots on Target",
    "Total de faltas": "Fouls",
}

ODDS_SELECTORS = [
    ".events-list__grid .events-list__event",
    '[data-cy="event-card"]',
    ".event-row",
]


def _parse_cookies(cookie_string: str) -> list[dict]:
    """Parsea cookies en formato name=value; name=value a objetos Playwright."""
    cookies = []
    for part in cookie_string.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            continue
        # Add both domains so at least one works
        cookies.append({"name": name, "value": value, "domain": ".betano.pe", "path": "/"})
        cookies.append({"name": name, "value": value, "domain": "www.betano.pe", "path": "/"})
    return cookies


async def _find_event_id(home_team: str, away_team: str) -> dict | None:
    """Busca el event ID en top-events-v2."""
    from curl_cffi import requests as cffi_requests

    home_norm = normalizar_nombre(home_team)
    away_norm = normalizar_nombre(away_team)

    session = cffi_requests.Session(impersonate="chrome")
    try:
        resp = session.get(
            f"{BETANO_BASE}/api/home/top-events-v2/",
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            timeout=15,
        )
        data = resp.json()
    except Exception:
        return None

    events = data.get("data", {}).get("topEventsV2", {}).get("events", {})
    for eid, ev in events.items():
        participants = ev.get("participants", [])
        if len(participants) < 2:
            continue
        p1 = normalizar_nombre(participants[0].get("name", ""))
        p2 = normalizar_nombre(participants[1].get("name", ""))

        if (home_norm in p1 or p1 in home_norm) and (away_norm in p2 or p2 in away_norm):
            return ev
        if (home_norm in p2 or p2 in home_norm) and (away_norm in p1 or p1 in away_norm):
            return ev

    return None


async def _scrape_cuotas_match(event: dict) -> dict:
    """Scrapea todos los mercados desde la pagina del partido."""
    if not BETANO_COOKIES:
        raise RuntimeError("BETANO_COOKIES no configurada en .env")

    event_id = event.get("id")
    url = event.get("url", f"/cuotas-de-partido/evento/{event_id}/")
    full_url = f"{BETANO_BASE}{url}" if not url.startswith("http") else url

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        cookies_list = _parse_cookies(BETANO_COOKIES)
        await ctx.add_cookies(cookies_list)
        page = await ctx.new_page()

        try:
            await page.goto(full_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(10000)

            # Scroll to ensure lazy content loads
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1000)

            # Click "Todo" tab to show all markets expanded
            try:
                todo_btn = page.locator('text=Todo').first
                if await todo_btn.count() > 0:
                    await todo_btn.click()
                    await page.wait_for_timeout(3000)
            except Exception:
                pass

            # Click collapsed market rows with BB (no odds visible)
            await page.evaluate("""
                () => {
                    // Only target elements that have 'BB' text AND look like market titles (no sidebar/etc)
                    const candidates = document.querySelectorAll('[class*="market"], [class*="event-row"], [class*="table-market"]');
                    let clicked = 0;
                    candidates.forEach(el => {
                        const text = el.textContent || '';
                        const hasBB = text.includes('BB');
                        const hasOdds = /\\d+\\.\\d{2}/.test(text);
                        if (hasBB && !hasOdds && text.length < 120) {
                            try {
                                // Click the clickable header, not the BB button itself
                                const header = el.querySelector('[class*="header"], [class*="title"], h3, h4') || el;
                                header.click();
                                clicked++;
                            } catch(e) {}
                        }
                    });
                    return clicked;
                }
            """)
            await page.wait_for_timeout(3000)

            all_markets = await _parse_dom_markets(page)

            if not all_markets:
                all_markets = await page.evaluate("""
                    () => {
                        const markets = [];
                        const sections = document.querySelectorAll('.table-market-header, [class*=\"market-header\"], [class*=\"MarketTitle\"], [data-testid*=\"market\"]');
                        
                        sections.forEach(section => {
                            const title = section.textContent?.trim() || '';
                            if (!title || title.length < 3) return;
                            
                            const parent = section.closest('[class*=\"market\"]') || section.parentElement;
                            if (!parent) return;
                            
                            const odds = parent.querySelectorAll('[class*=\"odd\"], [class*=\"price\"], [class*=\"oddValue\"], [data-testid*=\"odd\"], span[class*=\"selection\"]');
                            const selectionEls = parent.querySelectorAll('[class*=\"selection\"], [class*=\"outcome\"], [class*=\"participant\"]');
                            
                            if (odds.length === 0 && selectionEls.length === 0) return;
                            
                            const selections = [];
                            const allTexts = parent.innerText.split('\\n').filter(t => t.trim());
                            
                            const oddsPattern = /\\d+\\.\\d{2}/;
                            allTexts.forEach(t => {
                                const oddsMatch = t.match(oddsPattern);
                                if (oddsMatch) {
                                    const oddValue = parseFloat(oddsMatch[0]);
                                    const name = t.replace(oddsPattern, '').trim().split('\\n')[0].trim();
                                    if (name || oddValue) {
                                        selections.push({name: name || '?', odd: oddValue});
                                    }
                                }
                            });
                            
                            if (selections.length > 0) {
                                markets.push({name: title, selections: selections});
                            }
                        });
                        
                        return markets;
                    }
                """)

            if not all_markets:
                all_markets = await _parse_dom_markets(page)

            participants = event.get("participants", [])
            home_team = participants[0].get("name", "") if len(participants) > 0 else ""
            away_team = participants[1].get("name", "") if len(participants) > 1 else ""

            return {
                "event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "total_markets_available": event.get("totalMarketsAvailable", 0),
                "markets": all_markets,
            }

        except Exception as e:
            return {"event_id": event_id, "error": str(e)}
        finally:
            await browser.close()


async def _parse_dom_markets(page) -> list:
    """Parse DOM completo buscando mercados con nombres y odds."""
    return await page.evaluate("""
        () => {
            const text = document.body.innerText;
            const lines = text.split('\\n').map(l => l.trim());
            
            const markets = [];
            let currentMarket = null;
            let currentSelections = [];
            const oddPattern = /^\\d+\\.\\d{2}$/;
            
            // Known market title patterns (Betano Spanish)
            const marketTitles = [
                'Resultado del partido',
                'Resultado Final',
                'Doble oportunidad', 'Doble chance',
                'Ambos equipos anotan', 'Ambos anotan', 'Ambos marcan',
                'Total de goles', 'Goles totales',
                'Total de goles local', 'Goles local',
                'Total de goles visitante', 'Goles visitante',
                'Handicap', 'Handicap Asiatico', 'Handicap asiatico',
                'Handicap Resultado del Partido',
                'Marcador exacto', 'Marcador correcto', 'Resultado exacto',
                'Resultado/Ambos anotan', 'Resultado y ambos anotan',
                'Primer tiempo', '1er tiempo',
                'Segundo tiempo', '2do tiempo',
                'Total de corners', 'Corners', 'Escanteios',
                'Total de tarjetas', 'Tarjetas',
                'Total de tarjetas amarillas', 'Amarillas',
                'Total de remates', 'Remates', 'Chutes',
                'Total de remates al arco', 'Remates al arco', 'Chutes al arco',
                'Total de faltas', 'Faltas',
                'Goleador', 'Primer goleador', 'Anytime goleador',
                'Resultado Primer Tiempo',
                'Intervalo de goles',
                'Metodo de anotacion',
                'SuperCuotas',
                'Mercado de jugadores', 'Tarjetas de jugador',
                'Remates de jugador',
            ];
            
            // Flatten: collect all lines, find market titles, then capture odds under them
            const marketTitleSet = new Set(marketTitles.map(t => t.toLowerCase()));
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                if (!line) continue;
                
                const lineLower = line.toLowerCase();
                
                // Check if this line is a market title
                let isTitle = marketTitleSet.has(lineLower);
                if (!isTitle) {
                    // Fuzzy match: line starts with or exactly contains a known title
                    for (const t of marketTitles) {
                        const tl = t.toLowerCase();
                        if (lineLower === tl || lineLower.startsWith(tl + ' ')) {
                            isTitle = true;
                            break;
                        }
                    }
                }
                
                if (isTitle && line.length < 100) {
                    // Save previous market
                    if (currentMarket && currentSelections.length >= 2) {
                        markets.push({name: currentMarket, selections: [...currentSelections]});
                    }
                    currentMarket = line;
                    currentSelections = [];
                    continue;
                }
                
                // Look ahead for odds pattern
                if (currentMarket && i + 1 < lines.length) {
                    const nextLine = lines[i + 1];
                    if (oddPattern.test(nextLine)) {
                        currentSelections.push({
                            name: line,
                            odd: parseFloat(nextLine)
                        });
                        i++; // skip the odds line
                        continue;
                    }
                }
                
                // Check if current line has an odds number at the end
                const parts = line.split(/\\s+/);
                if (currentMarket && parts.length >= 2) {
                    const lastPart = parts[parts.length - 1];
                    if (oddPattern.test(lastPart)) {
                        const name = parts.slice(0, -1).join(' ');
                        currentSelections.push({
                            name: name,
                            odd: parseFloat(lastPart)
                        });
                        continue;
                    }
                }
            }
            
            // Save last market
            if (currentMarket && currentSelections.length >= 2) {
                markets.push({name: currentMarket, selections: [...currentSelections]});
            }
            
            return markets;
        }
    """)


def _normalize_odds(markets: list) -> dict:
    """Estandariza mercados a formato clave:nombre -> {selections}."""
    result = {}
    for m in markets:
        name = m.get("name", "").strip()
        selections = m.get("selections", [])
        if not selections:
            continue

        # Map to standardized name
        mapped = name
        name_lower = name.lower()
        if any(k in name_lower for k in ['resultado del partido', 'resultado final']):
            if 'supercuota' in name_lower:
                mapped = '1X2 SuperCuotas'
            else:
                mapped = '1X2'
        elif 'doble' in name_lower and ('oportunidad' in name_lower or 'chance' in name_lower):
            mapped = 'Double Chance'
        elif 'ambos' in name_lower and ('anotan' in name_lower or 'marcan' in name_lower):
            mapped = 'BTTS'
        elif 'goles' in name_lower and ('total' in name_lower or 'mas/menos' in name_lower):
            mapped = 'Over/Under'
        elif 'goles' in name_lower and 'local' in name_lower:
            mapped = 'Over/Under Local'
        elif 'goles' in name_lower and 'visitante' in name_lower:
            mapped = 'Over/Under Visitante'
        elif 'handicap' in name_lower and 'asiatico' in name_lower:
            mapped = 'Asian Handicap'
        elif 'handicap' in name_lower:
            mapped = 'Handicap'
        elif 'marcador' in name_lower and 'exacto' in name_lower:
            mapped = 'Correct Score'
        elif 'primer tiempo' in name_lower or '1er tiempo' in name_lower or '1t' in name_lower:
            if 'resultado' in name_lower:
                mapped = '1X2 1H'
            elif 'goles' in name_lower:
                mapped = 'Over/Under 1H'
            elif 'ambos' in name_lower:
                mapped = 'BTTS 1H'
            else:
                mapped = name
        elif 'corners' in name_lower or 'escanteios' in name_lower:
            mapped = 'Corners'
        elif 'tarjetas' in name_lower and 'amarilla' in name_lower:
            mapped = 'Yellow Cards'
        elif 'tarjetas' in name_lower:
            mapped = 'Cards'
        elif 'remates' in name_lower and 'arco' in name_lower:
            mapped = 'Shots on Target'
        elif 'remates' in name_lower or 'chutes' in name_lower:
            mapped = 'Shots'
        elif 'faltas' in name_lower:
            mapped = 'Fouls'

        # Deduplicate by mapped name
        if mapped in result:
            existing = {s["name"]: s for s in result[mapped].get("selections", [])}
            for s in selections:
                key = s.get("name", "")
                if key not in existing or key == "?":
                    existing[key] = s
            result[mapped]["selections"] = list(existing.values())
        else:
            result[mapped] = {"name": name, "selections": selections}
    return result


def obtener_cuotas_betano(home_team: str, away_team: str, league: str | None = None) -> dict:
    """
    Obtiene todas las cuotas disponibles en Betano para un partido.

    Args:
        home_team: Nombre del equipo local (BSD format).
        away_team: Nombre del equipo visitante (BSD format).
        league: Nombre de la liga (opcional, para busqueda mas precisa).

    Returns:
        Dict con event_id, teams, markets[] con todas las cuotas.
    """
    if not BETANO_COOKIES:
        return {"error": "BETANO_COOKIES no configurada en .env. Logueate en Betano, copia document.cookie y pegalo en .env"}

    async def _run():
        event = await _find_event_id(home_team, away_team)
        if not event:
            return {"error": f"No se encontro '{home_team} vs {away_team}' en Betano top-events"}

        result = await _scrape_cuotas_match(event)
        if "error" in result:
            return result

        result["markets"] = _normalize_odds(result.get("markets", []))
        return result

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_run())
        loop.close()
        return result
    except Exception as e:
        return {"error": str(e)}


# ── Formateo para el prompt del analyzer ──

def _formatear_cuotas_betano_para_prompt(result: dict) -> str:
    """
    Formatea las cuotas de Betano para incluir en el prompt del analyzer.
    """
    if not result or "error" in result:
        return ""

    markets = result.get("markets", {})
    if not markets:
        return ""

    partes = ["\n### CUOTAS BETANO (tiempo real)"]
    partes.append(f"**{result.get('home_team', '?')} vs {result.get('away_team', '?')}**")
    partes.append(f"Mercados disponibles: {result.get('total_markets_available', '?')}")
    partes.append("")

    priority = ["1X2", "Over/Under", "BTTS", "Double Chance", "Asian Handicap",
                "Handicap", "Corners", "Cards", "Yellow Cards", "Shots",
                "Shots on Target", "Fouls", "1X2 1H", "Over/Under 1H",
                "BTTS 1H", "Correct Score"]

    for market_name in priority:
        m = markets.get(market_name)
        if not m:
            continue
        selections = m.get("selections", [])
        if not selections:
            continue
        odds_str = " | ".join(f"{s.get('name', '?')}: @{s.get('odd', '?')}" for s in selections)
        partes.append(f"- {market_name}: {odds_str}")

    other = [k for k in markets if k not in priority]
    if other:
        partes.append(f"\nOtros mercados: {', '.join(other)}")

    partes.append("\nIMPORTANTE: Estas cuotas son de Betano en tiempo real. Usalas como las cuotas OFICIALES del bookmaker.")
    partes.append("Ignora cualquier otra cuota (BSD) que pueda aparecer en este prompt. Las cuotas de Betano mandan.")

    return "\n".join(partes)
