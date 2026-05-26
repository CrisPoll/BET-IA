"""
Betsafe client - cuotas completas en tiempo real.

Navega a la pagina del partido en Betsafe, intercepta las llamadas
API reales que hace el app React, y extrae TODOS los mercados con cuotas.

Hace click en TODOS los tabs de mercados para asegurar que el app
cargue absolutamente todos los mercados disponibles, incluyendo:
- Jugador (goleador, tiros, faltas, asistencias)
- Estadisticas del partido
- Handicaps
- Intervalos
- Especiales

Sin auth, sin cookies manuales - el browser se encarga de todo.
"""
import asyncio
import json
import re

from utils import normalizar_nombre

BETSAFE_BASE = "https://www.betsafe.pe"
BETSAFE_API_PREFIX = "/api/sb/v1/widgets/"

MARKET_LABELS = {
    "MW3W": "1X2",
    "MW3W2UPEP": "1X2 Early Payout",
    "DC": "Double Chance",
    "DNOB": "Draw No Bet",
    "BTTS": "BTTS",
    "BTTS1H": "BTTS 1H",
    "BTTS2H": "BTTS 2H",
    "FTCSR": "Correct Score",
    "HTFTFB": "Half/Full",
    "NGSNAB": "Clean Sheet Home",
    "AGSNAB": "Clean Sheet Away",
    "MWBHLF": "Win Both Halves",
    "M3WHCP": "Handicap 3-way",
    "MWOU": "Over/Under",
    "MGT": "Goals Total",
    "1HTG": "1H Goals Total",
    "2HTG": "2H Goals Total",
    "MTG2W": "Total Goals 2-way",
    "HWEH": "Home Win Either Half",
    "AWEH": "Away Win Either Half",
    "HTCS": "Half/Full Correct Score",
    "TOCO": "Total Corners",
    "TOYC": "Total Yellow Cards",
    "TORC": "Total Red Cards",
    "TOSG": "Total Shots on Goal",
    "TSTOUM": "Total Shots",
    "PLYPROPSHOT": "Player Shots",
    "PLYPROPGASM": "Player Goals",
    "SCMOFSH": "Method of Score",
    "FTCSWIN": "Score Cast",
}

ALL_TABS = [
    "Todos", "Populares", "Goles", "Tiempos", "Tiros de esquina",
    "Tarjetas", "Anotadores", "Estadísticas del jugador",
    "Estadísticas del partido", "Hándicaps", "Eventos del partido",
    "Marcador", "Intervalos", "Creador de Apuestas", "Pre-construida",
    "Especiales de jugador", "Carreras", "Apuestas Flash",
]


def _categorize_market(name: str) -> str:
    """Categoriza un mercado para agrupar en la salida."""
    nl = name.lower()
    if "ganador" in nl and "total" not in nl and "anotador" not in nl and "tiempo" not in nl:
        return "1X2"
    if nl.startswith("total de goles") or ("total de goles" in nl and any(c in nl for c in ["(", "mas", "menos"])):
        return "Over/Under"
    if "ambos equipos anotan" in nl:
        return "BTTS"
    if "handicap asiatico" in nl or "hándicap asiático" in nl:
        return "Asian Handicap"
    if "handicap" in nl and "tiro" not in nl and "tarjeta" not in nl:
        return "Handicap"
    if "tiro de esquina" in nl or "corner" in nl:
        return "Corners"
    if any(w in nl for w in ["tarjeta", "amonestado"]):
        return "Cards"
    if any(w in name for w in ["Anotador", "Goleador", "Hat-trick", "Hat Trick"]):
        return "Goalscorers"
    if "faltas cometidas" in nl:
        return "Player Stats"
    if "total de tiros" in nl or "tiros al arco" in nl:
        if "|" in name:
            return "Player Stats"
        return "Match Stats"
    if "fueras de juego" in nl or "asistencias" in nl:
        return "Player Stats"
    if "marcador" in nl:
        return "Correct Score"
    if "tiempo - ganador" in nl or "gol en ambos tiempos" in nl:
        return "Halves"
    if "ganador del partido despues" in nl or "minuto" in nl:
        return "Intervals"
    if "ganador del partido -" in name and "total" in nl:
        return "Combos"
    if any(w in name for w in ["Apuesta sin", "Doble oportunidad", "Gol en ambos arcos",
                                 "Fiesta de Goles", "Jugador recibe", "Libre directo"]):
        return "Specials"
    return "Other"


async def _obtener_cuotas_betsafe_async(home_team: str, away_team: str, event_id: str = None, competition: str = None) -> dict:
    """Obtiene TODAS las cuotas usando el mismo metodo que el scraper externo (headers + batch-fetch)."""
    from playwright.async_api import async_playwright

    api_responses = {}
    captured_headers: dict[str, str] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = await ctx.new_page()

        # ── Route: capture headers from event-market calls ──
        async def on_route(route):
            url = route.request.url
            if "event-market" in url and not captured_headers:
                headers = dict(route.request.headers)
                captured_headers.update({
                    k: v for k, v in headers.items()
                    if any(prefix in k.lower() for prefix in
                           ["x-sb-", "x-obg-", "brandid", "marketcode", "sessiontoken",
                            "accept", "content-type", "referer", "user-agent", "cookie"])
                })
            await route.continue_()

        await page.route("**/api/sb/v1/widgets/event-market/**", on_route)

        # ── Response: capture accordion + event-market data ──
        async def on_response(response):
            url = response.url
            try:
                body = await response.text()
                api_responses[url] = (response.status, body)
            except Exception:
                pass

        page.on("response", on_response)

        # ── Navigate ──
        if event_id:
            match_url = f"{BETSAFE_BASE}/es/apuestas-deportivas/buscar?eventId={event_id}&eti=0"
        else:
            match_url = f"{BETSAFE_BASE}/es/apuestas-deportivas/futbol"

        await page.goto(match_url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(12000)

        # ── If no event_id, try to find it ──
        if not event_id:
            event_id = await _find_event_id_on_page(page, home_team, away_team, competition)
            if not event_id:
                for _ in range(10):
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)
                    event_id = await _find_event_id_on_page(page, home_team, away_team, competition)
                    if event_id:
                        break
            if not event_id and competition:
                event_id = await _click_competition_filter(page, competition, home_team, away_team)
            if event_id:
                match_url = f"{BETSAFE_BASE}/es/apuestas-deportivas/buscar?eventId={event_id}&eti=0"
                await page.goto(match_url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(12000)

        # ── Click all tabs to trigger more API calls ──
        for name in ALL_TABS:
            try:
                el = page.locator(f"text={name}").first
                if await el.count() > 0:
                    await el.scroll_into_view_if_needed()
                    await el.click(timeout=2000)
                    await page.wait_for_timeout(1500)
            except Exception:
                pass

        for _ in range(5):
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)

        # ── Batch-fetch remaining market odds using captured headers ──
        accordion_markets: dict[str, dict] = {}
        event_market_data: dict[str, dict] = {}

        for url, (status, body) in api_responses.items():
            if status != 200:
                continue
            try:
                data = json.loads(body)
            except Exception:
                continue

            if "/accordion/v1" in url:
                accs = data.get("data", {}).get("accordions", {})
                for gid, gdata in accs.items():
                    for m in gdata.get("markets", []):
                        mid = m.get("id")
                        if mid:
                            accordion_markets[mid] = {
                                "name": m.get("marketFriendlyName") or m.get("label") or "",
                                "lineValue": m.get("lineValue", ""),
                                "marketTemplateId": m.get("marketTemplateId", ""),
                            }

            elif "/event-market/v1" in url:
                d = data.get("data", {})
                markets = d.get("markets", [])
                selections = d.get("marketSelections", [])
                sel_map = {}
                for s in selections:
                    sel_map.setdefault(s.get("marketId", ""), []).append(s)
                for m in markets:
                    mid = m.get("id")
                    if mid:
                        sl = sel_map.get(mid, [])
                        event_market_data[mid] = {
                            "name": m.get("marketFriendlyName") or m.get("label") or "",
                            "selections": [{"label": s.get("label", ""), "odd": s.get("odds", "")} for s in sl if s.get("odds") is not None],
                        }

        # Fetch odds for remaining markets
        remaining = [mid for mid in accordion_markets if mid not in event_market_data]
        if remaining and captured_headers:
            for i in range(0, len(remaining), 40):
                batch = remaining[i:i + 40]
                ids_param = ",".join(batch)
                api_url = f"{BETSAFE_BASE}/api/sb/v1/widgets/event-market/v1?includescoreboards=true&marketids={ids_param}"
                try:
                    resp = await page.request.get(api_url, headers=captured_headers)
                    if resp.ok:
                        body = await resp.json()
                        d = body.get("data", {})
                        markets = d.get("markets", [])
                        selections = d.get("marketSelections", [])
                        sm = {}
                        for s in selections:
                            sm.setdefault(s.get("marketId", ""), []).append(s)
                        for m in markets:
                            mid = m.get("id")
                            if mid:
                                sl = sm.get(mid, [])
                                event_market_data[mid] = {
                                    "name": m.get("marketFriendlyName") or m.get("label") or "",
                                    "selections": [{"label": s.get("label", ""), "odd": s.get("odds", "")} for s in sl if s.get("odds") is not None],
                                }
                except Exception:
                    pass

        await browser.close()

    # ── Merge accordion + event-market data ──
    all_markets = {}
    for mid, am in accordion_markets.items():
        em = event_market_data.get(mid, {})
        name = em.get("name") or am["name"]
        selections_list = em.get("selections", [])
        if not selections_list:
            continue

        lv = em.get("lineValue") or am.get("lineValue", "")
        full_label = name if not lv else f"{name} ({lv})"

        # Use a unique key to avoid merging different markets with same label
        unique_key = f"{mid}:{full_label}"

        all_markets[unique_key] = {
            "id": mid,
            "name": full_label,
            "category": _categorize_market(full_label),
            "selections": selections_list,
        }

    # Get event info
    event_name = f"{home_team} vs {away_team}"
    competition_name = ""
    for url, (status, body) in api_responses.items():
        if status != 200:
            continue
        try:
            data = json.loads(body)
        except Exception:
            continue
        if "/event/v2" in url:
            ev = data.get("data", {}).get("event", {})
            if ev:
                participants = ev.get("participants", [])
                h = participants[0].get("label", home_team) if len(participants) > 0 else home_team
                a = participants[1].get("label", away_team) if len(participants) > 1 else away_team
                home_team = h or home_team
                away_team = a or away_team
                event_name = f"{h} vs {a}"
                competition_name = ev.get("competitionName", "")

    if not all_markets:
        return {"error": f"No se encontro '{home_team} vs {away_team}' en Betsafe"}

    return {
        "event_id": event_id or "",
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition_name,
        "source": "betsafe",
        "markets": all_markets,
        "total_markets": len(all_markets),
    }


async def _find_event_id_on_page(page, home_team: str, away_team: str, competition: str = None) -> str | None:
    """Busca el event ID de Betsafe en la pagina actual usando DOM + texto."""
    home_norm = normalizar_nombre(home_team).lower()
    away_norm = normalizar_nombre(away_team).lower()

    # Buscar via DOM: todos los links/event-ids visibles
    try:
        ids = await page.evaluate("""
            () => {
                const ids = [];
                // Buscar data-event-id en cualquier elemento
                document.querySelectorAll('[data-event-id]').forEach(el => {
                    ids.push(el.getAttribute('data-event-id'));
                });
                // Buscar en links con eventId=
                document.querySelectorAll('a[href*="eventId="]').forEach(el => {
                    const m = el.href.match(/eventId=([\\w-]+)/);
                    if (m && !ids.includes(m[1])) ids.push(m[1]);
                });
                // Buscar en texto visible eventId=
                const text = document.body.innerText;
                const re = /eventId=([\\w-]+)/g;
                let match;
                while ((match = re.exec(text)) !== null) {
                    if (!ids.includes(match[1])) ids.push(match[1]);
                }
                return [...new Set(ids)];
            }
        """)

        for eid in ids[:20]:
            # Verificar si este event ID corresponde al partido buscado
            try:
                el = page.locator(f'[data-event-id="{eid}"]').first
                if await el.count() == 0:
                    el = page.locator(f'a[href*="eventId={eid}"]').first
                if await el.count() == 0:
                    continue
                context = (await el.inner_text()).lower()
                if home_norm in context and away_norm in context:
                    return eid
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: buscar en todo el texto de la pagina
    try:
        text = await page.evaluate("() => document.body.innerText")
        ids = re.findall(r'eventId=([\w-]+)', text)
        for eid in ids:
            idx = text.find(eid)
            if idx < 0:
                continue
            context = text[max(0, idx - 300):idx + 300].lower()
            if home_norm in context and away_norm in context:
                return eid
    except Exception:
        pass

    return None


async def _click_competition_filter(page, competition: str, home_team: str, away_team: str) -> str | None:
    """Intenta hacer click en el filtro de competicion y buscar el partido."""
    comp_lower = competition.lower()
    comp_map = {
        "champions league": ["champions", "uefa champions"],
        "europa league": ["europa league", "uefa europa"],
        "copa libertadores": ["libertadores", "copa libertadores"],
        "copa sudamericana": ["sudamericana", "copa sudamericana"],
        "premier league": ["premier league", "inglaterra"],
        "la liga": ["la liga", "laliga", "espana"],
        "bundesliga": ["bundesliga", "alemania"],
        "brasileirao serie a": ["brasileirao", "brasil"],
    }
    search_terms = comp_map.get(comp_lower, [competition])

    try:
        for term in search_terms:
            for sel in [f'text="{term}"', f'text="{term.title()}"', f'text="{term.upper()}"',
                        f'[aria-label*="{term}"]', f'a:has-text("{term}")']:
                el = page.locator(sel).first
                if await el.count() > 0:
                    try:
                        await el.scroll_into_view_if_needed()
                        await el.click(timeout=3000)
                        await page.wait_for_timeout(3000)
                        return await _find_event_id_on_page(page, home_team, away_team, competition)
                    except Exception:
                        continue
    except Exception:
        pass

    return None


def _find_event_in_text(text: str, home_team: str, away_team: str) -> str | None:
    """Fallback legacy: busca event ID en texto plano."""
    home_norm = normalizar_nombre(home_team).lower()
    away_norm = normalizar_nombre(away_team).lower()
    ids = re.findall(r'eventId=([\w-]+)', text)
    for eid in ids:
        idx = text.find(eid)
        context = text[max(0, idx - 200):idx + 200].lower()
        if home_norm in context and away_norm in context:
            return eid
    return None


def obtener_cuotas_betsafe(home_team: str, away_team: str, league: str | None = None, event_id: str = None) -> dict:
    """Wrapper sincrono para obtener TODAS las cuotas de Betsafe."""
    async def _run():
        return await _obtener_cuotas_betsafe_async(home_team, away_team, event_id, league)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_run())
        loop.close()
        return result
    except Exception as e:
        return {"error": str(e)}


def extract_event_id_desde_url(url: str) -> str | None:
    """Extrae el event ID de Betsafe desde una URL como:
    https://www.betsafe.pe/es/apuestas-deportivas?eventId=f-xxx&eti=0
    """
    m = re.search(r"eventId=([\w-]+)", url)
    if m:
        return m.group(1)
    if re.match(r"^[\w-]+$", url.strip()):
        return url.strip()
    return None


def obtener_cuotas_betsafe_desde_url(url: str) -> dict:
    """Wrapper para obtener cuotas de Betsafe directamente desde una URL del partido."""
    event_id = extract_event_id_desde_url(url)
    if not event_id:
        return {"error": f"No se pudo extraer eventId de: {url}"}
    return obtener_cuotas_betsafe("", "", event_id=event_id)


# ── Formateo para prompt ──

def _format_sel(s):
    return s.get("name") or s.get("label", "?")


def _is_team_total_market(name: str, suffix: str, home_team: str = "", away_team: str = "") -> bool:
    """Detecta mercados tipo 'Lanús - Total de tiros' sin mezclar mitades/jugadores."""
    normalized = re.sub(r"\s*[–—-]\s*", " - ", (name or "").strip().lower())
    normalized = re.sub(r"\s*\(\d+(?:[.,]\d+)?\)\s*$", "", normalized)
    suffix = suffix.lower()
    marker = f" - {suffix}"
    if not normalized.endswith(marker):
        return False
    if any(token in normalized for token in ["1er tiempo", "primer tiempo", "2º tiempo", "2do tiempo", "jugador", "|"]):
        return False

    team_part = normalized.rsplit(marker, 1)[0].strip()
    if not team_part or team_part == "total":
        return False

    teams = [home_team, away_team]
    team_norms = [normalizar_nombre(t).lower() for t in teams if t]
    if team_norms:
        team_part_norm = normalizar_nombre(team_part).lower()
        return any(team_part_norm in t or t in team_part_norm for t in team_norms)

    return True


def _is_team_total_shots_market(name: str, home_team: str = "", away_team: str = "") -> bool:
    return _is_team_total_market(name, "total de tiros", home_team, away_team)


def _is_team_total_corners_market(name: str, home_team: str = "", away_team: str = "") -> bool:
    return _is_team_total_market(name, "total de tiros de esquina", home_team, away_team)


def _formatear_cuotas_betsafe_para_prompt(result: dict) -> str:
    """Formatea cuotas de Betsafe para el prompt del analyzer - SOLO mercados de valor."""
    if not result or "error" in result:
        return ""

    markets = result.get("markets", {})
    if not markets:
        return ""

    partes = ["\n### CUOTAS BETSAFE (tiempo real)"]
    comp = result.get("competition", "")
    home_team = result.get("home_team", "")
    away_team = result.get("away_team", "")
    if comp:
        partes.append(f"Competicion: {comp}")
    partes.append("")

    # Only show high-value, predictor-friendly markets. Skip:
    # - Exact score, hat-tricks, "Fiesta de Goles", combo bets
    # - Per-player shots/fouls/offsides/assists/cards
    # - Interval winners after X minutes
    # - "Draw No Bet" per team, "Both Halves" variants
    # - Corner handicaps

    core_filters = {
        "1X2": [
            r"^ganador del partido$",
            r"^ganador del partido - pago anticipado$",
        ],
        "Over/Under": [
            r"^total de goles \(2\.5\)$",
            r"^total de goles \(3\.5\)$",
            r"^total asiático de goles \(2\.5\)$",
            r"^total asiático de goles \(2\.25\)$",
            r"^total asiático de goles \(2\.75\)$",
            r"^total asiático de goles \(2\)$",
            r"^total asiático de goles \(3\)$",
        ],
        "BTTS": [
            r"^ambos equipos anotan$",
        ],
        "Double Chance": [
            r"^doble oportunidad$",
        ],
        "Asian Handicap": [
            r"hándicap asiático \(0 - 1\)$",
            r"hándicap asiático \(0 - 1\.25\)$",
            r"hándicap asiático \(0 - 0\.75\)$",
            r"hándicap asiático \(0 - 0\.5\)$",
            r"hándicap asiático \(0 - 1\.5\)$",
        ],
        "Corners": [
            r"^total de tiros de esquina \(\d+(\.\d+)?\)$",
            r"^más tiros de esquina$",
        ],
        "Team Corners": [],
        "Cards": [
            r"^total de tarjetas \(4\.5\)$",
            r"^total de tarjetas \(3\.5\)$",
            r"^total de tarjetas \(5\.5\)$",
            r"^más tarjetas$",
            r"^más tarjetas \(3 opciones\)$",
        ],
        "Match Stats": [
            r"^total de tiros \(\d+(\.\d+)?\)$",
            r"^total de tiros al arco \(\d+(\.\d+)?\)$",
        ],
        "Team Match Stats": [],
        "Halves": [
            r"^1er tiempo - ganador$",
            r"^2º tiempo - ganador$",
            r"^gol en ambos tiempos$",
        ],
        "Specials": [
            r"^avanza$",
            r"^método de clasificación$",
        ],
    }

    for cat_key, patterns in core_filters.items():
        shown = []
        for mk, mdata in markets.items():
            name = mdata.get("name", "").lower()
            matches_filter = any(re.match(p, name) for p in patterns)
            if cat_key == "Team Match Stats":
                matches_filter = _is_team_total_shots_market(mdata.get("name", ""), home_team, away_team)
            elif cat_key == "Team Corners":
                matches_filter = _is_team_total_corners_market(mdata.get("name", ""), home_team, away_team)
            if matches_filter:
                selections = mdata.get("selections", [])
                if selections:
                    sels = []
                    for s in selections[:8]:
                        label = s.get("name") or s.get("label") or "?"
                        odd = s.get("odd", "?")
                        sels.append(f"{label}: @{odd}")
                    shown.append(f"  {mdata.get('name', mk)}: {' | '.join(sels)}")
        if shown:
            # Sort shot markets numerically by line value when possible.
            if cat_key in {"Corners", "Team Corners", "Match Stats", "Team Match Stats"}:
                def _sort_key(item):
                    m = re.search(r'\((\d+(?:\.\d+)?)\)', item)
                    if not m:
                        m = re.search(r'(?:más|mas|menos)\s+de\s+(\d+(?:\.\d+)?)', item, re.IGNORECASE)
                    return float(m.group(1)) if m else 999
                shown.sort(key=_sort_key)
            cat_label = {
                "1X2": "GANADOR", "Over/Under": "GOLES", "BTTS": "BTTS",
                "Double Chance": "DOBLE OPORTUNIDAD", "Asian Handicap": "HANDICAP ASIATICO",
                "Corners": "CORNERS", "Cards": "TARJETAS",
                "Team Corners": "CORNERS POR EQUIPO",
                "Match Stats": "TIROS", "Team Match Stats": "TIROS POR EQUIPO",
                "Halves": "TIEMPOS", "Specials": "CLASIFICACION",
            }.get(cat_key, cat_key.upper())
            partes.append(f"**{cat_label}**")
            for s in shown:
                partes.append(s)
            partes.append("")

    partes.append("IMPORTANTE: Cuotas oficiales de Betsafe en tiempo real.")

    return "\n".join(partes)
