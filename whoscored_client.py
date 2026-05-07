"""
WhoScored scraper - datos de arbitros para prediccion de tarjetas.

Extrae estadisticas via Playwright (headless Chromium):
  - Partidos arbitrados (total y por competicion)
  - Amarillas por partido (total y por competicion - clave para UCL vs liga local)
  - Rojas por partido
  - Faltas por partido
  - Penaltis por partido

Ejemplo URL: https://es.whoscored.com/referees/2356/show/joao-pinheiro
"""

import re
import asyncio
from playwright.async_api import async_playwright


def _extraer_stats(texto: str) -> dict:
    """Extrae los datos de la tabla de disciplina de WhoScored."""
    lineas = texto.split("\n")

    stats = {"competiciones": []}

    # Buscar la tabla de "Disciplina" > "General"
    in_table = False
    for i, line in enumerate(lineas):
        stripped = line.strip()

        if stripped == "Total / Promedio":
            # Agarrar la linea siguiente que tiene los totales
            if i + 1 < len(lineas):
                row = lineas[i + 1].strip().split("\t")
                if len(row) >= 8:
                    try:
                        stats["total_partidos"] = int(row[0])
                        stats["faltas_pp"] = float(row[1].replace(",", "."))
                        stats["faltas_pe"] = float(row[2].replace(",", "."))
                        stats["pen_pp"] = float(row[3].replace(",", "."))
                        stats["yc_pp"] = float(row[4].replace(",", "."))
                        stats["yc_total"] = int(row[5])
                        stats["rc_pp"] = float(row[6].replace(",", "."))
                        stats["rc_total"] = int(row[7])
                    except (ValueError, IndexError):
                        pass
            break

        if in_table and stripped and stripped[0].isdigit():
            # Es una fila de competicion: "13\t26.54\t0.79\t..."
            parts = stripped.split("\t")
            if len(parts) >= 8:
                # El nombre de la competicion esta en la linea anterior
                comp_name = lineas[i - 1].strip()
                try:
                    stats["competiciones"].append({
                        "nombre": comp_name,
                        "partidos": int(parts[0]),
                        "faltas_pp": float(parts[1].replace(",", ".")),
                        "yc_pp": float(parts[4].replace(",", ".")) if len(parts) > 4 else None,
                        "yc_total": int(parts[5]) if len(parts) > 5 else None,
                        "rc_pp": float(parts[6].replace(",", ".")) if len(parts) > 6 else None,
                        "rc_total": int(parts[7]) if len(parts) > 7 else None,
                    })
                except (ValueError, IndexError):
                    pass
            continue

        # Detectar inicio de la tabla de disciplina
        if "Jgdos\tFaltas pp\tFaltas pe\tPen pp\tAmar pp" in stripped:
            in_table = True

    return {k: v for k, v in stats.items() if v}


async def _scrapear_async(url: str) -> dict | None:
    """Scrapea los datos del arbitro desde WhoScored usando Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="es-ES",
        )
        page = await ctx.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(5000)
            text = await page.inner_text("body")
        except Exception:
            text = ""
        finally:
            await browser.close()

    if not text:
        return None

    # Extraer nombre del arbitro del title de la pagina
    nombre_match = re.search(r"<title>([^<]+)</title>", text)
    nombre = ""
    if not nombre:
        # Fallback: buscar texto grande antes de "Campeonatos" o "PARTIDOS"
        nombre_match = re.search(r"(\w[\w\s\.]{3,30}?)\n(?:Campeonatos|PARTIDOS)", text)
        if nombre_match:
            nombre = nombre_match.group(1).strip()

    stats = _extraer_stats(text)
    if not stats or not stats.get("total_partidos"):
        return None

    stats["nombre"] = nombre
    stats["fuente"] = "whoscored"
    return stats


def obtener_datos_arbitro_whoscored(url: str) -> dict | None:
    """Wrapper sincrono para scrapear WhoScored."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_scrapear_async(url))
        loop.close()
        return result
    except Exception:
        return None


def extraer_id_desde_url(url: str) -> str | None:
    """Extrae el ID de arbitro desde una URL de WhoScored."""
    m = re.search(r"/referees/(\d+)/", url)
    return m.group(1) if m else None


def enriquecer_arbitro_whoscored(datos_resumidos: dict, url: str) -> dict:
    """Enriquece los datos del arbitro con info de WhoScored."""
    try:
        ws_data = obtener_datos_arbitro_whoscored(url)
        if ws_data:
            merged = {**(datos_resumidos.get("arbitro") or {}), "_whoscored": ws_data}
            datos_resumidos["arbitro"] = merged
    except Exception:
        pass
    return datos_resumidos


def formatear_arbitro_whoscored_para_prompt(datos_resumidos: dict) -> str:
    """Formatea datos de WhoScored para el prompt de IA."""
    arb = datos_resumidos.get("arbitro", {})
    ws = arb.get("_whoscored", {})
    if not ws:
        return ""

    partes = [f"\n### ARBITRO (WhoScored): {ws.get('nombre', arb.get('nombre', '?'))}"]

    if ws.get("total_partidos"):
        partes.append(f"- Partidos dirigidos: {ws['total_partidos']}")

    yc = ws.get("yc_pp")
    if yc is not None:
        partes.append(f"- Amarillas/partido: {yc} (total: {ws.get('yc_total', '?')})")

    rc = ws.get("rc_pp")
    if rc is not None:
        partes.append(f"- Rojas/partido: {rc} (total: {ws.get('rc_total', '?')})")

    if ws.get("faltas_pp") is not None:
        partes.append(f"- Faltas/partido: {ws['faltas_pp']}")

    if ws.get("pen_pp") is not None:
        partes.append(f"- Penaltis/partido: {ws['pen_pp']}")

    # Desglose por competicion
    comps = ws.get("competiciones", [])
    if comps:
        partes.append("\n  Por competicion:")
        for c in comps:
            nombre = c.get("nombre", "?")
            partidos = c.get("partidos", "?")
            yc_c = c.get("yc_pp")
            rc_c = c.get("rc_pp")
            partes.append(f"    {nombre}: {partidos} part, {yc_c} YC/part, {rc_c} RC/part")

    partes.append("(Fuente: WhoScored - datos con desglose por competicion)")
    return "\n".join(partes)
