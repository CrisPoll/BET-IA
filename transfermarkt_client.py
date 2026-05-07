"""
Transfermarkt scraper - datos de arbitros para Sudamerica (Libertadores/Sudamericana).

Columnas extraidas:
  - Partidos arbitrados (total y por competicion)
  - Tarjetas amarillas
  - Tarjetas rojas
  - Penaltis

URL ejemplo: https://www.transfermarkt.es/guillermo-guerrero/profil/schiedsrichter/14446
"""

import re
import time

from curl_cffi import requests as curl_requests

TM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": "https://www.transfermarkt.es/",
}


def _extraer_id_desde_url(url: str) -> str | None:
    m = re.search(r"/schiedsrichter/(\d+)", url)
    return m.group(1) if m else None


def _fetch_html(url: str) -> str:
    try:
        time.sleep(2)
        resp = curl_requests.get(url, headers=TM_HEADERS, impersonate="chrome131", timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ""


def _parsear_stats(html: str) -> dict | None:
    """Extrae la tabla de disciplina del HTML de Transfermarkt."""
    # Buscar la tabla con los datos (la que contiene el tfoot con totales)
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    stats_table = None
    for t in tables:
        if "<tfoot>" in t and "zentriert" in t:
            stats_table = t
            break
    if not stats_table:
        return None

    # Extraer todas las filas
    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", stats_table, re.DOTALL)

    total = None
    competiciones = []
    nombre = ""

    for row_html in rows_html:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.DOTALL)
        # Limpiar cada celda
        clean = []
        for c in cells:
            # Extraer texto de enlaces
            link_texts = re.findall(r"<a[^>]*>([^<]+)</a>", c)
            # Extraer texto de spans con clase
            stripped = re.sub(r"<[^>]+>", "", c).strip().replace("&nbsp;", "")
            if stripped and stripped not in ("-", ""):
                clean.append(stripped)
            elif link_texts:
                clean.append(link_texts[0])

        if not clean:
            continue

        # Detectar si es fila de total (tfoot)
        if "<tfoot>" in row_html or (len(clean) == 4 and all(
            c.replace(".", "").isdigit() for c in clean
        )):
            # Puede tener 4 o 5 numeros: [partidos, amarillas, rojas, penaltis, ?]
            try:
                nums = [int(c) for c in clean]
                total = {
                    "partidos": nums[0],
                    "amarillas": nums[1] if len(nums) > 1 else None,
                    "rojas": nums[2] if len(nums) > 2 else None,
                    "penaltis": nums[3] if len(nums) > 3 else None,
                }
            except (ValueError, IndexError):
                pass
            continue

        # Detectar fila de competicion: nombre de competicion + numeros
        comp_name = None
        comp_nums = []
        for c in clean:
            if c.replace(".", "").replace(",", "").isdigit():
                comp_nums.append(int(c))
            elif not comp_name and not c.replace(".", "").isdigit():
                comp_name = c

        if comp_name and len(comp_nums) >= 3:
            competiciones.append({
                "nombre": comp_name,
                "partidos": comp_nums[0],
                "amarillas": comp_nums[1] if len(comp_nums) > 1 else None,
                "rojas": comp_nums[2] if len(comp_nums) > 2 else None,
                "penaltis": comp_nums[3] if len(comp_nums) > 3 else None,
            })

    if not total:
        return None

    return {
        "total": total,
        "competiciones": competiciones,
    }


def _extraer_nombre(html: str) -> str:
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return ""


def obtener_datos_arbitro_tm(url: str) -> dict | None:
    html = _fetch_html(url)
    if not html:
        return None

    stats = _parsear_stats(html)
    if not stats:
        return None

    nombre = _extraer_nombre(html)
    total = stats["total"]
    partidos = total["partidos"]

    result = {
        "fuente": "transfermarkt",
        "nombre": nombre,
        "total_partidos": partidos,
        "yc_total": total.get("amarillas"),
        "yc_pp": round(total["amarillas"] / partidos, 2) if partidos > 0 and total.get("amarillas") else None,
        "rc_total": total.get("rojas"),
        "rc_pp": round(total["rojas"] / partidos, 2) if partidos > 0 and total.get("rojas") else None,
        "penaltis_total": total.get("penaltis"),
        "penaltis_pp": round(total["penaltis"] / partidos, 2) if partidos > 0 and total.get("penaltis") else None,
        "competiciones": [],
    }

    for c in stats.get("competiciones", []):
        p = c["partidos"]
        result["competiciones"].append({
            "nombre": c["nombre"],
            "partidos": p,
            "yc_pp": round(c["amarillas"] / p, 2) if p > 0 and c.get("amarillas") else None,
            "rc_pp": round(c["rojas"] / p, 2) if p > 0 and c.get("rojas") else None,
        })

    return {k: v for k, v in result.items() if v not in (None, [], "")}


def enriquecer_arbitro_transfermarkt(datos_resumidos: dict, url: str) -> dict:
    try:
        tm_data = obtener_datos_arbitro_tm(url)
        if tm_data:
            merged = {**(datos_resumidos.get("arbitro") or {}), "_transfermarkt": tm_data}
            datos_resumidos["arbitro"] = merged
    except Exception:
        pass
    return datos_resumidos


def formatear_arbitro_transfermarkt_para_prompt(datos_resumidos: dict) -> str:
    arb = datos_resumidos.get("arbitro", {})
    tm = arb.get("_transfermarkt", {})
    if not tm:
        return ""

    partes = [f"\n### ARBITRO (Transfermarkt): {tm.get('nombre', arb.get('nombre', '?'))}"]

    if tm.get("total_partidos"):
        partes.append(f"- Partidos dirigidos: {tm['total_partidos']}")

    yc = tm.get("yc_pp")
    if yc is not None:
        partes.append(f"- Amarillas/partido: {yc} (total: {tm.get('yc_total', '?')})")

    rc = tm.get("rc_pp")
    if rc is not None:
        partes.append(f"- Rojas/partido: {rc} (total: {tm.get('rc_total', '?')})")

    if tm.get("penaltis_pp") is not None:
        partes.append(f"- Penaltis/partido: {tm['penaltis_pp']} (total: {tm.get('penaltis_total', '?')})")

    comps = tm.get("competiciones", [])
    if comps:
        partes.append("\n  Por competicion:")
        for c in comps:
            partes.append(
                f"    {c['nombre']}: {c.get('partidos', '?')} part, "
                f"{c.get('yc_pp', '?')} YC/part, {c.get('rc_pp', '?')} RC/part"
            )

    partes.append("(Fuente: Transfermarkt)")
    return "\n".join(partes)
