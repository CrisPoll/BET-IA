"""
ValueStats scraper - datos de arbitros para prediccion de tarjetas.

ValueStats tiene datos muy superiores a BSD v2 para arbitros:
  - Partidos totales de carrera (BSD v2 solo mostro 1 partido)
  - YC/partido, RC/partido
  - Faltas/partido
  - Tarjetas por mitad (1H vs 2H)
  - Probabilidad de amarilla temprana (< min 28)
  - Ambos equipos reciben tarjeta (%)

Fuente: https://valuestats.com/arbitros
"""

import re
from datetime import datetime, timedelta

import requests
from curl_cffi import requests as curl_requests

from utils import normalizar_nombre

VALUESTATS_BASE = "https://valuestats.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es-ES,es;q=0.9",
}

# Cache simple: {nombre_arbitro: {"data": {...}, "fetched_at": datetime}}
_cache: dict = {}


def _fetch_html(url: str) -> str:
    """Fetch HTML con bypass TLS via curl_cffi (impersona Chrome)."""
    try:
        resp = curl_requests.get(
            url,
            headers=HEADERS,
            impersonate="chrome131",
            timeout=20,
        )
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ""


def _buscar_arbitro(nombre: str) -> dict | None:
    """
    Busca un arbitro en el listado de ValueStats por nombre.
    Retorna dict con {id, slug} o None.
    """
    # Limpiar el nombre para la busqueda: quitar comas, puntos, normalizar espacios
    query = nombre.replace(",", " ").replace(".", " ")
    query = re.sub(r"\s+", " ", query).strip()
    query = query.replace(" ", "+")
    url = f"{VALUESTATS_BASE}/arbitros?name={query}&only_next_matches=false&page=1"
    html = _fetch_html(url)
    if not html:
        return None

    # Buscar enlaces: href="/arbitro/{id}-{slug}"
    patron = re.compile(r'href="(/arbitro/(\d+)-([^"]+))"')
    matches = patron.findall(html)

    # Desduplicar por ID
    seen = set()
    candidates = []
    for full_url, arb_id, slug in matches:
        if arb_id not in seen:
            seen.add(arb_id)
            slug_name = slug.replace("-", " ")
            score = _name_similarity(nombre, slug_name)
            candidates.append((int(arb_id), slug, score))

    if not candidates:
        return None

    # Tomar el mejor match o el unico resultado
    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    if len(candidates) == 1 or best[2] > 0.3:
        return {"id": best[0], "slug": best[1]}

    return None


def _name_similarity(buscado: str, encontrado: str) -> float:
    """Similitud entre dos nombres de arbitro (0 a 1)."""
    b = normalizar_nombre(buscado.replace(",", " ").replace(".", " "))
    e = normalizar_nombre(encontrado)
    if not b or not e:
        return 0.0
    palabras_b = set(b.split())
    palabras_e = set(e.split())
    comunes = palabras_b & palabras_e
    total = max(len(palabras_b), len(palabras_e))
    return len(comunes) / total if total > 0 else 0.0


def _extraer_estadisticas(html: str) -> dict:
    """Extrae las estadisticas clave del HTML de la pagina de detalle."""

    def _float_val(patron, default=None):
        m = re.search(patron, html, re.DOTALL)
        if not m:
            return default
        return float(m.group(1).replace(",", "."))

    def _int_val(patron, default=None):
        m = re.search(patron, html, re.DOTALL)
        if not m:
            return default
        return int(m.group(1).replace(",", "."))

    # Total de partidos arbitrados
    total_partidos = _int_val(r"(\d+)\s*partidos\s*arbitrados")

    # Seccion "por partido" para YC y RC
    # "4.79\n/ partido" cerca de "Tarjetas Amarillas por partido"
    yc_section = re.search(
        r"Tarjetas\s*Amarillas\s*por\s*partido.*?([\d,.]+).*?en\s*total.*?(\d+)",
        html, re.DOTALL,
    )
    yc_avg = float(yc_section.group(1).replace(",", ".")) if yc_section else None
    yc_total = int(yc_section.group(2)) if yc_section else None

    rc_section = re.search(
        r"Tarjetas\s*Rojas\s*por\s*partido.*?([\d,.]+).*?en\s*total.*?(\d+)",
        html, re.DOTALL,
    )
    rc_avg = float(rc_section.group(1).replace(",", ".")) if rc_section else None
    rc_total = int(rc_section.group(2)) if rc_section else None

    # Amarillas 1H / 2H (buscar en la tabla de "Estadisticas por partido")
    yc_1h_match = re.search(
        r"Tarjetas\s*amarillas\s*\(1H\)\s*([\d,.]+)",
        html,
    )
    yc_1h_avg = float(yc_1h_match.group(1).replace(",", ".")) if yc_1h_match else None

    yc_2h_match = re.search(
        r"Tarjetas\s*amarillas\s*\(2H\)\s*([\d,.]+)",
        html,
    )
    yc_2h_avg = float(yc_2h_match.group(1).replace(",", ".")) if yc_2h_match else None

    # Faltas promedio
    fouls_match = re.search(r"Faltas\s*([\d,.]+)", html)
    fouls_avg = float(fouls_match.group(1).replace(",", ".")) if fouls_match else None

    # Amarilla antes del min 28: "8/20" en la columna
    early_yc_match = re.search(r"T\.amarilla\s*antes\s*del\s*min\s*28\s*(\d+)/(\d+)", html)
    early_yc_pct = None
    if early_yc_match:
        num, den = int(early_yc_match.group(1)), int(early_yc_match.group(2))
        early_yc_pct = round(num / den * 100, 1) if den > 0 else None

    # Ambos equipos 1+ tarjeta y 2+ tarjetas
    both_1 = re.search(r"Ambos\s*equipos\s*1\+\s*tarjeta\s*(\d+)/(\d+)", html)
    both_1_pct = None
    if both_1:
        num, den = int(both_1.group(1)), int(both_1.group(2))
        both_1_pct = round(num / den * 100, 1) if den > 0 else None

    both_2 = re.search(r"Ambos\s*equipos\s*2\+\s*tarjetas\s*(\d+)/(\d+)", html)
    both_2_pct = None
    if both_2:
        num, den = int(both_2.group(1)), int(both_2.group(2))
        both_2_pct = round(num / den * 100, 1) if den > 0 else None

    return {
        "total_partidos": total_partidos,
        "yc_total": yc_total,
        "yc_promedio": yc_avg,
        "rc_total": rc_total,
        "rc_promedio": rc_avg,
        "faltas_promedio": fouls_avg,
        "yc_1h_promedio": yc_1h_avg,
        "yc_2h_promedio": yc_2h_avg,
        "yc_antes_min28_pct": early_yc_pct,
        "ambos_equipos_1_tarjeta_pct": both_1_pct,
        "ambos_equipos_2_tarjetas_pct": both_2_pct,
    }


def obtener_datos_arbitro(nombre: str) -> dict | None:
    """
    Obtiene estadisticas completas de un arbitro desde ValueStats.
    Usa cache interno de 2 horas.
    """
    if not nombre:
        return None

    # Verificar cache
    cache_key = normalizar_nombre(nombre)
    if cache_key in _cache:
        entry = _cache[cache_key]
        if datetime.now() - entry["fetched_at"] < timedelta(hours=2):
            return entry["data"]

    info = _buscar_arbitro(nombre)
    if not info:
        _cache[cache_key] = {"data": None, "fetched_at": datetime.now()}
        return None

    url = f"{VALUESTATS_BASE}/arbitro/{info['id']}-{info['slug']}"
    html = _fetch_html(url)
    if not html:
        _cache[cache_key] = {"data": None, "fetched_at": datetime.now()}
        return None

    stats = _extraer_estadisticas(html)
    result = {"fuente": "valuestats", "arbitro_id": info["id"], **stats}
    result = {k: v for k, v in result.items() if v is not None}

    _cache[cache_key] = {"data": result, "fetched_at": datetime.now()}
    return result


def enriquecer_arbitro_valuestats(datos_resumidos: dict) -> dict:
    """
    Enriquece los datos del arbitro con info de ValueStats.
    Se llama despues de enriquecer_con_v2, en el pipeline principal.
    """
    arb = datos_resumidos.get("arbitro")
    if not arb or not arb.get("nombre"):
        return datos_resumidos

    try:
        vs_data = obtener_datos_arbitro(arb["nombre"])
        if vs_data:
            merged = {**(datos_resumidos.get("arbitro") or {}), "_valuestats": vs_data}
            datos_resumidos["arbitro"] = merged
    except Exception:
        pass

    return datos_resumidos


def formatear_arbitro_valuestats_para_prompt(datos_resumidos: dict) -> str:
    """Formatea datos de ValueStats para el prompt de IA."""
    arb = datos_resumidos.get("arbitro", {})
    vs = arb.get("_valuestats", {})
    if not vs:
        return ""

    partes = [f"\n### ARBITRO (ValueStats): {arb.get('nombre', '?')}"]

    if vs.get("total_partidos"):
        partes.append(f"- Partidos dirigidos: {vs['total_partidos']}")

    yc = vs.get("yc_promedio")
    if yc is not None:
        partes.append(f"- Amarillas/partido: {yc} (total: {vs.get('yc_total', '?')})")

    rc = vs.get("rc_promedio")
    if rc is not None:
        partes.append(f"- Rojas/partido: {rc}")

    if vs.get("faltas_promedio") is not None:
        partes.append(f"- Faltas/partido: {vs['faltas_promedio']}")

    if vs.get("yc_1h_promedio") is not None and vs.get("yc_2h_promedio") is not None:
        partes.append(f"- Amarillas 1H: {vs['yc_1h_promedio']} / 2H: {vs['yc_2h_promedio']}")

    if vs.get("yc_antes_min28_pct") is not None:
        partes.append(f"- Prob. amarilla antes del min 28: {vs['yc_antes_min28_pct']}%")

    if vs.get("ambos_equipos_1_tarjeta_pct") is not None:
        partes.append(f"- Ambos equipos +1 tarjeta: {vs['ambos_equipos_1_tarjeta_pct']}%")

    if vs.get("ambos_equipos_2_tarjetas_pct") is not None:
        partes.append(f"- Ambos equipos +2 tarjetas: {vs['ambos_equipos_2_tarjetas_pct']}%")

    return "\n".join(partes)
