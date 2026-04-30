import re
import unicodedata

ABREVIATURAS = {
    "psg": "paris saint-germain",
    "fcb": "barcelona",
    "rma": "real madrid",
    "fc bayern": "bayern munich",
}

SUFIJOS_GENERICOS = [
    " football club",
    " futbol club",
    " association football club",
]


def normalizar_nombre(nombre: str) -> str:
    if not nombre:
        return ""

    n = unicodedata.normalize("NFKD", nombre)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().strip()

    for s in SUFIJOS_GENERICOS:
        if n.endswith(s):
            n = n[:-len(s)].strip()
            break

    for s in [" fc", " cf", " sc", " ac", " afc"]:
        if n.endswith(s):
            remaining = n[:-len(s)].strip()
            if len(remaining) >= 5:
                n = remaining
                break

    for p in ["fc ", "cf ", "sc ", "ac "]:
        if n.startswith(p):
            n = n[len(p):].strip()
            break

    n = n.replace("saint germain", "saint-germain")
    n = re.sub(r"\s+", " ", n)

    if n in ABREVIATURAS:
        n = ABREVIATURAS[n]

    return n
