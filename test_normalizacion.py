from utils import normalizar_nombre

print('=== Normalización corregida ===')
tests = [
    ('FC Barcelona', 'Barcelona', True),
    ('fc barcelona', 'Barcelona', True),
    ('CF Barcelona', 'Barcelona', True),
    ('FC Bayern Munich', 'Bayern Munich', True),
    ('Paris Saint-Germain', 'PSG', True),
    ('Paris Saint Germain', 'PSG', True),
    ('Atlético Madrid', 'Atletico Madrid', True),
    ('Manchester City', 'Manchester United', False),
]
for a, b, expected in tests:
    na = normalizar_nombre(a)
    nb = normalizar_nombre(b)
    result = na == nb
    status = '✓' if result == expected else '✗'
    print(f'  {status} "{a}" → "{na}" vs "{b}" → "{nb}" = {result} (esperado: {expected})')

print('\n=== Import de apifootball_client ===')
try:
    from apifootball_client import obtener_datos_completos_partido, obtener_proximos_partidos
    print('  ✓ apifootball_client importado correctamente')
except Exception as e:
    print(f'  ✗ Error: {e}')

print('\n=== Import de analyzer ===')
try:
    from analyzer import analizar_partido
    print('  ✓ analyzer importado correctamente')
except Exception as e:
    print(f'  ✗ Error: {e}')
