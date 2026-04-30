from sofascore_client import _normalizar_nombre

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
    na = _normalizar_nombre(a)
    nb = _normalizar_nombre(b)
    result = na == nb
    status = '✓' if result == expected else '✗'
    print(f'  {status} "{a}" → "{na}" vs "{b}" → "{nb}" = {result} (esperado: {expected})')

print('\n=== Prueba de conexión curl_cffi (sin hacer request real) ===')
try:
    from curl_cffi import requests as cffi_requests
    print('  ✓ curl_cffi importado')
    print('  Creando sesión con impersonate="chrome"...')
    session = cffi_requests.Session(impersonate="chrome")
    print('  ✓ Sesión creada')
except Exception as e:
    print(f'  ✗ Error: {e}')

print('\n=== Import de odds_client ===')
try:
    from odds_client import enriquecer_cuotas, _formatear_odds_para_prompt
    print('  ✓ odds_client importado correctamente')
except Exception as e:
    print(f'  ✗ Error: {e}')