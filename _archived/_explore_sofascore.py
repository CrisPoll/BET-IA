import json
from sofascore_client import _crear_sesion_sofascore, SOFASCORE_API, _normalizar_nombre
from bsd_client import obtener_proximos_partidos

partidos = obtener_proximos_partidos()
p = partidos[0]
home = p['home_team']
away = p['away_team']
fecha = p['event_date'][:10]

session = _crear_sesion_sofascore()
resp = session.get(f'{SOFASCORE_API}/sport/football/scheduled-events/{fecha}', timeout=15)
data = resp.json()

home_n = _normalizar_nombre(home)
away_n = _normalizar_nombre(away)

event = None
for e in data.get('events', []):
    eh = _normalizar_nombre(e.get('homeTeam', {}).get('name', ''))
    ea = _normalizar_nombre(e.get('awayTeam', {}).get('name', ''))
    if (home_n in eh or eh in home_n) and (away_n in ea or ea in away_n):
        event = e
        break

if event:
    ht = event.get('homeTeam', {})
    at = event.get('awayTeam', {})
    print('=== HOME TEAM ===')
    print(json.dumps({'id': ht.get('id'), 'name': ht.get('name')}, indent=2))
    print()
    print('=== AWAY TEAM ===')
    print(json.dumps({'id': at.get('id'), 'name': at.get('name')}, indent=2))
    print()

    home_tid = ht.get('id')
    print(f'=== SofaScore /team/{home_tid}/statistics ===')
    r = session.get(f'{SOFASCORE_API}/team/{home_tid}/statistics', timeout=15)
    if r.status_code == 200:
        stats = r.json()
        print('Top keys:', list(stats.keys())[:15])
        if 'statistics' in stats:
            if isinstance(stats['statistics'], list):
                for item in stats['statistics'][:5]:
                    keys = list(item.keys())
                    print(f'  item keys: {keys}')
            elif isinstance(stats['statistics'], dict):
                print(f'  statistics keys: {list(stats["statistics"].keys())[:10]}')
        print(json.dumps(stats, indent=2, ensure_ascii=False)[:2000])
    else:
        print(f'Status: {r.status_code}')

    print()
    print(f'=== SofaScore /team/{home_tid}/news ===')
    r = session.get(f'{SOFASCORE_API}/team/{home_tid}/news', timeout=15)
    print(f'Status: {r.status_code}')
    if r.status_code == 200:
        ndata = r.json()
        print('Top keys:', list(ndata.keys())[:10])
        if 'news' in ndata:
            print(f'News count: {len(ndata["news"])}')
            for n in ndata['news'][:2]:
                print(f'  Title: {n.get("title", "?")[:100]}')
                print(f'  Date: {n.get("updatedDateTimestamp", n.get("date", "?"))}')

    # Also try event news / pre-match preview
    eid = event.get('id')
    print()
    print(f'=== SofaScore /event/{eid}/incidents ===')
    r = session.get(f'{SOFASCORE_API}/event/{eid}/incidents', timeout=15)
    print(f'Status: {r.status_code}')
else:
    print('Partido no encontrado en SofaScore')
