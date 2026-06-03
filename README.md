# Betting AI

Sistema de analisis de mercados estadisticos para futbol. Combina datos de BSD Sports Data, SofaScore y Betano con una proyeccion cuantitativa propia y un LLM via OpenRouter para generar analisis orientados a mercados como tiros, tiros al arco, corners, tarjetas, faltas y goles.

> Este proyecto es una herramienta de analisis. No garantiza resultados ni debe tomarse como asesoramiento financiero.

## Que Hace

- Lista partidos cubiertos por las ligas configuradas en BSD y partidos extra desde SofaScore cuando aplica.
- Enriquece cada partido con forma reciente, tabla, contexto competitivo, alineaciones, bajas, arbitro, H2H y estadisticas por partido.
- Agrega medias por jugador cuando hay datos: tiros/90, tiros al arco/90, xG/90, xA/90, key passes, tarjetas, faltas, acciones defensivas y atajadas.
- Obtiene cuotas reales de Betano desde la URL publica del evento, incluyendo remates/tiros al arco de jugador cuando estan disponibles.
- Genera una proyeccion base con `quant_model.py`.
- Envia el contexto a OpenRouter para producir una lectura final en formato apto para Discord.
- Clasifica cada pick recomendado con stake fijo de 5, 10 o 15 unidades segun fuerza del edge.
- Guarda predicciones, resultados y apuestas en SQLite para medir MAE, ROI y calibracion.

## Fuentes De Datos

| Fuente | Uso Principal |
|---|---|
| BSD v1/v2 | Partidos, prediccion ML, stats, standings, fixtures, lineups de respaldo, odds comparativas |
| SofaScore | Alineaciones, bajas, arbitro, tablas, H2H, forma reciente y estadisticas por partido |
| BSD v2 jugadores | Medias recientes por jugador y respaldo de plantillas/alineaciones |
| Betano | Cuotas oficiales del partido y mercados estadisticos |
| OpenRouter | Analisis contextual con LLM |
| SQLite | Persistencia de predicciones, resultados y apuestas |

## Requisitos

- Python 3.10+
- Una API key de OpenRouter
- Una API key de BSD Sports Data
- Chromium de Playwright para leer cuotas de Betano
- Token de Discord si quieres usar el bot

## Instalacion

```bash
git clone https://github.com/CrisPoll/BET-IA.git
cd BET-IA
pip install -r requirements.txt
playwright install chromium
copy .env.example .env
```

En Linux/macOS usa:

```bash
cp .env.example .env
```

Luego completa `.env`:

```env
OPENROUTER_API_KEY=<openrouter-api-key>
OPENROUTER_MODEL=anthropic/claude-opus-4.8
BSD_API_KEY=<bsd-api-key>
DISCORD_BOT_TOKEN=<discord-bot-token>
```

Variables opcionales:

```env
DISCORD_HEALTH_WEBHOOK=<discord-webhook-url>
KELLY_FRACTION=0.25
BANKROLL_UNITS=1000
```

## Uso Por Consola

Ejecuta el menu principal:

```bash
python main.py
```

Flujo recomendado:

1. Elige `Ver proximos partidos`.
2. Elige `Seleccionar partido y analizar`.
3. Pega la URL completa del evento de Betano cuando el programa la solicite.
4. Revisa el analisis final y las recomendaciones.

Para ver el prompt sin llamar al LLM:

```bash
python main.py --show-prompt
```

Tambien puedes usar:

```bash
python show_prompt.py
```

Opciones utiles de `show_prompt.py`:

```bash
python show_prompt.py --save=prompt.txt
python show_prompt.py --no-color
python show_prompt.py --help
```

## Uso Con Discord

Primero aseguralo en `.env`:

```env
DISCORD_BOT_TOKEN=<discord-bot-token>
OPENROUTER_API_KEY=<openrouter-api-key>
BSD_API_KEY=<bsd-api-key>
```

Arranca el bot:

```bash
python bot.py
```

Comandos disponibles:

```text
!partidos
!p
!analizar <id> [url_betano] [url_arbitro]
!a <id> [url_betano] [url_arbitro]
```

Ejemplo:

```text
!a 9335 https://www.betano.pe/cuotas-de-partido/...
```

El bot usa locks en `output/locks` para evitar dos analisis simultaneos del mismo partido.

## Registrar Resultados

Despues del partido puedes cargar el resultado real y evaluar apuestas:

```bash
python post_match.py
```

Esto actualiza la base SQLite en `output/predictions.db` y permite calcular:

- MAE por metrica
- ROI por mercado
- calibracion de probabilidades
- historial de predicciones recientes

## Ligas Soportadas

Las competiciones activas se definen en `competition_config.py`.

Actualmente el flujo principal cubre:

- Brasileirao Serie A
- Premier League
- La Liga
- Bundesliga
- Serie A
- Ligue 1
- Champions League
- Europa League
- Copa Libertadores
- Copa Sudamericana
- World Cup 2026
- International Friendly Games

SofaScore se usa para enriquecer datos. El universo principal de ligas lo define BSD.

## Estructura Del Proyecto

```text
betting-ai/
|-- main.py                # CLI principal
|-- show_prompt.py         # Vista del prompt y capas de datos sin llamar al LLM
|-- bot.py                 # Entrada del bot de Discord
|-- cogs/betting.py        # Comandos de Discord y locks de analisis
|-- analyzer.py            # Prompt, armado de contexto y llamada OpenRouter
|-- quant_model.py         # Proyeccion cuantitativa base
|-- bankroll.py            # Kelly, calibracion y evaluacion de mercados
|-- prediction_db.py       # Persistencia SQLite
|-- bsd_client.py          # Cliente BSD base
|-- bsd_client_v2.py       # Enriquecimiento BSD v2
|-- sofascore_client.py    # Cliente SofaScore
|-- betano_client.py       # Extraccion de cuotas Betano
|-- competition_config.py  # Registro central de ligas/torneos
|-- tests/                 # Pruebas unitarias
|-- output/                # DB, locks y archivos locales ignorados por git
`-- requirements.txt
```

## Pruebas Y Validacion

Ejecuta las pruebas unitarias:

```bash
python -m unittest discover -s tests
```

Valida sintaxis de los modulos principales:

```bash
python -m py_compile competition_config.py analyzer.py quant_model.py sofascore_client.py betano_client.py bsd_client.py bsd_client_v2.py cogs\betting.py main.py show_prompt.py
```

En PowerShell, si quieres reiniciar el bot y evitar procesos duplicados:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|py' -and $_.CommandLine -match 'bot.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Process -FilePath python -ArgumentList '.\bot.py' -WorkingDirectory 'D:\Projects\NuevoAgenIa\betting-ai' -WindowStyle Hidden
```

## Seguridad

- No subas `.env`.
- Usa `.env.example` solo con placeholders.
- Si alguna API key se imprime o se comparte por error, rotala.
- `output/` contiene datos locales y esta ignorado por git.
- `README.local.md` queda reservado para notas internas de trabajo y no se publica.
