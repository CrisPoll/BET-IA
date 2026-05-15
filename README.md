# Betting AI

Sistema de análisis de value betting para fútbol, potenciado por DeepSeek V4 Pro via OpenRouter.

Agrega datos de múltiples fuentes (BSD, SofaScore, Flashscore, Betsafe, WhoScored, Transfermarkt), aplica un **modelo predictivo cuantitativo propio** con regresión a la media y ajustes por contexto, envía todo a un pipeline de agentes LLM para generar proyecciones estadísticas y detectar oportunidades de valor en cuotas de apuestas. Incluye **persistencia SQLite**, **Kelly Criterion** para gestión de stakes, y **feedback loop** post-partido para medir MAE, ROI y calibración.

## Ligas cubiertas

Brasileirao Serie A, Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Champions League, Europa League, Copa Libertadores, Copa Sudamericana, mas ligas adicionales via SofaScore (sin requerir BSD).

## Fuentes de datos

| Fuente | Metodo | Datos |
|---|---|---|
| **BSD API v1** | REST API | Partidos, cuotas, predicciones ML (CatBoost), H2H, standings |
| **BSD API v2** | REST API | xG/min, shotmap, momentum, metadata (derby, clima, viaje), player-stats, planteles, managers |
| **SofaScore** | API no oficial (curl_cffi) | Alineaciones, H2H detallado, arbitro/managers/lesiones, form/performance, standings, stats de jugadores y equipo |
| **Flashscore** | Web scraping | Estadisticas de tarjetas (arbitros y equipos) |
| **Betsafe** | Playwright (headless) | Cuotas en tiempo real de todos los mercados |
| **WhoScored** | Playwright (opcional) | Estadisticas de arbitro (YC/RC/faltas por partido) |
| **Transfermarkt** | curl_cffi (opcional) | Datos de arbitros sudamericanos |
| **SofaScore (arbitro)** | API (opcional) | Estadisticas de arbitro |

## Analisis de IA

El modelo DeepSeek V4 Pro analiza y proyecta por equipo y por mitad:

- Tiros totales y al arco
- Goles por mitad (1T/2T)
- Tarjetas amarillas
- Corners
- Faltas
- Laterales

Luego compara las proyecciones contra las cuotas reales de Betsafe para identificar valor esperado positivo.

## Requisitos

- Python 3.10+
- Playwright browsers: `playwright install chromium`

## Instalacion

```bash
git clone <repo-url>
cd betting-ai
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
# Editar .env con tus API keys
```

## Variables de entorno

| Variable | Requerida | Descripcion |
|---|---|---|
| `OPENROUTER_API_KEY` | Si | API key de OpenRouter |
| `BSD_API_KEY` | Si | API key de BSD Sports Data |
| `DISCORD_BOT_TOKEN` | No | Token del bot de Discord |
| `DISCORD_HEALTH_WEBHOOK` | No | Webhook para alertas del health monitor |
| `USE_AGENT_PIPELINE` | No | `true` para usar pipeline de 3 agentes LLM en vez de prompt monolítico |
| `KELLY_FRACTION` | No | Fracción Kelly Criterion (default: 0.25) |
| `BANKROLL_UNITS` | No | Bankroll base en unidades (default: 1000) |

## Uso

### Consola interactiva

```bash
python main.py
```

Menu interactivo para cargar partidos, seleccionar uno y ejecutar el analisis completo con IA. Opcionalmente acepta `--show-prompt` para mostrar el prompt sin llamar a la API.

### Bot de Discord

```bash
python bot.py
```

Comandos:
- `!partidos` / `!p` — Lista partidos con selector paginado
- `!analizar <id> [url_betsafe] [url_arbitro]` / `!a` — Analiza un partido

### Ver prompt

```bash
python show_prompt.py
```

Muestra el prompt completo enviado a la IA sin consumir la API.

### Health monitor

```bash
python _health_monitor.py
```

Monitoreo de endpoints de SofaScore (diseñado para ejecutarse vía Task Scheduler cada 30 min).

### Feedback loop — Registrar resultados post-partido

```bash
python post_match.py
```

Menú interactivo para:
- Registrar resultados reales de un partido (goles, tiros, corners, tarjetas, faltas)
- Evaluar apuestas realizadas (ganada/perdida/push)
- Ver predicciones pendientes de resultado

Toda la información se guarda en `output/predictions.db` y se usa para calcular automáticamente:
- **MAE** (Mean Absolute Error) por métrica
- **Log-loss** y calibración de probabilidades
- **ROI** por mercado y global

## Cómo funciona el análisis (v2)

1. **Recolección de datos**: se junta información de BSD v1/v2, SofaScore, Flashscore y Betsafe.
2. **Modelo cuantitativo (`quant_model.py`)**: genera una proyección base usando xG, forma ponderada, posición en tabla, distancia de viaje, regresión a la media y Poisson simplificado.
3. **Pipeline de agentes LLM (`agents_pipeline.py`)**, si `USE_AGENT_PIPELINE=true`:
   - Agente de contexto: motivación, estilo, dinámica esperada.
   - Agente de proyección: valida la base cuantitativa y ajusta por factores cualitativos.
   - Agente de valor: detecta edge y sugiere stakes.
4. **Persistencia (`prediction_db.py`)**: se guardan proyecciones, cuotas, stakes y features.
5. **Post-partido (`post_match.py`)**: se ingresan resultados reales y el sistema calcula errores y ROI automáticamente.

## Estructura

```
betting-ai/
├── main.py                # Entrada por consola (con menú de performance)
├── post_match.py          # CLI para registrar resultados y calcular métricas
├── bot.py                 # Entrada del bot de Discord
├── analyzer.py            # Prompt + llamada a DeepSeek + integración cuantitativa
├── quant_model.py         # Modelo predictivo base (xG, forma, regresión, Poisson)
├── agents_pipeline.py     # Pipeline de 3 agentes LLM (contexto → proyección → valor)
├── bankroll.py            # Kelly Criterion, calibración y recomendación de stakes
├── prediction_db.py       # SQLite: predicciones, resultados, apuestas, errores
├── bsd_client.py          # BSD API v1
├── bsd_client_v2.py       # BSD API v2
├── sofascore_client.py    # SofaScore
├── flashscore_client.py   # Flashscore (tarjetas)
├── betsafe_client.py      # Betsafe (cuotas)
├── whoscored_client.py    # WhoScored (árbitro, opcional)
├── transfermarkt_client.py # Transfermarkt (árbitro, opcional)
├── utils.py               # Normalización de nombres de equipos
├── cogs/                  # Comandos de Discord
├── output/                # Datos cacheados + base SQLite
├── _archived/             # Clientes obsoletos
└── requirements.txt
```
