# Betting AI

Sistema de análisis de value betting para fútbol, potenciado por DeepSeek V4 Pro via OpenRouter.

Agrega datos de múltiples fuentes (BSD, SofaScore, Betsafe), aplica un **modelo predictivo cuantitativo propio** con regresión a la media y ajustes por contexto, envía todo a DeepSeek V4 Pro para generar proyecciones estadísticas y detectar oportunidades de valor en cuotas de apuestas. Incluye **persistencia SQLite**, **Kelly Criterion** para gestión de stakes, y **feedback loop** post-partido para medir MAE, ROI y calibración.

## Ligas cubiertas

Brasileirao Serie A, Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Champions League, Europa League, Copa Libertadores, Copa Sudamericana, mas ligas adicionales via SofaScore (sin requerir BSD).

## Fuentes de datos

| Fuente | Metodo | Datos |
|---|---|---|---|
| **BSD API v1** | REST API | Partidos, cuotas, predicciones ML (CatBoost), H2H, standings |
| **BSD API v2** | REST API | xG/min, shotmap, momentum, metadata (derby, clima, viaje), player-stats, managers, arbitros |
| **SofaScore** | API no oficial (curl_cffi) | Alineaciones, H2H detallado, form/performance, standings, stats de jugadores y equipo |
| **Betsafe** | Playwright (headless) | Cuotas en tiempo real de todos los mercados |

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

1. **Recolección de datos**: se junta información de BSD v1/v2, SofaScore y Betsafe.
2. **Modelo cuantitativo (`quant_model.py`)**: genera una proyección base usando xG, forma ponderada, posición en tabla, distancia de viaje, regresión a la media y Poisson simplificado.
3. **LLM (DeepSeek V4 Pro)**: recibe la base cuantitativa + contexto cualitativo y emite proyecciones numéricas ajustadas con detección de valor contra cuotas.
4. **Persistencia (`prediction_db.py`)**: se guardan proyecciones, cuotas, stakes y features.
5. **Post-partido (`post_match.py`)**: se ingresan resultados reales y el sistema calcula errores y ROI automáticamente.

## Estructura

```
betting-ai/
├── main.py                # Entrada por consola (con menú de performance)
├── post_match.py          # CLI para registrar resultados y calcular métricas
├── bot.py                 # Entrada del bot de Discord
├── analyzer.py            # Prompt monolítico + DeepSeek + integración cuantitativa
├── quant_model.py         # Modelo predictivo base (xG, forma, regresión, Poisson)
├── bankroll.py            # Kelly Criterion, calibración y recomendación de stakes
├── prediction_db.py       # SQLite: predicciones, resultados, apuestas, errores
├── bsd_client.py          # BSD API v1
├── bsd_client_v2.py       # BSD API v2
├── sofascore_client.py    # SofaScore
├── betsafe_client.py      # Betsafe (cuotas)
├── utils.py               # Normalización de nombres de equipos
├── cogs/                  # Comandos de Discord
├── output/                # Datos cacheados + base SQLite
└── requirements.txt
```
