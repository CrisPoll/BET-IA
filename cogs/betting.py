import asyncio
import logging
import re
import discord
from discord.ext import commands

from bsd_client import (
    obtener_proximos_partidos,
    obtener_detalle_partido,
    obtener_predicciones,
    resumir_datos_partido,
    resumir_prediccion,
)
from bsd_client_v2 import enriquecer_con_v2
from sofascore_client import enriquecer_datos_partido as enriquecer_sofascore, verificar_salud_sofascore, obtener_partidos_sofascore_only, obtener_datos_completos_sofascore
from betsafe_client import obtener_cuotas_betsafe_desde_url
from analyzer import analizar_partido

logger = logging.getLogger(__name__)

PER_PAGE = 25
DISCORD_MSG_LIMIT = 2000


def _split_response(text: str, limit: int = DISCORD_MSG_LIMIT) -> list:
    chunks = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def _parse_urls(args: str) -> tuple:
    """Parsea URLs de Betsafe y arbitro desde string de argumentos."""
    betsafe_url = ""
    arbitro_url = ""
    words = args.strip().split()
    for w in words:
        w = w.strip()
        if "betsafe" in w:
            betsafe_url = w
        elif any(site in w for site in ("whoscored", "transfermarkt", "sofascore")):
            arbitro_url = w
    return betsafe_url, arbitro_url


class MatchSelectView(discord.ui.View):
    def __init__(self, matches: list):
        super().__init__(timeout=600)
        self.matches = matches
        self.page = 0
        self._build()

    def _build(self):
        self.clear_items()
        total_pages = max(1, (len(self.matches) - 1) // PER_PAGE + 1)
        start = self.page * PER_PAGE
        end = start + PER_PAGE

        select = discord.ui.Select(
            placeholder="Selecciona un partido...",
            min_values=1,
            max_values=1,
        )
        for m in self.matches[start:end]:
            match_id = m.get("id", 0)
            local = m.get("home_team", "?")
            visitante = m.get("away_team", "?")
            liga = m.get("_league_name", "?")
            fecha = m.get("event_date", "")
            if len(fecha) > 16:
                fecha = fecha[:16].replace("T", " ")

            select.add_option(
                label=f"[{liga}] {local} vs {visitante}",
                description=fecha,
                value=str(match_id),
            )

        select.callback = self._on_select
        self.add_item(select)

        if total_pages > 1:
            prev_btn = discord.ui.Button(label="\u25c0", disabled=self.page == 0)
            prev_btn.callback = self._prev
            self.add_item(prev_btn)

            page_btn = discord.ui.Button(
                label=f"{self.page + 1}/{total_pages}",
                disabled=True,
                style=discord.ButtonStyle.gray,
            )
            self.add_item(page_btn)

            next_btn = discord.ui.Button(
                label="\u25b6", disabled=self.page == total_pages - 1
            )
            next_btn.callback = self._next
            self.add_item(next_btn)

    async def _on_select(self, interaction: discord.Interaction):
        match_id = int(interaction.data["values"][0])
        match = next((m for m in self.matches if m.get("id") == match_id), None)
        if not match:
            await interaction.response.send_message("Partido no encontrado.", ephemeral=True)
            return
        self.stop()
        local = match.get("home_team", "?")
        visitante = match.get("away_team", "?")
        await interaction.response.edit_message(
            content=(
                f"**{local} vs {visitante}** (ID: `{match_id}`)\n"
                f"Usa el comando:\n"
                f"`!a {match_id} <url_betsafe> <url_arbitro>`\n"
                f"Ejemplo: `!a {match_id} https://www.betsafe.pe/...?eventId=f-... https://es.whoscored.com/referees/...`"
            ),
            view=None,
        )

    async def _prev(self, interaction: discord.Interaction):
        self.page -= 1
        self._build()
        await interaction.response.edit_message(view=self)

    async def _next(self, interaction: discord.Interaction):
        self.page += 1
        self._build()
        await interaction.response.edit_message(view=self)


class BettingCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._match_cache: list = []

    @commands.command(name="partidos", aliases=["p"])
    async def partidos(self, ctx: commands.Context):
        async with ctx.typing():
            try:
                verificar_salud_sofascore(detallado=False)
                self._match_cache = obtener_proximos_partidos()
                partidos_ss = obtener_partidos_sofascore_only()
                self._match_cache.extend(partidos_ss)
                self._match_cache.sort(key=lambda p: p.get("event_date", "") if isinstance(p.get("event_date"), str) else "")
            except Exception as e:
                await ctx.send(
                    f"\u274c Error al obtener partidos: {e}\n"
                    "Verifica que `BSD_API_KEY` este configurada en `.env`."
                )
                return

            if not self._match_cache:
                await ctx.send("No hay proximos partidos disponibles en este momento.")
                return

        total = len(self._match_cache)
        view = MatchSelectView(self._match_cache)
        await ctx.send(
            f"**Proximos partidos** ({total} encontrados)\n"
            "Selecciona uno para ver el comando de analisis:",
            view=view,
        )

    @commands.command(name="analizar", aliases=["a"])
    async def analizar(self, ctx: commands.Context, match_id: int, *, args: str = ""):
        betsafe_url, arbitro_url = _parse_urls(args) if args else ("", "")

        # Buscar en el cache para saber si es SofaScore-only
        cached = None
        for m in self._match_cache:
            if m.get("id") == match_id:
                cached = m
                break

        # Si el cache dice que es SofaScore-only, usar ese flujo directamente
        if cached and cached.get("_source") == "sofascore_only":
            async with ctx.typing():
                await self._analyze_match(ctx, cached, betsafe_url, arbitro_url)
            return

        async with ctx.typing():
            try:
                detalle = obtener_detalle_partido(match_id)
            except Exception as e:
                # Si BSD falla (404) y no lo encontramos en cache, intentar con SofaScore
                if cached:
                    await self._analyze_match(ctx, cached, betsafe_url, arbitro_url)
                    return
                await ctx.send(
                    f"\u274c Error al obtener el partido {match_id}: {e}\n"
                    f"Si es un partido de una liga no cubierta por BSD, usa `!partidos` primero y volve a intentar."
                )
                return

            local = detalle.get("home_team", "?")
            visitante = detalle.get("away_team", "?")

            match = {
                "id": match_id,
                "home_team": local,
                "away_team": visitante,
                "_league_name": detalle.get("_league_name") or detalle.get("league", {}).get("name", "?"),
                "league": detalle.get("league", {}),
            }

            await self._analyze_match(ctx, match, betsafe_url, arbitro_url)

    async def _analyze_match(self, channel, match: dict, betsafe_url: str = "", arbitro_url: str = ""):
        match_id = match.get("id")
        local = match.get("home_team", "?")
        visitante = match.get("away_team", "?")
        es_sofascore_only = match.get("_source") == "sofascore_only"

        total_steps = 3 if es_sofascore_only else 5
        step = 0

        progress_msg = await channel.send(
            f"\u23f3 Analizando **{local} vs {visitante}**..."
        )

        async def _update(text: str):
            nonlocal step
            step += 1
            lines = progress_msg.content.split("\n")
            lines.append(f"\u25ab {step}/{total_steps} {text}")
            await progress_msg.edit(content="\n".join(lines))

        datos_resumidos = None
        prediccion_resumida = {}

        try:
            if es_sofascore_only:
                await _update("Obteniendo datos via SofaScore...")
                datos_resumidos = obtener_datos_completos_sofascore(match)
                ss = datos_resumidos.get("_sofascore", {})
                if not ss.get("disponible"):
                    await progress_msg.edit(content=f"\u274c SofaScore no disponible: {ss.get('error','?')}")
                    return
            else:
                await _update("Obteniendo datos BSD + prediccion ML...")
                detalle = obtener_detalle_partido(match_id)
                try:
                    predicciones = obtener_predicciones(match_id=match_id)
                    if predicciones:
                        prediccion = predicciones[0]
                    else:
                        league_id = match.get("league", {}).get("id")
                        predicciones = obtener_predicciones(league_id=league_id)
                        prediccion = predicciones[0] if predicciones else {}
                except Exception:
                    prediccion = {}

                datos_resumidos = resumir_datos_partido(detalle)
                datos_resumidos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"
                prediccion_resumida = resumir_prediccion(prediccion)

                await _update("Enriqueciendo con BSD v2...")
                try:
                    datos_resumidos = enriquecer_con_v2(datos_resumidos, match_id)
                    prediccion_v2 = datos_resumidos.pop("_v2_prediction_raw", {})
                    if prediccion_v2 and not prediccion_resumida:
                        prediccion_resumida = resumir_prediccion(prediccion_v2)
                except Exception:
                    pass

                await _update("Enriqueciendo con SofaScore...")
                try:
                    datos_resumidos = enriquecer_sofascore(datos_resumidos)
                except Exception:
                    pass

            # Betsafe
            if betsafe_url:
                await _update("Obteniendo cuotas Betsafe...")
                try:
                    cuotas = await asyncio.to_thread(obtener_cuotas_betsafe_desde_url, betsafe_url)
                    if cuotas and "error" not in cuotas and cuotas.get("markets"):
                        datos_resumidos["_cuotas"] = cuotas
                except Exception:
                    pass

            # Arbitro SofaScore
            if arbitro_url:
                await _update("Scrapeando datos del arbitro...")
                try:
                    if "sofascore.com" in arbitro_url:
                        from sofascore_client import enriquecer_arbitro_sofascore
                        datos_resumidos = enriquecer_arbitro_sofascore(datos_resumidos, arbitro_url)
                except Exception:
                    pass

            await _update("Consultando a DeepSeek...")
            analisis = await asyncio.to_thread(analizar_partido, datos_resumidos, prediccion_resumida)

        except Exception as e:
            await progress_msg.edit(content=f"\u274c Error: {e}")
            return

        await progress_msg.edit(content=f"\u2705 Analisis completado: **{local} vs {visitante}**")
        chunks = _split_response(analisis)
        for chunk in chunks:
            await channel.send(chunk)


def setup(bot: commands.Bot):
    bot.add_cog(BettingCog(bot))
