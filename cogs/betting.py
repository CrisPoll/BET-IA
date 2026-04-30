import logging
import discord
from discord.ext import commands

from bsd_client import (
    obtener_proximos_partidos,
    obtener_detalle_partido,
    obtener_predicciones,
    resumir_datos_partido,
    resumir_prediccion,
)
from sofascore_client import enriquecer_datos_partido as enriquecer_sofascore
from odds_client import enriquecer_cuotas
from score365_client import enriquecer_datos_partido as enriquecer_365
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


class MatchSelectView(discord.ui.View):
    def __init__(self, matches: list, analyze_callback):
        super().__init__(timeout=600)
        self.matches = matches
        self.analyze_callback = analyze_callback
        self.page = 0
        self._build()

    def _build(self):
        self.clear_items()
        total_pages = max(1, (len(self.matches) - 1) // PER_PAGE + 1)
        start = self.page * PER_PAGE
        end = start + PER_PAGE

        select = discord.ui.Select(
            placeholder="Selecciona un partido para analizar...",
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
            prev_btn = discord.ui.Button(label="◀", disabled=self.page == 0)
            prev_btn.callback = self._prev
            self.add_item(prev_btn)

            page_btn = discord.ui.Button(
                label=f"{self.page + 1}/{total_pages}",
                disabled=True,
                style=discord.ButtonStyle.gray,
            )
            self.add_item(page_btn)

            next_btn = discord.ui.Button(
                label="▶", disabled=self.page == total_pages - 1
            )
            next_btn.callback = self._next
            self.add_item(next_btn)

    async def _on_select(self, interaction: discord.Interaction):
        match_id = int(interaction.data["values"][0])
        match = next((m for m in self.matches if m.get("id") == match_id), None)
        if not match:
            await interaction.response.send_message(
                "Partido no encontrado.", ephemeral=True
            )
            return
        self.stop()
        await interaction.response.edit_message(
            content="✅ Analizando partido seleccionado...", view=None
        )
        await self.analyze_callback(interaction.channel, match)

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
                self._match_cache = obtener_proximos_partidos()
            except Exception as e:
                await ctx.send(
                    f"❌ Error al obtener partidos: {e}\n"
                    "Verifica que `BSD_API_KEY` esté configurada en `.env`."
                )
                return

            if not self._match_cache:
                await ctx.send(
                    "No hay próximos partidos disponibles en este momento."
                )
                return

        total = len(self._match_cache)
        view = MatchSelectView(self._match_cache, self._analyze_match)
        await ctx.send(
            f"**Próximos partidos** ({total} encontrados)\n"
            "Selecciona uno del menú desplegable para analizar:",
            view=view,
        )

    @commands.command(name="analizar", aliases=["a"])
    async def analizar(self, ctx: commands.Context, match_id: int):
        async with ctx.typing():
            try:
                detalle = obtener_detalle_partido(match_id)
            except Exception as e:
                await ctx.send(
                    f"❌ Error al obtener el partido {match_id}: {e}"
                )
                return

            local = detalle.get("home_team", "?")
            visitante = detalle.get("away_team", "?")

            match = {
                "id": match_id,
                "home_team": local,
                "away_team": visitante,
                "_league_name": detalle.get("_league_name")
                or detalle.get("league", {}).get("name", "?"),
                "league": detalle.get("league", {}),
            }

            await self._analyze_match(ctx, match)

    async def _analyze_match(
        self, channel: discord.abc.Messageable, match: dict
    ):
        match_id = match.get("id")
        local = match.get("home_team", "?")
        visitante = match.get("away_team", "?")

        progress_msg = await channel.send(
            f"⏳ Analizando **{local} vs {visitante}**...\n"
            f"▫ 1/6 Obteniendo datos del partido..."
        )

        try:
            detalle = obtener_detalle_partido(match_id)
        except Exception as e:
            await progress_msg.edit(
                content=f"❌ Error al obtener detalle del partido: {e}"
            )
            return

        try:
            predicciones = obtener_predicciones(match_id=match_id)
            if predicciones:
                prediccion = predicciones[0]
            else:
                league_id = match.get("league", {}).get("id")
                predicciones = obtener_predicciones(league_id=league_id)
                prediccion = predicciones[0] if predicciones else {}
        except Exception as e:
            logger.warning(f"Error obteniendo predicciones: {e}")
            prediccion = {}

        datos_resumidos = resumir_datos_partido(detalle)
        datos_resumidos["_league_name"] = match.get(
            "_league_name", match.get("league", {}).get("name", "?")
        )
        datos_resumidos["partido"] = (
            f"{detalle.get('home_team', local)} vs "
            f"{detalle.get('away_team', visitante)}"
        )
        prediccion_resumida = resumir_prediccion(prediccion)

        await progress_msg.edit(
            content=f"⏳ Analizando **{local} vs {visitante}**...\n"
            f"✓ Datos del partido + predicción ML\n"
            f"▫ 2/6 Enriquiciendo con SofaScore..."
        )

        try:
            datos_resumidos = enriquecer_sofascore(datos_resumidos)
        except Exception as e:
            logger.warning(f"SofaScore falló: {e}")

        await progress_msg.edit(
            content=f"⏳ Analizando **{local} vs {visitante}**...\n"
            f"✓ Datos del partido + predicción ML + SofaScore\n"
            f"▫ 3/6 Enriquiciendo con 365Score..."
        )

        try:
            datos_resumidos = enriquecer_365(datos_resumidos)
        except Exception as e:
            logger.warning(f"365Score falló: {e}")

        await progress_msg.edit(
            content=f"⏳ Analizando **{local} vs {visitante}**...\n"
            f"✓ Datos del partido + predicción ML + SofaScore + 365Score\n"
            f"▫ 4/6 Enriquiciendo cuotas..."
        )

        try:
            datos_resumidos = enriquecer_cuotas(datos_resumidos)
        except Exception as e:
            logger.warning(f"Odds API falló: {e}")

        await progress_msg.edit(
            content=f"⏳ Analizando **{local} vs {visitante}**...\n"
            f"✓ Datos + predicción ML + SofaScore + 365Score + cuotas\n"
            f"▫ 5/6 Consultando a DeepSeek (esto puede tardar ~30s)..."
        )

        try:
            analisis = analizar_partido(datos_resumidos, prediccion_resumida)
        except Exception as e:
            await progress_msg.edit(
                content=f"❌ Error al consultar la IA: {e}"
            )
            return

        await progress_msg.edit(
            content=f"✅ Análisis completado: **{local} vs {visitante}**"
        )

        chunks = _split_response(analisis)
        for chunk in chunks:
            await channel.send(chunk)


def setup(bot: commands.Bot):
    bot.add_cog(BettingCog(bot))
