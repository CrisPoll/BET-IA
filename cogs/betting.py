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
from bsd_client_v2 import enriquecer_con_v2
from sofascore_client import enriquecer_datos_partido as enriquecer_sofascore, verificar_salud_sofascore, obtener_partidos_sofascore_only, obtener_datos_completos_sofascore
from flashscore_client import enriquecer_datos_partido as enriquecer_flashscore
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
            await interaction.response.send_message(
                "Partido no encontrado.", ephemeral=True
            )
            return
        self.stop()
        await interaction.response.edit_message(
            content="\u2705 Analizando partido seleccionado...", view=None
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
        self._last_analysis: dict = {}  # {channel_id: {datos, prediccion, local, visitante}}

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
                    f"\u274c Error al obtener el partido {match_id}: {e}"
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
        es_sofascore_only = match.get("_source") == "sofascore_only"

        if es_sofascore_only:
            progress_msg = await channel.send(
                f"\u23f3 Analizando **{local} vs {visitante}**...\n"
                f"\u25ab 1/2 Obteniendo datos via SofaScore..."
            )
            try:
                datos_resumidos = obtener_datos_completos_sofascore(match)
            except Exception as e:
                await progress_msg.edit(content=f"\u274c Error: {e}")
                return
            ss = datos_resumidos.get("_sofascore", {})
            if not ss.get("disponible"):
                await progress_msg.edit(content=f"\u274c SofaScore no disponible: {ss.get('error','?')}")
                return
            prediccion_resumida = {}
            try:
                datos_resumidos = enriquecer_flashscore(datos_resumidos)
            except Exception:
                pass
            await progress_msg.edit(
                content=f"\u23f3 Analizando **{local} vs {visitante}**...\n"
                f"\u2713 SofaScore (datos completos)\n"
                f"\u25ab 2/2 Consultando a DeepSeek..."
            )

        else:
            progress_msg = await channel.send(
                f"\u23f3 Analizando **{local} vs {visitante}**...\n"
                f"\u25ab 1/3 Obteniendo datos del partido..."
            )
            try:
                detalle = obtener_detalle_partido(match_id)
            except Exception as e:
                await progress_msg.edit(content=f"\u274c Error al obtener detalle del partido: {e}")
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
            datos_resumidos["partido"] = f"{detalle.get('home_team', local)} vs {detalle.get('away_team', visitante)}"
            prediccion_resumida = resumir_prediccion(prediccion)
            await progress_msg.edit(
                content=f"\u23f3 Analizando **{local} vs {visitante}**...\n"
                f"\u2713 Datos del partido + prediccion ML\n"
                f"\u25ab 2/4 Enrichiendo con BSD v2..."
            )
            try:
                datos_resumidos = enriquecer_con_v2(datos_resumidos, match_id)
                prediccion_v2 = datos_resumidos.pop("_v2_prediction_raw", {})
                if prediccion_v2 and not prediccion_resumida:
                    prediccion_resumida = resumir_prediccion(prediccion_v2)
            except Exception as e:
                logger.warning(f"BSD v2 enrichment fallo: {e}")
            await progress_msg.edit(
                content=f"\u23f3 Analizando **{local} vs {visitante}**...\n"
                f"\u2713 Datos del partido + prediccion ML + BSD v2\n"
                f"\u25ab 3/4 Enrichiendo con SofaScore..."
            )
            try:
                datos_resumidos = enriquecer_sofascore(datos_resumidos)
            except Exception as e:
                logger.warning(f"SofaScore fallo: {e}")
            try:
                datos_resumidos = enriquecer_flashscore(datos_resumidos)
            except Exception:
                pass
            await progress_msg.edit(
                content=f"\u23f3 Analizando **{local} vs {visitante}**...\n"
                f"\u2713 Datos del partido + prediccion ML + BSD v2 + SofaScore\n"
                f"\u25ab 4/4 Consultando a DeepSeek..."
            )

        self._last_analysis[channel.id] = {
            "datos": datos_resumidos,
            "prediccion": prediccion_resumida,
            "local": local,
            "visitante": visitante,
        }

        try:
            analisis = analizar_partido(datos_resumidos, prediccion_resumida)
        except Exception as e:
            await progress_msg.edit(content=f"\u274c Error al consultar la IA: {e}")
            return

        await progress_msg.edit(content=f"\u2705 Analisis completado: **{local} vs {visitante}**")
        chunks = _split_response(analisis)
        for chunk in chunks:
            await channel.send(chunk)

    @commands.command(name="cuotas", aliases=["c"])
    async def cuotas(self, ctx: commands.Context, url: str):
        """Agrega cuotas de Betsafe al ultimo analisis y re-ejecuta DeepSeek.
        Uso: !cuotas https://www.betsafe.pe/es/apuestas-deportivas?eventId=f-xxx&eti=0
        """
        last = self._last_analysis.get(ctx.channel.id)
        if not last:
            await ctx.send(
                "No hay un analisis previo en este canal.\n"
                "Usa `!partidos` o `!analizar <id>` primero."
            )
            return

        async with ctx.typing():
            cuotas = obtener_cuotas_betsafe_desde_url(url)
            if "error" in cuotas:
                await ctx.send(f"\u274c Error: {cuotas['error']}")
                return

            datos = last["datos"]
            datos["_cuotas"] = cuotas
            count = cuotas.get("total_markets", len(cuotas.get("markets", {})))
            await ctx.send(
                f"\u2705 {count} mercados de Betsafe cargados.\n"
                f"Re-analizando **{last['local']} vs {last['visitante']}**..."
            )

            try:
                analisis = analizar_partido(datos, last["prediccion"])
            except Exception as e:
                await ctx.send(f"\u274c Error al consultar la IA: {e}")
                return

            chunks = _split_response(analisis)
            for chunk in chunks:
                await ctx.send(chunk)


def setup(bot: commands.Bot):
    bot.add_cog(BettingCog(bot))
