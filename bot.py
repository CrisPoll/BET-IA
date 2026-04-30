import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents,
)


@bot.event
async def on_ready():
    print(f"Bot conectado como {bot.user} ({bot.user.id})")
    print(f"Servidores: {len(bot.guilds)}")
    print(f"Comandos: {[c.name for c in bot.commands]}")


if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        raise ValueError(
            "DISCORD_BOT_TOKEN no configurada en .env. "
            "Crea un bot en https://discord.com/developers/applications"
        )
    bot.load_extension("cogs.betting")
    print(f"Comandos antes de run: {[c.name for c in bot.commands]}")
    bot.run(DISCORD_BOT_TOKEN)
