import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BSD_API_KEY = os.getenv("BSD_API_KEY")

_REQUIRED_VARS = {
    "DISCORD_BOT_TOKEN": DISCORD_BOT_TOKEN,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "BSD_API_KEY": BSD_API_KEY,
}

_missing = [k for k, v in _REQUIRED_VARS.items() if not v]
if _missing:
    raise ValueError(
        f"Variables de entorno faltantes en .env: {', '.join(_missing)}. "
        "Completa el archivo .env basado en .env.example"
    )

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
    bot.load_extension("cogs.betting")
    print(f"Comandos antes de run: {[c.name for c in bot.commands]}")
    bot.run(DISCORD_BOT_TOKEN)
