import os, traceback
from dotenv import load_dotenv
load_dotenv()

print("=== TESTING BOT SETUP ===")

import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

print("Loading cog...")
try:
    bot.load_extension("cogs.betting")
    print(f"OK - Commands: {[c.name for c in bot.commands]}")
    print(f"Cogs: {list(bot.cogs.keys())}")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()
