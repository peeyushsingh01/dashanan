"""Incident tracker: collect from GDELT / USGS / RSS, classify with a small model, store."""
import asyncio
import os
import time

from . import cache, tools
from .gateway import chat, parse_json

FEEDS = os.getenv("INCIDENT_FEEDS", "https://feeds.bbci.co.uk/news/world/rss.xml,https://news.un.org/feed/subscribe/en/news/all/rss.xml").split(",")
KEY = "dashaanan:incidents"

SYSTEM = ("You are an incident tracker. From the headlines, list real-world incidents "
          "(disasters, outages, conflicts, accidents, major security events). Output ONLY JSON: "
          '{"incidents":[{"title":"","category":"","severity":1-5,"summary":"<=25 words","url":""}]}. '
          "Use only the given items. Max 8 incidents, most severe first.")


async def scan(query="breaking incident OR disaster OR outage", store=True):
    news, quakes, feeds = await asyncio.gather(
        tools.gdelt(query), tools.usgs_quakes(), tools.rss([f.strip() for f in FEEDS if f.strip()]))
    items = (quakes + news + feeds)[:24]
    if not items:
        return {"updated": int(time.time()), "incidents": [], "note": "no feed data"}
    text = "\n".join(f"- {i['title']} | {i['url']} | {i['snippet'][:80]}" for i in items)
    try:
        raw, used = await chat("incident", [{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": text}],
                               json_mode=True, max_tokens=900)
        data = parse_json(raw) or {"incidents": []}
    except Exception as e:
        data, used = {"incidents": [], "error": str(e)[:120]}, None
    out = {"updated": int(time.time()), "model": used, "incidents": data.get("incidents", [])}
    if store and out["incidents"]:
        await cache.set_json(KEY, out, ttl=86400 * 2)
    return out


async def latest():
    return await cache.get_json(KEY) or await scan()
