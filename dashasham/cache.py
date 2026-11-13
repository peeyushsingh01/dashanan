"""Optional Upstash Redis (REST). Everything degrades gracefully if not configured."""
import json
import os

import httpx


def _cfg():
    url = os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("KV_REST_API_URL")
    tok = os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("KV_REST_API_TOKEN")
    return (url, tok) if url and tok else (None, None)


async def cmd(*args):
    url, tok = _cfg()
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.post(url, headers={"Authorization": f"Bearer {tok}"}, json=list(args))
            return r.json().get("result")
    except Exception:
        return None


async def get_json(key):
    v = await cmd("GET", key)
    try:
        return json.loads(v) if v else None
    except Exception:
        return None


async def set_json(key, value, ttl=3600):
    await cmd("SET", key, json.dumps(value), "EX", ttl)
