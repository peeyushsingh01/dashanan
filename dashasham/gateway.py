"""One gateway for every LLM: OpenAI-compatible calls with automatic fallback."""
import contextvars
import json
import os
import re
import time

import httpx

from .config import PROVIDERS, chain

_THINK = re.compile(r"<think>.*?</think>", re.S)

_DEADLINE = contextvars.ContextVar("deadline", default=None)


def set_deadline(seconds):
    """Start a time budget for the current request (Vercel kills functions at maxDuration)."""
    _DEADLINE.set(time.monotonic() + seconds)


def remaining():
    d = _DEADLINE.get()
    return None if d is None else d - time.monotonic()


async def chat(role, messages, json_mode=False, temperature=0.3, max_tokens=900, timeout=25):
    """Returns (text, 'provider:model'). Tries each model in the role's chain."""
    errors = []
    async with httpx.AsyncClient() as c:
        for prov, model in chain(role):
            rem = remaining()
            if rem is not None and rem < 4:
                errors.append("time budget used up")
                break
            t = timeout if rem is None else max(2, min(timeout, rem - 1))
            p = PROVIDERS.get(prov)
            key = os.getenv(p["key"]) if p else None
            if not key:
                continue
            body = {"model": model, "messages": messages, "max_tokens": max_tokens}
            if prov == "gemini":          # Gemini 3.x: thinking counts toward max_tokens
                body["reasoning_effort"] = "low"
                body["max_tokens"] = max(max_tokens, 1500)
            else:
                body["temperature"] = temperature
            if model.startswith("openai/gpt-oss"):   # reasoning model on Groq
                body["reasoning_effort"] = "low"
                body["max_tokens"] = max(max_tokens, 1200)
            if json_mode and prov in ("groq", "gemini", "mistral", "cerebras"):
                body["response_format"] = {"type": "json_object"}
            try:
                r = await c.post(f"{p['base']}/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"}, json=body, timeout=t)
                if r.status_code >= 400:
                    errors.append(f"{prov}:{model} HTTP {r.status_code} {r.text[:120]}")
                    continue
                text = (r.json()["choices"][0]["message"].get("content") or "")
                text = _THINK.sub("", text).strip()
                if text:
                    return text, f"{prov}:{model}"
                errors.append(f"{prov}:{model} empty reply")
            except Exception as e:  # timeout, bad JSON -> next
                errors.append(f"{prov}:{model} {type(e).__name__} {str(e)[:70]}")
    raise RuntimeError(f"No model available for '{role}'. Tried: {errors or 'no API keys set'}")


def parse_json(text):
    """Pull the first JSON object out of a model reply."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


async def list_models():
    """Diagnostic: which model IDs each configured provider currently serves."""
    out = {}
    async with httpx.AsyncClient(timeout=20) as c:
        for prov, p in PROVIDERS.items():
            key = os.getenv(p["key"])
            if not key:
                continue
            try:
                r = await c.get(f"{p['base']}/models", headers={"Authorization": f"Bearer {key}"})
                ids = [m["id"].replace("models/", "") for m in r.json().get("data", [])]
                if prov == "openrouter":
                    ids = [i for i in ids if i.endswith(":free") or i == "openrouter/free"]
                out[prov] = sorted(ids)[:80]
            except Exception as e:
                out[prov] = f"error: {str(e)[:80]}"
    return out
