"""One gateway for every LLM: OpenAI-compatible calls with automatic fallback."""
import json
import os
import re

import httpx

from .config import PROVIDERS, chain

_THINK = re.compile(r"<think>.*?</think>", re.S)


async def chat(role, messages, json_mode=False, temperature=0.3, max_tokens=900, timeout=40):
    """Returns (text, 'provider:model'). Tries each model in the role's chain."""
    errors = []
    async with httpx.AsyncClient(timeout=timeout) as c:
        for prov, model in chain(role):
            p = PROVIDERS.get(prov)
            key = os.getenv(p["key"]) if p else None
            if not key:
                continue
            body = {"model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens}
            if json_mode and prov in ("groq", "gemini", "mistral", "cerebras"):
                body["response_format"] = {"type": "json_object"}
            try:
                r = await c.post(f"{p['base']}/chat/completions",
                                 headers={"Authorization": f"Bearer {key}"}, json=body)
                r.raise_for_status()
                text = (r.json()["choices"][0]["message"].get("content") or "")
                text = _THINK.sub("", text).strip()
                if text:
                    return text, f"{prov}:{model}"
            except Exception as e:  # rate limit, timeout, bad model name -> next
                errors.append(f"{prov}:{model} {str(e)[:70]}")
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
