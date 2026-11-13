"""Image / audio / video agents (two each). Free tiers: expect queues and failures."""
import asyncio
import base64
import os
import urllib.parse

import httpx

MAX_BYTES = 3_200_000  # Vercel response limit is ~4.5 MB
HF = "https://router.huggingface.co/hf-inference/models/"
IMAGE_MODELS = ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-xl-base-1.0"]
TTS_MODELS = ["hexgrad/Kokoro-82M", "facebook/mms-tts-eng"]
MUSIC_MODELS = ["facebook/musicgen-small", "stabilityai/stable-audio-open-1.0"]
VIDEO_MODELS = ["Lightricks/LTX-Video", "Wan-AI/Wan2.1-T2V-1.3B"]


async def _hf(model, prompt, kind, timeout=55):
    tok = os.getenv("HF_TOKEN")
    if not tok:
        return {"type": kind, "provider": model, "error": "HF_TOKEN not set"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(HF + model, headers={"Authorization": f"Bearer {tok}"},
                             json={"inputs": prompt})
            r.raise_for_status()
        if len(r.content) > MAX_BYTES:
            return {"type": kind, "provider": model, "error": "output too large; add Vercel Blob storage"}
        mime = r.headers.get("content-type", "application/octet-stream").split(";")[0]
        return {"type": kind, "provider": model,
                "data_uri": f"data:{mime};base64,{base64.b64encode(r.content).decode()}"}
    except Exception as e:
        return {"type": kind, "provider": model, "error": str(e)[:80]}


async def images(prompt):
    """Agent 1: Pollinations (keyless URL). Agent 2: FLUX.1-schnell on HF."""
    poll = {"type": "image", "provider": "pollinations",
            "url": f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?model=flux&nologo=true&width=768&height=768"}
    hf = await _hf(IMAGE_MODELS[0], prompt, "image")
    return [poll, hf]


async def audio(text, kind="tts"):
    models = MUSIC_MODELS if kind == "music" else TTS_MODELS
    res = await asyncio.gather(*[_hf(m, text, "audio") for m in models])
    return [r for r in res if "data_uri" in r][:1] or list(res)


async def video(prompt):
    res = await asyncio.gather(*[_hf(m, prompt, "video", timeout=58) for m in VIDEO_MODELS])
    return [r for r in res if "data_uri" in r][:1] or list(res)


async def speech_to_text(data: bytes, content_type: str):
    """Groq Whisper."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return {"error": "GROQ_API_KEY not set"}
    async with httpx.AsyncClient(timeout=50) as c:
        r = await c.post("https://api.groq.com/openai/v1/audio/transcriptions",
                         headers={"Authorization": f"Bearer {key}"},
                         data={"model": "whisper-large-v3-turbo"},
                         files={"file": ("audio", data, content_type or "audio/mpeg")})
        r.raise_for_status()
        return r.json()
