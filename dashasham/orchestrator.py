"""Talker -> Compressor/Router -> Specialists -> Validators -> Talker."""
import asyncio
import hashlib
import json
import os

from . import cache, gateway, incident, media, tools
from .config import DISCLAIMER, LIMITS
from .gateway import chat, parse_json

CLIP = lambda s, n=None: (s or "")[: n or LIMITS["evidence_chars"]]

COMPRESSOR = (
    "You are the Dashaanan router. Convert the user's request into a minimal JSON plan. "
    'Output ONLY JSON: {"tasks":[{"agent":"research|stock|incident|image|audio|video",'
    '"brief":"<=35 words, telegraphic","params":{}}]}. '
    "Use an empty tasks list for chat, writing, or explanations needing no tools. "
    "Use research for factual questions needing current or sourced info. "
    'Use stock only when a specific company or ticker is named (price, outlook, analysis); for stock ideas or screening without a name, use research. '
    'params: stock {"symbols":["AAPL"]} with Yahoo tickers (Indian NSE stocks end in .NS, e.g. RELIANCE.NS; BSE .BO); audio {"kind":"tts|music","text":"..."}; '
    "image/video: brief is the prompt."
)

TALKER = (
    "You are Dashaanan, a helpful assistant. Reply in the user's language, clearly and concisely. "
    "If EVIDENCE is provided, rely only on it for facts, cite sources as [n] or URLs, and say when "
    "something is uncertain. Never invent numbers or sources. Generated media is attached "
    "separately; just mention it.\n"
    "FORMAT: use simple Markdown for a phone screen: short paragraphs, bullet lists, **bold** for key "
    "terms. Prefer lists over tables; if you use a table, at most 3 columns with short cells.\n"
    "ABOUT YOURSELF: you are the Talker, one part of the Dashaanan system of 19 agents: "
    "1 Talker (you), 1 Compressor/Router, 2 image agents (Pollinations, FLUX), 2 audio agents "
    "(text-to-speech, music), 2 video agents, 4 research agents (web, academic, news, synthesis), "
    "2 stock agents (technical, fundamental), 1 incident tracker, 4 validators (different model "
    "families). Only the Talker speaks to the user; the others work behind the scenes. If asked "
    "about the system, describe this roster honestly. Do not claim other agents said or did "
    "anything unless EVIDENCE shows it."
)

VALIDATOR = (
    "You are a strict fact-checker. Given EVIDENCE and a DRAFT answer, check each factual claim "
    "against the evidence. Output ONLY JSON: "
    '{"verdict":"pass|fail|uncertain","unsupported":["claim", ...]}. '
    "pass = every factual claim is supported by the evidence or is common knowledge."
)


def _history(history):
    h = (history or [])[-LIMITS["history_turns"]:]
    return [{"role": m["role"], "content": CLIP(m["content"], LIMITS["history_chars"])}
            for m in h if m.get("role") in ("user", "assistant")]


async def compress(message, history):
    ctx = "\n".join(f"{m['role']}: {m['content']}" for m in _history(history)[-2:])
    raw, used = await chat("compressor", [
        {"role": "system", "content": COMPRESSOR},
        {"role": "user", "content": f"{ctx}\nuser: {CLIP(message, 1500)}"}], json_mode=True, max_tokens=400)
    plan = parse_json(raw) or {"tasks": []}
    tasks = [t for t in plan.get("tasks", []) if isinstance(t, dict) and t.get("agent")]
    return tasks[:4], used


# ---------- specialist agents ----------
def _src_block(items):
    return "\n".join(f"[{i+1}] {x['title']} ({x['url']}): {x['snippet']}" for i, x in enumerate(items))


async def _research_sub(role, label, items, brief):
    if not items:
        return label, None, []
    raw, _ = await chat(role, [
        {"role": "system", "content": "Summarise findings relevant to the task using ONLY the sources. "
                                      "<=150 words, cite as [n]. If sources lack the answer, say so."},
        {"role": "user", "content": f"TASK: {brief}\nSOURCES:\n{_src_block(items)}"}], max_tokens=450)
    return label, raw, items


async def research(brief, params):
    web, acad, news = await asyncio.gather(tools.web_search(brief), tools.arxiv(brief), tools.gdelt(brief))
    subs = await asyncio.gather(
        _research_sub("research_web", "web", web, brief),
        _research_sub("research_academic", "academic", acad, brief),
        _research_sub("research_news", "news", news, brief), return_exceptions=True)
    parts = [s for s in subs if not isinstance(s, Exception) and s[1]]
    if not parts:
        return {"text": "No research sources available (set TAVILY_API_KEY or BRAVE_API_KEY).", "sources": []}
    notes = "\n\n".join(f"## {l}\n{t}" for l, t, _ in parts)
    sources = [x for _, _, items in parts for x in items]
    raw, _ = await chat("research_synth", [
        {"role": "system", "content": "Merge the notes into one accurate briefing (<=250 words). "
                                      "Flag disagreements between sources. Keep source URLs for key claims."},
        {"role": "user", "content": f"TASK: {brief}\n{CLIP(notes, 3500)}"}], max_tokens=700)
    return {"text": raw, "sources": [{"title": s["title"], "url": s["url"]} for s in sources]}


async def stock(brief, params):
    symbols = [s.upper() for s in (params or {}).get("symbols", [])][:2]
    if not symbols:
        return {"text": "No ticker symbol identified.", "sources": []}
    tech = await asyncio.gather(*[tools.technicals(s) for s in symbols])
    fund = await asyncio.gather(*[tools.fundamentals(s) for s in symbols])
    (t_raw, _), (f_raw, _) = await asyncio.gather(
        chat("stock_technical", [
            {"role": "system", "content": "Technical analyst. Interpret ONLY the numbers given (trend, "
             "momentum, RSI, support/resistance). <=120 words. No buy/sell recommendation."},
            {"role": "user", "content": json.dumps(tech)}], max_tokens=400),
        chat("stock_fundamental", [
            {"role": "system", "content": "Fundamental analyst. Interpret ONLY the metrics/news given. "
             "<=120 words. No buy/sell recommendation."},
            {"role": "user", "content": CLIP(json.dumps(fund), 2500)}], max_tokens=400))
    srcs = [n for f in fund for n in f.get("news", [])]
    return {"text": f"Technical:\n{t_raw}\n\nFundamental:\n{f_raw}\n\nData: {json.dumps(tech)}\n{DISCLAIMER}",
            "sources": srcs}


async def incidents(brief, params):
    out = await incident.scan(brief or "breaking incident", store=False)
    return {"text": json.dumps(out["incidents"][:6]), "sources": [{"title": i["title"], "url": i.get("url", "")}
                                                                    for i in out["incidents"][:6]]}


async def dispatch(task):
    a, brief, params = task["agent"], task.get("brief", ""), task.get("params") or {}
    if a == "research": return a, await research(brief, params)
    if a == "stock": return a, await stock(brief, params)
    if a == "incident": return a, await incidents(brief, params)
    if a == "image": return a, {"media": await media.images(brief)}
    if a == "audio": return a, {"media": await media.audio(params.get("text") or brief, params.get("kind", "tts"))}
    if a == "video": return a, {"media": await media.video(brief)}
    return a, {"text": f"Unknown agent {a}"}


# ---------- validation ----------
async def validate(draft, evidence):
    msg = [{"role": "system", "content": VALIDATOR},
           {"role": "user", "content": f"EVIDENCE:\n{CLIP(evidence, 3500)}\n\nDRAFT:\n{CLIP(draft, 2500)}"}]
    res = await asyncio.gather(*[chat(f"validator_{i}", msg, json_mode=True, temperature=0, max_tokens=300, timeout=12)
                                 for i in range(1, 5)], return_exceptions=True)
    votes, issues = [], []
    for r in res:
        if isinstance(r, Exception):
            continue
        j = parse_json(r[0]) or {}
        votes.append({"model": r[1], "verdict": j.get("verdict", "uncertain")})
        issues += j.get("unsupported", [])[:3]
    passes = sum(v["verdict"] == "pass" for v in votes)
    need = 3 if len(votes) >= 4 else max(1, (len(votes) // 2) + 1)
    return {"passed": bool(votes) and passes >= need, "votes": votes, "issues": list(dict.fromkeys(issues))[:6]}


# ---------- main ----------
async def run(message, history=None):
    key = "dashaanan:chat:" + hashlib.sha1(message.strip().lower().encode()).hexdigest()
    if not history and (hit := await cache.get_json(key)):
        hit["cached"] = True
        return hit

    trace = []
    gateway.set_deadline(float(os.getenv("RUN_BUDGET_SECONDS", "50")))
    tasks, used = await compress(message, history)
    trace.append({"step": "compressor", "model": used, "tasks": [t["agent"] for t in tasks]})

    budget = max(8.0, (gateway.remaining() or 50) - 20)  # keep ~20s for talker + validators

    async def guarded(t):
        return await asyncio.wait_for(dispatch(t), timeout=budget)

    results = await asyncio.gather(*[guarded(t) for t in tasks], return_exceptions=True)
    evidence, media_out, sources = [], [], []
    for t, r in zip(tasks, results):
        if isinstance(r, Exception):
            trace.append({"step": t["agent"], "error": (str(r) or type(r).__name__)[:100]})
            continue
        agent, d = r
        trace.append({"step": agent, "ok": True})
        if d.get("text"): evidence.append(f"[{agent}]\n{CLIP(d['text'])}")
        media_out += d.get("media", [])
        sources += d.get("sources", [])
    if media_out:
        evidence.append("[media] generated: " + ", ".join(sorted({m['type'] for m in media_out})))

    ev = "\n\n".join(evidence)
    msgs = [{"role": "system", "content": TALKER}] + _history(history) + [
        {"role": "user", "content": (f"EVIDENCE:\n{ev}\n\n" if ev else "") + CLIP(message, 4000)}]
    draft, used = await chat("talker", msgs, max_tokens=1200, temperature=0.5)
    trace.append({"step": "talker", "model": used})

    validation = None
    if any(t["agent"] in ("research", "stock", "incident") for t in tasks) and (gateway.remaining() or 99) > 14:
        validation = await validate(draft, ev)
        trace.append({"step": "validators", "passed": validation["passed"], "votes": validation["votes"]})
        if not validation["passed"] and (gateway.remaining() or 99) <= 12:
            draft += "\n\n(Note: some details could not be fully verified.)"
        elif not validation["passed"]:
            fix = (f"Revise the DRAFT: remove or hedge these unsupported claims: {validation['issues']}. "
                   f"Use only EVIDENCE.\nEVIDENCE:\n{CLIP(ev,3500)}\nDRAFT:\n{draft}")
            draft, used = await chat("talker", [{"role": "system", "content": TALKER},
                                                {"role": "user", "content": fix}], max_tokens=1200, temperature=0.2)
            trace.append({"step": "revision", "model": used})
            draft += "\n\n(Note: some details could not be fully verified.)"

    if any(t["agent"] == "stock" for t in tasks) and DISCLAIMER not in draft:
        draft += f"\n\n{DISCLAIMER}"

    out = {"answer": draft, "media": media_out, "sources": sources[:10],
           "validation": validation, "trace": trace}
    if not history and not media_out:
        await cache.set_json(key, out, ttl=1800)
    return out
