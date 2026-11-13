"""Providers and per-role model chains. Free tiers change often: verify names,
or override any role with env MODEL_<ROLE>="provider:model,provider:model"."""
import os

PROVIDERS = {
    "gemini": {"base": "https://generativelanguage.googleapis.com/v1beta/openai", "key": "GEMINI_API_KEY"},
    "groq": {"base": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY"},
    "openrouter": {"base": "https://openrouter.ai/api/v1", "key": "OPENROUTER_API_KEY"},
    "mistral": {"base": "https://api.mistral.ai/v1", "key": "MISTRAL_API_KEY"},
    "cerebras": {"base": "https://api.cerebras.ai/v1", "key": "CEREBRAS_API_KEY"},
}

G_FLASH = ("gemini", "gemini-2.5-flash")
G_LITE = ("gemini", "gemini-2.5-flash-lite")
LLAMA70 = ("groq", "llama-3.3-70b-versatile")
LLAMA8 = ("groq", "llama-3.1-8b-instant")
OR_LLAMA = ("openrouter", "meta-llama/llama-3.3-70b-instruct:free")
OR_R1 = ("openrouter", "deepseek/deepseek-r1:free")
OR_QWEN = ("openrouter", "qwen/qwen3-235b-a22b:free")
OR_MISTRAL = ("openrouter", "mistralai/mistral-small-3.2-24b-instruct:free")
MISTRAL = ("mistral", "mistral-small-latest")
CEREBRAS = ("cerebras", "llama-3.3-70b")

DEFAULT_ROLES = {
    # 1 talker, 1 compressor/router
    "talker": [G_FLASH, LLAMA70, OR_LLAMA],
    "compressor": [LLAMA8, G_LITE, CEREBRAS],
    # 4 deep-research agents
    "research_web": [OR_QWEN, LLAMA70, G_FLASH],
    "research_academic": [OR_R1, G_FLASH, LLAMA70],
    "research_news": [G_FLASH, MISTRAL, OR_LLAMA],
    "research_synth": [OR_R1, G_FLASH, LLAMA70],
    # 2 stock agents, 1 incident tracker
    "stock_technical": [OR_R1, LLAMA70, G_FLASH],
    "stock_fundamental": [G_FLASH, OR_QWEN, LLAMA70],
    "incident": [LLAMA8, G_LITE, CEREBRAS],
    # 4 validators, four different model families
    "validator_1": [G_FLASH, G_LITE],
    "validator_2": [LLAMA70, CEREBRAS, OR_LLAMA],
    "validator_3": [MISTRAL, OR_MISTRAL],
    "validator_4": [OR_QWEN, OR_R1, LLAMA8],
}

LIMITS = {"evidence_chars": 2500, "history_turns": 4, "history_chars": 300}

DISCLAIMER = "Not financial advice. Analysis is informational only."


def chain(role: str):
    override = os.getenv("MODEL_" + role.upper())
    if override:
        return [tuple(x.strip().split(":", 1)) for x in override.split(",") if ":" in x]
    return DEFAULT_ROLES[role]
