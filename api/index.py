import hmac
from urllib.parse import urlencode
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from dashasham import gateway, incident, media, orchestrator  # noqa: E402
from dashasham.config import DEFAULT_ROLES, PROVIDERS  # noqa: E402

app = FastAPI(title="Dashaanan")


@app.middleware("http")
async def restore_path(request: Request, call_next):
    """vercel.json rewrites /api/x -> /api/index?route=x; put the real path back."""
    route = request.query_params.get("route")
    if route is not None:
        request.scope["path"] = "/api/" + route.strip("/")
        request.scope["raw_path"] = request.scope["path"].encode()
        rest = [(k, v) for k, v in request.query_params.multi_items() if k != "route"]
        request.scope["query_string"] = urlencode(rest).encode()
    return await call_next(request)


r = APIRouter(prefix="/api")


def _auth(x_api_key):
    want = os.getenv("DASHASHAM_API_KEY")
    if want and not hmac.compare_digest(x_api_key or "", want):
        raise HTTPException(401, "invalid or missing x-api-key")


class Turn(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    history: list[Turn] = []


@r.get("/health")
async def health():
    return {"ok": True, "version": "v4-models-fix",
            "providers_configured": [p for p, c in PROVIDERS.items() if os.getenv(c["key"])],
            "roles": list(DEFAULT_ROLES)}


@r.get("/models")
async def models(x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    return await gateway.list_models()


@r.post("/chat")
async def chat(body: ChatIn, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    try:
        return await orchestrator.run(body.message, [t.model_dump() for t in body.history])
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@r.post("/stt")
async def stt(request: Request, x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    data = await request.body()
    if not data or len(data) > 4_000_000:
        raise HTTPException(400, "send raw audio bytes (<4MB) as the request body")
    return await media.speech_to_text(data, request.headers.get("content-type", ""))


@r.get("/incidents")
async def incidents(x_api_key: str | None = Header(None)):
    _auth(x_api_key)
    return await incident.latest()


@r.get("/cron/incidents")
async def cron(authorization: str | None = Header(None)):
    secret = os.getenv("CRON_SECRET")
    if secret and authorization != f"Bearer {secret}":
        raise HTTPException(401, "unauthorized")
    out = await incident.scan()
    return {"stored": len(out["incidents"])}


app.include_router(r)


@app.api_route("/{full_path:path}", methods=["GET", "POST"], include_in_schema=False)
async def no_route(request: Request, full_path: str):
    """Debug: shows what path Vercel actually delivered to the function."""
    hdrs = {k: v for k, v in request.headers.items()
            if k.startswith(("x-vercel", "x-forwarded", "x-matched", "x-now"))}
    return {"error": "no route", "seen_path": request.url.path,
            "query": dict(request.query_params), "headers": hdrs, "version": "v4-models-fix"}
