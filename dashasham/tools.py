"""Data tools: web search, arXiv, GDELT, market data, feeds."""
import os
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import httpx

UA = {"User-Agent": "Mozilla/5.0 (Dashasham)"}


async def _get(url, **kw):
    headers = {**UA, **kw.pop("headers", {})}
    async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as c:
        return await c.get(url, **kw)


async def web_search(q, n=5):
    """Tavily first, Brave as fallback. Returns [{title,url,snippet}]."""
    try:
        if os.getenv("TAVILY_API_KEY"):
            async with httpx.AsyncClient(timeout=20) as c:
                r = await c.post("https://api.tavily.com/search",
                                 headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"},
                                 json={"query": q, "max_results": n})
                return [{"title": x["title"], "url": x["url"], "snippet": x["content"][:400]}
                        for x in r.json().get("results", [])]
        if os.getenv("BRAVE_API_KEY"):
            r = await _get("https://api.search.brave.com/res/v1/web/search",
                           params={"q": q, "count": n},
                           headers={"X-Subscription-Token": os.environ["BRAVE_API_KEY"]})
            return [{"title": x["title"], "url": x["url"], "snippet": x.get("description", "")[:400]}
                    for x in r.json().get("web", {}).get("results", [])]
    except Exception:
        pass
    return []


async def arxiv(q, n=5):
    try:
        r = await _get("https://export.arxiv.org/api/query",
                       params={"search_query": f"all:{q}", "max_results": n, "sortBy": "relevance"})
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            out.append({"title": " ".join(e.find("a:title", ns).text.split()),
                        "url": e.find("a:id", ns).text,
                        "snippet": " ".join(e.find("a:summary", ns).text.split())[:400]})
        return out
    except Exception:
        return []


async def gdelt(q, n=8):
    try:
        r = await _get("https://api.gdeltproject.org/api/v2/doc/doc",
                       params={"query": q, "mode": "artlist", "maxrecords": n,
                               "format": "json", "sort": "datedesc"})
        return [{"title": a.get("title", ""), "url": a.get("url", ""),
                 "snippet": f"{a.get('domain','')} {a.get('seendate','')}"}
                for a in r.json().get("articles", [])]
    except Exception:
        return []


async def usgs_quakes():
    try:
        r = await _get("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson")
        return [{"title": f["properties"]["title"], "url": f["properties"]["url"],
                 "snippet": f"magnitude {f['properties']['mag']}"} for f in r.json()["features"][:8]]
    except Exception:
        return []


async def rss(urls, per=5):
    out = []
    for u in urls:
        try:
            r = await _get(u)
            for i, it in enumerate(ET.fromstring(r.text).iter("item")):
                if i >= per:
                    break
                out.append({"title": (it.findtext("title") or "").strip(),
                            "url": (it.findtext("link") or "").strip(),
                            "snippet": (it.findtext("description") or "")[:200]})
        except Exception:
            continue
    return out


def _sma(v, n):
    return sum(v[-n:]) / n if len(v) >= n else None


def _rsi(v, n=14):
    if len(v) <= n:
        return None
    gains = [max(v[i] - v[i - 1], 0) for i in range(len(v) - n, len(v))]
    losses = [max(v[i - 1] - v[i], 0) for i in range(len(v) - n, len(v))]
    ag, al = sum(gains) / n, sum(losses) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


async def technicals(symbol):
    """Price + SMA20/50 + RSI14 from Yahoo's public chart endpoint."""
    try:
        r = await _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                       params={"range": "6mo", "interval": "1d"})
        res = r.json()["chart"]["result"][0]
        closes = [c for c in res["indicators"]["quote"][0]["close"] if c is not None]
        rnd = lambda x: None if x is None else round(x, 2)
        return {"symbol": symbol, "currency": res["meta"].get("currency"),
                "last": rnd(closes[-1]), "sma20": rnd(_sma(closes, 20)),
                "sma50": rnd(_sma(closes, 50)), "rsi14": rnd(_rsi(closes)),
                "high_6m": rnd(max(closes)), "low_6m": rnd(min(closes)),
                "chg_30d_pct": rnd((closes[-1] / closes[-22] - 1) * 100) if len(closes) > 22 else None}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)[:80]}


async def fundamentals(symbol):
    key = os.getenv("FINNHUB_API_KEY")
    if not key:
        return {"symbol": symbol, "error": "FINNHUB_API_KEY not set"}
    try:
        today = date.today()
        m = (await _get("https://finnhub.io/api/v1/stock/metric",
                        params={"symbol": symbol, "metric": "all", "token": key})).json().get("metric", {})
        news = (await _get("https://finnhub.io/api/v1/company-news",
                           params={"symbol": symbol, "from": str(today - timedelta(days=7)),
                                   "to": str(today), "token": key})).json()[:5]
        keep = ["peBasicExclExtraTTM", "epsTTM", "marketCapitalization", "roeTTM",
                "revenueGrowthTTMYoy", "52WeekHigh", "52WeekLow", "beta"]
        return {"symbol": symbol, "metrics": {k: m.get(k) for k in keep},
                "news": [{"title": n["headline"], "url": n["url"]} for n in news]}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)[:80]}
