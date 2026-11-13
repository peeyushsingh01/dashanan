# Dashaanan

Multi-agent AI ecosystem for Vercel (Python/FastAPI, serverless).

Flow: user -> Talker -> Compressor/Router -> specialists (research x4, stock x2, incident x1,
image x2, audio x2, video x2) -> Validators x4 (3-of-4 majority) -> Talker -> user.

## Deploy
1. Push this folder to GitHub.
2. Vercel -> New Project -> import repo (framework preset: Other).
3. Add env vars from `.env.example` (at least one LLM key; Tavily/Brave for research,
   HF_TOKEN for media, FINNHUB for fundamentals). Optionally add Upstash Redis from the Marketplace.
4. Deploy. Open `/` for the chat UI, `/api/health` to check configured providers.

CLI: `npm i -g vercel && vercel --prod`.  Local: `pip install -r requirements.txt fastapi[standard]`
then `vercel dev`.

## API
- `POST /api/chat` {message, history[]} -> {answer, media, sources, validation, trace}
- `POST /api/stt` raw audio body -> transcript (Groq Whisper)
- `GET /api/incidents` latest tracked incidents; `/api/cron/incidents` is run daily by Vercel Cron
- Set `DASHASHAM_API_KEY` to require header `x-api-key`.

## Notes
- Free-tier model names/limits change: override any role with `MODEL_<ROLE>` env vars.
- Request timeout is 60s (`vercel.json`). Video on free HF is slow/unreliable; media over ~3MB is
  rejected (add Vercel Blob to store bigger files).
- Hobby plan crons run once per day at most.
