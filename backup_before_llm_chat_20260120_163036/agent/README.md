# City Survey Agent

This workspace contains a FastAPI backend and a Vite+React frontend to run the `llm_survey_eval` Tier 1–4 analyses and call Deepseek for professional analysis.

## Backend

- Location: `agent/backend`
- Run locally:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r agent/backend/requirements.txt
uvicorn agent.backend.main:app --host 0.0.0.0 --port 8000
```

Set environment variables in a `.env` file (preferred) or your environment.

1. Copy the example and edit values:

```powershell
copy agent\backend\.env.example agent\backend\.env
# then edit agent\backend\.env and fill your DEEPSEEK_API_KEY
```

2. The backend will automatically load `agent/backend/.env`. Required variables:

- `DEEPSEEK_API_URL` (e.g. `https://api.deepseek.example/v1/analyze`)
- `DEEPSEEK_API_KEY`

## Frontend

- Location: `agent/frontend`

```bash
cd agent/frontend
npm install
npm run dev
```

To deploy the frontend on Vercel: set `VITE_BACKEND_URL` environment variable in Vercel dashboard pointing to your deployed backend (e.g. `https://your-backend.example`). Then deploy the `agent/frontend` folder as a Vite app.

## Notes

- The backend relies on the local package `llm_survey_eval` included in the repository.
- Deepseek integration uses `DEEPSEEK_API_URL` and `DEEPSEEK_API_KEY`. Adjust the request format in `agent/backend/main.py` to match Deepseek's API if necessary.
