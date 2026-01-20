import os
import tempfile
import json
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import math
import requests
from llm_survey_eval.summary import evaluate_all_tiers, summarise_tier_report

app = FastAPI(title="CitySurvey Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env from the backend folder if present (so you don't need to set vars in PowerShell)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_items=1)
    temperature: Optional[float] = 0.6


def call_deepseek(messages: list[dict], temperature: float = 0.6) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if msg.get("content"):
                return msg["content"]
    return str(data)


@app.post("/chat")
async def chat(payload: ChatRequest):
    try:
        reply = call_deepseek(
            messages=[m.model_dump() for m in payload.messages],
            temperature=payload.temperature or 0.6,
        )
        return JSONResponse(content={"reply": reply})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze")
async def analyze(
    ground_truth: UploadFile = File(...),
    generated: UploadFile = File(...),
    feature_schema: Optional[str] = Form(None),
    ordered_features: Optional[str] = Form(None),
    nominal_features: Optional[str] = Form(None),
):
    try:
        # save uploads to temp files
        tmpdir = tempfile.mkdtemp(prefix="city_agent_")
        gt_path = os.path.join(tmpdir, ground_truth.filename)
        gen_path = os.path.join(tmpdir, generated.filename)
        with open(gt_path, "wb") as f:
            f.write(await ground_truth.read())
        with open(gen_path, "wb") as f:
            f.write(await generated.read())

        # load into pandas
        try:
            human_df = pd.read_csv(gt_path)
            llm_df = pd.read_csv(gen_path)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Failed to read CSVs: {e}"})

        # parse optional JSON inputs
        try:
            schema = json.loads(feature_schema) if feature_schema else None
        except Exception:
            schema = None
        try:
            ordered = json.loads(ordered_features) if ordered_features else None
        except Exception:
            ordered = None
        try:
            nominal = json.loads(nominal_features) if nominal_features else None
        except Exception:
            nominal = None

        # Auto-detect if not provided (prefer example-like defaults)
        if ordered is None and nominal is None:
            preferred_ordered = [c for c in ["travel_time_min", "satisfaction"] if c in human_df.columns]
            preferred_nominal = [c for c in ["city", "mode", "trip_purpose"] if c in human_df.columns]
            if preferred_ordered and preferred_nominal:
                ordered = preferred_ordered
                nominal = preferred_nominal
            else:
                ordered = []
                nominal = []
                ordered_keywords = {"satisfaction", "rating", "score", "frequency"}
                for c in human_df.columns:
                    if pd.api.types.is_numeric_dtype(human_df[c]):
                        s = pd.to_numeric(human_df[c], errors="coerce").dropna()
                        is_int = not s.empty and (s % 1 == 0).all()
                        nunique = s.nunique()
                        if is_int and nunique <= 7 and not any(k in c.lower() for k in ordered_keywords):
                            nominal.append(c)
                        else:
                            ordered.append(c)
                    else:
                        nominal.append(c)

        # Remove ID-like columns from outcomes
        def _is_id_column(col: str) -> bool:
            if "id" not in col.lower():
                return False
            return human_df[col].nunique(dropna=True) == len(human_df)

        ordered = [c for c in ordered if not _is_id_column(c)]
        nominal = [c for c in nominal if not _is_id_column(c)]

        # Encode nominal columns to integer codes (required by tier eval)
        for col in nominal:
            combined = pd.concat([human_df[col], llm_df[col]], ignore_index=True)
            categories = sorted(combined.dropna().unique().tolist())
            human_df[col] = pd.Categorical(human_df[col], categories=categories).codes
            llm_df[col] = pd.Categorical(llm_df[col], categories=categories).codes

        # Ensure ordered columns are numeric
        for col in ordered:
            human_df[col] = pd.to_numeric(human_df[col], errors="coerce")
            llm_df[col] = pd.to_numeric(llm_df[col], errors="coerce")

        # Build a fallback feature schema for Tier 4 if none provided
        if not schema:
            schema = {}
            preferred = {
                "gender": "binary",
                "car_ownership": "binary",
                "season": "nominal",
                "education": "nominal",
                "job_type": "nominal",
                "income": "continuous",
                "age": "continuous",
                "household_size": "continuous",
            }
            for col, ftype in preferred.items():
                if col not in human_df.columns:
                    continue
                if _is_id_column(col):
                    continue
                if ftype == "nominal":
                    combined = pd.concat([human_df[col], llm_df[col]], ignore_index=True)
                    cats = sorted([c for c in combined.dropna().unique().tolist() if c != -1])
                    schema[col] = {"type": "nominal", "categories": cats}
                else:
                    schema[col] = {"type": ftype}

            if not schema:
                candidates = []
                for col in human_df.columns:
                    if _is_id_column(col):
                        continue
                    if pd.api.types.is_numeric_dtype(human_df[col]) and human_df[col].nunique(dropna=True) > 2:
                        candidates.append(col)
                for col in candidates[:3]:
                    schema[col] = {"type": "continuous"}

        # Ensure predictor columns are not treated as outcomes
        if schema:
            predictors = set(schema.keys())
            ordered = [c for c in ordered if c not in predictors]
            nominal = [c for c in nominal if c not in predictors]

        # Run evaluation (may be heavy)
        combined = evaluate_all_tiers(human_df, llm_df, schema or {}, ordered, nominal)
        report_text = summarise_tier_report(combined)

        cleaned = combined.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.where(pd.notnull(cleaned), None)

        def _sanitize(value):
            if isinstance(value, (list, tuple)):
                return [_sanitize(v) for v in value]
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            if isinstance(value, (float, np.floating)):
                return value if math.isfinite(value) else None
            return value

        payload = {
            "report": report_text,
            "combined_table": cleaned.to_dict(orient="records"),
        }
        return JSONResponse(content=jsonable_encoder(_sanitize(payload)))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {e}"})
