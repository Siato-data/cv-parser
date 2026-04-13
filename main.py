import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

from extractor import extract, SUPPORTED
from llm import parse

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

MAX_FILE_MB = 10
COST_PER_1K_TOKENS = 0.000020


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    app.state.total_tokens = 0
    app.state.total_calls = 0
    logger.info("Prêt")
    yield
    cost = round(app.state.total_tokens * COST_PER_1K_TOKENS / 1000, 4)
    logger.info(f"Shutdown — {app.state.total_calls} appels, {app.state.total_tokens} tokens, ~${cost}")


app = FastAPI(title="CV Parser — AutoSiato", version="2.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health():
    cost = round(app.state.total_tokens * COST_PER_1K_TOKENS / 1000, 4)
    return {"status": "ok", "calls": app.state.total_calls, "tokens": app.state.total_tokens, "cost_usd": cost}


@app.post("/parse")
async def parse_cv(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
):
    filename = file.filename or "cv.pdf"
    ext = "." + filename.lower().rsplit(".", 1)[-1]

    if ext not in SUPPORTED:
        raise HTTPException(415, f"Format non supporté : {ext}")

    raw_bytes = await file.read()
    size_mb = len(raw_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, f"Fichier trop lourd : {size_mb:.1f} MB (max {MAX_FILE_MB} MB)")

    t0 = time.perf_counter()
    logger.info(f"→ {filename} ({size_mb:.2f} MB)")

    try:
        text = extract(raw_bytes, filename)
    except ValueError as e:
        raise HTTPException(422, str(e))

    try:
        data = parse(app.state.client, text, job_description)
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    tokens = data.pop("_tokens", 0)
    app.state.total_tokens += tokens
    app.state.total_calls += 1

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(f"✓ {filename} {elapsed}s {tokens} tokens")

    data["_meta"] = {
        "filename": filename,
        "size_mb": round(size_mb, 3),
        "processing_s": elapsed,
        "tokens_used": tokens,
        "cost_usd": round(tokens * COST_PER_1K_TOKENS / 1000, 5),
    }

    return data
