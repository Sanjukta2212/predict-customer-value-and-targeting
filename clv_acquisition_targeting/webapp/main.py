"""
CSR web app: insurance-friendly form → churn + CLV with streamed calculation steps.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from webapp.prediction import build_calculation_trace, train_model

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CustomerPayload(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure_months: int = Field(..., ge=1, le=600, description="Months insured / on file")
    num_orders: int = Field(
        ...,
        ge=1,
        le=400,
        description="Count of billable interactions (claims, endorsements, payments)",
    )
    total_spend: float = Field(..., ge=0, description="Lifetime premiums or payments received")
    avg_order_value: float = Field(..., ge=0, description="Typical payment size")
    days_since_last_order: int = Field(..., ge=0, le=730, description="Days since last touchpoint")
    email_opens_30d: int = Field(..., ge=0, le=60, description="Communications opened (30 days)")
    app_sessions_30d: int = Field(..., ge=0, le=60, description="Digital sessions (30 days)")
    region: str = Field(..., description="US region bucket")


REGIONS = {"NE", "SE", "MW", "W", "S"}

app = FastAPI(title="Insurance CSR — CLV assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    train_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _ndjson_lines(payload: CustomerPayload) -> Iterator[bytes]:
    body = payload.model_dump()
    if body["region"] not in REGIONS:
        err = {"type": "error", "message": f"region must be one of {sorted(REGIONS)}"}
        yield (json.dumps(err) + "\n").encode("utf-8")
        return
    result, steps = build_calculation_trace(body)
    for step in steps:
        line = json.dumps({"type": "step", "step": step}) + "\n"
        yield line.encode("utf-8")
        time.sleep(0.12)
    yield (json.dumps({"type": "result", "result": result}) + "\n").encode("utf-8")


@app.post("/api/predict/stream")
def predict_stream(payload: CustomerPayload) -> StreamingResponse:
    return StreamingResponse(
        _ndjson_lines(payload),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/predict")
def predict_sync(payload: CustomerPayload) -> dict:
    if payload.region not in REGIONS:
        raise HTTPException(400, detail=f"region must be one of {sorted(REGIONS)}")
    result, steps = build_calculation_trace(payload.model_dump())
    return {"result": result, "steps": steps}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
