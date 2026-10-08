import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat

from model_runner import ECGModel
from preprocessing import preprocess_ecg

logger = logging.getLogger(__name__)
RIndex = Annotated[int, Field(strict=True, ge=0)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ECGModel()
    yield
    del app.state.model


app = FastAPI(
    title="HTTM ECG API",
    version="0.2.0",
    lifespan=lifespan,
)


class ECGRequest(BaseModel):
    ecg: list[FiniteFloat] = Field(
        min_length=234,
        max_length=1500000,
        description="ECG gốc, chưa lọc và chưa chuẩn hóa, biên độ theo mV.",
    )
    fs: Literal[360]
    r_peaks: list[RIndex] = Field(
        min_length=1,
        max_length=20000,
        description="Các đỉnh R của nhịp được chọn theo TV1, gồm cả nhịp biên.",
    )


@app.get("/health")
def health():
    model = app.state.model
    return {
        "status": "ok",
        "mode": "onnx",
        "model_version": model.version,
    }


@app.post("/predict")
def predict(data: ECGRequest):
    try:
        waveform, rr, kept_peaks, skipped = preprocess_ecg(
            data.ecg, data.r_peaks, data.fs
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    model = app.state.model

    try:
        probabilities = model.predict(waveform, rr)
    except Exception as error:
        logger.exception("ECG model inference failed")
        raise HTTPException(
            status_code=500,
            detail="Không thể chạy mô hình phân loại.",
        ) from error

    beats = []

    for r, scores in zip(kept_peaks, probabilities, strict=True):
        class_index = int(scores.argmax())

        beats.append({
            "r_peak_index": r,
            "predicted_class": model.classes[class_index],
            "probabilities": {
                label: float(score)
                for label, score in zip(
                    model.classes, scores, strict=True
                )
            },
        })

    return {
        "mode": "onnx",
        "model_version": model.version,
        "class_order": model.classes,
        "processed_count": len(beats),
        "skipped_count": len(skipped),
        "beats": beats,
        "skipped": skipped,
    }


@app.middleware("http")
async def utf8_json(request, call_next):
    response = await call_next(request)

    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"

    return response