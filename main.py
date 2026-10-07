from typing import Annotated, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat

app = FastAPI(title="HTTM ECG API", version="0.1.0")

CLASSES = ["N", "S", "V", "F", "Q"]
RIndex = Annotated[int, Field(strict=True, ge=0)]


class ECGRequest(BaseModel):
    ecg: list[FiniteFloat] = Field(min_length=234)
    fs: Literal[360]
    r_peaks: list[RIndex] = Field(min_length=1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "mock"
    }


@app.post("/predict")
def predict(data: ECGRequest):
    if any(r >= len(data.ecg) for r in data.r_peaks):
        raise HTTPException(
            status_code=422,
            detail="Vị trí R nằm ngoài tín hiệu"
        )

    if any(
        a >= b
        for a, b in zip(data.r_peaks, data.r_peaks[1:])
    ):
        raise HTTPException(
            status_code=422,
            detail="Danh sách R phải tăng dần và không trùng"
        )

    beats = []
    skipped = []

    for r in data.r_peaks:
        if r < 90 or r + 144 > len(data.ecg):
            skipped.append({
                "r_peak_index": r,
                "reason": "Không đủ 234 mẫu quanh đỉnh R"
            })
            continue

        beats.append({
            "r_peak_index": r,
            "predicted_class": None,
            "probabilities": None
        })

    return {
        "mode": "mock",
        "model_version": None,
        "class_order": CLASSES,
        "processed_count": len(beats),
        "skipped_count": len(skipped),
        "beats": beats,
        "skipped": skipped}
@app.middleware("http")
async def utf8_json(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response 
