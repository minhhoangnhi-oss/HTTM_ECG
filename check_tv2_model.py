import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "tv2"
CLASSES = ["N", "S", "V", "F", "Q"]

parameters = json.loads(
    (MODEL_DIR / "normalization.json").read_text(encoding="utf-8-sig")
)

with np.load(ROOT / "data" / "tv2" / "sample_io.npz", allow_pickle=False) as data:
    sample = {name: data[name].copy() for name in data.files}

ecg = np.ascontiguousarray(
    (sample["ecg_raw"].astype(np.float64) - parameters["ecg_mean"])
    / parameters["ecg_std"],
    dtype=np.float32,
)

rr = np.ascontiguousarray(
    (sample["rr_raw"].astype(np.float64) - np.asarray(parameters["rr_mean"]))
    / np.asarray(parameters["rr_std"]),
    dtype=np.float32,
)

np.testing.assert_allclose(
    ecg, sample["ecg_normalized"], rtol=1e-5, atol=1e-6
)
np.testing.assert_allclose(
    rr, sample["rr_normalized"], rtol=1e-5, atol=1e-6
)
print("OK: Chuan hoa ECG va RR khop du lieu TV2.")

session = ort.InferenceSession(
    str(MODEL_DIR / "exports" / "cnn_rr_fp32.onnx"),
    providers=["CPUExecutionProvider"],
)

for count in [1, len(ecg)]:
    probabilities = session.run(
        ["probabilities"], {"ecg": ecg[:count], "rr": rr[:count]}
    )[0]

    np.testing.assert_equal(probabilities.shape, (count, 5))

    if not np.isfinite(probabilities).all():
        raise ValueError("Mo hinh tra ve gia tri khong huu han.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Xac suat nam ngoai khoang 0 den 1.")

    np.testing.assert_allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
    )

    for reference in ["pytorch_probabilities", "onnx_fp32_probabilities"]:
        expected = sample[reference][:count]
        np.testing.assert_allclose(
            probabilities, expected, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_array_equal(
            probabilities.argmax(axis=1), expected.argmax(axis=1)
        )

    labels = [CLASSES[index] for index in probabilities.argmax(axis=1)]
    error = np.max(
        np.abs(probabilities - sample["pytorch_probabilities"][:count])
    )
    print(f"OK: Batch {count}, nhan {labels}, sai so lon nhat {error:.8g}")

print("KIEM TRA MO HINH TV2 THANH CONG.")