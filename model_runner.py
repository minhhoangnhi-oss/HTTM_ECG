import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(__file__).resolve().parent / "models" / "tv2"
CLASSES = ["N", "S", "V", "F", "Q"]


class ECGModel:
    def __init__(self):
        config = json.loads(
            (MODEL_DIR / "inference_config.json").read_text(
                encoding="utf-8-sig"
            )
        )

        if config["class_names"] != CLASSES:
            raise ValueError("Thu tu lop cua mo hinh khong dung.")

        if config["sampling_rate"] != 360:
            raise ValueError("Mo hinh khong su dung tan so 360 Hz.")

        if config["onnx_output"] != "probabilities":
            raise ValueError("Dau ra mo hinh phai la xac suat.")

        if config["normalization_location"] != "outside_ONNX":
            raise ValueError("Vi tri chuan hoa khong dung.")

        self.version = config["model_version"]
        self.classes = CLASSES.copy()

        parameters = json.loads(
            (MODEL_DIR / config["normalization_file"]).read_text(
                encoding="utf-8-sig"
            )
        )

        self.ecg_mean = float(parameters["ecg_mean"])
        self.ecg_std = float(parameters["ecg_std"])
        self.rr_mean = np.asarray(parameters["rr_mean"], dtype=np.float64)
        self.rr_std = np.asarray(parameters["rr_std"], dtype=np.float64)

        if self.rr_mean.shape != (3,) or self.rr_std.shape != (3,):
            raise ValueError("Thong so chuan hoa RR phai co 3 gia tri.")

        values = np.concatenate([
            [self.ecg_mean, self.ecg_std],
            self.rr_mean,
            self.rr_std,
        ])

        if not np.isfinite(values).all():
            raise ValueError("Thong so chuan hoa khong hop le.")

        if self.ecg_std <= 0 or np.any(self.rr_std <= 0):
            raise ValueError("Do lech chuan phai lon hon 0.")

        self.session = ort.InferenceSession(
            str(MODEL_DIR / config["deployment_model"]),
            providers=["CPUExecutionProvider"],
        )

        inputs = {
            item.name: item for item in self.session.get_inputs()
        }

        if set(inputs) != {"ecg", "rr"}:
            raise ValueError("Ten dau vao ONNX khong dung.")

        if inputs["ecg"].shape[1:] != [1, 234]:
            raise ValueError("Kich thuoc dau vao ECG khong dung.")

        if inputs["rr"].shape[1:] != [3]:
            raise ValueError("Kich thuoc dau vao RR khong dung.")

        if any(item.type != "tensor(float)" for item in inputs.values()):
            raise ValueError("Dau vao ONNX phai la float32.")

        self.output_name = config["output_name"]

    def predict(self, waveform, rr):
        waveform = np.asarray(waveform, dtype=np.float64)
        rr = np.asarray(rr, dtype=np.float64)

        if waveform.ndim != 3 or waveform.shape[1:] != (1, 234):
            raise ValueError("ECG phai co kich thuoc [N, 1, 234].")

        count = len(waveform)

        if rr.shape != (count, 3):
            raise ValueError("RR phai co kich thuoc [N, 3].")

        if not np.isfinite(waveform).all() or not np.isfinite(rr).all():
            raise ValueError("Dau vao chua gia tri khong huu han.")

        if np.any(rr <= 0):
            raise ValueError("Khoang RR phai lon hon 0.")

        if count == 0:
            return np.empty((0, 5), dtype=np.float32)

        ecg_input = np.ascontiguousarray(
            (waveform - self.ecg_mean) / self.ecg_std,
            dtype=np.float32,
        )
        rr_input = np.ascontiguousarray(
            (rr - self.rr_mean) / self.rr_std,
            dtype=np.float32,
        )

        if not np.isfinite(ecg_input).all() or not np.isfinite(rr_input).all():
            raise ValueError("Dau vao sau chuan hoa khong hop le.")

        results = []

        for start in range(0, count, 256):
            end = min(start + 256, count)

            probabilities = self.session.run(
                [self.output_name],
                {
                    "ecg": ecg_input[start:end],
                    "rr": rr_input[start:end],
                },
            )[0]

            if probabilities.shape != (end - start, 5):
                raise RuntimeError("Kich thuoc dau ra ONNX khong dung.")

            if not np.isfinite(probabilities).all():
                raise RuntimeError("Mo hinh tra ve gia tri khong huu han.")

            if np.any(probabilities < 0) or np.any(probabilities > 1):
                raise RuntimeError("Xac suat nam ngoai khoang 0 den 1.")

            if not np.allclose(
                probabilities.sum(axis=1),
                1.0,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise RuntimeError("Tong xac suat khong bang 1.")

            results.append(probabilities)

        return np.concatenate(results, axis=0)