"""Compare the two handed-over TV2 checkpoints on one CPU backend.

Run from the HTTM_ECG project folder using its Python virtual environment.
Only inference is measured. No training or changes to existing files occur.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
CLASSES = ["N", "S", "V", "F", "Q"]
EXPECTED_HASHES = {
    "checkpoints/final_baseline_seed42.pt": "c35e7709d88ca722cd9a599b7d282e21b6dd88558ab401553a2fbb4e5f59b2a2",
    "checkpoints/final_cnn_rr_seed42.pt": "66b5d9e8f3fba80acf47acccc31e35a4913e10fab27a0bcbd6a8fb463dbf5035",
    "normalization.json": "2d92ef83f86480ac33f4b27c429ed372b24d220205dc4b8bba48291abf5fc333",
    "exports/sample_io.npz": "965bb6f50c57497da3a3b48032f2442af76280589ec54ed495632b6c90a3e0a2",
    "results/ds2_predictions.csv": "4903856949519a5a03a0e5ab00ee9f92f6eb492d9cd0c6a7c12ae0311e4c8cb9",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ECGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, ecg):
        return self.layers(ecg)


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ECGEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 5),
        )

    def forward(self, ecg, rr=None):
        return self.classifier(self.encoder(ecg))


class CNNWithRR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ECGEncoder()
        self.rr_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 5),
        )

    def forward(self, ecg, rr):
        ecg_features = self.encoder(ecg)
        rr_features = self.rr_branch(rr)
        features = torch.cat([ecg_features, rr_features], dim=1)
        return self.classifier(features)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tv2-dir", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.threads < 1 or args.rounds < 100 or args.warmup < 1:
        parser.error("threads >= 1, rounds >= 100, warmup >= 1 are required")
    return args


def resolve_bundle(explicit):
    candidates = [explicit] if explicit else [
        ROOT.parent / "TV2_MITBIH_20261008_094922",
        ROOT / "TV2_MITBIH_20261008_094922",
    ]
    matches = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / name).is_file() for name in EXPECTED_HASHES):
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "Cannot identify exactly one complete TV2 bundle. "
            "Use --tv2-dir followed by its full directory path."
        )
    return matches[0]


def baseline_reference(path):
    first = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["model_variant"] != "baseline" or int(row["seed"]) != 42:
                continue
            index = int(row["sample_index"])
            if index in range(4):
                if index in first:
                    raise ValueError("Duplicate sample in baseline reference.")
                first[index] = [float(row["p_" + name]) for name in CLASSES]
                if len(first) == 4:
                    break
    if set(first) != set(range(4)):
        raise ValueError("The baseline reference must contain samples 0..3.")
    return np.asarray([first[i] for i in range(4)], dtype=np.float32)


def normalize_samples(samples, norm):
    ecg = np.ascontiguousarray(
        (samples["ecg_raw"].astype(np.float64) - norm["ecg_mean"])
        / norm["ecg_std"], dtype=np.float32,
    )
    rr = np.ascontiguousarray(
        (samples["rr_raw"].astype(np.float64) - np.asarray(norm["rr_mean"]))
        / np.asarray(norm["rr_std"]), dtype=np.float32,
    )
    if ecg.shape != (4, 1, 234) or rr.shape != (4, 3):
        raise ValueError("Unexpected sample shapes.")
    if not np.isfinite(ecg).all() or not np.isfinite(rr).all():
        raise ValueError("Nonfinite normalized samples.")
    np.testing.assert_allclose(ecg, samples["ecg_normalized"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rr, samples["rr_normalized"], rtol=1e-6, atol=1e-6)
    return ecg, rr


def load_models(bundle, norm):
    constructors = {"baseline": BaselineCNN, "cnn_rr": CNNWithRR}
    models, metadata = {}, {}
    for name, constructor in constructors.items():
        path = bundle / "checkpoints" / ("final_" + name + "_seed42.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["model_variant"] != name or checkpoint["seed"] != 42:
            raise ValueError("Unexpected checkpoint identity.")
        if checkpoint["class_names"] != CLASSES:
            raise ValueError("Unexpected class order.")
        for field in ("ecg_mean", "ecg_std", "rr_mean", "rr_std"):
            np.testing.assert_allclose(checkpoint["normalization"][field], norm[field], rtol=1e-12, atol=1e-12)
        model = constructor().cpu().float()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        expected = {"baseline": 11173, "cnn_rr": 12021}[name]
        if parameter_count != expected:
            raise ValueError("Unexpected parameter count.")
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise ValueError("Nonfinite model weights.")
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        models[name] = model
        metadata[name] = {
            "parameter_count": parameter_count,
            "tensor_storage_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
            "state_dict_serialized_bytes": len(buffer.getvalue()),
            "checkpoint_file_bytes": path.stat().st_size,
            "checkpoint_sha256": sha256(path),
            "training_seed": 42,
        }
    return models, metadata


def timed_forward(model, ecg, rr):
    start = time.perf_counter_ns()
    output = torch.softmax(model(ecg, rr), dim=1)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return elapsed_ms, output


def main():
    args = parse_arguments()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    bundle = resolve_bundle(args.tv2_dir)
    print("TV2 bundle:", bundle)
    hashes = {name: sha256(bundle / name) for name in EXPECTED_HASHES}
    if hashes != EXPECTED_HASHES:
        changed = [name for name in EXPECTED_HASHES if hashes[name] != EXPECTED_HASHES[name]]
        raise ValueError("Files differ from the reviewed TV2 handover: " + ", ".join(changed))
    print("OK: checkpoint and reference hashes match TV2.")
    norm = json.loads((bundle / "normalization.json").read_text(encoding="utf-8-sig"))
    with np.load(bundle / "exports/sample_io.npz", allow_pickle=False) as source:
        samples = {name: source[name] for name in source.files}
    ecg, rr = normalize_samples(samples, norm)
    models, metadata = load_models(bundle, norm)
    references = {
        "baseline": baseline_reference(bundle / "results/ds2_predictions.csv"),
        "cnn_rr": samples["pytorch_probabilities"],
    }
    validation = {}
    with torch.inference_mode():
        for name, model in models.items():
            predicted = torch.softmax(model(torch.from_numpy(ecg), torch.from_numpy(rr)), dim=1).numpy()
            np.testing.assert_allclose(predicted, references[name], rtol=1e-4, atol=1e-5)
            if not np.array_equal(predicted.argmax(1), references[name].argmax(1)):
                raise ValueError("Reference label mismatch: " + name)
            validation[name] = {"reference_samples": 4, "max_abs_probability_error": float(np.max(np.abs(predicted - references[name]))), "labels_match": True}
    print("OK: both checkpoints reproduce their TV2 reference samples.")
    rng = random.Random(42)
    raw_times, summaries = [], []
    batches = (1, 64, 256)
    with torch.inference_mode():
        for batch in batches:
            indices = np.arange(batch) % len(ecg)
            x = torch.from_numpy(np.ascontiguousarray(ecg[indices]))
            r = torch.from_numpy(np.ascontiguousarray(rr[indices]))
            for _ in range(args.warmup):
                for model in models.values():
                    torch.softmax(model(x, r), dim=1)
            timings = {name: [] for name in models}
            for iteration in range(args.rounds):
                order = list(models)
                rng.shuffle(order)
                for position, name in enumerate(order):
                    elapsed_ms, output = timed_forward(models[name], x, r)
                    if not torch.isfinite(output).all().item():
                        raise ValueError("Nonfinite timed output.")
                    timings[name].append(elapsed_ms)
                    raw_times.append({"batch_size": batch, "round": iteration + 1, "order_in_round": position + 1, "model_variant": name, "latency_ms": elapsed_ms})
            for name in models:
                durations = np.asarray(timings[name], dtype=np.float64)
                row = {
                    "model_variant": name, "batch_size": batch,
                    "measured_calls": len(durations),
                    "median_ms": float(np.median(durations)),
                    "p95_ms": float(np.percentile(durations, 95)),
                    "mean_ms": float(durations.mean()),
                    "sample_sd_ms": float(durations.std(ddof=1)),
                    "min_ms": float(durations.min()), "max_ms": float(durations.max()),
                    "median_ms_per_beat": float(np.median(durations)) / batch,
                }
                summaries.append(row)
                print(f"{name:8s} | batch {batch:3d} | median {row['median_ms']:.4f} ms | P95 {row['p95_ms']:.4f} ms")
    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "protocol": {
            "device": "cpu", "dtype": "float32", "backend": "PyTorch eager",
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "batch_sizes": list(batches), "warmup_calls_per_model_per_batch": args.warmup,
            "measured_calls_per_model_per_batch": args.rounds,
            "order": "seeded random model order within every paired round",
            "random_seed": 42,
            "timed_scope": "CPU forward plus softmax; preallocated normalized input tensors",
            "excluded": ["checkpoint loading", "normalization", "tensor creation", "HTTP", "LLM", "output validation"],
            "input": "four handed-over DS2 reference beats, repeated to obtain the requested batch size",
            "limitations": "A repeated-input model microbenchmark, not a new accuracy evaluation, full DS2 timing, or API/ONNX latency.",
        },
        "host": {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "logical_cpu_count": os.cpu_count(), "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "mkldnn_available": torch.backends.mkldnn.is_available()},
        "source_bundle": str(bundle), "input_sha256": hashes,
        "models": metadata, "validation": validation,
        "latency_summary": summaries,
        "size_notes": "tensor_storage_bytes counts FP32 state tensors; state_dict_serialized_bytes uses identical torch.save serialization. Checkpoint size includes metadata. These sizes are not ONNX artifact sizes.",
    }
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    stem = "model_comparison_" + uuid.uuid4().hex
    json_path = output_dir / (stem + ".json")
    csv_path = output_dir / (stem + ".csv")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_times[0]))
        writer.writeheader()
        writer.writerows(raw_times)
    for name, item in metadata.items():
        print(f"{name}: {item['parameter_count']} parameters; {item['tensor_storage_bytes']} tensor bytes; {item['state_dict_serialized_bytes']} serialized state_dict bytes")
    print("Saved JSON:", json_path)
    print("Saved CSV:", csv_path)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as error:
        print("STOP:", error, file=sys.stderr)
        sys.exit(1)
