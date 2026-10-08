import json
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
CONTAINER = "httm-ecg-api"
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_SYMBOLS = {
    "N", "L", "R", "e", "j",
    "A", "a", "J", "S",
    "V", "E", "F", "/", "f", "Q",
}

CONTAINER_PROBE = """
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

def read_value(paths):
    for name in paths:
        try:
            return Path(name).read_text().strip()
        except OSError:
            pass
    return None

def read_bytes(paths):
    value = read_value(paths)
    return int(value) if value and value.isdigit() else None

model = Path("/app/models/tv2/exports/cnn_rr_fp32.onnx")
config = json.loads(
    Path("/app/models/tv2/inference_config.json").read_text(
        encoding="utf-8-sig"
    )
)

print(json.dumps({
    "python": platform.python_version(),
    "packages": {
        name: version(name)
        for name in ["fastapi", "uvicorn", "onnxruntime", "numpy", "scipy", "wfdb"]
    },
    "model_version": config["model_version"],
    "model_bytes": model.stat().st_size,
    "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    "memory_current_bytes": read_bytes([
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    ]),
    "memory_peak_bytes": read_bytes([
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"
    ]),
    "memory_limit": read_value([
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    ]),
    "cpu_quota": read_value(["/sys/fs/cgroup/cpu.max"]),
    "cpu_set": read_value(["/sys/fs/cgroup/cpuset.cpus.effective"])
}))
"""


def run_command(arguments):
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=25,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def read_json_command(arguments):
    output = run_command(arguments)
    if output is None:
        return None
    try:
        return json.loads(output)
    except ValueError:
        return None


def host_information():
    details = {
        "os": platform.platform(),
        "client_python": platform.python_version(),
        "cpu": None,
        "ram_bytes": None,
    }

    if platform.system() == "Windows":
        command = """
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ecgProcessors = Get-CimInstance Win32_Processor
$ecgComputer = Get-CimInstance Win32_ComputerSystem
[ordered]@{
    cpu = @($ecgProcessors | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
    ram_bytes = $ecgComputer.TotalPhysicalMemory
} | ConvertTo-Json -Depth 4 -Compress
"""
        measured = read_json_command([
            "powershell.exe", "-NoProfile", "-Command", command,
        ])
        if isinstance(measured, dict):
            details.update(measured)

    return details


def container_information():
    return read_json_command([
        "docker", "exec", CONTAINER, "python", "-c", CONTAINER_PROBE,
    ])


def check_prediction(result, payload, model_version):
    peaks = payload["r_peaks"]
    samples = len(payload["ecg"])
    kept = [r for r in peaks if r >= 90 and r + 144 <= samples]
    skipped = [r for r in peaks if r < 90 or r + 144 > samples]

    if result["mode"] != "onnx":
        raise ValueError("API khong chay ONNX.")
    if result["model_version"] != model_version:
        raise ValueError("Phien ban mo hinh thay doi trong khi do.")
    if result["class_order"] != CLASSES:
        raise ValueError("Thu tu lop khong dung.")
    if result["processed_count"] != len(kept):
        raise ValueError("So nhip xu ly khong dung.")
    if result["skipped_count"] != len(skipped):
        raise ValueError("So nhip bo qua khong dung.")
    if [beat["r_peak_index"] for beat in result["beats"]] != kept:
        raise ValueError("Vi tri R xu ly khong dung.")
    if [beat["r_peak_index"] for beat in result["skipped"]] != skipped:
        raise ValueError("Vi tri R bo qua khong dung.")

    for beat in result["beats"]:
        scores = beat["probabilities"]
        if set(scores) != set(CLASSES):
            raise ValueError("Thieu xac suat cac lop.")
        probabilities = np.asarray(
            [scores[label] for label in CLASSES],
            dtype=np.float64,
        )
        if not np.isfinite(probabilities).all():
            raise ValueError("Xac suat khong huu han.")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Xac suat ngoai khoang 0 den 1.")
        if not np.isclose(probabilities.sum(), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("Tong xac suat khong bang 1.")
        if beat["predicted_class"] != CLASSES[int(probabilities.argmax())]:
            raise ValueError("Nhan khong khop xac suat.")


def measure_request(client, payload, model_version):
    response = None
    elapsed = None
    started = time.perf_counter()

    try:
        response = client.post(f"{API_URL}/predict", json=payload)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        result = response.json()
        check_prediction(result, payload, model_version)

        return {
            "success": True,
            "seconds": elapsed,
            "status_code": response.status_code,
            "processed_count": result["processed_count"],
            "skipped_count": result["skipped_count"],
        }

    except (
        httpx.HTTPError, ValueError, KeyError, TypeError, IndexError,
    ) as error:
        if elapsed is None:
            elapsed = time.perf_counter() - started
        return {
            "success": False,
            "seconds": elapsed,
            "status_code": response.status_code if response is not None else None,
            "error_type": type(error).__name__,
        }


def main():
    record_path = str(ROOT / "data" / "mitdb" / "100")
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    channel = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
    if record.fs != 360 or record.units[channel] != "mV":
        raise ValueError("Ban ghi phai la ECG 360 Hz, don vi mV.")

    signal = record.p_signal[:, channel]
    selected = np.array([
        symbol in AAMI_SYMBOLS
        for symbol in annotation.symbol
    ])
    peaks = annotation.sample[selected]

    short_samples = 3600
    short_payload = {
        "ecg": signal[:short_samples].tolist(),
        "fs": 360,
        "r_peaks": peaks[peaks < short_samples].tolist(),
    }
    full_payload = {
        "ecg": signal.tolist(),
        "fs": 360,
        "r_peaks": peaks.tolist(),
    }

    hardware = host_information()
    docker_host = read_json_command([
        "docker", "info", "--format",
        '{"cpu_count":{{.NCPU}},"ram_bytes":{{.MemTotal}},"version":"{{.ServerVersion}}"}',
    ])
    image_id = run_command([
        "docker", "inspect", "--format", "{{.Image}}", CONTAINER,
    ])
    image_size = None
    if image_id:
        size = run_command([
            "docker", "image", "inspect", "--format", "{{.Size}}", image_id,
        ])
        if size and size.isdigit():
            image_size = int(size)

    before = container_information()

    with httpx.Client(timeout=180.0, trust_env=False) as client:
        response = client.get(f"{API_URL}/health")
        response.raise_for_status()
        health = response.json()

        if health["mode"] != "onnx":
            raise ValueError("API chua chay mo hinh ONNX.")

        model_version = health["model_version"]
        if before and before["model_version"] != model_version:
            raise ValueError("Container va API dung mo hinh khac nhau.")

        print("Dang chay 5 request lam nong...", flush=True)

        for _ in range(5):
            warmup = measure_request(client, short_payload, model_version)
            if not warmup["success"]:
                raise RuntimeError(f"Warm-up that bai: {warmup}")

        print("Dang do 100 request, moi request la ECG 10 giay...", flush=True)
        requests = []

        for index in range(100):
            measured = measure_request(client, short_payload, model_version)
            measured["request_number"] = index + 1
            requests.append(measured)
            if (index + 1) % 20 == 0:
                print(f"Da do {index + 1}/100 request.", flush=True)

        print("Dang do 2 luot ban ghi day du, tai tuong duong khoang 1 gio...", flush=True)
        full_requests = []

        for index in range(2):
            measured = measure_request(client, full_payload, model_version)
            measured["request_number"] = index + 1
            full_requests.append(measured)
            print(
                f"Luot {index + 1}: {measured['seconds']:.3f} giay, "
                f"thanh cong={measured['success']}",
                flush=True,
            )

    after = container_information()
    successful = [item for item in requests if item["success"]]
    durations_ms = np.array(
        [item["seconds"] * 1000 for item in successful],
        dtype=np.float64,
    )

    latency = {
        "request_count": 100,
        "warmup_count": 5,
        "success_count": len(successful),
        "failure_count": 100 - len(successful),
        "samples_per_request": short_samples,
        "r_peaks_per_request": len(short_payload["r_peaks"]),
        "signal_seconds_per_request": 10,
        "median_ms": float(np.median(durations_ms)) if len(durations_ms) else None,
        "p95_ms": float(np.percentile(durations_ms, 95, method="linear")) if len(durations_ms) else None,
        "min_ms": float(durations_ms.min()) if len(durations_ms) else None,
        "max_ms": float(durations_ms.max()) if len(durations_ms) else None,
        "statistics_use_successful_requests_only": True,
        "response_json_validation_excluded_from_timing": True,
        "requests": requests,
    }

    full_success = all(item["success"] for item in full_requests)
    full_seconds = sum(item["seconds"] for item in full_requests)
    peak_bytes = after.get("memory_peak_bytes") if after else None
    model_bytes = after.get("model_bytes") if after else None

    report = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "container_name": CONTAINER,
        "image_id": image_id,
        "docker_image_bytes": image_size,
        "model_version": model_version,
        "record_id": "100",
        "source": "physionet_mit_bih",
        "host": hardware,
        "docker_host": docker_host,
        "container_before": before,
        "container_after": after,
        "latency": latency,
        "one_hour_equivalent": {
            "method": "same_record_repeated_in_two_separate_requests",
            "samples_per_request": len(signal),
            "r_peaks_per_request": len(peaks),
            "equivalent_signal_minutes": 2 * len(signal) / 360 / 60,
            "success_count": sum(item["success"] for item in full_requests),
            "total_http_seconds": full_seconds,
            "processed_count": sum(
                item.get("processed_count", 0) for item in full_requests
            ),
            "requests": full_requests,
        },
        "criteria": {
            "all_100_requests_successful": len(successful) == 100,
            "p95_within_2000_ms": latency["p95_ms"] <= 2000 if latency["p95_ms"] is not None else None,
            "one_hour_equivalent_within_60_seconds": full_success and full_seconds <= 60,
            "container_lifetime_peak_within_1_gib": peak_bytes <= 1024**3 if peak_bytes is not None else None,
            "onnx_model_within_1_mib": model_bytes <= 1024**2 if model_bytes is not None else None,
        },
        "notes": [
            "Sequential local HTTP requests with a reused client connection.",
            "HTTP timing includes JSON encoding and complete response reception.",
            "Response JSON decoding and validation are outside the HTTP timer.",
            "Memory peak covers container lifetime, including earlier Agent runs and measurement helpers.",
            "Repeated record 100 is a workload test, not an independent one-hour recording.",
            "Background processes were not controlled by this script.",
            "Proposed engineering targets are not teacher-specified thresholds.",
        ],
    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    destination = output_dir / f"benchmark_{uuid.uuid4().hex}.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nKET QUA BENCHMARK")
    print(f"Request thanh cong: {len(successful)}/100")

    if len(durations_ms):
        print(f"Median: {latency['median_ms']:.3f} ms")
        print(f"P95: {latency['p95_ms']:.3f} ms")
        print(f"Min: {latency['min_ms']:.3f} ms")
        print(f"Max: {latency['max_ms']:.3f} ms")

    print(f"Tai tuong duong {2 * len(signal) / 360 / 60:.2f} phut")
    print(f"Tong thoi gian 2 luot: {full_seconds:.3f} giay")
    print(f"So luot day du thanh cong: {sum(item['success'] for item in full_requests)}/2")

    if peak_bytes is not None:
        print(f"RAM peak container tu khi tao: {peak_bytes / 1024**2:.2f} MiB")
    else:
        print("Chua doc duoc RAM peak container.")

    if model_bytes is not None:
        print(f"Kich thuoc ONNX: {model_bytes} byte")

    print(f"Da luu: {destination}")


if __name__ == "__main__":
    main()