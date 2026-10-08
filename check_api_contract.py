import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent_tools import execute_tool

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"


def main():
    base = {
        "ecg": [0.0] * 1000,
        "fs": 360,
        "r_peaks": [20, 400, 700],
    }

    cases = [
        ("valid_request", {}, 200),
        ("wrong_sampling_rate", {"fs": 250}, 422),
        ("signal_too_short", {"ecg": [0.0] * 233}, 422),
        ("null_signal_value", {"ecg": [None] + [0.0] * 999}, 422),
        ("nonfinite_string", {"ecg": ["NaN"] + [0.0] * 999}, 422),
        ("negative_r_index", {"r_peaks": [-1]}, 422),
        ("r_index_outside_signal", {"r_peaks": [1000]}, 422),
        ("noninteger_r_index", {"r_peaks": [400.5]}, 422),
        ("duplicate_r_indices", {"r_peaks": [400, 400]}, 422),
        ("unsorted_r_indices", {"r_peaks": [700, 400]}, 422),
        ("empty_r_indices", {"r_peaks": []}, 422),
        ("all_beats_skipped", {"r_peaks": [20]}, 200),
    ]

    results = []

    with httpx.Client(timeout=30.0, trust_env=False) as client:
        for name, changes, expected_status in cases:
            payload = {**base, **changes}

            try:
                response = client.post(
                    f"{API_URL}/predict",
                    json=payload,
                )
                passed = response.status_code == expected_status

                if passed and expected_status == 200:
                    data = response.json()
                    passed = (
                        data.get("mode") == "onnx"
                        and data.get("model_version") == "20261008_094922"
                    )

                    if name == "valid_request":
                        passed = passed and (
                            data.get("processed_count") == 2
                            and data.get("skipped_count") == 1
                            and [
                                beat["r_peak_index"]
                                for beat in data.get("beats", [])
                            ] == [400, 700]
                        )

                    if name == "all_beats_skipped":
                        passed = passed and (
                            data.get("processed_count") == 0
                            and data.get("skipped_count") == 1
                            and data.get("beats") == []
                            and [
                                beat["r_peak_index"]
                                for beat in data.get("skipped", [])
                            ] == [20]
                        )

                result = {
                    "test": name,
                    "expected_status": expected_status,
                    "actual_status": response.status_code,
                    "passed": passed,
                }

            except (
                httpx.HTTPError, ValueError, KeyError, TypeError,
            ) as error:
                result = {
                    "test": name,
                    "passed": False,
                    "error_type": type(error).__name__,
                }

            results.append(result)
            print(
                f"{'PASS' if result['passed'] else 'FAIL'}: {name}",
                flush=True,
            )

    denied_cases = [
        ("unknown_tool", "run_shell", {}),
        (
            "unauthorized_record",
            "classify_record",
            {"record_id": "not_allowed"},
        ),
        (
            "extra_path_argument",
            "classify_record",
            {"record_id": "100", "path": "outside_allowlist"},
        ),
        (
            "unexpected_list_argument",
            "list_records",
            {"record_id": "100"},
        ),
    ]

    for name, tool, arguments in denied_cases:
        try:
            execute_tool(tool, arguments)
            result = {
                "test": name,
                "passed": False,
                "reason": "Tool unexpectedly accepted the request.",
            }
        except ValueError:
            result = {
                "test": name,
                "passed": True,
                "rejected_with": "ValueError",
            }
        except Exception as error:
            result = {
                "test": name,
                "passed": False,
                "error_type": type(error).__name__,
            }

        results.append(result)
        print(
            f"{'PASS' if result['passed'] else 'FAIL'}: {name}",
            flush=True,
        )

    passed_count = sum(item["passed"] for item in results)
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "api_test_count": len(cases),
        "agent_guard_test_count": len(denied_cases),
        "passed_count": passed_count,
        "total_count": len(results),
        "all_passed": passed_count == len(results),
        "notes": [
            "API cases run against Docker on port 8000.",
            "Agent guard cases run against the local agent_tools module.",
            "Zero-valued signals test the interface, not model accuracy.",
            "The NaN case uses a JSON string, not a nonstandard JSON numeric literal.",
        ],
        "tests": results,
    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    destination = output_dir / f"contract_check_{uuid.uuid4().hex}.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nKET QUA: {passed_count}/{len(results)} kiem tra dat.")
    print(f"Da luu: {destination}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()