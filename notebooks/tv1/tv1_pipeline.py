"""TV1 MIT-BIH preprocessing and record-level bootstrap.

Preprocessing retains the float32-RR implementation from TV1's v2 source.
Each output directory is new; existing runs are never overwritten.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

SAMPLING_RATE = 360

BEAT_LEFT_SAMPLES = 90

BEAT_RIGHT_SAMPLES = 144

FILTER_LOWCUT = 0.5

FILTER_HIGHCUT = 40.0

FILTER_ORDER = 2

EXCLUDED_RECORDS = ['102', '104', '107', '202', '217']

DS1_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
]

DS2_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]

AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,             # N (0)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                    # S (1)
    'V': 2, 'E': 2,                                    # V (2)
    'F': 3,                                            # F (3)
    '/': 4, 'f': 4, 'Q': 4                             # Q (4)
}

def apply_bandpass_filter(signal):
    """Lọc thông dải 0.5 - 40 Hz (KHÔNG chuẩn hóa Z-score để giữ dữ liệu gốc)"""
    nyq = 0.5 * SAMPLING_RATE
    b, a = butter(FILTER_ORDER, [FILTER_LOWCUT / nyq, FILTER_HIGHCUT / nyq], btype='band')
    return filtfilt(b, a, signal)

def process_single_record(record_path):
    record_id = str(os.path.basename(record_path))
    patient_id = patient_for(record_id)  # Mã bệnh nhân tương ứng với bản ghi

    rec = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    if rec.fs != SAMPLING_RATE:
        raise ValueError(f'Unexpected sampling rate in record {record_id}: {rec.fs}')

    # Chọn kênh MLII nếu có, mặc định kênh 0
    channel_idx = 0
    if 'MLII' in rec.sig_name:
        channel_idx = rec.sig_name.index('MLII')

    raw_signal = rec.p_signal[:, channel_idx]

    # Lọc thông dải (GIỮ NGUYÊN DỮ LIỆU TỰ NHIÊN, CHƯA CHUẨN HÓA Z-SCORE)
    filtered_sig = apply_bandpass_filter(raw_signal)

    valid_indices = []
    valid_labels = []

    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        if symbol in AAMI_MAPPING:
            valid_indices.append(sample_idx)
            valid_labels.append(AAMI_MAPPING[symbol])

    valid_indices = np.array(valid_indices, dtype=np.int64)
    n_beats = len(valid_indices)

    if n_beats == 0:
        return None

    # Tính khoảng RR theo GIÂY
    rr_intervals = np.diff(valid_indices) / SAMPLING_RATE

    rr_pre = np.zeros(n_beats, dtype=np.float32)
    rr_post = np.zeros(n_beats, dtype=np.float32)

    # Xử lý nhịp biên ở đầu/cuối bản ghi
    if len(rr_intervals) > 0:
        rr_pre[1:] = rr_intervals
        rr_pre[0] = rr_intervals[0]

        rr_post[:-1] = rr_intervals
        rr_post[-1] = rr_intervals[-1]
    else:
        rr_pre.fill(0.8)
        rr_post.fill(0.8)

    X_waveform_list, X_rr_list, y_list = [], [], []
    rec_ids, pat_ids, r_peaks = [], [], []

    sig_len = len(filtered_sig)

    for k in range(n_beats):
        r_peak = valid_indices[k]

        # Cửa sổ nhịp [R-90 : R+144]
        start_idx = r_peak - BEAT_LEFT_SAMPLES
        end_idx = r_peak + BEAT_RIGHT_SAMPLES

        if start_idx < 0 or end_idx > sig_len:
            continue

        waveform = filtered_sig[start_idx:end_idx]

        # Tính RR_local_avg (Trung bình tối đa 10 nhịp RR_pre liền trước)
        start_win = max(0, k - 10)
        if k == 0:
            local_avg = rr_post[0]
        else:
            local_avg = float(np.mean(rr_pre[start_win:k]))

        rr_feat = [rr_pre[k], rr_post[k], local_avg]

        X_waveform_list.append(waveform)
        X_rr_list.append(rr_feat)
        y_list.append(valid_labels[k])
        rec_ids.append(record_id)
        pat_ids.append(patient_id)
        r_peaks.append(r_peak)

    return {
        "waveform": np.array(X_waveform_list, dtype=np.float32),
        "rr": np.array(X_rr_list, dtype=np.float32),
        "y": np.array(y_list, dtype=np.int64),
        "record_id": np.array(rec_ids, dtype='U10'),
        "patient_id": np.array(pat_ids, dtype='U10'),
        "r_peak_index": np.array(r_peaks, dtype=np.int64)
    }

def process_dataset_split(record_list, data_dir):
    waveforms, rrs, ys = [], [], []
    rec_ids, pat_ids, r_peaks = [], [], []

    for rec_id in record_list:
        if rec_id in EXCLUDED_RECORDS:
            continue

        rec_path = os.path.join(data_dir, rec_id)
        res = process_single_record(rec_path)

        if res is not None and len(res["y"]) > 0:
            waveforms.append(res["waveform"])
            rrs.append(res["rr"])
            ys.append(res["y"])
            rec_ids.append(res["record_id"])
            pat_ids.append(res["patient_id"])
            r_peaks.append(res["r_peak_index"])

    # Đưa về Layout Tensor quy ước
    X_waveform = np.vstack(waveforms)[:, np.newaxis, :]  # Shape [N, 1, 234]
    X_rr = np.vstack(rrs)                                # Shape [N, 3]
    y = np.concatenate(ys)                               # Shape [N]

    record_id = np.concatenate(rec_ids)                  # Shape [N]
    patient_id = np.concatenate(pat_ids)                  # Shape [N]
    r_peak_index = np.concatenate(r_peaks)              # Shape [N]

    return X_waveform, X_rr, y, record_id, patient_id, r_peak_index
PREPROCESS_CONFIG = {'sampling_rate_hz': 360, 'ecg_channel': 'MLII (Ưu tiên MLII, mặc định lấy kênh 0)', 'beat_window': {'left_samples': 90, 'right_samples': 144, 'total_samples': 234, 'ecg_tensor_shape': '[N, 1, 234]'}, 'r_peak_indexing': '0-based indexing (chỉ số bắt đầu từ 0)', 'filtering': {'method': 'Butterworth bandpass filter', 'lowcut_hz': 0.5, 'highcut_hz': 40.0, 'order': 2, 'zero_phase': True}, 'normalization': 'none (Dữ liệu chưa chuẩn hóa Z-score, giữ nguyên biên độ)', 'rr_features': {'unit': 'seconds (giây)', 'rr_tensor_shape': '[N, 3]', 'vector_format': ['RR_pre', 'RR_post', 'RR_local_avg'], 'rr_local_avg_calculation': 'mean(RR_pre[max(0, k-10) : k])', 'edge_beats_handling': 'Nhịp đầu: RR_pre = RR_post[0], RR_local_avg = RR_post[0]. Nhịp cuối: RR_post = RR_pre[-1].'}, 'class_mapping': {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}, 'ds1_ds2_partition': {'ds1_records_count': 22, 'ds2_records_count': 21, 'excluded_records': ['102', '104', '107', '202', '217'], 'patient_overlap': False}}
REFERENCE_NORMALIZATION = {'ecg_mean': 0.010119017217793718, 'ecg_std': 0.3775945190048688, 'rr_mean': [0.7785235945552929, 0.7785457610047392, 0.7785428904380115], 'rr_std': [0.49549133582073135, 0.4955129232710033, 0.2272689599105361]}
REFERENCE_INPUT_HASHES = {'ds1.npz': {'md5': '610764294f469688dc11e6ed3adead9d', 'sha256': '11b030d3cabbf51e39fa93fb89d48411662c4e671d98e646c8c1093b4cb500b3'}, 'ds2.npz': {'md5': '264a5c4e63c0555ff377c0c1ab3ccf99', 'sha256': 'a789409757f35fee38a45d5e6097fbebd3e8b6cc73ef7f096cc2c8b02d2fad6e'}, 'preprocess_config.json': {'md5': '2ae22eefa342053f118e1fb4212c2d87', 'sha256': '0526749c35ce3336fdf07e3e6fb0adff1a712498be8c9d156c460cee968dcfe1'}}
REFERENCE_METRICS = {'baseline_42': {'se_S': 0.9051627384960719, 'ppv_S': 0.07730649412892404, 'se_V': 0.767260231177757, 'ppv_V': 0.6451273969004465, 'f1_macro': 0.33505829455776137, 'accuracy': 0.524234077002334}, 'baseline_123': {'se_S': 0.9197530864197531, 'ppv_S': 0.060771227289581016, 'se_V': 0.8906591690096844, 'ppv_V': 0.45153626860943935, 'f1_macro': 0.24261814294681278, 'accuracy': 0.3549214626658536}, 'baseline_2026': {'se_S': 0.8153759820426487, 'ppv_S': 0.14885769900624937, 'se_V': 0.8019368947203999, 'ppv_V': 0.6435196791175734, 'f1_macro': 0.35518283506663095, 'accuracy': 0.6430178522614968}, 'cnn_rr_42': {'se_S': 0.8557800224466892, 'ppv_S': 0.3369421122403889, 'se_V': 0.8394251796313652, 'ppv_V': 0.9236851151598487, 'f1_macro': 0.4886416277352323, 'accuracy': 0.855520743528818}, 'cnn_rr_123': {'se_S': 0.8131313131313131, 'ppv_S': 0.47445972495088407, 'se_V': 0.9137769447047798, 'ppv_V': 0.8149902479799387, 'f1_macro': 0.49211227979089706, 'accuracy': 0.8347456736127173}, 'cnn_rr_2026': {'se_S': 0.8417508417508418, 'ppv_S': 0.4351610095735422, 'se_V': 0.8962824117463293, 'ppv_V': 0.7925414364640884, 'f1_macro': 0.4818431534124941, 'accuracy': 0.8109006034863427}}


def patient_for(record_id):
    return "P_201" if str(record_id) == "202" else "P_" + str(record_id)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def environment():
    packages = {}
    for name in ["numpy", "scipy", "pandas", "wfdb"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def new_run(output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    run = root / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    run.mkdir(exist_ok=False)
    return run


def verify_patient_independence():
    records = sorted(set(DS1_RECORDS) & set(DS2_RECORDS))
    patients = sorted({patient_for(r) for r in DS1_RECORDS} & {patient_for(r) for r in DS2_RECORDS})
    if records or patients or "202" in DS2_RECORDS:
        raise ValueError("DS1 and DS2 overlap or contain excluded record 202.")
    print("Record intersection:", records, "Patient intersection:", patients)
    return {
        "DS1": {"records": DS1_RECORDS, "patients": [patient_for(r) for r in DS1_RECORDS]},
        "DS2": {"records": DS2_RECORDS, "patients": [patient_for(r) for r in DS2_RECORDS]},
        "record_intersection": records, "patient_intersection": patients,
        "patient_mapping_note": "201 and 202 share P_201; 202 is excluded from DS2.",
        "excluded_records": EXCLUDED_RECORDS,
    }


def find_dataset_dir(input_root):
    required = DS1_RECORDS + DS2_RECORDS
    root = Path(input_root)
    candidates = {root.resolve()} if (root / "100.hea").is_file() else {p.parent.resolve() for p in root.rglob("100.hea")}
    complete = sorted(p for p in candidates if all((p / (r + ext)).is_file() for r in required for ext in [".hea", ".dat", ".atr"]))
    if len(complete) != 1:
        raise ValueError("Expected one complete MIT-BIH directory containing all 43 required records; set RAW_DIR explicitly.")
    return complete[0]


def resolve_predictions(input_root, explicit=None):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(Path(input_root).rglob("ds2_predictions.csv"))
    if not candidates:
        raise FileNotFoundError("Add the original TV2 ds2_predictions.csv to Input, or set PREDICTIONS_CSV.")
    if len({file_hash(p) for p in candidates}) != 1:
        raise ValueError("Different prediction CSVs found; set PREDICTIONS_CSV explicitly.")
    return candidates[0]


def read_predictions(path):
    frame = pd.read_csv(path, dtype={"record_id": str, "patient_id": str})
    required = {"model_variant", "seed", "record_id", "r_peak_index", "y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError("Prediction CSV is missing required columns.")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise")
    expected_groups = {(name, seed) for name in ["baseline", "cnn_rr"] for seed in [42, 123, 2026]}
    if set(zip(frame["model_variant"], frame["seed"])) != expected_groups:
        raise ValueError("Need exactly two models times three seeds.")
    for (name, seed), group in frame.groupby(["model_variant", "seed"], sort=False):
        if len(group) != 47557 or set(group["record_id"]) != set(DS2_RECORDS):
            raise ValueError(f"Unexpected DS2 coverage for {name}, seed {seed}.")
        if group.duplicated(["record_id", "r_peak_index"]).any():
            raise ValueError("Duplicate beat identifiers in predictions.")
        if not group[["y_true", "y_pred"]].isin(range(5)).all().all():
            raise ValueError("Invalid class labels.")
        if "patient_id" in group:
            if not np.array_equal(group["patient_id"].to_numpy(), np.asarray([patient_for(r) for r in group["record_id"]])):
                raise ValueError("Unexpected patient mapping in predictions.")
    anchor = frame[(frame["model_variant"] == "cnn_rr") & (frame["seed"] == 42)]
    anchor_keys = anchor[["record_id", "r_peak_index", "y_true"]].reset_index(drop=True)
    for _, group in frame.groupby(["model_variant", "seed"], sort=False):
        if not group[["record_id", "r_peak_index", "y_true"]].reset_index(drop=True).equals(anchor_keys):
            raise ValueError("Beat order or true labels differ across model/seed groups.")
    return frame


def quality_summary(arrays, expected_count, expected_records):
    if arrays["X_waveform"].shape != (expected_count, 1, 234) or arrays["X_rr"].shape != (expected_count, 3):
        raise ValueError("Unexpected beat count or tensor shape.")
    for name, values in arrays.items():
        if values.shape[0] != expected_count:
            raise ValueError("Metadata length mismatch: " + name)
    for name in ["X_waveform", "X_rr"]:
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError("Unexpected dtype or nonfinite data: " + name)
    if not np.isin(arrays["y"], range(5)).all() or set(arrays["record_id"]) != set(expected_records):
        raise ValueError("Unexpected labels or records.")
    if not np.array_equal(arrays["patient_id"], np.asarray([patient_for(r) for r in arrays["record_id"]])):
        raise ValueError("Unexpected patient metadata.")
    keys = set()
    for record, peak in zip(arrays["record_id"], arrays["r_peak_index"]):
        key = (str(record), int(peak))
        if key in keys:
            raise ValueError("Duplicate beat identifiers in preprocessed data.")
        keys.add(key)
    waveform_hashes = [hashlib.sha256(row.tobytes()).digest() for row in arrays["X_waveform"]]
    return {
        "beats": expected_count, "records": len(expected_records),
        "class_counts_N_S_V_F_Q": np.bincount(arrays["y"], minlength=5).tolist(),
        "nonfinite_waveform": 0, "nonfinite_rr": 0, "duplicate_record_r_peak": 0,
        "exact_duplicate_waveforms": expected_count - len(set(waveform_hashes)),
        "ecg_min_max_mV": [float(arrays["X_waveform"].min()), float(arrays["X_waveform"].max())],
        "rr_min_max_seconds_by_feature": [arrays["X_rr"].min(axis=0).tolist(), arrays["X_rr"].max(axis=0).tolist()],
        "abnormal_rr_note": "Extremes are recorded without clipping, preserving the handed-over pipeline.",
    }


def preprocess_all(data_dir, run_dir, predictions_csv=None):
    import wfdb as wfdb_module
    globals()["wfdb"] = wfdb_module
    import os as os_module
    globals()["os"] = os_module
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    split = verify_patient_independence()
    save_json(run_dir / "preprocess_config.json", PREPROCESS_CONFIG)
    save_json(run_dir / "split.json", split)
    summaries, hashes, datasets = {}, {}, {}
    for name, records, expected in [("ds1", DS1_RECORDS, 51003), ("ds2", DS2_RECORDS, 47557)]:
        print("Preprocessing", name, "...")
        result = process_dataset_split(records, str(data_dir))
        arrays = dict(zip(["X_waveform", "X_rr", "y", "record_id", "patient_id", "r_peak_index"], result))
        summary = quality_summary(arrays, expected, records)
        expected_classes = {"ds1": [45849, 944, 3788, 414, 8], "ds2": [42180, 1782, 3201, 387, 7]}[name]
        if summary["class_counts_N_S_V_F_Q"] != expected_classes:
            raise ValueError("Class distribution differs from the training handover.")
        datasets[name] = arrays
        summaries[name] = summary
        path = run_dir / (name + ".npz")
        np.savez_compressed(path, **arrays)
        hashes[path.name] = {"md5": file_hash(path, "md5"), "sha256": file_hash(path)}
        print(name, summary["beats"], "beats; class counts", summary["class_counts_N_S_V_F_Q"])
    train = datasets["ds1"]
    statistics = {
        "ecg_mean": float(train["X_waveform"].mean(dtype=np.float64)),
        "ecg_std": float(train["X_waveform"].std(dtype=np.float64)),
        "rr_mean": train["X_rr"].mean(axis=0, dtype=np.float64).tolist(),
        "rr_std": train["X_rr"].std(axis=0, dtype=np.float64).tolist(),
    }
    for name in statistics:
        np.testing.assert_allclose(statistics[name], REFERENCE_NORMALIZATION[name], rtol=1e-6, atol=1e-7, err_msg="DS1 statistics differ from TV2: " + name)
    summaries["ds1_statistics_match_tv2"] = True
    summaries["ds1_statistics_for_validation_only"] = statistics
    if predictions_csv is not None:
        frame = read_predictions(predictions_csv)
        anchor = frame[(frame["model_variant"] == "cnn_rr") & (frame["seed"] == 42)]
        for key, column in [("record_id", "record_id"), ("patient_id", "patient_id"), ("r_peak_index", "r_peak_index"), ("y", "y_true")]:
            if column in anchor and not np.array_equal(datasets["ds2"][key], anchor[column].to_numpy()):
                raise ValueError("DS2 metadata differs from TV2 predictions: " + key)
        summaries["ds2_metadata_matches_tv2_predictions"] = True
    md5 = {}
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".hea", ".dat", ".atr"}:
            md5[path.name] = {"md5": file_hash(path, "md5"), "size_bytes": path.stat().st_size}
    save_json(run_dir / "md5_checksums.json", md5)
    save_json(run_dir / "processed_input_hashes.json", hashes)
    save_json(run_dir / "reference_tv2_input_hashes.json", REFERENCE_INPUT_HASHES)
    save_json(run_dir / "preprocessing_checks.json", summaries)
    return summaries


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    actual, predicted = matrix.sum(axis=-1), matrix.sum(axis=-2)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros_like(tp), where=(actual + predicted) > 0)
    se = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    ppv = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    return {
        "Accuracy": tp.sum(axis=-1) / matrix.sum(axis=(-2, -1)),
        "Macro_F1": f1.mean(axis=-1),
        **{"F1_" + label: f1[..., i] for i, label in enumerate(["N", "S", "V", "F", "Q"])},
        "Se_S": se[..., 1], "PPV_S": ppv[..., 1], "Se_V": se[..., 2], "PPV_V": ppv[..., 2],
    }


def bootstrap_all(predictions_csv, run_dir, n_bootstraps=1000, random_seed=42):
    if n_bootstraps < 1000:
        raise ValueError("At least 1,000 bootstrap replicates are required.")
    frame = read_predictions(predictions_csv)
    all_results = []
    reference_names = {"se_S": "Se_S", "ppv_S": "PPV_S", "se_V": "Se_V", "ppv_V": "PPV_V", "f1_macro": "Macro_F1", "accuracy": "Accuracy"}
    for variant in ["baseline", "cnn_rr"]:
        for seed in [42, 123, 2026]:
            group = frame[(frame["model_variant"] == variant) & (frame["seed"] == seed)]
            codes, records = pd.factorize(group["record_id"], sort=False)
            per_record = np.zeros((len(records), 5, 5), dtype=np.int64)
            np.add.at(per_record, (codes, group["y_true"].to_numpy(dtype=np.int64), group["y_pred"].to_numpy(dtype=np.int64)), 1)
            original = metrics_from_confusion(per_record.sum(axis=0))
            for old_name, new_name in reference_names.items():
                np.testing.assert_allclose(original[new_name], REFERENCE_METRICS[f"{variant}_{seed}"][old_name], rtol=1e-12, atol=1e-12)
            rng = np.random.RandomState(random_seed)
            draws = rng.choice(len(records), size=(n_bootstraps, len(records)), replace=True)
            sampled_matrices = per_record[draws].sum(axis=1)
            replicates = metrics_from_confusion(sampled_matrices)
            for metric, original_value in original.items():
                values = replicates[metric] * 100
                lower, upper = np.percentile(values, [2.5, 97.5])
                mean = values.mean()
                value = float(original_value) * 100
                all_results.append({
                    "Model": variant, "Seed": seed, "Is_Deployed": "YES" if variant == "cnn_rr" and seed == 42 else "No",
                    "Metric": metric, "Original_DS2 (%)": round(value, 2),
                    "Bootstrap_Mean (%)": round(float(mean), 2), "CI_95_Lower (%)": round(float(lower), 2), "CI_95_Upper (%)": round(float(upper), 2),
                    "95% CI Formatted": f"{value:.2f}% (Mean: {mean:.2f}%) [{lower:.2f}% - {upper:.2f}%]",
                })
            print("Bootstrap OK:", variant, "seed", seed, len(group), "beats;", len(records), "records")
    result = pd.DataFrame(all_results)
    if len(result) != 66:
        raise ValueError("Expected exactly 66 metric rows.")
    result.to_csv(Path(run_dir) / "ds2_bootstrap_ci_95_all_models.csv", index=False)
    save_json(Path(run_dir) / "bootstrap_metadata.json", {
        "input_sha256": file_hash(predictions_csv), "replicates": n_bootstraps, "rng": "numpy RandomState", "seed_reset_per_model_seed": random_seed,
        "resampling_unit": "record; 202 excluded so records represent different patients", "records_per_group": 21,
        "class_labels": [0, 1, 2, 3, 4], "zero_division": 0, "ci_method": "percentile 2.5/97.5, numpy linear interpolation",
        "implementation": "Sum complete per-record confusion matrices with replacement, including multiplicity; equivalent to concatenating resampled records.",
    })
    print(result[(result["Model"] == "cnn_rr") & (result["Seed"] == 42)].to_string(index=False))
    return result


def finish_run(run_dir):
    run_dir = Path(run_dir)
    env = environment()
    save_json(run_dir / "environment.json", env)
    requirements = "\n".join(f"{name}=={version}" for name, version in env["packages"].items() if version) + "\n"
    (run_dir / "requirements_run.txt").write_text(requirements, encoding="utf-8")
    manifest = {p.name: file_hash(p) for p in sorted(run_dir.iterdir()) if p.is_file() and p.name != "sha256_manifest.json"}
    save_json(run_dir / "sha256_manifest.json", manifest)
    archive = run_dir.parent / (run_dir.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run_dir.iterdir()):
            if path.is_file():
                z.write(path, path.name)
    print("Output directory:", run_dir)
    print("Output archive:", archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("TV1_FINAL"))
    parser.add_argument("--bootstrap-only", action="store_true")
    args = parser.parse_args()
    predictions = resolve_predictions(args.input_root, args.predictions)
    data_dir = None if args.bootstrap_only else find_dataset_dir(args.data_dir or args.input_root)
    run = new_run(args.output_root)
    if data_dir is not None:
        preprocess_all(data_dir, run, predictions)
    bootstrap_all(predictions, run)
    finish_run(run)


if __name__ == "__main__":
    main()
