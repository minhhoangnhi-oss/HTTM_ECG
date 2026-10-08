import numpy as np
from scipy.signal import butter, filtfilt

FS = 360
LEFT = 90
RIGHT = 144


def preprocess_ecg(ecg, r_peaks, fs=360):
    if fs != FS:
        raise ValueError("Tan so lay mau phai la 360 Hz.")

    signal = np.asarray(ecg, dtype=np.float64)
    peaks = np.asarray(r_peaks)

    if signal.ndim != 1 or len(signal) < LEFT + RIGHT:
        raise ValueError("ECG phai la mang 1 chieu co it nhat 234 mau.")

    if not np.isfinite(signal).all():
        raise ValueError("ECG chua gia tri khong huu han.")

    if peaks.ndim != 1 or len(peaks) == 0 or peaks.dtype.kind not in "iu":
        raise ValueError("Danh sach R phai gom cac so nguyen.")

    if np.any(peaks < 0) or np.any(peaks >= len(signal)):
        raise ValueError("Vi tri R nam ngoai tin hieu.")

    if np.any(peaks[1:] <= peaks[:-1]):
        raise ValueError("Danh sach R phai tang dan va khong trung.")

    peaks = peaks.astype(np.int64)

    b, a = butter(
        2,
        [0.5 / (0.5 * FS), 40.0 / (0.5 * FS)],
        btype="band",
    )
    filtered = filtfilt(b, a, signal)

    count = len(peaks)
    intervals = np.diff(peaks) / FS
    rr_pre = np.zeros(count, dtype=np.float32)
    rr_post = np.zeros(count, dtype=np.float32)

    if count > 1:
        rr_pre[1:] = intervals
        rr_pre[0] = intervals[0]
        rr_post[:-1] = intervals
        rr_post[-1] = intervals[-1]
    else:
        rr_pre.fill(0.8)
        rr_post.fill(0.8)

    waveforms = []
    rr_features = []
    kept_peaks = []
    skipped = []

    for k, peak in enumerate(peaks):
        r = int(peak)

        if r < LEFT or r + RIGHT > len(signal):
            skipped.append({
                "r_peak_index": r,
                "reason": "Không đủ 234 mẫu quanh đỉnh R",
            })
            continue

        local_avg = rr_post[0] if k == 0 else float(
            np.mean(rr_pre[max(0, k - 10):k])
        )

        waveforms.append(filtered[r - LEFT:r + RIGHT])
        rr_features.append([rr_pre[k], rr_post[k], local_avg])
        kept_peaks.append(r)

    waveform = np.asarray(
        waveforms, dtype=np.float32
    ).reshape(-1, 1, 234)

    rr = np.asarray(
        rr_features, dtype=np.float32
    ).reshape(-1, 3)

    if not np.isfinite(waveform).all() or not np.isfinite(rr).all():
        raise ValueError("Tien xu ly tao ra gia tri khong huu han.")

    return waveform, rr, kept_peaks, skipped