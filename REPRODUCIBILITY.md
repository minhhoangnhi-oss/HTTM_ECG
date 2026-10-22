# Chạy lại thí nghiệm và dịch vụ ECG

> Cập nhật phạm vi demo ngày 22/10/2026: giao diện và Agent được mở rộng sang 21 bản ghi DS2, có bộ chọn bản ghi và trạng thái dữ liệu. DS1 và 202 không được mở quyền. Các phép đo/kiểm thử ghi bên dưới vẫn thuộc đúng phiên bản và thời điểm đã nêu; đã đối chiếu API trên laptop: 21/21 bản khớp số nhịp và số lượng nhãn với bảng TV2. Xem [DS2_DEMO_GUIDE.md](DS2_DEMO_GUIDE.md) và chạy `check_ds2_records.py` để đối chiếu.

## 1. Ba môi trường có mục đích riêng

| Phần | Môi trường | Cần dữ liệu nào? |
|---|---|---|
| Demo API/web/Agent | Docker Python 3.14 theo Dockerfile và requirements.txt gốc | Bản ghi 100; key OpenRouter riêng nếu dùng Agent |
| TV1: tiền xử lý và bootstrap | Python 3.14, requirements_tv1.txt; CPU đủ | 43 bản ghi WFDB và CSV dự đoán sáu cặp model–seed |
| TV2: huấn luyện/đánh giá | Kaggle Python 3.13, GPU T4; notebooks/tv2/requirements.txt | DS1/DS2 v2 và cấu hình TV1 chưa Z-score |

Không cài requirements TV1 và TV2 vào cùng một môi trường: NumPy trong hồ sơ hai phần khác phiên bản. Docker suy luận không cần PyTorch để huấn luyện. Môi trường lịch sử TV2 được ghi tại notebooks/tv2/environment.json; pin torch==2.11.0 chưa bảo đảm đúng build CUDA lịch sử 2.11.0+cu128. Khi chạy mới, kiểm tra phiên bản và GPU thực tế rồi lưu environment.json của lượt đó.

## 2. Tệp giữ trong kho và tệp gửi riêng

Kho mã có ONNX FP32, cấu hình suy luận, notebook/code TV1/TV2, split, protocol, seed và các bảng kết quả nhỏ. reports/README.md mô tả nguồn bằng chứng.

Tệp gửi riêng, không có trong ZIP GitHub:

- WFDB: các tệp .dat/.hea/.atr gốc; tải MIT-BIH từ PhysioNet, không commit dữ liệu gốc.
- DS1/DS2 lịch sử đã dùng train: ds1.npz, ds2.npz, preprocess_config.json từ TV1 v2. Đối chiếu hash notebooks/tv2/processed_input_hashes.json.
- Gói TV2_MITBIH_20261008_094922 đầy đủ: checkpoints/, exports/sample_io.npz, results/ds2_predictions.csv, logs/, saliency/ và các kết quả khác. Cần cho đối chiếu checkpoint, bootstrap và vẽ lại hình lịch sử.
- record_202.npz bản cuối của TV1; dùng với evaluate_record_202.ipynb. Hash đầu vào nằm trong reports/record_202/evaluation_config.json.
- .env của người chạy; không gửi key trong báo cáo hay Git.

Bản NPZ tạo lại ngày 22/10 có hash khác bản TV2 lịch sử. Giữ riêng hai bản; không đổi tên bản mới thành bản cũ rồi cập nhật hash lịch sử. Các kiểm tra thống kê và metadata không chứng minh mọi phần tử ECG/RR giống hệt.

## 3. Demo từ bản GitHub mới tải

Theo README.md mục 4: tạo .env riêng, tạo data/mitdb, logs, outputs; build image; tải bản ghi 100 bằng image; khởi động API với mount dữ liệu read-only. Chỉ cần Docker, không cần Python trên máy chủ.

Đầu ra /health cần có mode=onnx, model_version=20261008_094922. Mở http://127.0.0.1:8000/ và yêu cầu phân loại 100. Kết quả tham chiếu: R=2.273; xử lý 2.271; bỏ 2; N/S/V/F/Q=2.236/34/1/0/0. Đỉnh R lấy từ chú thích có sẵn. Chọn lưu mới tạo báo cáo; ghi lại run_id, JSON và audit của chính lượt thử.

Không dùng số nhịp này để suy ra độ chính xác mô hình. Đánh giá chính dùng toàn DS2 đã khóa từ TV2.

## 4. Chạy lại TV1

Notebook tự chứa mã nguồn: notebooks/tv1/TV1_preprocess_bootstrap_final.ipynb. Hướng dẫn Kaggle và tham số chi tiết: notebooks/tv1/README.md.

Trên Windows, mở PowerShell ở thư mục gốc dự án; dùng môi trường riêng mới nếu cần kiểm tra khả năng chạy từ đầu:

```powershell
python -m venv .venv-tv1
```

```powershell
.\.venv-tv1\Scripts\python.exe -m pip install -r .\notebooks\tv1\requirements_tv1.txt
```

Ví dụ gói TV2 đặt cạnh thư mục HTTM_ECG, dữ liệu WFDB ở data/mitdb:

```powershell
.\.venv-tv1\Scripts\python.exe .\notebooks\tv1\tv1_pipeline.py --data-dir .\data\mitdb --predictions ..\TV2_MITBIH_20261008_094922\results\ds2_predictions.csv --output-root .\outputs\tv1_final
```

Chạy từ đầu cần có đủ 43 bản ghi được chỉ định trong split; script không tự tải thiếu. Mỗi bản ghi cần ba tệp .dat/.hea/.atr ở cùng thư mục. Danh sách DS1/DS2 trong notebooks/tv1/verification/pc_run_20261022/split.json.

Kỳ vọng: DS1 51.003 nhịp; DS2 47.557 nhịp; giao bản ghi/bệnh nhân rỗng; sáu cặp bootstrap đủ 66 dòng. Script tạo thư mục run mới, NPZ, cấu hình, MD5 gốc, môi trường, metadata và ZIP. Xem preprocessing_checks.json để đánh giá kiểm tra cụ thể, không chỉ nhìn dòng cuối “thành công”.

Chỉ tái tính CI từ dự đoán lịch sử:

```powershell
.\.venv-tv1\Scripts\python.exe .\notebooks\tv1\tv1_pipeline.py --bootstrap-only --predictions ..\TV2_MITBIH_20261008_094922\results\ds2_predictions.csv --output-root .\outputs\tv1_ci_check
```

Nếu vừa huấn luyện mới, dùng ds2_predictions.csv của đúng lượt đó; CI sẽ mô tả mô hình mới, không thay bảng CI lịch sử.

## 5. Chạy lại TV2 từ đầu

1. Import notebooks/tv2/train_evaluate.ipynb vào Kaggle; tạo session mới, bật Internet cho cài thư viện, chọn GPU T4 nếu có.
2. Thêm dataset chứa ds1.npz, ds2.npz và preprocess_config.json bản v2. Kiểm tra hash với processed_input_hashes.json nếu muốn tái hiện dữ liệu lịch sử. Không đưa NPZ đã Z-score vào notebook.
3. Sửa DATA_DIR trong cell cấu hình và cell đọc DS1 nếu dataset được mount ở đường dẫn khác mặc định `/kaggle/input/datasets/quanghuy10/mitbih-tv1-v2`.
4. Bấm Run All. Cell đầu cài các thư viện đã pin; nếu cài làm thay phiên bản thư viện đã import trong session cũ, restart session trước khi chạy các cell xử lý.
5. Kiểm tra torch.cuda.is_available(), phiên bản PyTorch/CUDA và environment.json đầu ra. Không tuyên bố đã chạy GPU chỉ vì chọn GPU trong giao diện.
6. Chạy đủ 60 lượt CV, chọn cấu hình bằng DS1, huấn luyện cuối, đánh giá DS2, saliency, xuất ONNX/INT8 và đối chiếu toàn DS2. Lưu toàn bộ output và ZIP tạo cuối notebook.

Notebook đã được tách khỏi các cell khôi phục kết quả cũ. Không cần dataset tv2-results để chạy luồng huấn luyện. Số liệu báo cáo thuộc lượt TV2 20261008_094922.

Số lượt, seed và quy tắc chọn cấu hình phải khớp protocol.json. Lượt gốc chọn lr=0,001, batch=128 và bốn epoch cuối. Nếu môi trường mới cho kết quả số khác, lưu thành lượt mới và giải thích; không sửa output mới để khớp con số cũ, không tự thay ONNX đang triển khai bằng kết quả mới.

### Vẽ hình từ kết quả lịch sử

Dùng notebooks/tv2/figures_from_results.ipynb, không dùng notebook train. Thêm gói TV2 đã giải nén thành dataset tv2-results; sửa source_dir nếu cần. Gói cần cả logs, results, architecture.json, inference_config.json và 12 ảnh saliency. Run All tạo thư mục figures_report và ZIP. Không huấn luyện lại.

### Đánh giá thêm bản ghi 202

Dùng notebooks/tv2/evaluate_record_202.ipynb. Chuẩn bị record_202.npz bản cuối và gói TV2 đầy đủ; sửa DATA_202 và RESULT_DIR theo Input. Giữ model CNN + RR seed 42 và normalization lịch sử. Đầu ra ba bảng CSV và evaluation_config.json. Kết quả chính vẫn loại 202 vì cùng người với 201; không train lại cho phép so sánh này.

## 6. So sánh CPU hai checkpoint

Script cần PyTorch và NumPy, không cần GPU hoặc API key. Không cài PyTorch vào requirements runtime/Docker chỉ để chạy phép đo này. Ví dụ môi trường riêng Python 3.14 trên PC:

```powershell
python -m venv .venv-cpu-benchmark
```

```powershell
.\.venv-cpu-benchmark\Scripts\python.exe -m pip install numpy==2.5.3
```

```powershell
.\.venv-cpu-benchmark\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

```powershell
.\.venv-cpu-benchmark\Scripts\python.exe .\benchmark_model_comparison.py --tv2-dir ..\TV2_MITBIH_20261008_094922
```

Gói TV2 cần hai checkpoint seed 42, normalization.json, sample_io.npz và ds2_predictions.csv; script kiểm tra SHA256 trước khi đo.

Kỳ vọng: cả hai checkpoint khớp mẫu tham chiếu; 6 hàng model/batch, mỗi hàng 100 lượt đo; JSON/CSV mới nằm trong outputs/. Độ trễ phụ thuộc máy. Không bắt laptop tái hiện đúng mili giây của desktop.

## 7. Trạng thái xác nhận

| Phép kiểm tra | Bằng chứng hiện có |
|---|---|
| TV1 đủ 43 bản ghi trên PC hiện có | Đạt; hồ sơ ngày 22/10 trong notebooks/tv1/verification/pc_run_20261022 |
| API, web, quyền công cụ, lưu báo cáo | Các kiểm tra ghi trong PERFORMANCE_RESULTS.md |
| API trên laptop, toàn bộ DS2 | 21/21 bản; 47.557 nhịp, số nhịp và số lượng nhãn khớp TV2; xem reports/deployment/ |
| Hai checkpoint trên cùng CPU | Đạt; reports/cpu/ |
| Đánh giá thêm 202 không train lại | Có bảng và cấu hình; reports/record_202/ |
| Notebook TV2 | Đã tách luồng huấn luyện và vẽ hình; hướng dẫn tại notebooks/tv2/README.md |
