# TV1 tiền xử lý MIT-BIH và bootstrap bản bàn giao

Bản dọn từ notebook `xulith (2).ipynb` của TV1. Giữ đoạn tiền xử lý v2 có metadata `P_<record>` và RR float32. Dọn thứ tự thực thi, bỏ các cell lặp và các bản xử lý 202 cũ. Mô hình đã huấn luyện và kết quả DS2 chính vẫn dùng hồ sơ TV2 `20261008_094922`.

## Tệp trong gói

- `TV1_preprocess_bootstrap_final.ipynb`: notebook tự chứa mã nguồn, chạy trên Kaggle.
- `tv1_pipeline.py`: cùng mã nguồn dưới dạng script độc lập.
- `requirements_tv1.txt`: phiên bản đề xuất để chạy xác nhận bản bàn giao này; không phải tuyên bố về môi trường tiền xử lý lịch sử chưa được ghi lại.
- `verification/`: kết quả bootstrap chạy lại từ CSV dự đoán TV2 và biên bản kiểm tra code.

## Chạy trên Kaggle

1. Import `TV1_preprocess_bootstrap_final.ipynb`.
2. Thêm bộ MIT-BIH gốc đủ các tệp `.hea`, `.dat`, `.atr` cho 43 bản ghi được dùng.
3. Thêm `results/ds2_predictions.csv` từ ZIP TV2 `20261008_094922`. Mỗi cặp mô hình–seed cần 47.557 nhịp của 21 bản ghi, không có 202; tổng cộng 6 cặp.
4. Bật Internet cho cell cài thư viện. CPU đủ cho tiền xử lý và bootstrap.
5. Restart Session, chạy Run All từ trên xuống dưới. Nếu Input có nhiều nguồn WFDB đầy đủ khác nhau hoặc các CSV dự đoán khác nhau, điền `RAW_DIR` / `PREDICTIONS_CSV` ở cell cấu hình thay cho `None`.
6. Lấy thư mục `TV1_FINAL/run_<thời gian UTC>` và ZIP tương ứng trong Output. Mỗi lần chạy tạo bản mới để tránh ghi đè dữ liệu đã dùng train.

## Đầu ra và tiêu chí đối chiếu

- `ds1.npz`: 51.003 nhịp, 22 bản ghi; số nhịp N/S/V/F/Q = 45.849/944/3.788/414/8.
- `ds2.npz`: 47.557 nhịp, 21 bản ghi; số nhịp N/S/V/F/Q = 42.180/1.782/3.201/387/7; không có 202.
- Sáu trường NPZ: `X_waveform [N,1,234]`, `X_rr [N,3]`, `y`, `record_id`, `patient_id`, `r_peak_index`.
- Chọn MLII nếu có; lọc Butterworth 0,5–40 Hz, order 2, filtfilt; cắt `[R-90:R+144]`.
- Tính RR trên toàn bộ chú thích thuộc ánh xạ AAMI trước khi bỏ nhịp thiếu cửa sổ; RR_local_avg là trung bình tối đa 10 RR_pre trước nhịp hiện tại. ECG/RR đầu ra chưa Z-score.
- `preprocess_config.json`, `split.json`, `md5_checksums.json`, `processed_input_hashes.json`, `reference_tv2_input_hashes.json`, `preprocessing_checks.json`.
- Thống kê toàn DS1 được đối chiếu với normalization của TV2, chỉ để kiểm tra tương thích; không áp dụng phép chuẩn hóa đó lên NPZ đầu ra.
- Metadata, thứ tự R và nhãn DS2 được đối chiếu với CSV dự đoán TV2.
- `ds2_bootstrap_ci_95_all_models.csv`: 66 dòng = 2 mô hình × 3 seed × 11 chỉ số; 1.000 lần lấy mẫu có hoàn lại theo bản ghi, cố định 5 lớp, seed 42 cho từng cặp; CI percentile 2,5/97,5%.
- `environment.json`, `requirements_run.txt`, `sha256_manifest.json`: lưu môi trường và hash đúng lần chạy thực tế.

Hash tệp NPZ có thể khác khi cách đóng gói hoặc phiên bản thư viện khác. Các kiểm tra số lượng, metadata và thống kê DS1 không đủ chứng minh mọi tensor trùng từng phần tử với dữ liệu đã train. Giữ nguyên provenance đầu vào TV2, không thay hash lịch sử bằng hash bản mới.

## Chạy script độc lập

```powershell
python -m pip install -r requirements_tv1.txt
```

```powershell
python tv1_pipeline.py --data-dir "DUONG_DAN_THU_MUC_MITDB" --predictions "DUONG_DAN_ds2_predictions.csv" --output-root "TV1_FINAL"
```

Nếu chỉ kiểm tra lại CI từ dự đoán đã có:

```powershell
python tv1_pipeline.py --bootstrap-only --predictions "DUONG_DAN_ds2_predictions.csv" --output-root "TV1_CI_CHECK"
```

Thay các chuỗi DUONG_DAN bằng đường dẫn thực tế. Với notebook Kaggle, mã nguồn đã nằm ngay trong notebook, không cần upload thêm script này để chạy.

## Bản ghi 202

Bộ 202 và phép so sánh có/không có 202 đã được làm riêng với mô hình cố định. Notebook này không tạo lại bộ 202, không thay thế kết quả bổ sung đã chốt. Kết quả DS2 không có 202 vẫn là kết quả chính; 201 và 202 cùng một người.

## Trạng thái kiểm tra

Đã kiểm tra:

- Mảng đầu ra của hàm tiền xử lý khớp từng phần tử với đoạn v2 đã chọn từ notebook TV1 trên các ca giả lập nhịp biên, một nhịp và nhiều nhịp.
- Bootstrap từ CSV TV2 thực tế chạy đủ 6 cặp, mỗi cặp 1.000 lượt, tạo đủ 66 dòng; chỉ số gốc cả sáu cặp khớp bảng TV2.
- Cách cộng ma trận nhầm lẫn theo bản ghi khớp cách ghép lại các nhịp và tính bằng scikit-learn trên 48 lượt lấy mẫu đối chiếu; thứ tự rút mẫu RNG giống nhau.
- Đủ 11 dòng CI mô hình CNN + RR seed 42 khớp bảng đã kiểm tra trước đó.

Ngày 22/10/2026, đã chạy script trên đủ 43 bản ghi WFDB trong môi trường Windows hiện có. DS1/DS2 đúng số nhịp, không giao bệnh nhân; thống kê DS1 và metadata DS2 khớp kiểm tra TV2. Toàn bộ 66 dòng CI khớp bảng đã kiểm chứng. Hồ sơ lần chạy nằm trong verification/pc_run_20261022/. Đây không phải phép thử trên môi trường mới cài hoàn toàn.

## Đưa vào kho nộp

Đưa notebook, script, README, requirements và bảng kết quả/metadata cần thiết vào kho riêng tư. Dữ liệu WFDB gốc và NPZ là dữ liệu chạy cục bộ, gửi riêng hoặc tải theo hướng dẫn, không đưa lên Git. TV2 dùng `ds1.npz`, `ds2.npz` và `preprocess_config.json` làm đầu vào; giữ dữ liệu lịch sử đã huấn luyện để đối chiếu phiên bản.
