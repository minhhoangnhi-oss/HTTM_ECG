# Bằng chứng số liệu dùng đối chiếu báo cáo

## Nguồn

- tv2/: các CSV nhỏ từ gói TV2_MITBIH_20261008_094922, giữ nguyên byte. Gồm DS1 OOF, DS2 theo seed/tổng hợp/ablation, ma trận nhầm lẫn, chỉ số từng lớp, chuyển đổi và kích thước. Bảng lớn dự đoán từng nhịp, checkpoint, logs và hình được giữ trong gói TV2 gửi riêng.
- record_202/: ba bảng/cấu hình từ TV2_202_COMPARISON bản cuối. Giữ nguyên model và normalization, không train lại; kết quả có 202 là bổ sung vì 201/202 cùng người.
- cpu/: JSON gốc benchmark_model_comparison, ngày đo 22/10/2026 giờ Việt Nam. Gồm cấu hình, input hash, mẫu tham chiếu, kích thước và thống kê độ trễ. CSV 600 lượt đo chi tiết vẫn nằm trong gói bằng chứng riêng của nhóm.
- CI và xác nhận tiền xử lý: ../notebooks/tv1/verification/, đặc biệt pc_run_20261022/.

Các tệp ở đây bổ sung hồ sơ kết quả đã có, không phải kết quả huấn luyện mới. Không có tín hiệu ECG gốc hoặc API key. Bảng dự đoán từng nhịp của gói TV2 cần được cung cấp riêng khi tái tính CI.

## Đơn vị và phạm vi

CSV TV2 lưu phần lớn chỉ số dưới dạng tỷ lệ 0–1; bảng CI TV1 lưu phần trăm theo tên cột. Kiểm tra cột trước khi nhân 100. SD ba seed khác CI bootstrap theo bản ghi. Benchmark CPU PyTorch khác benchmark HTTP/ONNX; không gộp hai loại thời gian.

sha256_manifest.json trong thư mục này ghi hash các tệp bằng chứng được sao chép; không chứa chính manifest. Hash dùng kiểm tra tính toàn vẹn của tệp.
