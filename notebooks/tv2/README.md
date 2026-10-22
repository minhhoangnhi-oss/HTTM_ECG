# Notebook TV2

- train_evaluate.ipynb: Run All cho luồng huấn luyện, chọn cấu hình DS1, đánh giá DS2, saliency và xuất/đối chiếu ONNX. Yêu cầu DS1/DS2 v2, Python 3.13 và môi trường Kaggle phù hợp.
- figures_from_results.ipynb: vẽ lại hình từ gói kết quả đã lưu; không train. Yêu cầu dataset tv2-results.
- evaluate_record_202.ipynb: đánh giá bổ sung bản 202 trên model và normalization đã khóa; không train.
- requirements.txt và environment.json: phiên bản thư viện/hồ sơ môi trường TV2 lịch sử.
- protocol.json, selection.json, split.json, cv_split.json: ngân sách, seed, cách chọn cấu hình và chia tập.
- processed_input_hashes.json: hash dữ liệu lịch sử TV2, không cập nhật bằng hash NPZ tái tạo mới.

Hướng dẫn chi tiết, vị trí đầu vào và bằng chứng chạy lại: [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).

Ngày đồng bộ: 22/10/2026. Tách phần vẽ hình khỏi notebook train; giữ nguyên mã huấn luyện, đánh giá và xuất mô hình, trừ cell cài thư viện được pin đầy đủ hơn và đường dẫn cấu hình được thống nhất với v2. Kết quả đã báo cáo thuộc gói TV2_MITBIH_20261008_094922.
