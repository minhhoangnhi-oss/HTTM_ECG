# Bảng yêu cầu kỹ thuật — HTTM_ECG

> Cập nhật phạm vi demo ngày 22/10/2026: giao diện và Agent được mở rộng sang 21 bản ghi DS2, có bộ chọn bản ghi và trạng thái dữ liệu. DS1 và 202 không được mở quyền. Các phép đo/kiểm thử ghi bên dưới vẫn thuộc đúng phiên bản và thời điểm đã nêu; đã đối chiếu API trên laptop: 21/21 bản khớp số nhịp và số lượng nhãn với bảng TV2. Xem [DS2_DEMO_GUIDE.md](DS2_DEMO_GUIDE.md) và chạy `check_ds2_records.py` để đối chiếu.

## 1. Phạm vi và thời điểm lập

Hệ thống hỗ trợ phân loại nhịp trên bản ghi ECG đã thu nhận, dùng dữ liệu MIT-BIH và vị trí đỉnh R từ tệp chú thích.

Hệ thống gồm phần tiền xử lý của TV1, mô hình CNN + RR của TV2, API FastAPI và AI Agent do TV3 tích hợp.

Tài liệu được lập sau khi đã kiểm tra tích hợp trên bản ghi 100, trước khi chạy benchmark hiệu năng được mô tả dưới đây. Không trình bày tài liệu này như một đặc tả được lập trước toàn bộ quá trình huấn luyện và đánh giá mô hình.

Các ngưỡng thời gian và RAM là mục tiêu kỹ thuật đề xuất của nhóm. Nếu đề bài chính thức có ngưỡng khác, cần ghi rõ và điều chỉnh trước lần đo tiếp theo.

## 2. Quy ước đầu vào và tiền xử lý

| Thành phần | Yêu cầu |
|---|---|
| Tần số lấy mẫu | 360 Hz |
| Tín hiệu đưa vào API | ECG gốc, chưa lọc và chưa chuẩn hóa |
| Đơn vị biên độ | mV |
| Kênh ECG | Ưu tiên MLII; nếu không có, chọn kênh 0 |
| Vị trí R | Chỉ số nguyên, bắt đầu từ 0, tăng dần, không trùng |
| Nguồn vị trí R | Chú thích được cung cấp; hệ thống chưa có bộ phát hiện R |
| Chọn chú thích | Giữ các ký hiệu thuộc ánh xạ AAMI của TV1 |
| Lọc | Butterworth bandpass 0,5–40 Hz, order=2, dùng scipy.signal.butter và filtfilt |
| Phạm vi lọc | Lọc toàn bộ tín hiệu đầu vào trước khi cắt nhịp |
| Cửa sổ nhịp | signal[R-90:R+144], đúng 234 mẫu |
| Tensor ECG trước chuẩn hóa | float32, shape [N, 1, 234] |
| Nhịp thiếu cửa sổ | Bỏ qua và trả lý do; không tự đệm thêm mẫu |
| Giới hạn tín hiệu mỗi request | Từ 234 đến 1.500.000 mẫu |
| Giới hạn số vị trí R | Từ 1 đến 20.000 vị trí |
| Giá trị không hữu hạn | Từ chối NaN và vô cực |

Không đưa X_waveform trong ds1_data.npz hoặc ds2_data.npz vào trường ecg của API. Các đoạn đó đã được lọc và cắt nhịp.

### Ánh xạ lớp

| Lớp | Mã | Ký hiệu chú thích |
|---|---:|---|
| N | 0 | N, L, R, e, j |
| S | 1 | A, a, J, S |
| V | 2 | V, E |
| F | 3 | F |
| Q | 4 | /, f, Q |

### Đặc trưng RR

| Thành phần | Quy tắc |
|---|---|
| Thứ tự vector | RR_pre, RR_post, RR_local_avg |
| Đơn vị trước chuẩn hóa | Giây |
| Tensor | float32, shape [N, 3] |
| RR_pre | Khoảng cách từ đỉnh R trước đến đỉnh R hiện tại, chia 360 |
| RR_post | Khoảng cách từ đỉnh R hiện tại đến đỉnh R sau, chia 360 |
| RR_local_avg | mean(RR_pre[max(0, k-10):k]) |
| Nhịp đầu | RR_pre và RR_local_avg lấy khoảng RR đầu tiên |
| Nhịp cuối | RR_post lấy khoảng RR cuối cùng |
| Chỉ có một đỉnh R | Dùng 0,8 giây cho các đặc trưng RR, theo code TV1 |
| Thứ tự xử lý | Tính RR trên toàn bộ đỉnh R đã chọn, trước khi bỏ nhịp thiếu cửa sổ |

API phải nhận cả các đỉnh R ở biên để giữ đúng ngữ cảnh RR.

Do có RR_post và filtfilt, phiên bản hiện tại phục vụ xem lại ECG đã thu nhận. Không mô tả là hệ thống phân loại thời gian thực.

## 3. Mô hình triển khai

| Thành phần | Yêu cầu |
|---|---|
| Mô hình | CNN + RR, ONNX FP32 |
| Phiên bản hiện tại | 20261008_094922 |
| Tệp triển khai | models/tv2/exports/cnn_rr_fp32.onnx |
| Backend | ONNX Runtime, CPUExecutionProvider |
| Đầu vào ONNX | ecg: [N, 1, 234]; rr: [N, 3] |
| Kiểu dữ liệu ONNX | float32 |
| Thứ tự lớp | N, S, V, F, Q |
| Chuẩn hóa | Ngoài ONNX, dùng normalization.json của TV2 |
| Số lần chuẩn hóa | Đúng một lần |
| Thông số chuẩn hóa | Giữ nguyên thông số TV2; không tính lại bằng DS2 |
| Đầu ra | probabilities, shape [N, 5] |
| Softmax | Đã có trong ONNX; không áp dụng thêm |
| Quy tắc chọn nhãn | Argmax theo thứ tự lớp |
| Batch nội bộ | Tối đa 256 nhịp mỗi lần gọi ONNX |
| Nạp mô hình | Một lần khi API khởi động |
| Thiếu mô hình hoặc cấu hình sai | Khởi động thất bại; không tự chuyển sang mock |

Không thay FP32 bằng mô hình INT8 thử nghiệm nếu chưa đánh giá và ghi nhận lại sai lệch.

## 4. API và Agent

| Thành phần | Yêu cầu |
|---|---|
| GET /health | Trả status, mode=onnx và model_version |
| POST /predict | Nhận ecg, fs, r_peaks |
| Kết quả từng nhịp | Có r_peak_index, predicted_class và xác suất đủ 5 lớp |
| Tổng số nhịp | processed_count + skipped_count bằng số đỉnh R đầu vào hợp lệ |
| Xác suất | Hữu hạn, trong [0, 1], tổng xấp xỉ 1 |
| Đầu vào sai | HTTP 422 |
| Lỗi chạy mô hình | HTTP 500, không tạo kết quả phân loại giả |
| Công cụ Agent | Chỉ list_records và classify_record |
| Bản ghi Agent hiện được phép | 100 |
| API Agent được phép gọi | localhost hoặc 127.0.0.1, cổng 8000 |
| Đọc dữ liệu | Từ đường dẫn cố định trên máy; LLM không tự chọn đường dẫn |
| Dữ liệu gửi LLM | Metadata và thống kê tổng hợp; không gửi toàn bộ ECG |
| Nội dung tóm tắt | Dựa trên số liệu công cụ, không tự tạo nhãn hoặc số lượng |
| Lưu báo cáo | Chỉ lưu khi người dùng nhập y |
| Nhật ký | Ghi run_id, công cụ, trạng thái, mô hình LLM và quyết định lưu |
| Diễn giải | Kết quả dự đoán hỗ trợ xem lại dữ liệu, không phải chẩn đoán |

## 5. Tiêu chí kiểm tra tính nhất quán

| Kiểm tra | Tiêu chí |
|---|---|
| ECG tiền xử lý so với TV1 | numpy.allclose, rtol=1e-5, atol=1e-6 |
| RR so với TV1 | numpy.allclose, rtol=1e-5, atol=1e-6 |
| Vị trí R và nhãn tham chiếu TV1 | Khớp hoàn toàn |
| ONNX so với sample_io của TV2 | Xác suất: rtol=1e-4, atol=1e-5; nhãn khớp hoàn toàn |
| API so với model_runner trên cùng dữ liệu | Xác suất: rtol=1e-4, atol=1e-5; nhãn khớp hoàn toàn |
| Docker | Khởi động được API và chạy được Agent trên bản ghi thật |

Đối chiếu API với model_runner kiểm tra tính nhất quán của triển khai. Việc này không thay thế đánh giá độ chính xác trên toàn bộ DS2.

## 6. Mục tiêu và phương pháp đo hiệu năng

### Độ trễ API

- Dùng API trong Docker tại http://127.0.0.1:8000.

- Dùng đoạn ECG đầu tiên dài 10 giây của bản ghi 100.

- Chỉ lấy đỉnh R nằm trong đoạn tín hiệu đó.

- Chạy 5 request làm nóng, không đưa vào thống kê.

- Đo 100 request liên tiếp, không chạy song song.

- Đo thời gian từ trước khi gọi HTTP đến khi nhận đầy đủ response, gồm mã hóa JSON và truyền dữ liệu qua localhost.

- Báo cáo median, P95, min, max và số request thành công.

- Mục tiêu đề xuất: 100/100 request thành công và P95 không quá 2 giây.

### Tải tương đương khoảng một giờ ECG

- Xử lý toàn bộ bản ghi 100 hai lần, bằng hai request riêng.

- Mỗi lượt có 650.000 mẫu tại 360 Hz.

- Tổng lượng tín hiệu tương đương khoảng 60,19 phút.

- Giữ riêng ngữ cảnh lọc và RR của mỗi request.

- Báo cáo thời gian từng lượt, tổng thời gian và tổng nhịp xử lý.

- Mục tiêu đề xuất: tổng thời gian không quá 60 giây.

Đây là phép đo tải bằng cách lặp lại cùng một bản ghi. Không gọi đây là đánh giá trên một bản ghi một giờ độc lập.

### Bộ nhớ và kích thước

- Ghi RAM hiện tại và RAM peak của container nếu môi trường hỗ trợ.

- Nếu dùng memory.peak của cgroup, ghi rõ đó là peak từ khi container được tạo, có thể bao gồm các lần chạy trước.

- Nếu không đọc được peak, ghi chưa đo được; không thay bằng số RAM tức thời.

- Mục tiêu đề xuất: RAM peak container không quá 1 GiB trong cấu hình thử.

- Mục tiêu kích thước tệp ONNX triển khai: không quá 1 MiB.

- Báo cáo riêng kích thước model và Docker image; không coi hai giá trị là một.

### Điều kiện đo cần ghi lại

- CPU, số lõi và RAM máy chủ.

- Hệ điều hành.

- Phiên bản Docker và Python.

- Phiên bản ONNX Runtime, NumPy và SciPy.

- CPU/RAM cấp cho Docker nếu có giới hạn.

- Phiên bản mô hình.

- Số mẫu và số nhịp mỗi request.

- Trạng thái có tải nền hoặc các dịch vụ khác đang chạy.

## 7. Bằng chứng đã có và phần còn cần hoàn thành

| Nội dung | Trạng thái tại thời điểm lập tài liệu |
|---|---|
| Sample_io: batch 1 và batch 4 khớp TV2 | Đã kiểm tra |
| Tiền xử lý bản ghi 100 khớp TV1 trên 2.271 nhịp | Đã kiểm tra |
| API trên máy và trong Docker khớp model_runner với bản ghi 100 | Đã kiểm tra |
| Agent dùng ECG thật và lưu báo cáo sau xác nhận | Đã kiểm tra trên máy và trong Docker |
| Benchmark 100 request | Chưa đo |
| Tải tương đương một giờ ECG | Chưa đo |
| RAM peak và thông tin phần cứng | Chưa ghi nhận |
| Đối chiếu trên toàn bộ DS2 qua API | Chưa thực hiện |

Kết quả kiểm tra bản ghi 100 được lưu trong outputs/check_real_ecg_100.json. Báo cáo Agent và nhật ký nằm trong outputs/ và logs/, không được commit mặc định lên GitHub.

## 8. Cập nhật trạng thái ngày 22/10/2026

Bảng ở mục 7 là ghi nhận tại thời điểm lập đặc tả, không phải danh sách công việc hiện tại. Sau đó đã hoàn thành benchmark API 100/100 request, tải tương đương một giờ và đo RAM (ngày 09/10); có giao diện web, so sánh CPU hai mô hình và hồ sơ chạy TV1 trên đủ 43 bản ghi. Các kết quả và giới hạn được trình bày tại PERFORMANCE_RESULTS.md và reports/.

Đã có notebook/code và cấu hình cho TV1/TV2 trong kho mã. Bootstrap cố định năm lớp, 1.000 lượt theo bản ghi, đủ sáu cặp mô hình–seed. Đánh giá 202 là bổ sung trên mô hình cố định; DS2 chính vẫn loại 202 để tránh trùng người với 201.

Đã đối chiếu API trên laptop với đủ 21 bản DS2, tổng 47.557 nhịp: số nhịp và số lượng nhãn từng lớp khớp bảng TV2. Phép đối chiếu không so xác suất từng nhịp. Hướng dẫn chạy lại nằm tại REPRODUCIBILITY.md.
