\# HTTM ECG — Phát hiện loạn nhịp trên MIT-BIH



\## Trạng thái hiện tại



API đang ở chế độ mock, chưa tích hợp tiền xử lý và mô hình thật.

Nhãn, xác suất và phiên bản mô hình hiện trả về null.



Đã có:

\- API kiểm tra đầu vào.

\- Kiểm tra cửa sổ 234 mẫu quanh R.

\- Thông tin nhịp bị bỏ qua và lý do.

\- Đóng gói và chạy bằng Docker.



\## Yêu cầu chạy



Docker đã cài và engine đang chạy ở chế độ Linux containers.

Không cần cài Python riêng trên máy nếu chạy bằng Docker.



\## Đóng gói



Chạy tại thư mục chứa Dockerfile:



```bash

docker build -t httm-ecg .

```



\## Khởi động



```bash

docker run --rm --name httm-ecg-api -p 127.0.0.1:8000:8000 httm-ecg

```



Giữ cửa sổ terminal mở. Nhấn Ctrl+C để dừng.



\## Truy cập



\- Kiểm tra dịch vụ: http://127.0.0.1:8000/health

\- Tài liệu và thử API: http://127.0.0.1:8000/docs



\## Đầu vào POST /predict



\- ecg: mảng tín hiệu một kênh, ít nhất 234 mẫu.

\- fs: 360 Hz.

\- r\_peaks: chỉ số mẫu của các đỉnh R, bắt đầu từ 0, tăng dần và không trùng.



Nhịp được kiểm tra theo cửa sổ ecg\[R-90:R+144].

Nhịp thiếu mẫu ở biên được ghi vào skipped.



\## Phần cần tích hợp



\- Thành viên 1: module tiền xử lý và trích đặc trưng RR.

\- Thành viên 2: mô hình ONNX và cấu hình chuẩn hóa.

\- Thành viên 3: tích hợp mô hình, AI agent, kiểm thử và đo hiệu năng.



\## Quản lý dữ liệu



Không đưa dữ liệu gốc MIT-BIH, thông tin định danh hoặc API key lên Git.

Dữ liệu được tải riêng theo hướng dẫn của phần xử lý dữ liệu.

