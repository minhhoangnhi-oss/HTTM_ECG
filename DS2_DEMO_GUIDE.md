# Chọn bản ghi DS2 để demo

## Thay đổi

Web có danh sách 21 bản ghi DS2: 100, 103, 105, 111, 113, 117, 121, 123, 200, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234. Đây là tập kiểm tra đã dùng đánh giá, không có bản ghi DS1 và không có 202. Tải thêm dữ liệu không mở quyền truy cập các bản ngoài danh sách này.

Mô hình, normalization, tiền xử lý và endpoint /predict giữ nguyên. Mỗi lần phân loại vẫn xử lý một bản ghi. Không huấn luyện lại và không thay ONNX.

Chọn bản ghi trong ô bên trái, rồi bấm gợi ý “Phân tích bản ghi …”. Có thể nhập “Phân tích bản đang chọn”. Nếu câu hỏi nêu rõ mã khác, mã trong câu hỏi được ưu tiên; bản ngoài quyền bị từ chối, không đổi sang bản đang chọn để phân tích thay. Đổi lựa chọn sẽ xóa lịch sử gửi LLM để tránh lẫn bản ghi, nhưng các thẻ chat/báo cáo cũ vẫn hiện đúng mã của lượt cũ.

Danh sách cho biết bản nào đủ ba tệp .dat/.hea/.atr. Thiếu dữ liệu sẽ báo rõ; danh sách không tự tải tệp. Đọc header và kiểm tra tín hiệu tiếp tục được thực hiện khi gọi công cụ.

Chữ chờ giữ là “Đang nghĩ…”. Báo cáo chỉ lưu sau xác nhận và chứa record_id thực tế cùng lựa chọn giao diện lúc gửi. Giá trị selected_record_id không thay thế record_id thực tế của kết quả API.

## Dữ liệu

Đặt các tệp trực tiếp ở data/mitdb/, ví dụ data/mitdb/103.dat, 103.hea, 103.atr. Dữ liệu trong thư mục x_mitdb không thay thế các bản ghi chính. Không đưa dữ liệu gốc lên GitHub.

## Chạy bản mới trên laptop

Chép các tệp trong gói cập nhật vào thư mục dự án đúng đường dẫn tương đối, sau đó build lại image và tạo lại container từ image mới. Giữ nguyên .env, data, logs, outputs. Với container đặt tên httm-ecg-laptop-api:

```powershell
docker build -t httm-ecg-laptop-check .
```

```powershell
docker stop httm-ecg-laptop-api
```

```powershell
docker rm httm-ecg-laptop-api
```

```powershell
docker run -d --name httm-ecg-laptop-api --env-file .\.env -p 127.0.0.1:8000:8000 --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" --mount "type=bind,source=$($PWD.Path)\logs,target=/app/logs" --mount "type=bind,source=$($PWD.Path)\outputs,target=/app/outputs" httm-ecg-laptop-check
```

Mở lại http://127.0.0.1:8000/ và nhấn Ctrl+F5. Các lần sau dùng nút Start/Stop trong Docker Desktop, không xóa container nếu chỉ muốn tắt/mở lại bản này.

## Kiểm tra trước khi quay

1. /health vẫn trả onnx, phiên bản 20261008_094922.
2. Danh sách có 21 bản DS2; chọn 100/103/105 và phân loại từng bản. Kết quả và báo cáo phải mang đúng mã.
3. Chọn 103, nhập câu hỏi về 105: chỉ phân loại 105 theo câu hỏi, không lấy kết quả 103.
4. Yêu cầu 101 hoặc 202: từ chối, không tạo kết quả API để lưu.
5. Lưu một lượt, bỏ lưu một lượt; kiểm tra JSON đã lưu có đúng mã và confirmed=true.
6. Đối chiếu cả 21 bản với số lượng nhãn từ TV2 bằng lệnh dưới. Không cần Python trên Windows hoặc key LLM để chạy phép đối chiếu này:

```powershell
docker exec httm-ecg-laptop-api python check_ds2_records.py
```

Phép đối chiếu gọi API thật cho từng bản và so số nhịp, số lượng nhãn N/S/V/F/Q với dự đoán TV2 lịch sử seed42. Bảng tham chiếu gồm 21 bản, tổng 47.557 nhịp, trích từ ds2_predictions.csv. Phép đối chiếu này kiểm tra số nhịp và số lượng nhãn theo từng bản, không so xác suất từng nhịp. Nếu không khớp, giữ log/JSON để điều tra, không sửa bảng chuẩn để ép đạt.

## Agent dòng lệnh

```powershell
docker exec -it httm-ecg-laptop-api python agent_main.py --record-id 103
```

Không truyền --record-id thì CLI vẫn mặc định dùng 100.

## Phạm vi đã kiểm tra khi chuẩn bị gói

Đã kiểm tra điều phối bằng mô phỏng và đã chạy API trên laptop cho đủ 21 bản DS2. Số nhịp và số lượng nhãn từng lớp khớp bảng TV2, tổng 47.557 nhịp. Hồ sơ: `reports/deployment/ds2_records_check_b9a4a1c058d04fdeaef6dfedf6c8ee6e.json`. Các benchmark cũ vẫn thuộc bản triển khai và máy đã ghi trong tài liệu.
