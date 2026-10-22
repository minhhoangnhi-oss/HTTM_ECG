# Đồng bộ hồ sơ ngày 22/10/2026

Bản gốc đối chiếu: commit 4e069929545a0d6cbd042945d9dce821c9d8edad. Bản cập nhật này chưa được đẩy lên GitHub.

## Đã cập nhật

- README, thiết kế hệ thống, kết quả và trạng thái yêu cầu kỹ thuật theo hồ sơ TV1/TV2 mới.
- REPRODUCIBILITY.md: hướng dẫn riêng Docker, TV1, TV2 và benchmark CPU; liệt kê dữ liệu/gói cần gửi riêng.
- reports/: CSV nhỏ gốc của TV2, so sánh 202 và JSON benchmark CPU; không đưa dữ liệu gốc hoặc bảng dự đoán lớn lên Git.
- Tách notebook train TV2 và notebook vẽ hình từ kết quả lịch sử. Giữ nguyên mã mô hình/huấn luyện/đánh giá, pin đủ thư viện và thống nhất đường dẫn v2.
- Bỏ các môi trường .venv-* khỏi Git/Docker context.
- Cập nhật hash README TV1 trong manifest theo đúng nội dung hiện tại; không thay hash NPZ lịch sử.

## Kiểm tra trong lần này

- Toàn bộ Python runtime, ONNX và cấu hình mô hình giữ nguyên byte so với ZIP gốc.
- Kiểm tra cú pháp Python và code cell notebook; kiểm tra liên kết tài liệu nội bộ.
- Đối chiếu bảng DS2 trung bình/SD từ CSV gốc; hash script CPU khớp hồ sơ đã đo.
- Bằng chứng CSV/JSON bổ sung giữ nguyên byte từ gói bàn giao.


## Chép về PC

Giải nén ZIP ra một thư mục tạm. Chép toàn bộ nội dung bên trong HTTM_ECG-main vào thư mục dự án HTTM_ECG hiện tại; chọn thay thế tệp trùng. Không xóa thư mục dự án cũ trước khi chép. ZIP không chứa .git, .env, dữ liệu, logs hoặc outputs nên các thư mục cục bộ đó được giữ lại.

Sau khi chép, kiểm tra git status --short và git diff --check trước khi commit/push. Ghi commit mới thực tế vào bản báo cáo cuối sau khi push thành công; không tiếp tục dùng commit gốc nếu nội dung nộp đã thay đổi.
