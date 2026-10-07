# HTTM_ECG — API và AI Agent cho phân loại nhịp ECG

Dự án môn Hệ thống thông minh, sử dụng dữ liệu ECG MIT-BIH.

Repository hiện cung cấp API, Docker và AI Agent để chuẩn bị tích hợp phần tiền xử lý và mô hình của các thành viên.

## 1. Trạng thái hiện tại

| Thành phần | Trạng thái |
|---|---|
| API `/health` và `/predict` | Đã chạy thử trên máy và trong Docker |
| Kiểm tra đầu vào và vị trí đỉnh R | Đã có |
| Phản hồi tiếng Việt UTF-8 | Đã có |
| AI Agent gọi công cụ qua OpenRouter | Đã chạy thử |
| Công cụ liệt kê và xử lý bản ghi được phép | Đã có bản demo |
| Tổng hợp kết quả API bằng LLM | Đã có |
| Hỏi người dùng trước khi lưu báo cáo | Đã kiểm tra chọn `y` và `n` |
| Nhật ký hoạt động của agent | Đã có |
| Tiền xử lý ECG và tính RR thực tế | Chờ tích hợp |
| Mô hình phân loại thật | Chờ tích hợp |
| Đọc bản ghi MIT-BIH thực tế | Chờ tích hợp |
| Đo hiệu năng hệ thống với mô hình thật | Chưa thực hiện |

**API hiện chạy ở chế độ `mock`.**

Các trường `predicted_class`, `probabilities` và `model_version` hiện là `null`. Kết quả này chỉ dùng kiểm tra luồng hoạt động, chưa phải kết quả phân loại của mô hình.

## 2. Cấu trúc project

| File hoặc thư mục | Chức năng |
|---|---|
| `main.py` | FastAPI, kiểm tra đầu vào và trả kết quả |
| `agent_tools.py` | Khai báo công cụ, giới hạn bản ghi và gọi API |
| `agent_main.py` | Vòng lặp agent, gọi LLM, ghi nhật ký và hỏi xác nhận lưu |
| `check_openrouter.py` | Kiểm tra riêng kết nối OpenRouter và yêu cầu gọi công cụ |
| `requirements.txt` | Các thư viện Python và phiên bản |
| `Dockerfile` | Đóng gói API và agent |
| `.env.example` | Mẫu cấu hình |
| `.env` | Cấu hình và API key riêng trên từng máy |
| `logs/` | Nhật ký hoạt động |
| `outputs/` | Báo cáo đã được người dùng xác nhận lưu |

`.env`, `logs/` và `outputs/` được loại khỏi Git.

Docker image chứa API và agent. API key được truyền vào khi chạy container qua `--env-file`, không được chép vào image.

## 3. Quy ước kỹ thuật chung

Các thành viên cần tuân theo quy ước trong tài liệu phân công.

| Nội dung | Quy ước |
|---|---|
| Tần số lấy mẫu | 360 Hz |
| Tín hiệu đầu vào | Một kênh ECG đã thống nhất |
| Chỉ số đỉnh R | Bắt đầu từ 0 |
| Danh sách đỉnh R | Tăng dần, không trùng |
| Cửa sổ nhịp | `signal[R-90:R+144]` |
| Số mẫu mỗi nhịp | 234 |
| Tensor ECG dùng khi tích hợp | `[N, 1, 234]` |
| Vector RR | `[RR_pre, RR_post, RR_local_avg]` |
| Tensor RR | `[N, 3]` |
| Đơn vị RR | Giây |
| Thứ tự lớp | `N, S, V, F, Q` |

Nếu mô hình xuất ra yêu cầu layout khác, thành viên 2 cần ghi rõ để phần tích hợp chuyển đổi đúng.

Nhóm cần thống nhất thêm cách lọc, chuẩn hóa, tính `RR_local_avg` và xử lý nhịp thiếu RR ở đầu/cuối bản ghi. API phải sử dụng cùng cách xử lý với quá trình huấn luyện và đánh giá.

Do có sử dụng `RR_post`, hệ thống được xây dựng cho việc xem lại bản ghi ECG đã thu.

## 4. Chạy bằng Docker trên Windows

Các lệnh dưới đây dùng PowerShell. Chạy từng lệnh riêng.

### 4.1. Chuẩn bị

Máy cần có Git và Docker Desktop đang hoạt động.

Kiểm tra Git:

```powershell
git --version
```

Kiểm tra Docker:

```powershell
docker version
```

Docker cần hiển thị cả phần `Client` và `Server`.

### 4.2. Tải project

```powershell
git clone https://github.com/minhhoangnhi-oss/HTTM_ECG.git
```

```powershell
cd .\HTTM_ECG
```

Nếu repository riêng tư, tài khoản GitHub cần được cấp quyền truy cập.

### 4.3. Tạo cấu hình riêng

```powershell
Copy-Item .\.env.example .\.env
```

```powershell
notepad .\.env
```

Điền API key OpenRouter của mình vào:

```dotenv
OPENROUTER_API_KEY=DIEN_API_KEY_CUA_BAN
OPENROUTER_MODEL=openrouter/free
ECG_API_URL=http://127.0.0.1:8000
```

Lưu file rồi đóng Notepad.

Chỉ cần API key khi chạy agent. API mock có thể chạy mà chưa điền key.

Không đưa API key thật vào `.env.example`, README hoặc code.

### 4.4. Tạo thư mục lưu kết quả

```powershell
New-Item -ItemType Directory -Force .\logs | Out-Null
```

```powershell
New-Item -ItemType Directory -Force .\outputs | Out-Null
```

### 4.5. Build image

```powershell
docker build -t httm-ecg .
```

### 4.6. Chạy API

```powershell
docker run -d --rm --name httm-ecg-api --env-file .\.env -p 127.0.0.1:8000:8000 --mount "type=bind,source=$($PWD.Path)\logs,target=/app/logs" --mount "type=bind,source=$($PWD.Path)\outputs,target=/app/outputs" httm-ecg
```

Nếu đã có container cùng tên đang chạy, dừng container đó trước khi chạy lại.

Kiểm tra API:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Kết quả hiện tại:

```json
{
  "status": "ok",
  "mode": "mock"
}
```

Tài liệu API tương tác:

http://127.0.0.1:8000/docs

### 4.7. Chạy AI Agent

```powershell
docker exec -it httm-ecg-api python agent_main.py
```

Agent sẽ gọi công cụ, hiển thị số liệu từ API và bản tóm tắt.

Khi chương trình hỏi:

```text
Lưu báo cáo? Nhập y để lưu, n để bỏ:
```

- Nhập `y`: lưu báo cáo JSON vào `outputs/`.
- Nhập `n`: không tạo báo cáo.
- Ngắt chương trình tại bước xác nhận: không lưu báo cáo.

Các thư mục `logs/` và `outputs/` được gắn với thư mục trên máy, nên dữ liệu đã ghi vẫn còn sau khi container dừng.

### 4.8. Xem log hoặc dừng API

Xem log của dịch vụ API:

```powershell
docker logs httm-ecg-api
```

Dừng container:

```powershell
docker stop httm-ecg-api
```

Container được chạy với `--rm`, nên sẽ bị xóa sau khi dừng. Image và các file trong thư mục trên máy vẫn còn.

### 4.9. Khi cập nhật code

Container đang chạy sử dụng code của lần build trước.

Sau khi cập nhật code, cần dừng container, build lại image rồi chạy lại lệnh ở mục 4.6.

Sau khi sửa `.env`, cũng cần tạo lại container để nạp cấu hình mới.

## 5. Kiểm tra API `/predict`

API nhận JSON với ba trường:

| Trường | Ý nghĩa |
|---|---|
| `ecg` | Mảng giá trị tín hiệu ECG, tối thiểu 234 mẫu |
| `fs` | Tần số lấy mẫu, hiện chỉ chấp nhận 360 |
| `r_peaks` | Danh sách chỉ số đỉnh R |

### Ví dụ bằng PowerShell

Tạo tín hiệu giả 1000 mẫu:

```powershell
$ecg = @(0..999 | ForEach-Object { [Math]::Sin(2 * [Math]::PI * 1.2 * $_ / 360) })
```

Tạo nội dung yêu cầu:

```powershell
$body = @{ ecg = $ecg; fs = 360; r_peaks = @(20, 400, 700) } | ConvertTo-Json -Depth 10
```

Gọi API:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

Kết quả mong đợi trong chế độ mock:

- `processed_count`: 2.
- `skipped_count`: 1.
- Các nhịp tại R = 400 và R = 700 đủ cửa sổ.
- Nhịp tại R = 20 bị bỏ qua vì thiếu mẫu trước đỉnh R.
- Nhãn và xác suất của hai nhịp được xử lý đều là `null`.

### Các trường phản hồi

| Trường | Ý nghĩa |
|---|---|
| `mode` | Chế độ hoạt động |
| `model_version` | Phiên bản mô hình |
| `class_order` | Thứ tự lớp tương ứng với vector xác suất |
| `processed_count` | Số nhịp được xử lý |
| `skipped_count` | Số nhịp bị bỏ qua |
| `beats` | Kết quả theo từng đỉnh R |
| `skipped` | Các đỉnh R bị bỏ qua và lý do |

Mỗi phần tử trong `beats` có:

```json
{
  "r_peak_index": 400,
  "predicted_class": null,
  "probabilities": null
}
```

Khi tích hợp mô hình thật, cần trả vector xác suất theo thứ tự `N, S, V, F, Q`.

### Kiểm tra đầu vào hiện có

API từ chối yêu cầu không hợp lệ, gồm:

- Tần số lấy mẫu khác 360.
- Tín hiệu quá ngắn hoặc chứa giá trị không hữu hạn.
- Danh sách đỉnh R rỗng.
- Chỉ số R âm, không phải số nguyên hoặc vượt ngoài tín hiệu.
- Danh sách R không tăng dần hoặc có vị trí trùng.

Đỉnh R nằm trong tín hiệu nhưng thiếu cửa sổ 234 mẫu được đưa vào `skipped`, kèm lý do.

## 6. Cách AI Agent hoạt động

Luồng hiện tại:

1. LLM gọi `list_records` để xem các bản ghi được phép.
2. LLM gọi `classify_record` cho bản ghi `demo_001`.
3. Chương trình thực thi công cụ và gọi API `/predict`.
4. Công cụ tổng hợp số lượng nhịp theo lớp.
5. LLM nhận kết quả công cụ và viết bản tóm tắt.
6. Chương trình hiển thị số liệu API cùng bản tóm tắt.
7. Người dùng xác nhận trước khi chương trình lưu báo cáo.

### Công cụ được phép

| Công cụ | Chức năng |
|---|---|
| `list_records` | Liệt kê bản ghi được phép |
| `classify_record` | Gọi API cho bản ghi được phép |

Hiện chỉ có bản ghi thử `demo_001`, là tín hiệu sine tổng hợp 1000 mẫu.

Agent chưa đọc dữ liệu MIT-BIH thật.

Trong bản hiện tại, LLM nhận metadata và số liệu tổng hợp. Mảng ECG được tạo và xử lý trong công cụ, không đưa vào nội dung gửi đến OpenRouter.

Công cụ chỉ chấp nhận API tại:

- `http://127.0.0.1:8000`
- `http://localhost:8000`

LLM không được cung cấp công cụ chạy lệnh hệ thống, truy cập đường dẫn tùy ý, ghi file hoặc gửi báo cáo. Việc lưu do chương trình thực hiện sau khi người dùng xác nhận.

Vòng lặp có giới hạn 6 lượt gọi LLM và tối đa 8 lượt gọi công cụ.

### Nhật ký và báo cáo

Nhật ký:

```text
logs/agent_audit.jsonl
```

Nhật ký ghi nhận lần chạy, công cụ được yêu cầu, thành công hoặc lỗi, mô hình LLM thực tế và quyết định lưu báo cáo.

Báo cáo:

```text
outputs/report_<run_id>.json
```

Báo cáo chứa kết quả API, bản tóm tắt, thời gian tạo, trạng thái xác nhận và thông tin mô hình LLM.

`openrouter/free` là cấu hình thử hiện tại. Mô hình thực tế được ghi lại trong nhật ký và báo cáo. Khi đánh giá chính thức, nhóm cần thống nhất cấu hình LLM dùng cho thử nghiệm.

## 7. Phần cần bàn giao để tích hợp

### Thành viên 1 — Tiền xử lý và RR

Cần bàn giao:

- Code đọc dữ liệu và chọn kênh ECG.
- Code lọc và chuẩn hóa theo phương pháp nhóm đã thống nhất.
- Code cắt nhịp 234 mẫu và tạo đặc trưng RR.
- Quy tắc xử lý biên bản ghi, nhịp bị bỏ qua và thiếu RR.
- Định nghĩa cụ thể của `RR_local_avg`.
- Cách đối chiếu mỗi hàng tensor với chỉ số đỉnh R.
- Một ví dụ đầu vào và đầu ra để kiểm tra tích hợp.

Các tham số tiền xử lý cần dùng chung giữa huấn luyện, đánh giá và API.

### Thành viên 2 — Mô hình

Cần bàn giao:

- File mô hình dùng để suy luận, gồm bản ONNX theo yêu cầu.
- Bản INT8 nếu đã hoàn thành phần lượng tử hóa.
- Tên, shape và dtype của từng đầu vào.
- Xác nhận layout ECG và thứ tự đặc trưng RR.
- Tên và ý nghĩa đầu ra: logits hay xác suất.
- Thứ tự lớp.
- Phiên bản mô hình và thư viện cần thêm.
- Script suy luận mẫu và kết quả tham chiếu.

Nếu đầu ra là logits, phần tích hợp cần chuyển sang xác suất đúng cách. Nếu đầu ra đã là xác suất, không áp dụng softmax thêm lần nữa.

### Thành viên 3 — Tích hợp hệ thống

Sau khi nhận phần bàn giao:

- Kết nối tiền xử lý và mô hình vào `/predict`.
- Giữ đúng liên kết giữa kết quả và chỉ số đỉnh R.
- Cập nhật nhãn, xác suất và phiên bản mô hình.
- Thay bản ghi tổng hợp trong công cụ bằng các bản ghi thực được phép.
- Cập nhật dependencies và Docker.
- So sánh kết quả API với script suy luận tham chiếu.
- Đo hiệu năng hệ thống và hoàn thiện tài liệu.

Giữ rõ chế độ mock cho đến khi mô hình thật đã được nạp và chạy thành công.

## 8. Các việc còn lại trước khi hoàn thành bài

- Tích hợp tiền xử lý và mô hình thật.
- Kiểm tra API và script đánh giá trả kết quả thống nhất.
- Kiểm tra agent với lỗi API, bản ghi không được phép và công cụ không được phép.
- Kiểm tra lưu báo cáo và nhật ký khi chạy hoàn toàn trong Docker.
- Đo latency ít nhất 100 lần gọi; báo cáo median và P95.
- Đo thời gian xử lý lượng ECG tương đương 1 giờ.
- Khi xử lý nhiều bản ghi, tính RR riêng từng bản ghi.
- Ghi nhận kích thước mô hình, RAM và cấu hình máy thử.
- Hoàn thiện sơ đồ hệ thống, bảng yêu cầu kỹ thuật và phần báo cáo được phân công.
- Chuẩn bị video demo theo yêu cầu môn học.

Các con số hiệu năng phải được đo bằng phiên bản hệ thống và mô hình thực tế. Project hiện chưa có số liệu hiệu năng của mô hình thật.

## 9. Lưu ý khi làm việc nhóm

- Mỗi người giữ `.env` và API key riêng.
- Không commit dữ liệu ECG thô, API key, log hoặc báo cáo chạy thử.
- Thống nhất giao diện tiền xử lý và mô hình trước khi tích hợp.
- Ghi rõ thay đổi dependencies và cách chạy trong README.
- Khi sửa quy ước tensor, RR hoặc thứ tự lớp, cần cập nhật các phần liên quan.