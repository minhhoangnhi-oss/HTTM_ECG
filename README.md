# HTTM_ECG — API và AI Agent phân loại nhịp ECG

Dự án môn Hệ thống thông minh, sử dụng dữ liệu ECG MIT-BIH.

Repository tích hợp tiền xử lý của TV1, mô hình CNN + RR của TV2, API FastAPI, Docker và AI Agent gọi công cụ qua OpenRouter.

Phiên bản hiện tại chạy mô hình ONNX FP32 thật trên CPU, không còn dùng mock trong API.

## 1. Trạng thái hiện tại

| Thành phần | Trạng thái |
|---|---|
| Tiền xử lý ECG và RR | Đã tích hợp theo notebook TV1 |
| CNN + RR ONNX FP32 | Đã tích hợp, phiên bản 20261008_094922 |
| API /health và /predict | Đã chạy trên máy và trong Docker |
| Đối chiếu dữ liệu thật | Đã kiểm tra bản ghi 100 trên 2.271 nhịp |
| Agent trên ECG thật | Đã chạy trên máy và trong Docker |
| Xác nhận trước khi lưu | Đã thử lưu báo cáo bằng y |
| Kiểm tra API và giới hạn công cụ | 16/16 kiểm tra đạt |
| Benchmark API | 100/100 request thành công |
| Tải tương đương khoảng một giờ | Đã đo bằng hai lượt bản ghi 100 |
| Đối chiếu toàn bộ DS2 qua API | Chưa thực hiện |

Docker này phục vụ suy luận và chạy Agent. Đây không phải môi trường huấn luyện GPU cho TV1/TV2.

## 2. Tài liệu cần đọc

- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md): sơ đồ hệ thống, luồng dữ liệu và trách nhiệm các thành viên.
- [TECHNICAL_REQUIREMENTS.md](TECHNICAL_REQUIREMENTS.md): quy ước dữ liệu, giao diện và mục tiêu đo.
- [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md): kết quả tích hợp, benchmark, cấu hình máy và giới hạn.

Bảng trạng thái trong TECHNICAL_REQUIREMENTS.md ghi nhận thời điểm lập đặc tả trước benchmark. Kết quả đo sau đó nằm trong PERFORMANCE_RESULTS.md.

## 3. Cấu trúc repository

| Tệp hoặc thư mục | Nội dung |
|---|---|
| main.py | API FastAPI |
| preprocessing.py | Lọc, RR và cắt nhịp theo TV1 |
| model_runner.py | Chuẩn hóa và chạy ONNX |
| agent_tools.py | Công cụ được phép, đọc bản ghi và tổng hợp số liệu |
| agent_main.py | Điều phối LLM, audit và xác nhận lưu |
| check_openrouter.py | Kiểm tra kết nối OpenRouter |
| check_tv2_model.py | Đối chiếu sample_io với TV2 |
| check_real_ecg.py | Đối chiếu bản ghi thật với TV1 và API |
| check_api_contract.py | Kiểm tra giao diện và giới hạn công cụ |
| benchmark_api.py | Benchmark API và tài nguyên |
| models/tv2/ | Cấu hình, chuẩn hóa và ONNX triển khai |
| data/ | Dữ liệu cục bộ, không commit mặc định |
| logs/ | Nhật ký Agent, không commit mặc định |
| outputs/ | Báo cáo và bằng chứng, không commit mặc định |
| .env.example | Mẫu biến môi trường |
| Dockerfile | Đóng gói API và Agent |
| requirements.txt | Thư viện Python |

## 4. Chạy trên máy mới bằng Docker

### 4.1. Chuẩn bị

Cần:

- Docker Desktop với Linux Engine đang chạy.
- WSL và ảo hóa đã được bật nếu dùng Docker Desktop trên Windows.
- Mạng cho lần build đầu, tải bản ghi và gọi OpenRouter.
- API key OpenRouter để chạy Agent.

Không cần cài Python trên máy chủ nếu chỉ chạy API và Agent bằng Docker.

Có thể tải ZIP repository từ GitHub rồi giải nén. Thư mục làm việc phải chứa Dockerfile, main.py và requirements.txt.

Nếu đã có Git, có thể dùng:

```powershell
git clone https://github.com/minhhoangnhi-oss/HTTM_ECG.git
```

```powershell
cd .\HTTM_ECG
```

### 4.2. Tạo biến môi trường

Chỉ chạy lệnh sao chép khi chưa có .env, tránh ghi đè cấu hình đang dùng:

```powershell
Copy-Item .\.env.example .\.env
```

```powershell
notepad .\.env
```

Điền API key của mình. Cấu hình hiện tại dùng:

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
ECG_API_URL=http://127.0.0.1:8000
```

Agent ghi lại mô hình LLM thực tế do OpenRouter trả về. Không coi tên router là tên một mô hình cố định.

Không gửi .env hoặc API key lên GitHub.

### 4.3. Tạo thư mục dữ liệu và kết quả

```powershell
New-Item -ItemType Directory -Force -Path ".\data\mitdb", ".\logs", ".\outputs"
```

### 4.4. Build image

```powershell
docker build -t httm-ecg .
```

Lần đầu cần tải image nền và thư viện. Những lần build sau có thể sử dụng cache.

### 4.5. Chuẩn bị bản ghi 100

Nếu đã có đủ 100.dat, 100.hea và 100.atr, đặt chúng vào data/mitdb/.

Nếu chưa có, tải bằng image vừa build:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)\data,target=/app/data" httm-ecg python -c "import wfdb; wfdb.dl_database('mitdb', '/app/data/mitdb', records=['100'], annotators=['atr'], overwrite=False)"
```

Lệnh này dùng mount ghi được để tải dữ liệu. Container API ở bước tiếp theo chỉ được đọc data/.

### 4.6. Chạy API

```powershell
docker run -d --rm --name httm-ecg-api --env-file .\.env -p 127.0.0.1:8000:8000 --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" --mount "type=bind,source=$($PWD.Path)\logs,target=/app/logs" --mount "type=bind,source=$($PWD.Path)\outputs,target=/app/outputs" httm-ecg
```

Kiểm tra:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json
```

Kết quả mong đợi:

```json
{
  "status": "ok",
  "mode": "onnx",
  "model_version": "20261008_094922"
}
```

### 4.7. Chạy Agent

```powershell
docker exec -it httm-ecg-api python agent_main.py
```

Agent liệt kê bản ghi được phép, gọi API cho bản ghi 100 và tóm tắt số liệu.

Khi được hỏi lưu:

- Nhập y để lưu.
- Nhập n hoặc hủy để không lưu.

Báo cáo được lưu trong outputs/ trên máy chủ. Nhật ký nằm trong logs/agent_audit.jsonl.

Chỉ metadata và thống kê tổng hợp được đưa vào nội dung công cụ gửi LLM; toàn bộ ECG không được gửi.

### 4.8. Dừng và chạy lại

```powershell
docker stop httm-ecg-api
```

Container dùng --rm nên sẽ bị xóa sau khi dừng. Image và các thư mục mount trên máy chủ vẫn còn.

Muốn chạy lại, dùng lại lệnh docker run ở mục 4.6.

Khi sửa code, cần build lại image, dừng container cũ và tạo container mới. Build image không tự cập nhật container đang chạy.

## 5. Giao diện API

### GET /health

Trả trạng thái dịch vụ, chế độ và phiên bản mô hình.

Model được nạp một lần khi khởi động. Nếu thiếu model hoặc cấu hình sai, API khởi động thất bại.

### POST /predict

| Trường | Yêu cầu |
|---|---|
| ecg | Mảng ECG gốc, chưa lọc, chưa chuẩn hóa, đơn vị mV |
| fs | 360 |
| r_peaks | Mảng chỉ số nguyên, 0-based, tăng dần, không trùng |

Cần gửi toàn bộ đỉnh R đã chọn theo TV1, kể cả nhịp biên, để RR có đúng ngữ cảnh.

Không đưa các đoạn X_waveform đã xử lý trong NPZ vào trường ecg. API sẽ lọc và cắt nhịp, nên cách đó gây xử lý hai lần.

Đầu ra gồm:

- mode, model_version và class_order.
- processed_count và skipped_count.
- beats: vị trí R, nhãn và xác suất theo từng lớp.
- skipped: vị trí R và lý do bỏ qua.

Thứ tự lớp: N, S, V, F, Q.

API trả HTTP 422 cho đầu vào không hợp lệ. Lỗi suy luận trả HTTP 500, không chuyển sang mock.

### Thử giao diện bằng tín hiệu tổng hợp

```powershell
$ecg = @(0..999 | ForEach-Object { [Math]::Sin(2 * [Math]::PI * 1.2 * $_ / 360) })
```

```powershell
$body = @{ ecg = $ecg; fs = 360; r_peaks = @(20, 400, 700) } | ConvertTo-Json -Depth 5
```

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

Hai lệnh tạo biến không in kết quả là bình thường.

Ví dụ này phải xử lý 2 nhịp và bỏ 1 nhịp biên. Nhãn trên tín hiệu tổng hợp chỉ kiểm tra luồng xử lý, không dùng đánh giá độ chính xác.

## 6. Chạy bộ kiểm tra trên Windows

Phần này cần Python 3.14 trên máy chủ. Docker API vẫn chạy ở cổng 8000.

Nếu chưa có môi trường:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Kiểm tra TV2

Cần đặt sample_io.npz của TV2 tại data/tv2/sample_io.npz:

```powershell
.\.venv\Scripts\python.exe .\check_tv2_model.py
```

### Kiểm tra dữ liệu thật và API Docker

Cần đặt ds2_data.npz của TV1 tại data/tv1/ds2_data.npz:

```powershell
.\.venv\Scripts\python.exe -c "import check_real_ecg as check; check.API_URL = 'http://127.0.0.1:8000'; check.main()"
```

Script kiểm tra ECG, RR, vị trí R và nhãn tham chiếu của bản ghi 100 so với TV1, rồi đối chiếu xác suất API với model_runner.

Nếu thiếu bản ghi gốc 100, script tải từ PhysioNet.

### Kiểm tra giao diện và công cụ

```powershell
.\.venv\Scripts\python.exe .\check_api_contract.py
```

### Benchmark

Cần có bản ghi 100 tại data/mitdb/ và container httm-ecg-api đang chạy:

```powershell
.\.venv\Scripts\python.exe .\benchmark_api.py
```

Benchmark gồm 5 request làm nóng, 100 request ECG 10 giây và hai request toàn bộ bản ghi 100.

Các script lưu bằng chứng JSON vào outputs/. Không cần API key OpenRouter để chạy các phép kiểm tra và benchmark này.

### Chạy API trực tiếp để phát triển

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Giữ cửa sổ server mở. check_real_ecg.py mặc định kiểm tra API máy chủ ở cổng 8001:

```powershell
.\.venv\Scripts\python.exe .\check_real_ecg.py
```

Agent hiện chỉ cho phép API cổng 8000. Cổng 8001 dùng riêng để phát triển và đối chiếu, tránh nhầm với Docker.

## 7. Kết quả đã ghi nhận

Cấu hình đo: Ryzen 5 5600X, 6 lõi/12 luồng, khoảng 32 GiB RAM máy chủ; Docker nhìn thấy 12 CPU logic và khoảng 15,57 GiB RAM.

| Chỉ số | Kết quả |
|---|---:|
| Bản ghi thật đã đối chiếu end-to-end | 100, 2.271 nhịp xử lý |
| Kiểm tra giao diện và giới hạn công cụ | 16/16 đạt |
| Request benchmark thành công | 100/100 |
| Median, request ECG 10 giây | 5,123 ms |
| P95, request ECG 10 giây | 5,644 ms |
| Hai lượt bản ghi 100, tương đương 60,19 phút | 0,708 giây |
| RAM peak container từ khi tạo, tại thời điểm đo | 312,87 MiB |
| Tệp ONNX | 82.719 byte |
| Docker image | 956.832.376 byte |

Chi tiết phương pháp và giới hạn nằm trong PERFORMANCE_RESULTS.md.

Tải tương đương một giờ là hai lượt của cùng bản ghi, không phải một bản ghi một giờ độc lập. RAM peak bao gồm vòng đời container, không phải RAM riêng của model.

## 8. Quy ước tích hợp cho TV1 và TV2

### TV1

Giữ đồng bộ code tiền xử lý và dữ liệu đã xuất:

- Butterworth 0,5–40 Hz, order=2, filtfilt.
- Cửa sổ signal[R-90:R+144], 234 mẫu.
- ECG [N, 1, 234], chưa chuẩn hóa.
- RR [N, 3], đơn vị giây.
- Thứ tự RR_pre, RR_post, RR_local_avg.
- Tính RR trước khi bỏ nhịp biên.
- Chọn chú thích theo ánh xạ AAMI.
- Giữ đúng cách chia DS1/DS2 và loại bản ghi theo phân công.

Nếu thay đổi cách chọn nhịp, lọc hoặc RR, cần báo TV2 và TV3, xuất lại dữ liệu và kiểm tra lại tính nhất quán.

### TV2

Bộ triển khai hiện tại gồm:

- exports/cnn_rr_fp32.onnx.
- inference_config.json.
- normalization.json.
- preprocess_config.json.

ONNX phải nhận ecg [N, 1, 234] và rr [N, 3], float32; trả probabilities [N, 5] theo thứ tự N, S, V, F, Q.

Chuẩn hóa ngoài ONNX đúng một lần. Không tự tính lại mean/std bằng DS2.

Khi đổi model, bàn giao kèm cấu hình, phiên bản, thông số chuẩn hóa và sample_io tham chiếu. Không chỉ thay tệp ONNX rồi giữ cấu hình cũ.

Model INT8 trong bundle TV2 là bản thử nghiệm, chưa được chọn để triển khai ở repository này.

### Bàn giao báo cáo

TV1/TV2 tiếp tục bàn giao các kết quả theo phân công: chia dữ liệu, quá trình huấn luyện, đánh giá, khoảng tin cậy và phân tích sai số.

TV3 đã có code tích hợp, sơ đồ, bảng yêu cầu và kết quả kiểm thử. Các bằng chứng JSON và audit cần gửi riêng khi tổng hợp báo cáo vì không được commit mặc định.

## 9. Giới hạn cần nêu trong báo cáo

- Chưa có bộ phát hiện R; cần vị trí R được cung cấp.
- filtfilt và RR_post dùng thông tin sau nhịp, phục vụ xem lại bản ghi.
- Agent chỉ cho phép bản ghi 100 và API cục bộ ở cổng 8000.
- Đối chiếu dữ liệu thật end-to-end mới thực hiện trên bản ghi 100.
- Chưa đối chiếu toàn bộ DS2 qua API.
- Chưa đo nhiều client đồng thời, mạng ngoài localhost hoặc chạy dài hạn.
- Benchmark API không bao gồm thời gian gọi OpenRouter.
- Docker suy luận không thay thế môi trường huấn luyện của TV2.
- Nhãn dự đoán và bản tóm tắt Agent không phải chẩn đoán hoặc đề xuất điều trị.
