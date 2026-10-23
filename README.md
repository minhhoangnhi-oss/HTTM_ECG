Phát hiện loạn nhịp ECG trên dữ liệu MIT-BIH

Dự án môn **Hệ thống thông minh**, xây dựng hệ thống phân loại từng nhịp ECG vào năm nhóm **N, S, V, F, Q**. Nhóm sử dụng CNN 1D kết hợp đặc trưng khoảng cách giữa các nhịp tim (RR), sau đó đưa mô hình vào một dịch vụ có giao diện chat trên trình duyệt.

Giao diện cho chọn 21 bản ghi thuộc tập kiểm tra DS2. Người dùng có thể yêu cầu phân tích bản ghi, xem số nhịp dự đoán theo từng lớp và chọn lưu báo cáo JSON. **Mô hình ECG thực hiện phân loại; AI Agent gọi công cụ và tóm tắt kết quả bằng tiếng Việt.**

Mô hình triển khai hiện tại là **CNN + RR, seed 42, ONNX FP32**, phiên bản `20261008_094922`, chạy trên CPU. Kết quả phục vụ học tập và nghiên cứu, không dùng làm chẩn đoán bệnh.

## 1. Nên bắt đầu từ đâu?

| Bạn muốn làm gì? | Đọc phần nào? |
|---|---|
| Mở web và chạy thử | [Mục 2](#2-mở-web-và-chạy-thử) |
| Biết nên hỏi Agent như thế nào | [Mục 3](#3-sử-dụng-giao-diện-chat) |
| Tắt chương trình hoặc mở lại hôm sau | [Mục 4](#4-tắt-và-mở-lại) |
| Kiểm tra API và kết quả tích hợp | [Mục 5](#5-kiểm-tra-api-và-kết-quả-tích-hợp) |
| Chạy lại tiền xử lý, huấn luyện hoặc vẽ hình | [Mục 6](#6-chạy-lại-thí-nghiệm-của-nhóm) |
| Xem kết quả đã có và phạm vi đã kiểm tra | [Mục 7](#7-kết-quả-đã-ghi-nhận) |

## 2. Mở web và chạy thử

Các lệnh dưới đây dùng **PowerShell trên Windows**. Chạy từng khối lệnh riêng, trong thư mục dự án có `Dockerfile`, `main.py` và `requirements.txt`.

### Bước 1 — Chuẩn bị máy và mã nguồn

Máy cần Docker Desktop với **Linux Engine đang chạy**. Trên Windows, cần bật ảo hóa và WSL theo hướng dẫn của Docker Desktop. Cần mạng để tải thư viện lần đầu, tải bản ghi ECG và gọi OpenRouter.

**Chỉ chạy bản demo bằng Docker thì không cần cài Python hoặc PyTorch trên Windows.**

Tải ZIP từ GitHub rồi giải nén. Nếu dùng Git, có thể tải bằng:

```powershell
git clone https://github.com/minhhoangnhi-oss/HTTM_ECG.git
```

```powershell
cd .\HTTM_ECG
```

Kiểm tra Docker đã hoạt động:

```powershell
docker version
```

Kết quả cần có cả phần `Client` và `Server`. Nếu báo không kết nối được Engine, mở Docker Desktop và đợi Engine khởi động trước khi tiếp tục.

### Bước 2 — Tạo cấu hình riêng

Tạo `.env` từ tệp mẫu nếu máy chưa có. Lệnh này giữ nguyên `.env` đã tồn tại:

```powershell
if (-not (Test-Path .\.env)) { Copy-Item .\.env.example .\.env }
```

Mở tệp để điền cấu hình:

```powershell
notepad .\.env
```

Nội dung mẫu:

```dotenv
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
ECG_API_URL=http://127.0.0.1:8000
```

Điền API key OpenRouter của bạn sau dấu `=` ở dòng đầu, rồi lưu tệp. Key cần cho chức năng chat; API phân loại ECG vẫn có thể chạy khi chưa có key. Không đưa `.env` hoặc API key lên GitHub.

### Bước 3 — Tạo thư mục và đóng gói chương trình

```powershell
New-Item -ItemType Directory -Force -Path ".\data\mitdb", ".\logs", ".\outputs"
```

```powershell
docker build -t httm-ecg .
```

Lần đầu có thể mất thời gian vì cần tải image Python và thư viện. Những lần sau Docker có thể dùng lại phần đã tải.

### Bước 4 — Chuẩn bị bản ghi ECG để demo

Giao diện Agent cho chọn **21 bản ghi DS2 của MIT-BIH**. Bản 100 là ví dụ mặc định bên dưới. Danh sách cho biết bản nào đã có dữ liệu trên máy; xem [hướng dẫn DS2](DS2_DEMO_GUIDE.md) để kiểm tra đủ 21 bản.

Nếu đã có `100.dat`, `100.hea` và `100.atr`, đặt cả ba tệp vào `data/mitdb/` và bỏ qua lệnh tải dưới đây.

Nếu chưa có, tải bằng image vừa tạo:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)\data,target=/app/data" httm-ecg python -c "import wfdb; wfdb.dl_database('mitdb', '/app/data/mitdb', records=['100'], annotators=['atr'], overwrite=False)"
```

Dữ liệu gốc không nằm trong ZIP GitHub. Chỉ cần bản 100 để chạy ví dụ mặc định. Muốn chọn đủ 21 bản DS2, tải chúng bằng lệnh dưới, không cần tải các bản DS1:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)\data,target=/app/data" httm-ecg python -c "import wfdb; wfdb.dl_database('mitdb', '/app/data/mitdb', records='100 103 105 111 113 117 121 123 200 210 212 213 214 219 221 222 228 231 232 233 234'.split(), annotators=['atr'], overwrite=False)"
```

### Bước 5 — Khởi động dịch vụ

```powershell
docker run -d --rm --name httm-ecg-api --env-file .\.env -p 127.0.0.1:8000:8000 --mount "type=bind,source=$($PWD.Path)\data,target=/app/data,readonly" --mount "type=bind,source=$($PWD.Path)\logs,target=/app/logs" --mount "type=bind,source=$($PWD.Path)\outputs,target=/app/outputs" httm-ecg
```

Lệnh trả về một chuỗi mã container là bình thường. Kiểm tra dịch vụ:

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

### Bước 6 — Mở giao diện

```powershell
Start-Process "http://127.0.0.1:8000/"
```

Bạn cũng có thể tự nhập địa chỉ trên vào trình duyệt. Máy khác sẽ không mở được dịch vụ này bằng địa chỉ `127.0.0.1` của họ; cấu hình hiện tại chạy trên máy của bạn.

## 3. Sử dụng giao diện chat

Kiểm tra trạng thái dịch vụ, API key và dữ liệu trên giao diện. Chọn mã bản ghi trong ô bên trái, rồi bấm gợi ý phân tích hoặc nhập câu hỏi. Nếu câu hỏi ghi mã khác, hệ thống ưu tiên mã bạn gõ; bản ngoài DS2 bị từ chối. Có thể gửi:

- “Liệt kê các bản ghi được phép truy cập.”
- “Phân loại bản đang chọn và thống kê số nhịp từng lớp.”
- “Phân tích bản ghi 103.”
- “Bản ghi 100 có bao nhiêu nhịp bị bỏ qua và chưa được phân loại?”
- “Đỉnh R của bản ghi 100 được lấy từ đâu?”

Agent sẽ gọi công cụ để lấy thông tin rồi trả lời. Mở **Thông tin lần chạy** để xem công cụ đã dùng và tên mô hình LLM thực tế. Với `openrouter/free`, mô hình được chọn có thể khác giữa các lượt; thời gian phản hồi cũng có thể thay đổi.

Kết quả tham chiếu của bản ghi 100:

| Nội dung | Giá trị |
|---|---:|
| Số mẫu tín hiệu | 650.000 |
| Số đỉnh R từ chú thích có sẵn | 2.273 |
| Số nhịp xử lý được | 2.271 |
| Số nhịp bỏ qua do thiếu cửa sổ | 2 |
| Nhãn dự đoán N / S / V / F / Q | 2.236 / 34 / 1 / 0 / 0 |
| Số nhịp chưa phân loại | 0 |

Đây là **số lượng nhãn mô hình dự đoán**, không phải số nhịp dự đoán đúng. `samples` đếm mẫu tín hiệu; `processed_count` và `skipped_count` đếm nhịp.

Sau khi xem kết quả, chọn **Lưu báo cáo** để tạo tệp JSON hoặc **Bỏ qua lưu** để không tạo tệp. Báo cáo đã lưu nằm trong `outputs/` trên Windows và có thể tải từ giao diện. Nhật ký hoạt động nằm trong `logs/agent_audit.jsonl`.

Bản xem trước chưa lưu chỉ được giữ tạm tối đa một giờ và mất khi chương trình khởi động lại. Agent không có công cụ chạy shell hoặc đọc tệp tùy ý. Công cụ gửi cho LLM metadata và thống kê tổng hợp, không gửi toàn bộ sóng ECG. Khi nhập yêu cầu chat, chỉ dùng thông tin cần thiết cho bản ghi demo.

Nếu muốn chạy Agent ngay trong PowerShell thay vì dùng web:

```powershell
docker exec -it httm-ecg-api python agent_main.py
```

Khi được hỏi lưu, nhập `y` để lưu hoặc `n` để bỏ.

## 4. Tắt và mở lại

Dừng dịch vụ trước khi kết thúc:

```powershell
docker stop httm-ecg-api
```

Sau đó có thể đóng PowerShell và Docker Desktop. Dữ liệu, nhật ký và báo cáo đã lưu trong các thư mục trên Windows vẫn còn.

Lần sau, mở Docker Desktop, mở PowerShell trong thư mục dự án và chạy lại lệnh **Bước 5**, rồi mở web. Không cần tạo lại `.env`, tải lại dữ liệu hoặc build lại nếu mã nguồn và thư viện không thay đổi.

Khi cập nhật code, build lại image, dừng container cũ rồi chạy lại Bước 5. Việc build image không tự thay code trong container đang chạy.

<details>
<summary>Một số lỗi thường gặp</summary>

| Hiện tượng | Cách xử lý |
|---|---|
| Không kết nối được Docker Engine | Mở Docker Desktop, đợi Engine chạy rồi thử lại `docker version` |
| Tên `httm-ecg-api` đã được dùng | Kiểm tra `docker ps -a`; nếu là container demo cũ đang chạy, dừng bằng `docker stop httm-ecg-api` |
| Không mở được web | Kiểm tra `docker ps` và `/health`; xem lỗi bằng `docker logs httm-ecg-api` |
| Chat báo chưa có API key | Kiểm tra `.env`, sau đó dừng và tạo lại container để nạp cấu hình mới |
| Bản đang chọn chưa có dữ liệu | Kiểm tra đủ ba tệp `<mã>.dat`, `<mã>.hea`, `<mã>.atr` trong `data/mitdb/` |
| Chat không lấy được kết quả công cụ | Xem trạng thái dịch vụ và thông tin lần chạy; lỗi có thể do API, tham số công cụ hoặc dịch vụ LLM |

</details>

## 5. Kiểm tra API và kết quả tích hợp

Phần này dành cho người kiểm thử hoặc tích hợp code. Người chỉ cần mở web có thể bỏ qua.

### Gửi dữ liệu trực tiếp vào API

`GET /health` cho biết dịch vụ và phiên bản mô hình. `POST /predict` nhận:

| Trường | Nội dung |
|---|---|
| `ecg` | Mảng ECG gốc theo mV, chưa lọc, chưa Z-score |
| `fs` | Tần số lấy mẫu, bằng 360 Hz |
| `r_peaks` | Chỉ số đỉnh R bắt đầu từ 0, kiểu nguyên, tăng dần và không trùng |

API lọc tín hiệu, tính RR, cắt nhịp, chuẩn hóa theo cấu hình đã chốt và chạy ONNX. Kết quả gồm nhãn/xác suất từng nhịp, số nhịp đã xử lý, nhịp bị bỏ qua và lý do. Thứ tự lớp là **N, S, V, F, Q**.

Gửi đủ các đỉnh R đã chọn, kể cả nhịp biên, để giữ ngữ cảnh tính RR. Không đưa `X_waveform` đã xử lý trong NPZ vào trường `ecg`, vì như vậy tín hiệu sẽ bị xử lý lại. Đầu vào sai trả HTTP 422; lỗi suy luận trả HTTP 500. Nếu không tải được mô hình, dịch vụ khởi động thất bại.

<details>
<summary>Ví dụ thử API bằng tín hiệu tổng hợp</summary>

Chạy từng khối trong PowerShell khi API đang hoạt động:

```powershell
$ecg = @(0..999 | ForEach-Object { [Math]::Sin(2 * [Math]::PI * 1.2 * $_ / 360) })
```

```powershell
$body = @{ ecg = $ecg; fs = 360; r_peaks = @(20, 400, 700) } | ConvertTo-Json -Depth 5
```

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

Hai lệnh tạo biến không in kết quả là bình thường. Ví dụ cần xử lý hai nhịp và bỏ một nhịp biên. Nhãn dự đoán trên tín hiệu tổng hợp chỉ dùng kiểm tra luồng chương trình, không dùng đánh giá độ chính xác.

</details>

### Chạy các script kiểm tra

Cần Python 3.14 trên Windows và API Docker ở cổng 8000. Tạo môi trường nếu chưa có:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

| Script | Mục đích | Đầu vào cần thêm |
|---|---|---|
| `check_api_contract.py` | Kiểm tra đầu vào API và giới hạn công cụ | Không cần gói TV1/TV2 |
| `check_tv2_model.py` | Đối chiếu ONNX với mẫu tham chiếu TV2 | `data/tv2/sample_io.npz` |
| `check_real_ecg.py` | Đối chiếu tiền xử lý bản ghi 100 với TV1, rồi so với API | `data/tv1/ds2_data.npz` |
| `benchmark_api.py` | Đo độ trễ API, tải tương đương một giờ và RAM container | Bản ghi 100, container `httm-ecg-api` |

Chạy bộ kiểm tra đầu vào và quyền công cụ:

```powershell
.\.venv\Scripts\python.exe .\check_api_contract.py
```

Đối chiếu mô hình với mẫu TV2:

```powershell
.\.venv\Scripts\python.exe .\check_tv2_model.py
```

Đối chiếu bản ghi thật với API Docker:

```powershell
.\.venv\Scripts\python.exe -c "import check_real_ecg as check; check.API_URL = 'http://127.0.0.1:8000'; check.main()"
```

Đo hiệu năng API:

```powershell
.\.venv\Scripts\python.exe .\benchmark_api.py
```

Các phép kiểm tra trên không cần key OpenRouter. Các script có xuất bằng chứng sẽ lưu vào `outputs/`. Nếu tệp TV1 v2 tên `ds2.npz`, đặt một bản sao tại đường dẫn `data/tv1/ds2_data.npz` mà script đối chiếu yêu cầu; giữ nguyên bản dữ liệu nguồn.

<details>
<summary>Chạy API trực tiếp bằng Python để phát triển</summary>

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Giữ cửa sổ server mở. `check_real_ecg.py` mặc định dùng cổng 8001:

```powershell
.\.venv\Scripts\python.exe .\check_real_ecg.py
```

Agent hiện chỉ chấp nhận API cục bộ ở cổng 8000. Cổng 8001 dùng để phát triển và đối chiếu riêng.

</details>

## 6. Chạy lại thí nghiệm của nhóm

Bản demo sử dụng mô hình đã huấn luyện. Nếu muốn tái hiện quá trình tạo dữ liệu và đánh giá, đọc [REPRODUCIBILITY.md](REPRODUCIBILITY.md), trong đó có môi trường, đầu vào và lệnh chạy chi tiết cho từng phần.

| Phần việc | Code hoặc notebook |
|---|---|
| TV1: tiền xử lý DS1/DS2 và tính CI bootstrap | [notebooks/tv1/README.md](notebooks/tv1/README.md) |
| TV2: huấn luyện, đánh giá DS2, phân tích sai và xuất ONNX | [train_evaluate.ipynb](notebooks/tv2/train_evaluate.ipynb) |
| TV2: vẽ lại hình từ kết quả đã lưu | [figures_from_results.ipynb](notebooks/tv2/figures_from_results.ipynb) |
| TV2: đánh giá thêm bản ghi 202, giữ nguyên mô hình | [evaluate_record_202.ipynb](notebooks/tv2/evaluate_record_202.ipynb) |
| TV3: so sánh hai checkpoint trên cùng CPU | [benchmark_model_comparison.py](benchmark_model_comparison.py) |

Dữ liệu WFDB, NPZ và gói TV2 đầy đủ được tải hoặc bàn giao riêng. Kho mã lưu các cấu hình, hash dữ liệu và bảng kết quả cần đối chiếu. Không cài môi trường huấn luyện TV2 chung với môi trường TV1; phiên bản thư viện của hai phần khác nhau. Docker demo chỉ phục vụ suy luận và Agent.

**Cải tiến đã thực hiện là thêm nhánh RR vào CNN 1D.** Khi thay đổi pipeline hoặc mô hình, cần cập nhật đồng bộ cấu hình, chuẩn hóa và mẫu tham chiếu. Chuẩn hóa đầu vào ONNX thực hiện bên ngoài mô hình đúng một lần; không tính lại mean/std bằng DS2. Bản triển khai chọn FP32; INT8 nằm trong phần thử nghiệm lượng tử hóa.

## 7. Kết quả đã ghi nhận

### Đánh giá mô hình trên DS2

DS1 gồm **51.003 nhịp/22 bản ghi**; DS2 chính gồm **47.557 nhịp/21 bản ghi**. Bản ghi 202 bị loại khỏi DS2 chính vì cùng người với bản ghi 201 trong DS1. Phép đánh giá thêm 202 được giữ riêng.

| Chỉ số trên DS2 | CNN cơ sở | CNN + RR |
|---|---:|---:|
| F1 macro, trung bình ± SD ba seed | 31,10 ± 6,00% | 48,75 ± 0,52% |
| Accuracy, trung bình ± SD ba seed | 50,74 ± 14,48% | 83,37 ± 2,23% |

Riêng CNN + RR seed 42 đang triển khai có F1 macro **48,86%** và accuracy **85,55%**. Đây là kết quả một seed, khác với trung bình ba seed ở bảng trên.

Nhánh RR giúp tăng F1 macro trung bình, nhưng độ nhạy lớp S trung bình giảm; cần xem đầy đủ các chỉ số trong [reports/tv2/](reports/tv2/) trước khi nhận xét mô hình tốt hơn ở mọi mặt.

### Kiểm thử và hiệu năng

| Phép kiểm tra | Kết quả đã ghi nhận |
|---|---|
| API và giới hạn công cụ, bản triển khai trước khi mở rộng danh sách | 16/16 kiểm tra đạt |
| Bản ghi 100 qua toàn bộ luồng API | 2.271 nhịp xử lý, khớp kết quả đối chiếu |
| Tiền xử lý TV1 trên PC ngày 22/10 | Đủ 43 bản ghi; số nhịp, metadata và 66 dòng CI đã đối chiếu |
| So sánh CNN/CNN + RR trên cùng CPU | Đã đo, bằng chứng tại `reports/cpu/` |
| 100 request API | 100/100 thành công; median 5,123 ms, P95 5,644 ms |
| Hai lượt bản ghi 100, tương đương 60,19 phút ECG | Tổng thời gian 0,708 giây |
| RAM peak container từ khi tạo | 312,87 MiB |
| Kích thước ONNX FP32 | 82.719 byte |

Số liệu benchmark API/RAM thuộc phép đo ngày **09/10/2026, trước khi thêm web**, trên desktop Ryzen 5 5600X. Chưa đo lại benchmark API trên bản có web. Thời gian trả lời chat còn bao gồm gọi OpenRouter nên không so trực tiếp với số mili giây của API.

Tải tương đương một giờ là hai lượt của cùng bản ghi, không phải một bản ghi một giờ độc lập. RAM peak tính cả vòng đời container. Xem [PERFORMANCE_RESULTS.md](PERFORMANCE_RESULTS.md) để biết cách đo và giới hạn từng kết quả.

Đã đối chiếu đủ **21/21 bản DS2 trên API Docker ở laptop**, tổng **47.557 nhịp**. Số nhịp và số lượng nhãn N/S/V/F/Q từng bản khớp bảng TV2. Bằng chứng: [ds2_records_check_b9a4a1c058d04fdeaef6dfedf6c8ee6e.json](reports/deployment/ds2_records_check_b9a4a1c058d04fdeaef6dfedf6c8ee6e.json). Phép kiểm tra này không so xác suất từng nhịp.

### Phạm vi hiện tại

- Agent hỗ trợ 21 bản DS2 trong danh sách được phép; DS1 và bản 202 nằm ngoài demo. API `/predict` nhận mảng ECG theo quy ước ở mục 5.
- Đỉnh R lấy từ chú thích có sẵn hoặc do bên gọi cung cấp; hệ thống chưa có bộ phát hiện R tự động.
- Lọc `filtfilt` và đặc trưng `RR_post` cần thông tin sau nhịp, phù hợp xem lại ECG đã thu nhận.
- Đã đối chiếu số nhịp và số lượng nhãn từng lớp qua API cho toàn bộ 21 bản DS2; chưa đối chiếu xác suất từng nhịp qua HTTP.
- Chưa đo nhiều người dùng đồng thời, mạng ngoài localhost hoặc hoạt động dài hạn. Cấu hình cục bộ này chưa có xác thực người dùng cho triển khai công khai.
- Nội dung LLM tóm tắt cần được đối chiếu với số liệu công cụ; kết quả không thay thế đánh giá y khoa.

## 8. Tìm code và tài liệu ở đâu?

| Tệp hoặc thư mục | Nội dung |
|---|---|
| `main.py` | Dịch vụ FastAPI và trang web |
| `preprocessing.py` | Lọc tín hiệu, tính RR và cắt nhịp |
| `model_runner.py` | Chuẩn hóa và chạy ONNX |
| `agent_tools.py` | Công cụ đọc bản ghi và gọi API |
| `agent_main.py` | Agent chạy từ dòng lệnh |
| `web_agent.py`, `web/` | Backend và giao diện chat |
| `models/tv2/` | ONNX FP32 và cấu hình triển khai |
| `notebooks/tv1/`, `notebooks/tv2/` | Code tiền xử lý, huấn luyện và đánh giá |
| `reports/` | Các bảng kết quả và bằng chứng nhỏ lưu cùng kho mã |
| `data/`, `logs/`, `outputs/` | Dữ liệu, nhật ký và báo cáo trên máy người chạy; được loại khỏi Git |
| `Dockerfile`, `requirements.txt` | Môi trường chạy dịch vụ |
| `.env.example` | Mẫu cấu hình, không có key thật |

Tài liệu chi tiết:

- [Thiết kế hệ thống](SYSTEM_DESIGN.md): kiến trúc và luồng xử lý.
- [Yêu cầu kỹ thuật](TECHNICAL_REQUIREMENTS.md): quy ước dữ liệu và mục tiêu kiểm tra. Bảng trạng thái ban đầu được giữ theo thời điểm lập; phần cập nhật nằm cuối tài liệu.
- [Kết quả kiểm thử và hiệu năng](PERFORMANCE_RESULTS.md): số liệu, cấu hình máy và cách đo.
- [Hướng dẫn chạy lại](REPRODUCIBILITY.md): chuẩn bị dữ liệu và tái hiện từng phần thí nghiệm.
- [Chọn và kiểm tra 21 bản DS2](DS2_DEMO_GUIDE.md): cách dùng bản ghi, đối chiếu và cập nhật container.
- [Nguồn các bảng kết quả](reports/README.md): xuất xứ và đơn vị của bằng chứng đã lưu.
