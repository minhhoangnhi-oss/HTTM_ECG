# Kết quả kiểm tra tích hợp và hiệu năng — HTTM_ECG

## 1. Phạm vi

Ngày kiểm tra: 09/10/2026, giờ Việt Nam.

Phiên bản mô hình: 20261008_094922.

Các kết quả dưới đây kiểm tra triển khai API, tiền xử lý, Docker và Agent. Chúng không thay thế đánh giá phân loại trên toàn bộ DS2 của TV2.

Benchmark sử dụng bản ghi 100 từ PhysioNet MIT-BIH. Không dùng nhãn dự đoán để tự suy ra độ chính xác hoặc kết luận bệnh.

## 2. Môi trường đo

| Thành phần | Cấu hình ghi nhận |
|---|---|
| CPU máy chủ | AMD Ryzen 5 5600X |
| Số lõi / luồng máy chủ | 6 lõi / 12 luồng |
| Hệ điều hành máy chủ | Windows 11, build 26100 |
| RAM máy chủ ghi nhận | 34.243.350.528 byte, khoảng 31,89 GiB |
| Docker Engine | 29.8.1 |
| CPU Docker nhìn thấy | 12 CPU logic |
| RAM Docker nhìn thấy | 16.713.752.576 byte, khoảng 15,57 GiB |
| Giới hạn RAM riêng của container | memory.max = max |
| Giới hạn CPU riêng của container | cpu.max = max 100000 |
| CPU set | 0-11 |
| Python máy chủ và container | 3.14.8 |
| FastAPI | 0.142.2 |
| Uvicorn | 0.54.0 |
| ONNX Runtime | 1.30.0 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| WFDB | 4.3.1 |
| Backend suy luận | CPUExecutionProvider |

Không đặt giới hạn CPU/RAM riêng cho container trong lần đo này. Tài nguyên thực tế vẫn chịu giới hạn của máy chủ và môi trường Docker/WSL.

Benchmark không kiểm soát các tiến trình nền trên máy.

## 3. Mô hình và bản đóng gói

| Thành phần | Giá trị |
|---|---:|
| Tệp ONNX FP32 | 82.719 byte, khoảng 80,78 KiB |
| Docker image | 956.832.376 byte, khoảng 912,51 MiB |
| RAM container tại thời điểm lấy mẫu cuối | 177.602.560 byte, khoảng 169,38 MiB |
| RAM peak container từ khi tạo, tại thời điểm benchmark | 328.065.024 byte, khoảng 312,87 MiB |

SHA-256 của tệp ONNX:

`7d19389f652832c31e94cd6f1b168dd64e1d27df68fa573efbb4768cd5ca20cf`

RAM peak lấy từ cgroup của container. Giá trị này bao gồm các lần chạy trước trong cùng container, có thể gồm Agent và tiến trình đọc thông tin đo. Đây không phải phép đo RAM riêng của ONNX Runtime.

Kích thước Docker image bao gồm Python và thư viện; không được trình bày như kích thước mô hình.

## 4. Đối chiếu dữ liệu thật với TV1 và TV2

Bản ghi 100 có:

- 650.000 mẫu tại 360 Hz.

- Kênh MLII, đơn vị mV.

- 2.273 đỉnh R thuộc các ký hiệu được chọn theo TV1.

- 2.271 nhịp đủ cửa sổ.

- 2 nhịp biên bị bỏ qua.

### Kết quả đối chiếu

| Nội dung | Kết quả |
|---|---|
| Vị trí R giữ lại so với ds2_data.npz của TV1 | Khớp hoàn toàn |
| Nhãn tham chiếu so với TV1 | Khớp hoàn toàn |
| ECG sau lọc và cắt nhịp | Sai số tuyệt đối lớn nhất 7,4505806 × 10^-9 |
| RR | Sai số tuyệt đối lớn nhất 0 |
| API trên máy so với model_runner trên dữ liệu TV1 | Nhãn khớp; sai số xác suất lớn nhất 0 |
| API trong Docker so với model_runner trên dữ liệu TV1 | Nhãn khớp; sai số xác suất lớn nhất 0 |

Sai số bằng 0 là kết quả quan sát trên bản ghi và cấu hình này; không cam kết mọi nền tảng hoặc bản ghi đều có sai số bằng 0.

Trước đó, kiểm tra sample_io của TV2 trên batch 1 và batch 4 cũng đạt. Sai số xác suất lớn nhất được ghi nhận lần lượt khoảng 1,12 × 10^-8 và 1,21 × 10^-8.

## 5. Độ trễ API: 100 request

### Phương pháp

- API Docker: http://127.0.0.1:8000.

- Mỗi request dùng đoạn ECG đầu tiên dài 10 giây của bản ghi 100, gồm 3.600 mẫu.

- Lấy các đỉnh R nằm trong đoạn đó.

- Chạy 5 request làm nóng, loại khỏi thống kê.

- Đo 100 request nối tiếp bằng một HTTP client được tái sử dụng.

- Thời gian gồm mã hóa JSON, truyền qua localhost, xử lý API và nhận đầy đủ response.

- Giải mã JSON và kiểm tra nội dung response thực hiện ngoài bộ đếm thời gian.

- P95 dùng numpy.percentile với phương pháp linear.

### Kết quả

| Chỉ số | Giá trị |
|---|---:|
| Request thành công | 100/100 |
| Median | 5,123 ms |
| P95 | 5,644 ms |
| Min | 4,890 ms |
| Max | 9,150 ms |

Đây là độ trễ cho request ECG dài 10 giây. Không áp dụng trực tiếp các giá trị này cho request 30 phút, toàn bộ một giờ hoặc độ trễ Agent qua OpenRouter.

## 6. Tải tương đương khoảng một giờ ECG

### Phương pháp

Xử lý toàn bộ bản ghi 100 hai lần bằng hai request riêng.

Tổng số mẫu là 1.300.000, tương đương khoảng 60,19 phút ECG tại 360 Hz. Mỗi request giữ riêng ngữ cảnh lọc và RR.

### Kết quả

| Chỉ số | Giá trị |
|---|---:|
| Request toàn bản ghi thành công | 2/2 |
| Tổng lượng tín hiệu tương đương | 60,19 phút |
| Tổng thời gian HTTP của hai lượt | 0,708 giây |
| Tổng nhịp được xử lý | 4.542 |

Đây là phép thử tải lặp lại cùng một bản ghi khoảng 30 phút. Không phải phép đánh giá trên một bản ghi một giờ độc lập hoặc trên nhiều bệnh nhân.

## 7. Đối chiếu với mục tiêu đề xuất

Các mục tiêu được ghi trong TECHNICAL_REQUIREMENTS.md trước lần chạy benchmark này.

| Tiêu chí | Mục tiêu đề xuất | Kết quả | Đánh giá |
|---|---|---|---|
| Request 10 giây thành công | 100/100 | 100/100 | Đạt |
| P95 cho request 10 giây | Không quá 2.000 ms | 5,644 ms | Đạt |
| Tải tương đương một giờ | Không quá 60 giây, cả hai request thành công | 0,708 giây; 2/2 | Đạt |
| RAM peak container | Không quá 1 GiB | 312,87 MiB | Đạt |
| Kích thước ONNX | Không quá 1 MiB | 80,78 KiB | Đạt |

Các ngưỡng trên là mục tiêu kỹ thuật của nhóm, không phải ngưỡng do thầy quy định.

## 8. Kiểm tra giao diện và giới hạn công cụ

Tổng cộng 16/16 kiểm tra đạt:

| Nhóm | Trường hợp |
|---|---|
| API hợp lệ | Request có nhịp đủ cửa sổ; request có toàn bộ nhịp bị bỏ qua |
| API từ chối đầu vào sai | Sai fs; tín hiệu quá ngắn; giá trị null; chuỗi NaN; R âm; R ngoài tín hiệu; R không nguyên; R trùng; R không tăng dần; danh sách R rỗng |
| Agent từ chối yêu cầu ngoài phạm vi | Công cụ run_shell; bản ghi không được phép; tham số đường dẫn bổ sung; tham số thừa cho list_records |

Các trường hợp API chạy trên container Docker ở cổng 8000.

Các trường hợp giới hạn công cụ chạy trên module agent_tools của máy chủ. Không mô tả chúng như những lần gọi LLM độc hại hoặc bài kiểm tra xâm nhập toàn hệ thống.

Trường hợp NaN sử dụng chuỗi JSON "NaN". Chưa kiểm tra literal số NaN không thuộc chuẩn JSON.

## 9. Agent trên ECG thật

Agent đã thực hiện luồng:

1\. Liệt kê bản ghi được phép.

2\. Gọi API phân loại bản ghi 100.

3\. Tổng hợp số liệu từ công cụ.

4\. Hỏi người dùng trước khi lưu.

5\. Lưu khi người dùng nhập y.

Luồng được chạy thành công trên máy chủ và trong Docker.

Thống kê dự đoán được ghi nhận:

| Lớp | Số nhịp dự đoán |
|---|---:|
| N | 2.236 |
| S | 34 |
| V | 1 |
| F | 0 |
| Q | 0 |
| Chưa phân loại | 0 |

Các số lượng này là nhãn dự đoán của mô hình, không phải số lượng nhãn đúng hay độ chính xác.

Báo cáo Agent đã lưu:

- Máy chủ: outputs/report_e284c849ecb64223b80b85979eba61cc.json

- Docker, được mount về máy chủ: outputs/report_1e006ab5789949249de1d33656e6935e.json

Độ trễ LLM và tổng thời gian Agent chưa được benchmark trong phép đo API.

## 10. Bằng chứng và giới hạn

Các script tái hiện:

- check_tv2_model.py

- check_real_ecg.py

- check_api_contract.py

- benchmark_api.py

Các tệp bằng chứng nằm trong outputs/:

- check_real_ecg_100.json

- benchmark_*.json

- contract_check_*.json

- report_*.json

Nhật ký Agent nằm trong logs/agent_audit.jsonl.

Các tệp data/, logs/, outputs/ và .env được loại khỏi Git theo cấu hình hiện tại. Khi bàn giao báo cáo, cần gửi riêng các bằng chứng phù hợp đã kiểm tra nội dung.

Chưa thực hiện đối chiếu toàn bộ DS2 qua API. Chưa đo độ trễ nhiều client đồng thời, mạng ngoài localhost, khởi động lạnh hoặc hoạt động lâu dài.

API chưa có bộ phát hiện R. Agent hiện chỉ được phép truy cập bản ghi 100.

Hệ thống dùng lọc zero-phase và RR_post nên phục vụ xem lại ECG đã thu nhận, chưa phải xử lý thời gian thực.

## Kiểm tra giao diện web — tối 09/10/2026

Giao diện được chạy trên máy desktop bằng Docker, truy cập tại http://127.0.0.1:8000/.

### Kết quả đã kiểm tra

| Tình huống | Kết quả quan sát |
|---|---|
| Phân loại bản ghi 100 qua web | Khớp API: N=2236, S=34, V=1, F=0, Q=0 |
| Lưu báo cáo bằng nút trên web | Đã kiểm tra file JSON có `source: web`, `confirmed: true`, số liệu API và bản tóm tắt |
| Bỏ qua lưu | Giao diện thông báo “Đã bỏ qua. Không tạo tệp báo cáo.”; chưa kiểm đếm thư mục độc lập trong lượt này |
| Yêu cầu phân loại 101 và đọc `.env` bằng shell | Agent từ chối, không trả kết quả phân loại hay nội dung tệp |
| Chat tiếp nối trên bản backend mới | Nhật ký lượt hiển thị `classify_record: thành công`; trả đúng nguồn đỉnh R, 2271 nhịp xử lý, 2 nhịp bỏ qua, 0 nhịp chưa phân loại |
| Kiểm tra hồi quy API và giới hạn công cụ | Bộ `check_api_contract.py` đạt 16/16 sau khi thêm web |
| Đối chiếu bản ghi ECG thật qua API Docker | 2271 nhịp; sai số ECG lớn nhất 7.4505806e-09, RR bằng 0; nhãn khớp và sai số xác suất API bằng 0 |

Báo cáo web đã kiểm tra: `report_55f32cd8a9e1479c97fa2736df100051.json`.

Lượt chat tiếp nối trên bản mới: `93aa0f91d18b40ce98b62dc5f8b137ae`.

### Lỗi phát hiện và điều chỉnh

- Có lượt LLM trả lời mà không gọi công cụ, khiến backend từ chối câu trả lời. Đã thay `tool_choice` để bắt buộc gọi công cụ khi lượt hiện tại chưa có công cụ thành công.
- Bản tóm tắt ban đầu có cách diễn đạt sai về nguồn đỉnh R, ý nghĩa lớp F và đơn vị đếm nhịp. Đã bổ sung hướng dẫn trong system prompt; các lượt kiểm tra sau trả đúng thông tin được yêu cầu. Đây chưa phải bảo đảm mọi câu trả lời đều chính xác.
- Một lượt `classify_record` gặp `ValueError`. Gọi trực tiếp công cụ trong container thành công. Nhật ký cũ chưa chứa thông báo lỗi chi tiết nên chưa xác định nguyên nhân; đã bổ sung ghi thông báo `ValueError` để phục vụ điều tra nếu tái diễn.
- Sau khi cập nhật backend, đã có lượt phân loại và chat tiếp nối thành công. Điều này chưa chứng minh lỗi `ValueError` trước đó đã được giải quyết dứt điểm.

### Giới hạn của kiểm tra

Các tình huống web trên là kiểm tra chức năng thủ công, chưa phải đánh giá toàn diện về độ tin cậy của LLM hoặc an toàn hệ thống.

Chưa đo lại benchmark trên bản có giao diện web. Các số liệu benchmark ở phần trước thuộc bản triển khai trước khi thêm web.

Thời gian hoàn thành hiển thị trên giao diện gồm gọi OpenRouter và công cụ; không được dùng thay cho độ trễ API `/predict`.

Kiểm tra từ chối yêu cầu ngoài quyền đã thực hiện trước lần cập nhật system prompt cuối. Các kiểm tra API và ECG thật đã thực hiện trước lần thay system prompt và bổ sung thông báo lỗi; hai thay đổi này không sửa đường xử lý `/predict`.
