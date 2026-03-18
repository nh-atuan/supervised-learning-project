# Dataset Summary

## 1. Dataset Regression - Bike Sharing

### 1.1 Tổng quan
- Dataset: Bike Sharing Dataset (UCI)
- Nguồn: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Bài toán: Regression (dự đoán số lượng xe được thuê)
- Miền ứng dụng: Dự báo nhu cầu theo chuỗi thời gian
- Mục tiêu modeling: Dự đoán tổng số lượt thuê xe dựa trên yếu tố thời gian và thời tiết

### 1.2 Cấu trúc dữ liệu
- File sử dụng: `data/raw/regression/hour.csv`
- Số lượng mẫu: 17,379 bản ghi theo giờ
- Số cột: 17 cột
- Ghi chú: Chọn `hour.csv` vì đáp ứng yêu cầu dữ liệu lớn (>10k mẫu)

### 1.3 Biến mục tiêu
- Target: `cnt`
- Kiểu dữ liệu: Biến liên tục
- Ý nghĩa: Tổng số xe được thuê trong mỗi giờ

### 1.4 Mô tả đặc trưng
#### Nhóm đặc trưng thời gian
- `season`, `yr`, `mnth`, `hr`, `weekday`
- `holiday`, `workingday`

#### Nhóm đặc trưng thời tiết
- `temp`, `atemp`, `hum`, `windspeed`, `weathersit`

#### Nhóm đặc trưng có nguy cơ rò rỉ dữ liệu
- `casual`, `registered`
- Cảnh báo: `casual + registered = cnt`, vì vậy cần loại bỏ 2 cột này khi huấn luyện

### 1.5 Chất lượng dữ liệu
- Giá trị thiếu: Không có
- Trùng lặp: Không kỳ vọng trong định dạng gốc của bộ dữ liệu
- Ngoại lệ: Có thể xuất hiện các đỉnh nhu cầu hoặc điều kiện thời tiết cực đoan
- Kiểu dữ liệu: Hỗn hợp giữa biến số liên tục và biến mã hóa dạng phân loại

### 1.6 Nhận xét ban đầu
- Nhu cầu thuê xe phụ thuộc mạnh vào yếu tố thời gian (`hr`, `weekday`, `season`)
- Điều kiện thời tiết ảnh hưởng đáng kể đến số lượng thuê
- Quan hệ có khả năng phi tuyến, nên cân nhắc feature engineering và basis expansion

---

## 2. Dataset Classification - Covertype

### 2.1 Tổng quan
- Dataset: Forest Cover Type (Covertype)
- Nguồn: https://archive.ics.uci.edu/dataset/31/covertype
- Bài toán: Phân loại đa lớp
- Mục tiêu: Dự đoán loại thảm phủ rừng từ các biến địa hình

### 2.2 Cấu trúc dữ liệu
- File sử dụng: `data/raw/classification/covtype.csv`
- Số lượng mẫu: 581,012
- Số đặc trưng đầu vào: 54
- Số lớp: 7 (`Cover_Type`)
- Ghi chú: Dataset lớn, phù hợp để đánh giá khả năng mở rộng và độ ổn định khi huấn luyện

### 2.3 Biến mục tiêu
- Target: `Cover_Type`
- Kiểu dữ liệu: Biến phân loại (7 lớp)
- Ý nghĩa: Nhãn loại thảm phủ rừng

### 2.4 Mô tả đặc trưng
#### Nhóm đặc trưng số
- Elevation, Aspect, Slope
- Các khoảng cách ngang/dọc đến thủy văn, đường sá và điểm cháy
- Các chỉ số Hillshade

#### Nhóm Wilderness area (one-hot)
- 4 đặc trưng nhị phân (`Wilderness_Area1` đến `Wilderness_Area4`)

#### Nhóm Soil type (one-hot)
- 40 đặc trưng nhị phân (`Soil_Type1` đến `Soil_Type40`)

### 2.5 Chất lượng dữ liệu
- Giá trị thiếu: Không có
- Mất cân bằng lớp: Có (phân bố lớp không đồng đều)
- Số chiều dữ liệu: Cao (nhiều đặc trưng nhị phân thưa)
- Kiểu dữ liệu: Kết hợp đặc trưng số và đặc trưng one-hot

### 2.6 Nhận xét ban đầu
- Không gian đặc trưng có số chiều cao, nhiều cột nhị phân thưa
- Một số lớp có thể tách tuyến tính trong các không gian con
- Phù hợp với Logistic Regression, LDA và các mô hình tuyến tính có regularization

---

## 3. Bảng tổng quan

| Tiêu chí | Bike Sharing | Covertype |
| --- | --- | --- |
| Bài toán | Regression | Classification |
| Số mẫu | 17,379 | 581,012 |
| Số đặc trưng | ~12 đặc trưng cốt lõi (sau khi loại cột leakage) | 54 |
| Biến mục tiêu | Biến liên tục `cnt` | 7 lớp (`Cover_Type`) |
| Mức độ phức tạp | Trung bình | Cao |
| Thách thức chính | Phi tuyến theo thời gian và thời tiết | Không gian đặc trưng cao chiều, thưa |
