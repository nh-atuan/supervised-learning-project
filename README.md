# Đồ Án 1: Học Có Giám Sát - Nhóm 14

Tài liệu này cung cấp hướng dẫn chi tiết để cài đặt môi trường, chuẩn bị dữ liệu, thực thi toàn bộ mã nguồn của dự án và biên dịch báo cáo LaTeX.

---

## 1. Thông Tin về Đồ Án và Nhóm

**Thông Tin Đồ án**
- **Tên đồ án**: Nhập môn Học máy - Học có giám sát và ứng dụng.
- **Môn học**: Nhập môn Học máy (CSC14005).
- **Giảng viên hướng dẫn**: Thầy Lê Nhựt Nam.
- **Đơn vị**: Khoa học Máy tính, Trường Đại học Khoa học Tự nhiên - ĐHQG.HCM.

**Thông Tin Nhóm 14**

| Họ và tên | MSSV | Vai trò chính |
|-----------|------|---------------|
| Lê Xuân Trí | 23120099 | Phụ trách xử lý dữ liệu và đánh giá (Regression) |
| Dương Tuấn Anh | 23120208 | Xây dựng các mô hình nền tảng (Regression) |
| Đàm Tiến Đạt | 23120118 | Triển khai mô hình phân loại (Classification) |
| Tống Thanh Phúc | 23120158 | Xử lý dữ liệu và đánh giá (Classification) |
| Nguyễn Hồ Anh Tuấn | 23120185 | Nghiên cứu tổng hợp (Part 3), Đảm bảo chất lượng mã nguồn |


---
## 2. Cài Đặt Môi Trường

Mã nguồn được phát triển và kiểm thử trên môi trường Python 3.11. Để đảm bảo tính tương thích và khả năng tái lập thực nghiệm, vui lòng làm theo các bước thiết lập dưới đây.

### 2.1 Khởi tạo môi trường ảo

**Trên hệ điều hành Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Trên hệ điều hành macOS hoặc Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2.2 Cài đặt các thư viện phụ thuộc

Tiến hành cài đặt các gói phần mềm cần thiết từ tệp `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2.3 Đăng ký Kernel cho Jupyter Notebook

Để các tệp Notebook nhận diện đúng môi trường ảo vừa tạo, thực hiện đăng ký kernel như sau:
```bash
python -m ipykernel install --user --name ml-project-py311
```
Khi mở bất kỳ tệp Notebook nào trong dự án, vui lòng chọn kernel có tên là **`ml-project-py311`**.

---

## 3. Chuẩn Bị Dữ Liệu (Bắt Buộc)

Do giới hạn về kích thước, các tệp dữ liệu thô (raw data) không được đưa lên kho lưu trữ. Bạn cần tải dữ liệu thủ công theo các bước sau trước khi tiến hành chạy mã nguồn:

### 3.1 Dữ liệu bài toán Hồi quy (Bike Sharing Dataset)
1. Truy cập trang: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
2. Tải về và giải nén.
3. Đặt 2 tệp `day.csv` và `hour.csv` vào thư mục: `data/raw/regression/`

### 3.2 Dữ liệu bài toán Phân loại (Forest Cover Type)
1. Truy cập trang: https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset (Yêu cầu có tài khoản Kaggle).
2. Tải về bộ dữ liệu.
3. Đặt tệp `covtype.csv` vào thư mục: `data/raw/classification/`

Lưu ý: Tuyệt đối không chỉnh sửa nội dung của các tệp dữ liệu thô này sau khi tải về để đảm bảo tính nguyên bản.

---

## 4. Hướng Dẫn Chạy Mã Nguồn

Để tái lập toàn bộ kết quả của đồ án, cần chạy lần lượt các phần theo trình tự sau đây.

**Lưu ý quan trọng khi chạy Notebook:**
- Hãy đảm bảo bạn đã chọn đúng kernel là **`ml-project-py311`** (đã thiết lập ở mục cài đặt).
- Trước khi thực thi, vui lòng chọn **Restart and Run All** để đảm bảo trạng thái môi trường sạch nhất.

### 4.1 Phần 1 - Bài toán Hồi quy (Regression)

Thực thi tuần tự các tệp Notebook trong thư mục `code/Part1_Regression/`:

1. `eda_preprocessing/01_eda.ipynb`
2. `eda_preprocessing/02_preprocessing.ipynb`
3. `models/03_linear_regression.ipynb`
4. `models/04_regularized_regression.ipynb`
5. `models/05_nonlinear_models.ipynb`
6. `models/06_advanced_regression.ipynb`
7. `evaluation/07_evaluation.ipynb`

### 4.2 Phần 2 - Bài toán Phân loại (Classification)

Tiếp tục thực thi tuần tự các tệp Notebook trong thư mục `code/Part2_Classification/`:

1. `eda_preprocessing/01_eda.ipynb`
2. `eda_preprocessing/02_preprocessing.ipynb`
3. `models/03_classification_model_training.ipynb`
4. `models/04_advanced_classification.ipynb`
5. `evaluation/05_evaluation.ipynb`

### 4.3 Phần 3 - So sánh và Phân tích chuyên sâu (Comparison)

Đối với Phần 3, bạn có thể lựa chọn một trong hai phương pháp để chạy các thí nghiệm tổng hợp (Split Sensitivity, Noise Robustness, Feature Corruption, Convergence Analysis):

**Cách 1: Sử dụng Jupyter Notebook**
Mở và chạy toàn bộ tệp: `code/Part3_Comparison/08_part3_comparison_research.ipynb`

**Cách 2: Sử dụng tập lệnh Python tự động**
Chạy trực tiếp tệp pipeline qua giao diện dòng lệnh:
```bash
python code/Part3_Comparison/part3_pipeline.py
```
Quá trình này sẽ tự động tổng hợp kết quả từ Phần 1 và Phần 2, sau đó chạy các thí nghiệm mở rộng và xuất dữ liệu ra thư mục `code/Part3_Comparison/results/`.

---

