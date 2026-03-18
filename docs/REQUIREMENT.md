# CSC14005 - Đồ Án 1: Học Có Giám Sát

## 1. Tổng Quan

### Mục Tiêu
Đồ án tập trung vào việc áp dụng các phương pháp học có giám sát cho:
- Hồi quy (Regression)
- Phân loại (Classification)

Sinh viên cần:
- Nắm vững nền tảng lý thuyết
- Tự cài đặt mô hình từ đầu
- Thực nghiệm trên dữ liệu thực tế
- Đánh giá và phân tích kết quả một cách phản biện

### Tài Liệu Tham Khảo
- Bishop (2006) - PRML
  - Chương 3: Regression
  - Chương 4: Classification

---

## 2. Yêu Cầu Chung

### Nhóm
- Tối đa 5 sinh viên mỗi nhóm

### Lập Trình
- Python 3.x
- Thư viện được phép:
  - numpy, pandas
  - matplotlib, seaborn, plotly
  - scikit-learn (chỉ dùng để so sánh, không dùng cho phần cài đặt chính)

### Dữ Liệu
- Phải là dữ liệu thực tế
- Nguồn dữ liệu:
  - UCI, Kaggle, v.v.
- Không sử dụng dữ liệu tổng hợp (synthetic)

### Báo Cáo
- Định dạng: PDF (khuyến khích LaTeX)
- Bắt buộc có:
  - Giải thích toán học
  - Hình minh họa có chú thích
  - Trích dẫn tài liệu (APA/IEEE)

---

## 3. Phần 1: Hồi Quy (50 điểm)

### 3.1 Yêu Cầu Dữ Liệu
- >= 10,000 mẫu
- >= 5 đặc trưng (trong đó >= 3 đặc trưng liên tục)

### 3.2 Nhiệm Vụ

#### (1) Phân Tích Dữ Liệu và Tiền Xử Lý
- Mô tả bộ dữ liệu
- EDA:
  - Thống kê mô tả (mean, std, quartiles)
  - Biểu đồ phân phối
  - Ma trận tương quan
  - Phát hiện ngoại lệ (IQR/Z-score)
- Tiền xử lý:
  - Giá trị thiếu
  - Chuẩn hóa / tiêu chuẩn hóa
  - Chia tập train/val/test (ví dụ 70/10/20)

#### (2) Cài Đặt Mô Hình

##### A. Linear Regression
- Cài đặt:
  - Normal Equation (từ đầu)
  - Mini-batch Gradient Descent
- Learning rate schedule
- Kiểm định giả định Gauss-Markov:
  - Residual plot
  - QQ-plot
  - Breusch-Pagan test
- Nếu cần:
  - Weighted Least Squares (WLS)

##### B. Regularization và Chọn Đặc Trưng
- Ridge (L2), Lasso (L1)
- Tối ưu siêu tham số:
  - k-fold CV (k=10)
- Vẽ:
  - Regularization path
- Cài đặt:
  - Elastic Net
- Chọn đặc trưng:
  - Forward
  - Backward
  - Dựa trên Lasso

##### C. Mô Hình Phi Tuyến
- Sử dụng >= 3 hàm cơ sở:
  - Polynomial
  - RBF
  - Tự thiết kế
- Validation curve
- Ablation study
- Tương tác đặc trưng (xi * xj)

#### (3) Đánh Giá
Chỉ số:
- MSE, RMSE, MAE, R^2

Bổ sung:
- Learning curve
- Residual plot
- Predicted vs actual
- k-fold CV (k=10)
- Kiểm định thống kê:
  - t-test / Wilcoxon

#### (4) Thảo Luận
- So sánh mô hình
- Bias-variance tradeoff
- Overfitting / underfitting
- Độ phức tạp tính toán
- Đề xuất cải tiến

#### (5) Bonus (+5)
- Bayesian Regression
- Kernel Ridge
- Gaussian Process
- Robust Regression
- Bias-Variance (bootstrap)

---

## 4. Phần 2: Phân Loại (50 điểm)

### 4.1 Yêu Cầu Dữ Liệu
- >= 10,000 mẫu
- >= 5 đặc trưng
- Bài toán nhị phân hoặc đa lớp

### 4.2 Nhiệm Vụ

#### (1) Cài Đặt Mô Hình

##### A. Logistic Regression
- Cài đặt:
  - Gradient Descent
  - Newton-Raphson (IRLS)
- Đa lớp:
  - OvR, OvO, Softmax
- Suy diễn gradient và Hessian

##### B. LDA và QDA
- Cài đặt cả hai mô hình
- Fisher ratio
- Chiếu 2D và decision boundary
- So sánh LDA và QDA

##### C. Perceptron và Regularization
- Cài đặt Perceptron
- Logistic Regression với:
  - L1, L2
- Xử lý mất cân bằng lớp:
  - Class-weighted loss
- Stratified k-fold (k=5)

#### (2) Đánh Giá
Chỉ số:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- ROC + AUC

Bổ sung:
- PR curve
- Calibration plot
- k-fold CV (k=5)
- McNemar test

#### (3) Thảo Luận
- So sánh mô hình
- Logistic và LDA
- Ảnh hưởng của regularization
- Phân tích dữ liệu mất cân bằng
- Phân tích lỗi

#### (4) Bonus (+5)
- Probit model
- Laplace approximation
- Kernel Logistic Regression
- GNB và LDA
- VC dimension

---

## 5. Phần 3: Phân Tích So Sánh

### Bắt Buộc
- Liên hệ giữa Regression và Classification
- Khung GLM
- Họ phân phối mũ (Exponential family)

### Bảng So Sánh
Bao gồm:
- Giả định
- Độ phức tạp
- Khả năng diễn giải
- Hiệu năng

### Phân Tích Độ Bền Vững
- Các tỉ lệ chia train/test (60/40, 70/30, 80/20)
- Bơm nhiễu (noise injection)
- Làm hỏng đặc trưng (feature corruption)
- Phân tích hội tụ

### Tái Lập Kết Quả
- Cố định random seed
- Ghi log thực nghiệm
- Báo cáo thời gian chạy
- Cung cấp:
  - requirements.txt / environment.yml

---

## 6. Nộp Bài

### Cấu Trúc Thư Mục
```text
Group_ID/
|
|-- report/
|   |-- report.pdf
|   `-- report.tex
|
|-- code/
|   |-- Part1_Regression/
|   `-- Part2_Classification/
|
|-- data/
|   `-- README.md
|
`-- README.md
```

### Yêu Cầu Mã Nguồn
- Chạy end-to-end được
- Tách riêng:
  - Phần cài đặt từ đầu
  - Phần dùng thư viện
- Cố định random seed

### Yêu Cầu Báo Cáo
- Cấu trúc rõ ràng
- Toán học chính xác
- Trực quan hóa tốt
- Phân tích sâu
- Trích dẫn đúng chuẩn

### Quy Định Nộp
- Định dạng: Group_<ID>.zip
- Trừ điểm nộp trễ:
  - -10% mỗi ngày
  - >3 ngày: 0 điểm

---

## 7. Thang Điểm

| Phần | Điểm |
|------|-----:|
| Regression | 50 |
| Classification | 50 |
| Bonus | +10 |
| **Tổng** | 110 |

---

## 8. Trung Thực Học Thuật
- Không đạo văn
- AI chỉ được dùng để hỗ trợ
- Phải hiểu toàn bộ nội dung đã nộp
