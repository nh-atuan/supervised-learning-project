# Kế hoạch

* Phạm vi: gồm **Phần 1 (Regression), Phần 2 (Classification), Phần 3 (So sánh & báo cáo)**
* Không bao gồm yêu cầu nâng cao (bonus làm sau nếu còn thời gian)

---

## Danh sách thành viên

| Mã | MSSV     | Họ và tên          |
| -- | -------- | ------------------ |
| A  | 23120185 | Nguyễn Hồ Anh Tuấn |
| B  | 23120099 | Lê Xuân Trí        |
| C  | 23120208 | Dương Tuấn Anh     |
| D  | 23120118 | Đàm Tiến Đạt       |
| E  | 23120158 | Tống Thanh Phúc    |

---

## Phân công

## Giai đoạn 0 - Chọn dataset

* Cả nhóm:

  * Chọn **1 dataset regression + 1 dataset classification**
  * Đảm bảo >= 10k samples, >= 5 features
  * Thống nhất cách chia train/val/test

---

## Giai đoạn 1 - EDA + Tiền xử lý

| Thành viên   | Regression (Part 1 - 2.2.2) | Classification (Part 2 - 3.2.1 + EDA) |
| ------------ | --------------------------- | ------------------------------------- |
| A (Tuấn)     | Mô tả dữ liệu + thống kê    | Mô tả dataset + class distribution    |
| B (Trí)      | Histogram, correlation      | EDA + imbalance analysis              |
| C (Tuấn Anh) | Outlier detection (IQR/Z)   | Feature analysis + visualization      |
| D (Đạt)      | Missing + normalization     | Preprocessing + encoding              |
| E (Phúc)     | Data split + pipeline       | Data split + pipeline                 |

---

## Giai đoạn 2 - Model Implementation

### Phần 1 - Regression (50 điểm)

| Thành viên   | Nội dung                                      |
| ------------ | --------------------------------------------- |
| A (Tuấn)     | Linear Regression (Normal Equation)           |
| B (Trí)      | Gradient Descent + learning rate schedule     |
| C (Tuấn Anh) | Ridge + Lasso + Elastic Net                   |
| D (Đạt)      | Feature selection (Forward/Backward/Lasso)    |
| E (Phúc)     | Non-linear basis (Polynomial, RBF) + ablation |

---

### Phần 2 - Classification (50 điểm)

| Thành viên   | Nội dung                                    |
| ------------ | ------------------------------------------- |
| A (Tuấn)     | Logistic Regression (GD - binary + softmax) |
| B (Trí)      | Newton-Raphson / IRLS                       |
| C (Tuấn Anh) | LDA + QDA                                   |
| D (Đạt)      | Perceptron                                  |
| E (Phúc)     | Regularization + class-weighted + CV        |

---

## Giai đoạn 3 - Evaluation & Visualization

| Thành viên | Regression                   | Classification         |
| ---------- | ---------------------------- | ---------------------- |
| A          | MSE, RMSE, MAE, R²           | Accuracy               |
| B          | Learning curve               | Precision, Recall, F1  |
| C          | Residual plot                | Confusion Matrix       |
| D          | Pred vs Actual               | ROC + AUC              |
| E          | k-fold CV + statistical test | PR curve + calibration |

---

## Giai đoạn 4 - Analysis & Discussion

| Thành viên | Nội dung                                 |
| ---------- | ---------------------------------------- |
| A          | So sánh model Regression                 |
| B          | Bias-Variance + regularization           |
| C          | So sánh model Classification             |
| D          | Error analysis                           |
| E          | Overfitting / Underfitting + improvement |

---

## Giai đoạn 5 - Phần 3 (So sánh & Research)

| Thành viên | Nội dung                            |
| ---------- | ----------------------------------- |
| A          | Kết nối Regression - Classification |
| B          | GLM + exponential family            |
| C          | Bảng so sánh toàn bộ model          |
| D          | Sensitivity + robustness            |
| E          | Reproducibility + logging           |

---

## Giai đoạn 6 - Viết báo cáo

| Thành viên | Nội dung                |
| ---------- | ----------------------- |
| A          | Part 1 - Theory         |
| B          | Part 1 - Experiment     |
| C          | Part 2 - Theory         |
| D          | Part 2 - Experiment     |
| E          | Tổng hợp + format LaTeX |

---

## Đầu ra mong đợi

### Regression

* Code:

  * Linear Regression (NE + GD)
  * Ridge / Lasso / Elastic
  * Non-linear models
* Output:

  * MSE, RMSE, MAE, R²
  * Learning curve, residual plot
  * Comparison table

---

### Classification

* Code:

  * Logistic Regression (GD + Newton)
  * LDA / QDA
  * Perceptron
* Output:

  * Accuracy, Precision, Recall, F1
  * Confusion matrix, ROC, PR curve
  * Model comparison

---

### Báo cáo

* LaTeX đầy đủ:

  * Lý thuyết
  * Thực nghiệm
  * Phân tích
* Có:

  * Hình vẽ rõ ràng
  * Bảng so sánh
  * Trích dẫn chuẩn

---

## Branching và commit

* Branch:
  `main` (stable), `dev` (integration), `feature/*` (task branch)
  (vd: `feature/eda`, `feature/regression-model`, `feature/classification-model`, `feature/report`)

* Mỗi phần:

  * 1 người code chính
  * 1 người review

* Commit:

  * rõ ràng, theo format:

    ```
    feat(regression): implement ridge regression
    ```

---

## Yêu cầu về code

* Python 3.x + venv
* Thư viện:

  * numpy, pandas, matplotlib, sklearn
* Có:

  * seed cố định
  * code chạy lại được
  * requirements.txt

---

## Lưu ý triển khai

* Các giai đoạn sẽ được thực hiện tuần tự để đảm bảo phụ thuộc dữ liệu.
* Để tránh xung đột mã nguồn khi merge:

  * Notebook chia theo section rõ ràng cho từng yêu cầu
  * Mỗi thành viên phụ trách chính phần được phân công và review chéo

---

## Tiêu chí chung cho tất cả yêu cầu

* Mỗi yêu cầu trong notebook phải có đủ 3 phần:

  * Markdown lý thuyết
  * Code cài đặt
  * Markdown phân tích kết quả
* Tất cả chỉ số số học (MSE, RMSE, MAE, R², Accuracy, Precision, Recall, F1, AUC, ...) phải:

  * In ra màn hình
  * Lưu vào biến Python để tái sử dụng cho tổng hợp
* Biểu đồ cần có đầy đủ tiêu đề, nhãn trục, chú thích để đưa thẳng vào báo cáo.

---

## Cấu trúc thư mục mã nguồn đề xuất

```text
src/
├── utils.py              # Hàm tiện ích chung (seed, logging, I/O)
├── dataset.py            # Load dữ liệu, split train/val/test
├── visualize.py          # Hàm vẽ biểu đồ cho regression/classification
├── stats_tests.py        # Kiểm định thống kê và so sánh mô hình
├── metrics.py            # Metrics và helper đánh giá
└── preprocessing/
    ├── regression.py     # Pipeline tiền xử lý cho Part 1
    └── classification.py # Pipeline tiền xử lý cho Part 2
```

---

## Đảm bảo tính reproducibility

Trước các bước có ngẫu nhiên (data split, CV shuffle, khởi tạo model), chạy cell cố định seed ở đầu notebook:

```python
from code.Part1_Regression.utils import set_seed as set_seed_reg
from code.Part2_Classification.utils import set_seed as set_seed_cls

set_seed_reg(42)
set_seed_cls(42)
```

---

## Tập dữ liệu

* Mô tả và hướng dẫn đặt dữ liệu trong [data/README.md](data/README.md).
* README tổng quan dự án nằm ở [README.md](README.md).
* Khi chốt dataset, cần cập nhật lại thống kê mẫu/đặc trưng vào notebook của từng phần.

