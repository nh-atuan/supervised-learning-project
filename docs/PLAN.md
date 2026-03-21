# 1. Kế hoạch

* Gồm **Phần 1 (Regression), Phần 2 (Classification), Phần 3 (So sánh & báo cáo)**
* Yêu cầu nâng cao (bonus) được tích hợp có kiểm soát vào timeline

---

## 2. Danh sách thành viên

| Mã | MSSV     | Họ và tên          |
| -- | -------- | ------------------ |
| A  | 2312099 | Lê Xuân Trí        |
| B  | 231228 | Dương Tuấn Anh     |
| C  | 2312118 | Đàm Tiến Đạt       |
| D  | 2312158 | Tống Thanh Phúc    |
| E  | 2312185 | Nguyễn Hồ Anh Tuấn |

---

## 3. Phân công

### Giai đoạn 0 - Chọn dataset

* Cả nhóm:

  * Chọn **1 dataset regression + 1 dataset classification**
  * Đảm bảo >= 10k samples, >= 5 features
  * Thống nhất cách chia train/val/test

---

### Giai đoạn 1 - EDA + Tiền xử lý

| Thành viên   | Regression (Part 1 - 2.2.2) | Classification (Part 2 - 3.2.1 + EDA) |
| :-----------: | --------------------------- | ------------------------------------- |
| A      | Mô tả dữ liệu + thống kê    | Mô tả dataset + class distribution    |
| B       | Histogram, correlation      | EDA + imbalance analysis              |
| C  | Outlier detection (IQR/Z)   | Feature analysis + visualization      |
| D       | Missing + normalization     | Preprocessing + encoding              |
| E      | Data split + pipeline       | Data split + pipeline                 |

---

### Giai đoạn 2 - Model Implementation

#### Phần 1 - Regression

| Thành viên   | Nội dung                                      |
| :------------: | --------------------------------------------- |
| A     | Linear Regression (Normal Equation) + residual diagnostics design |
| B     | Gradient Descent + learning rate schedule + Breusch-Pagan hook |
| C  | Ridge/Lasso/Elastic Net + Lasso path + điều kiện phát hiện heteroscedasticity |
| D     | Feature selection (Forward/Backward/Lasso) + khung so sánh OLS vs WLS + template bảng tổng hợp |
| E     | Core pipeline owner (split/preprocess/logging/artifact) + tích hợp **Kernel Ridge (bonus)** + chuẩn bị khung bootstrap cho Bias-Variance (đầu tuần 5) |

---

#### Phần 2 - Classification

| Thành viên   | Nội dung                                    |
| :------------: | ------------------------------------------- |
| A     | Logistic Regression (GD - binary + softmax) + đảm bảo output xác suất + calibration hook |
| B     | Newton-Raphson / IRLS + theo dõi hội tụ + lưu convergence log |
| C  | LDA + QDA + decision boundary (2D projection khi phù hợp) + McNemar hook |
| D     | Perceptron + owner template so sánh trước/sau calibration + bảng tổng hợp classification |
| E     | Regularization + class-weighted + tích hợp pipeline lưu prediction/probability + **Kernel LR (bonus)** + hỗ trợ Probit refine (đầu tuần 5) + CV hỗ trợ |

---

### Giai đoạn 3 - Evaluation & Visualization

#### Phần 1 - Regression

| Thành viên | Regression                                      |
| :----------: | ----------------------------------------------- |
| A          | MSE, RMSE, MAE, R² + QQ-plot                    |
| B          | Learning curve + Breusch-Pagan test             |
| C          | Residual plot + phát hiện heteroscedasticity    |
| D          | Pred vs Actual + WLS + so sánh OLS vs WLS       |
| E          | k-fold CV (k=10) + statistical test + validate schema output + chuẩn hóa logging cho kết quả bonus (Kernel Ridge/Bias-Variance) |

#### Phần 2 - Classification

| Thành viên | Classification                                  |
| :----------: | ----------------------------------------------- |
| A          | Accuracy + calibration curve (reliability diagram) |
| B          | Precision, Recall, F1 + hội tụ Newton/IRLS      |
| C          | Confusion Matrix + decision boundary (nếu 2D)   |
| D          | ROC + AUC + so sánh trước/sau calibration + owner bảng tổng hợp |
| E          | PR curve + k-fold CV (k=5) + kiểm tra tính nhất quán output + chuẩn hóa output xác suất cho Kernel Logistic/Probit |

---

### Giai đoạn 4 - Analysis & Discussion

| Thành viên | Nội dung                                 |
| :----------: | ---------------------------------------- |
| A          | So sánh model Regression + diễn giải QQ-plot/giả định |
| B          | Bias-Variance decomposition (định lượng) + regularization |
| C          | So sánh model Classification + phân tích McNemar/calibration |
| D          | Error analysis + hội tụ IRLS/Newton-Raphson + OLS vs WLS |
| E          | Overfitting / Underfitting + improvement + tổng hợp kiểm định + tích hợp bảng so sánh core vs bonus |

---

### Giai đoạn 5 - Phần 3 (So sánh & Research)

| Thành viên | Nội dung                            |
| :----------: | ----------------------------------- |
| A          | Kết nối Regression - Classification (góc nhìn GLM) + viết narrative so sánh |
| B          | GLM + exponential family + thiết kế thí nghiệm split variation (3 seed) |
| C          | Hessian/Jacobian cho Logistic Regression + thí nghiệm feature corruption (5%-10%-20%) |
| D          | Gaussian-Markov assumptions + thí nghiệm noise injection (Gaussian noise theo nhiều mức) |
| E          | Reproducibility + logging + sensitivity/robustness summary table +  bonus  |

---

### Giai đoạn 6 - Viết báo cáo

| Thành viên | Nội dung                |
| :----------: | ----------------------- |
| A          | Part 1 - Theory + giả định/QQ-plot/Breusch-Pagan |
| B          | Part 1 - Experiment + OLS vs WLS + learning analysis |
| C          | Part 2 - Theory + Hessian/Jacobian + calibration theory |
| D          | Part 2 - Experiment + McNemar + trước/sau calibration |
| E         | Tổng hợp + format LaTeX + bảo đảm đầu ra + tích hợp narrative bonus vào bảng so sánh/analysis hiện có |

---

## 4. Deadline cho từng giai đoạn

| Giai đoạn | Deadline |
| :---------: | :--------: |
| 1 | 21/3 |
| 2 | 26/3 |
| 3 | 28/3 |
| 4 | 1/4 |
| 5 | 6/4 |
| 6 | 13/4 |

---

## 5. Đầu ra mong đợi

### 5.1 Regression

* Code:

  * Linear Regression (NE + GD)
  * Ridge / Lasso / Elastic
  * Non-linear models
  * Bonus tích hợp: Kernel Ridge, Bias-Variance bootstrap
* Output:

  * MSE, RMSE, MAE, R²
  * Learning curve, residual plot
  * Comparison table

---

### 5.2 Classification

* Code:

  * Logistic Regression (GD + Newton)
  * LDA / QDA
  * Perceptron
  * Bonus tích hợp: Kernel Logistic hoặc Probit
* Output:

  * Accuracy, Precision, Recall, F1
  * Confusion matrix, ROC, PR curve
  * Model comparison

---

### 5.3 Báo cáo

* LaTeX đầy đủ:

  * Lý thuyết
  * Thực nghiệm
  * Phân tích
* Có:

  * Hình vẽ rõ ràng
  * Bảng so sánh
  * Trích dẫn chuẩn
* Yêu cầu formal:

  * File nộp dưới dạng PDF.
  * Không giới hạn số trang (không tính phụ lục, bìa,...).
  * Khuyến nghị sử dụng LaTeX (nộp kèm source `.tex` nếu dùng).
  * Mỗi thành viên phải có phần đóng góp rõ ràng.

---

## 6. Branching và commit

### 6.1 Branch flow

* Branch flow:

  * `main` (stable)
  * `task/*` (task branch)

### 6.2 Branch format

* Branch format (bắt buộc):

  ```
  task/<phase>/<member>-<task>
  ```

  Trong đó:

  * `<phase>`: số giai đoạn theo PLAN (0 -> 6)
  * `<member>`: mã thành viên `A | B | C | D | E`
  * `<task>`: mô tả ngắn theo kebab-case

### 6.3 Mapping phase -> prefix

* Mapping phase -> prefix:

  * Giai đoạn 0 (Dataset): `task/0/`
  * Giai đoạn 1 (EDA + preprocess): `task/1/`
  * Giai đoạn 2 (Model): `task/2/`
  * Giai đoạn 3 (Evaluation): `task/3/`
  * Giai đoạn 4 (Analysis): `task/4/`
  * Giai đoạn 5 (Research): `task/5/`
  * Giai đoạn 6 (Report): `task/6/`

### 6.4 Ví dụ branch

* Ví dụ branch:

  * `task/1/A-data-description`
  * `task/2/B-gradient-descent`
  * `task/2/C-regularization-models`
  * `task/3/D-roc-auc`
  * `task/6/E-final-report`

### 6.5 Quy tắc chống conflict

* Quy tắc chống conflict (đặc biệt cho notebook):

  * Mỗi người chỉ sửa đúng section được phân công.
  * Notebook phải tách section rõ theo người và task (vd: `## [A] Linear Regression`, `## [B] Gradient Descent`).
  * Không sửa cell thuộc phần của thành viên khác nếu chưa thống nhất.

### 6.6 Naming guideline

* Naming guideline:

  * Nên dùng: `linear-regression`, `ridge-lasso`, `feature-selection`, `kernel-ridge`
  * Không dùng: `fix1`, `test`, `code`, `update`

---

## 7. Yêu cầu về code (Updated)

* Python 3.11.x + venv
* Thư viện:

  * numpy, pandas, matplotlib, sklearn
* Có:

  * seed cố định
  * code chạy lại được
  * requirements.txt
* Tất cả code phải chạy lại được từ đầu (end-to-end).
* Phần cài đặt từ đầu phải tách biệt với phần dùng thư viện kiểm chứng.
* Ghi rõ version Python và thư viện trong requirements.txt hoặc environment.yml.
* Đặt random seed để đảm bảo reproducibility.

---

## 8. Lưu ý triển khai

* Các giai đoạn sẽ được thực hiện tuần tự để đảm bảo phụ thuộc dữ liệu.
* Để tránh xung đột mã nguồn khi merge:

  * Notebook chia theo section rõ ràng cho từng yêu cầu
  * Mỗi thành viên phụ trách chính phần được phân công và review chéo

---

## 9. Tiêu chí chung cho tất cả yêu cầu

* Mỗi yêu cầu trong notebook phải có đủ 3 phần:

  * Markdown lý thuyết
  * Code cài đặt
  * Markdown phân tích kết quả
* Tất cả chỉ số số học (MSE, RMSE, MAE, R², Accuracy, Precision, Recall, F1, AUC, ...) phải:

  * In ra màn hình
  * Lưu vào biến Python để tái sử dụng cho tổng hợp
* Biểu đồ cần có đầy đủ tiêu đề, nhãn trục, chú thích để đưa thẳng vào báo cáo.

---


## 10. Đảm bảo tính reproducibility

Trước các bước có ngẫu nhiên (data split, CV shuffle, khởi tạo model), chạy cell cố định seed ở đầu notebook:

```python
from code.Part1_Regression.utils import set_seed as set_seed_reg
from code.Part2_Classification.utils import set_seed as set_seed_cls

set_seed_reg(42)
set_seed_cls(42)
```

---

## 11. Khả năng tái hiện thí nghiệm (Reproducibility)

Để đảm bảo kết quả có thể tái hiện:

* Cố định tất cả random seed (data split, model initialization, CV,...).
* Ghi log đầy đủ:

  * hyperparameters
  * train/val/test split indices
  * metric theo từng fold
* Báo cáo:

  * thời gian train / inference
  * bộ nhớ tiêu tốn (nếu có thể)
* Cung cấp:

  * requirements.txt hoặc environment.yml với version cụ thể
* Kết quả phải tái hiện được khi chạy lại toàn bộ notebook từ đầu.

---

## 12. Tập dữ liệu

* Mô tả và hướng dẫn đặt dữ liệu trong [data/README.md](data/README.md).
* README tổng quan dự án nằm ở [README.md](README.md).
* Khi chốt dataset, cần cập nhật lại thống kê mẫu/đặc trưng vào notebook của từng phần.

---

## 13. Tiêu chí đánh giá báo cáo

Báo cáo cần đảm bảo:

* Trình bày rõ ràng, mạch lạc, có cấu trúc logic.
* Công thức toán học chính xác, có giải thích ký hiệu và thống nhất xuyên suốt.
* Biểu đồ đầy đủ: tiêu đề, nhãn trục, chú thích.
* Code sạch, có comment, có thể chạy lại và kiểm chứng.
* Phân tích kết quả sâu sắc, thể hiện hiểu biết về mô hình/thuật toán.
* Trích dẫn tài liệu đúng chuẩn (IEEE hoặc APA).
