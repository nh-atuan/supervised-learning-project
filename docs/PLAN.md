# 1. Kế hoạch

* Gồm **Phần 1 (Regression), Phần 2 (Classification), Phần 3 (So sánh & báo cáo)**
* Yêu cầu nâng cao (bonus) được tích hợp có kiểm soát vào timeline
* Data pipeline và Model pipeline tách biệt hoàn toàn. Thành viên phụ trách data không làm model và ngược lại.

---

## 2. Danh sách thành viên

| Mã | MSSV     | Họ và tên          |
| -- | -------- | ------------------ |
| A  | 23120099 | Lê Xuân Trí        |
| B  | 23120208 | Dương Tuấn Anh     |
| C  | 23120118 | Đàm Tiến Đạt       |
| D  | 23120158 | Tống Thanh Phúc    |
| E  | 23120185 | Nguyễn Hồ Anh Tuấn |

---

## 3. Phân công

### Giai đoạn 0 - Chọn dataset

* Cả nhóm:

  * Chọn **1 dataset regression + 1 dataset classification**
  * Đảm bảo >= 10k samples, >= 5 features
  * Thống nhất cách chia train/val/test

---

### Giai đoạn 1 - EDA + Tiền xử lý

| Phần                  | Thành viên         | Nội dung                                                                                                   |
| --------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| Regression - Data     | A (Xuân Trí)  | 2.1.1 + 2.2.2 + Mục 2 trong 2.3 (Mô tả dataset, EDA, visualization, xử lý missing, normalization, chia  train/valid/test, xây dựng dataset hoàn chỉnh, kiểm định Breusch–Pagan, WLS nếu cần,...) |
| Classification - Data | D (Tống Phúc) | 3.2.1  + Mục 2 trong 3.3 (Mô tả dataset, EDA, class distribution, imbalance, preprocessing, encoding, chia tập train/valid/test,...)                   |
| Tổng hợp       | E     | Kiểm tra + hỗ trợ + đảm bảo chất lượng |

---

### Giai đoạn 2 - Model Implementation

#### Phần 1 - Regression

| Thành viên        | Vai trò | Nội dung                                                                   |
| ------------ | ------- | -------------------------------------------------------------------------- |
| B (Tuấn Anh) | Model   | 2.2.3 + Mục 3, 4, 5 trong 2.3 (Xây dựng model, train, regularization, feature selection, nonlinear models,...)|

---

#### Phần 2 - Classification

| Thành viên         | Vai trò | Nội dung                                      |
| ------------- | ------- | --------------------------------------------- |
| C (Đàm Đạt)   | Model   | 3.2.2 + Mục 3, 4, 5 trong 3.3 (Logistic, LDA/QDA, Perceptron, regularization,...) |

---

#### Bonus + System

| Thành viên        | Nội dung                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| E (Anh Tuấn) | Kiểm tra + hỗ trợ + đảm bảo chất lượng & Cài đặt để lấy điểm bonus của 2 phần|

---

### Giai đoạn 3 - Evaluation & Visualization

| Phần           | Thành viên     | Nội dung                                                   |
| -------------- | --------- | ---------------------------------------------------------- |
| Regression     | A  |2.2.4 + Mục 6 trong 2.3 (Metric (MSE, RMSE, MAE, R²), learning curve, residual, CV,...) |
| Classification | D | 3.2.3 + Mục 6 trong 3.3 (Accuracy, Precision, Recall, F1, ROC, PR, CV,...)              |
| Tổng hợp       | E     | Kiểm tra + hỗ trợ + đảm bảo chất lượng |

---

### Giai đoạn 4 - Analysis & Discussion

| Phần           | Thành viên | Nội dung                                      |
| -------------- | ----- | --------------------------------------------- |
| Regression     | A + B | Phân tích (A: data insights, B: model insights) |
| Classification | D + C | Phân tích (D: data insights, C: model insights) |
| Tổng hợp       | E     | Phân tích các model insights của thuật toán bonus |

---

### Giai đoạn 5 - Phần 3 (So sánh & Research)
Sẽ chia sau

| Thành viên | Nội dung                            |
| :----------: | ----------------------------------- |
| A          | |
| B          |  |
| C          | |
| D          | |
| E          |   |

---

### Giai đoạn 6 - Viết báo cáo
Ai làm phần nào viết báo cáo phần đó

---

## 4. Deadline cho từng giai đoạn

| Giai đoạn | Deadline |
| :---------: | :--------: |
| 1 | 25/3 |
| 2 | 30/3 |
| 3 | 2/4 |
| 4 | 5/4 |
| 5 | 8/4 |
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
  task/<phase>/<member>
  ```

  Trong đó:

  * `<phase>`: mô tả giai đoạn
  * `<member>`: mã thành viên `A | B | C | D | E`

### 6.3 Mapping phase -> prefix

* Mapping phase -> prefix:

  * Giai đoạn 1: `task/eda-preprocessing/`
  * Giai đoạn 2: `task/model/`
  * Giai đoạn 3: `task/evaluation/`
  * Giai đoạn 4: `task/analysis/`
  * Giai đoạn 5 : `task/research/`
  * Giai đoạn 6: `task/report/`

### 6.4 Ví dụ branch

* Ví dụ branch:

  * `task/eda-preprocessing/memA`
  * `task/model/memB`

### 6.5 Quy tắc chống conflict

* Quy tắc chống conflict (đặc biệt cho notebook):

  * Mỗi Thành viên chỉ sửa đúng section được phân công.
  * Notebook phải tách section rõ theo Thành viên và task (vd: `## [A] Linear Regression`, `## [B] Gradient Descent`).
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
