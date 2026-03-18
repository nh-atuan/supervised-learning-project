# Kế hoạch

* Phạm vi: gồm **Phần 1 (Regression), Phần 2 (Classification), Phần 3 (So sánh & báo cáo)**
* Yêu cầu nâng cao (bonus) được tích hợp có kiểm soát vào timeline hiện tại, không tạo phase riêng

## Chiến lược bonus (tích hợp, không overload)

* Mục tiêu: tối ưu cơ hội +10 điểm bonus nhưng không phá vỡ critical path của phần core.
* Nguyên tắc triển khai:

  * Không làm sớm trước khi model core ổn định.
  * Không dồn vào cuối tuần 5.
  * Mỗi phần chỉ chọn 1-2 bonus có khả năng tái sử dụng pipeline.

* Bonus được chọn:

  * Regression: **Kernel Ridge** (ưu tiên) + **Bias-Variance bootstrap**.
  * Classification: **Kernel Logistic** (ưu tiên) hoặc **Probit** (refine đầu tuần 5).

* Khung thời gian tích hợp bonus:

  * Cuối tuần 3 (30/3): bonus nhẹ dựa trên pipeline sẵn có (Kernel).
  * Đầu tuần 5 (7/4 -> 8/4): bonus nặng cần model ổn định (Bias-Variance/Probit).
  * Tuần 4: không thêm bonus để tránh quá tải.

---

## Danh sách thành viên

| Mã | MSSV     | Họ và tên          |
| -- | -------- | ------------------ |
| A  | 2312099 | Lê Xuân Trí        |
| B  | 231228 | Dương Tuấn Anh     |
| C  | 2312118 | Đàm Tiến Đạt       |
| D  | 2312158 | Tống Thanh Phúc    |
| E  | 2312185 | Nguyễn Hồ Anh Tuấn |

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
| :-----------: | --------------------------- | ------------------------------------- |
| A      | Mô tả dữ liệu + thống kê    | Mô tả dataset + class distribution    |
| B       | Histogram, correlation      | EDA + imbalance analysis              |
| C  | Outlier detection (IQR/Z)   | Feature analysis + visualization      |
| D       | Missing + normalization     | Preprocessing + encoding              |
| E      | Data split + pipeline       | Data split + pipeline                 |

---

## Giai đoạn 2 - Model Implementation

### Phần 1 - Regression

| Thành viên   | Nội dung                                      |
| :------------: | --------------------------------------------- |
| A     | Linear Regression (Normal Equation) + residual diagnostics design |
| B     | Gradient Descent + learning rate schedule + Breusch-Pagan hook |
| C  | Ridge/Lasso/Elastic Net + Lasso path + điều kiện phát hiện heteroscedasticity |
| D     | Feature selection (Forward/Backward/Lasso) + khung so sánh OLS vs WLS + template bảng tổng hợp |
| E     | Core pipeline owner (split/preprocess/logging/artifact) + tích hợp **Kernel Ridge (bonus)** + chuẩn bị khung bootstrap cho Bias-Variance (đầu tuần 5) |

---

### Phần 2 - Classification

| Thành viên   | Nội dung                                    |
| :------------: | ------------------------------------------- |
| A     | Logistic Regression (GD - binary + softmax) + đảm bảo output xác suất + calibration hook |
| B     | Newton-Raphson / IRLS + theo dõi hội tụ + lưu convergence log |
| C  | LDA + QDA + decision boundary (2D projection khi phù hợp) + McNemar hook |
| D     | Perceptron + owner template so sánh trước/sau calibration + bảng tổng hợp classification |
| E     | Regularization + class-weighted + tích hợp pipeline lưu prediction/probability + **Kernel LR (bonus)** + hỗ trợ Probit refine (đầu tuần 5) + CV hỗ trợ |

---

## Giai đoạn 3 - Evaluation & Visualization

### Phần 1 - Regression

| Thành viên | Regression                                      |
| :----------: | ----------------------------------------------- |
| A          | MSE, RMSE, MAE, R² + QQ-plot                    |
| B          | Learning curve + Breusch-Pagan test             |
| C          | Residual plot + phát hiện heteroscedasticity    |
| D          | Pred vs Actual + WLS + so sánh OLS vs WLS       |
| E          | k-fold CV (k=10) + statistical test + validate schema output + chuẩn hóa logging cho kết quả bonus (Kernel Ridge/Bias-Variance) |

### Phần 2 - Classification

| Thành viên | Classification                                  |
| :----------: | ----------------------------------------------- |
| A          | Accuracy + calibration curve (reliability diagram) |
| B          | Precision, Recall, F1 + hội tụ Newton/IRLS      |
| C          | Confusion Matrix + decision boundary (nếu 2D)   |
| D          | ROC + AUC + so sánh trước/sau calibration + owner bảng tổng hợp |
| E          | PR curve + k-fold CV (k=5) + kiểm tra tính nhất quán output + chuẩn hóa output xác suất cho Kernel Logistic/Probit |

---

## Giai đoạn 4 - Analysis & Discussion

| Thành viên | Nội dung                                 |
| :----------: | ---------------------------------------- |
| A          | So sánh model Regression + diễn giải QQ-plot/giả định |
| B          | Bias-Variance decomposition (định lượng) + regularization |
| C          | So sánh model Classification + phân tích McNemar/calibration |
| D          | Error analysis + hội tụ IRLS/Newton-Raphson + OLS vs WLS |
| E          | Overfitting / Underfitting + improvement + tổng hợp kiểm định + tích hợp bảng so sánh core vs bonus |

---

## Giai đoạn 5 - Phần 3 (So sánh & Research)

| Thành viên | Nội dung                            |
| :----------: | ----------------------------------- |
| A          | Kết nối Regression - Classification (góc nhìn GLM) + viết narrative so sánh |
| B          | GLM + exponential family + thiết kế thí nghiệm split variation (3 seed) |
| C          | Hessian/Jacobian cho Logistic Regression + thí nghiệm feature corruption (5%-10%-20%) |
| D          | Gaussian-Markov assumptions + thí nghiệm noise injection (Gaussian noise theo nhiều mức) |
| E          | Reproducibility + logging + sensitivity/robustness summary table +  bonus  |

---

## Giai đoạn 6 - Viết báo cáo

| Thành viên | Nội dung                |
| :----------: | ----------------------- |
| A          | Part 1 - Theory + giả định/QQ-plot/Breusch-Pagan |
| B          | Part 1 - Experiment + OLS vs WLS + learning analysis |
| C          | Part 2 - Theory + Hessian/Jacobian + calibration theory |
| D          | Part 2 - Experiment + McNemar + trước/sau calibration |
| E         | Tổng hợp + format LaTeX + bảo đảm đầu ra + tích hợp narrative bonus vào bảng so sánh/analysis hiện có |

---

## Timeline triển khai 

* Start: **9/3**
* End: **13/4**
* Tổng thời lượng: **5 tuần (35 ngày)**

| Tuần   | Khung ngày thực      | Giai đoạn chính                                  | Mục tiêu chính                                  | Deliverable |
| ------ | -------------------- | ------------------------------------------------ | ----------------------------------------------- | ----------- |
| Tuần 1 + 2 | 9/3 -> 23/3       | Dataset + EDA + Preprocessing                    | Chốt dataset, làm sạch dữ liệu, hoàn thiện EDA | Notebook EDA + pipeline tiền xử lý |
| Tuần 3 | 24/3 -> 30/3       | Model Implementation + Integration Checkpoint 1 + bonus nhẹ | Hoàn thành model cốt lõi + hook QQ/BP/calibration + freeze schema output + implement bonus (Kernel) | Code model chạy được + regression integration freeze + kernel baseline |
| Tuần 4 | 31/3 -> 6/4       | Evaluation + Visualization + Integration Checkpoint 2 | Hoàn thiện chỉ số, biểu đồ, test thống kê và so sánh bắt buộc | Bảng kết quả đầy đủ + classification integration freeze |
| Tuần 5 | 7/4 -> 13/4       | Bonus window + Research + Report + Finalize      | Chốt bonus nặng đầu tuần, sau đó phân tích sâu và hoàn thiện báo cáo | PDF hoàn chỉnh + repo sẵn nộp + bonus comparison integrated |

### Chi tiết theo mốc ngày thực

#### Tuần 1 + 2 (9/3 -> 23/3): DATA + EDA

* 9/3 - 18/3:

  * Chốt 1 regression dataset + 1 classification dataset.
  * Kiểm tra tiêu chí >= 10k samples, >= 5 features.
  * Chốt split train/val/test (70/10/20), classification dùng stratified split.

* 19/3 - 20/3:

  * Regression: descriptive stats, histogram, boxplot, correlation, scatter.
  * Classification: class distribution, imbalance analysis, feature visualization.

* 21/3 - 22/3:

  * Missing handling, normalization/standardization, encoding.
  * Hoàn thiện pipeline dùng lại được cho cả train/val/test.

* Deadline tuần 1 + 2: **23/3 - Hoàn thành Dataset + EDA**.

#### Tuần 3 (24/3 -> 30/3): MODEL IMPLEMENTATION

* 24/3 - 26/3:

  * Regression core: Linear Regression (NE + mini-batch GD), LR schedule.
  * Ridge/Lasso/Elastic Net, feature selection (Forward/Backward/Lasso).
  * Tích hợp chuẩn bị residual diagnostics và dữ liệu cần cho QQ-plot.

* 27/3 - 29/3:

  * Non-linear basis: Polynomial, RBF, validation curve, ablation.
  * Gắn hook cho Breusch-Pagan test; chuẩn bị flow fallback WLS khi phát hiện heteroscedasticity.

* 30/3:

  * Classification core start: Logistic Regression (GD + softmax), chuẩn bị Newton/IRLS.
  * Đảm bảo toàn bộ model classification trả xác suất để phục vụ calibration ở tuần 4.
  * Integration Checkpoint 1: khóa schema output regression, chạy dry-run bảng tổng hợp.
  * Bonus nhẹ (không tách phase): implement Kernel Ridge + Kernel Logistic ở mức baseline (reuse pipeline hiện có).

* Deadline tuần 2: **30/3: Model core phải chạy được**.

#### Tuần 4 (31/3 -> 6/4): EVALUATION + ANALYSIS

* 31/3 - 2/4 (Regression evaluation):

  * MSE, RMSE, MAE, R2; learning curve, residual plot, pred vs actual.
  * Bắt buộc: QQ-plot, Breusch-Pagan test; nếu có heteroscedasticity thì chạy WLS và so sánh OLS vs WLS.

* 3/4 - 5/4 (Classification evaluation):

  * Accuracy, Precision, Recall, F1; confusion matrix, ROC/AUC, PR curve.
  * Bắt buộc: calibration curve (reliability diagram), McNemar's test, decision boundary (nếu 2D) và so sánh trước/sau calibration.

* 6/4:

  * k-fold CV (Regression k=10, Classification k=5).
  * Statistical test (t-test/Wilcoxon) cho so sánh mô hình.
  * Integration Checkpoint 2: freeze bảng so sánh thống nhất và checklist rubric coverage.

* Deadline tuần 3: **6/4 - Hoàn thành Evaluation full**.

#### Tuần 5 (7/4 -> 13/4): RESEARCH + REPORT + FINALIZE

* 7/4 - 8/4 (BONUS):

  * Regression bonus: Bias-Variance decomposition bằng bootstrap.
  * Classification bonus: Probit hoặc refine Kernel Logistic (chọn 1 theo độ ổn định kết quả).
  * Chuẩn hóa logging/seed/schema để đưa trực tiếp vào bảng so sánh tổng hợp.

* 9/4:

  * Analysis: so sánh model, bias-variance decomposition (định lượng), overfitting/underfitting, error analysis.
  * Tổng hợp diễn giải OLS vs WLS, trước/sau calibration, ổn định hội tụ IRLS/Newton.

* 10/4:

  * Research: GLM, exponential family, Hessian/Jacobian, Gaussian-Markov, sensitivity/robustness.

* 11/4:

  * Viết và hợp nhất báo cáo LaTeX (theory + experiment + discussion) với nội dung rubric tích hợp theo từng phần.
  * Report Freeze v1: khóa toàn bộ hình/bảng để chỉ sửa lỗi trình bày sau mốc này.

* 12/4:

  * Final check: reproducibility, requirements, seed, run lại notebook từ đầu.
  * Buffer day cho lỗi phát sinh (merge conflict, metric mismatch, rerun experiment).
  * Hoàn tất repo sẵn sàng nộp.

* Deadline cuối: **13/4 - Nộp hoàn chỉnh**.

### Critical Path (mốc bắt buộc)

| Mốc bắt buộc      | Deadline  |
| ----------------- | --------- |
| Dataset + EDA     | 23/3     |
| Model core + Integration Checkpoint 1 | 30/3  |
| Evaluation full + Integration Checkpoint 2   | 6/4     |
| Bonus window (Kernel/Bias-Var/Probit) | 8/4 |
| Report full     | 12/4  |
| Final submission  | 13/4     |

---

## Đầu ra mong đợi

### Regression

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

### Classification

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
* Bonus triển khai theo nguyên tắc "gài vào pipeline": không tạo phase riêng, không thêm task ở tuần 4.
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

