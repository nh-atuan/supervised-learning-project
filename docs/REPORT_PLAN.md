# KẾ HOẠCH VIẾT REPORT PHẦN 1 & 2 TRONG ĐỀ BÀI

---

# 1. NGUYÊN TẮC CHIA VIỆC 

Dựa theo đề:
* Mỗi phần đều có 6 block chính:
  ```
  Theory → Data → EDA & Preprocessing → Model → Evaluation → Discussion
  ```

| Loại phần                   | Ai viết          |
| --------------------------- | ---------------- |
| Theory                      | E                |
| Data + EDA + Preprocessing  | A (reg), D (clf) |
| Model                       | B (reg), C (clf) |
| Evaluation                  | A (reg), D (clf) |
| Discussion                  | B (reg), C (clf) |
| Tổng hợp / kiểm tra / bonus | E                |

---

# 2. PHẦN 1 – REGRESSION (CHIA VIỆC REPORT)

## 2.1 Theory → E
Bám đúng mục 2.1 + 2.3 trong đề 

**File:** `content/part1/theory.tex`

**Gợi ý (không giới hạn chỉ trong đây):**
* Linear model + basis functions
* OLS + Normal Equation
* Regularization (Ridge, Lasso)
* MLE vs MAP
* Bias–Variance
* GD variants
* (bonus: Bayesian, Kernel, GP)

---

## 2.2 Data + EDA + Preprocessing → A

**File:** `content/part1/data_eda.tex`

**A viết dựa code đã làm - Gợi ý (không giới hạn chỉ trong đây):**
* Dataset description (đề bắt buộc)
* EDA:
  * histogram, correlation, scatter
  * outlier (IQR/Z-score)
* Preprocessing:
  * missing
  * scaling
  * split (70/10/20)
* Breusch–Pagan test (nếu có)
* WLS (nếu có)

Đây là phần **ăn 7 điểm riêng** trong đề.

---

## 2.3 Model → B

**File:** `content/part1/model.tex`

**B viết toàn bộ phần đã code - Gợi ý (không giới hạn chỉ trong đây):**

### (1) Linear Regression
* Normal Equation (tự cài)
* Mini-batch GD
* Learning rate schedule
* So sánh convergence

### (2) Regularization + Feature Selection
* Ridge / Lasso
* CV chọn λ
* Lasso path
* Elastic Net
* Forward / Backward selection

### (3) Non-linear models
* Polynomial
* RBF
* model thứ 3
* Validation curve
* Ablation study

Đây là **core 25+ điểm**

---

## 2.4 Evaluation → A

**File:** `content/part1/evaluation.tex`

**A viết dựa vào code đã làm - Gợi ý (không giới hạn chỉ trong đây):**
* MSE, RMSE, MAE, R²
* Learning curve
* Residual plot
* Predicted vs Actual
* k-fold CV (k=10)
* Statistical test (t-test / Wilcoxon)

Vì A đang xử lý data + pipeline → hợp lý nhất

---

## 2.5 Discussion → B

**File:** `content/part1/discussion.tex`

**B viết bám vào các yêu cầu của phần 2.2.5 trong file đề bài - Gợi ý (không giới hạn chỉ trong đây):**
* So sánh models
* Effect của λ
* Overfitting / underfitting
* Bias–variance
* Complexity O(N,M)
* Hướng cải tiến
* ...

Vì B hiểu model sâu nhất

---

# 3. PHẦN 2 – CLASSIFICATION (CHIA VIỆC REPORT)

---

## 3.1 Theory → E

**File:** `content/part2/theory.tex`

**E viết (bám mục 3.1 & 3.3 của đề):**
* Classification problem
* Logistic regression
* Cross-entropy
* LDA / QDA
* Perceptron
* Optimization
* (bonus: Probit, Laplace, Kernel LR)

---

## 3.2 Data + EDA → D

**File:** `content/part2/data_eda.tex`

**D viết dựa vào code đã làm - Gợi ý (không giới hạn chỉ trong đây):**
* Dataset description
* Class distribution
* Imbalance analysis
* Visualization
* Encoding
* Scaling
* Stratified split

Đây là phần **6 điểm riêng**

---

## 3.3 Model → C

**File:** `content/part2/model.tex`

**C viết dựa vào code đã làm - Gợi ý (không giới hạn chỉ trong đây):**

### (1) Logistic Regression
* GD (binary + multi)
* Newton / IRLS
* OvR / OvO / Softmax

### (2) LDA + QDA
* Implementation
* Fisher ratio
* Decision boundary

### (3) Perceptron + Regularization
* Perceptron
* Logistic + L1/L2
* Class-weighted loss
* Stratified CV

---

## 3.4 Evaluation → D

**File:** `content/part2/evaluation.tex`

**D viết dựa vào code đã làm - Gợi ý (không giới hạn chỉ trong đây):**
* Accuracy, Precision, Recall, F1
* Confusion matrix
* ROC + AUC
* PR curve
* k-fold CV (k=5)
* McNemar test
* Calibration

---

## 3.5 Discussion → C

**File:** `content/part2/discussion.tex`

**C viết bám sát vào phần 3.2.4. của đề - Gợi ý (không giới hạn chỉ trong đây):**
* Logistic vs LDA
* Model comparison
* Class imbalance impact
* Error analysis
* Linear limitation
* ...

---

# 5. MAP FINAL

| Phần           | Data | Model | Eval | Discussion | Theory + bonus|
| -------------- | ---- | ----- | ---- | ---------- | ------ |
| Regression     | A    | B     | A    | B          | E      |
| Classification | D    | C     | D    | C          | E      |

---

# 6. QUY ĐỊNH CHUNG & TIÊU CHÍ ĐÁNH GIÁ ĐÁNH GIÁ CHẤT LƯỢNG BÁO CÁO (TỪ ĐỀ BÀI)
* **Hình thức báo cáo:** Báo cáo nên được viết bằng **LaTeX**, trình bày sạch đẹp, có đánh số công thức, bảng biểu và hình vẽ rõ ràng. Các báo cáo thiếu mạch lạc, hình / đồ thị kém chất lượng đều bị trừ điểm.
* **Trình bày:** Rõ ràng, mạch lạc, có cấu trúc logic.
* **Toán học:** Công thức toán học chính xác, có giải thích ý nghĩa các ký hiệu, tính thống nhất trên toàn báo cáo. *(E đặc biệt chú ý khi viết Theory)*
* **Biểu đồ:** Biểu đồ đẹp, phải có tên trục, chú thích, tiêu đề. *(A, D chú ý phần Data pipeline và Evaluation)*
* **Phân tích:** Phân tích kết quả sâu sắc, thể hiện hiểu biết phần mô hình/thuật toán. *(B, C chú ý phần Models + Discussion)*
* **Trích dẫn:** Tài liệu trích dẫn đúng chuẩn, Cần ghi rõ ràng theo chuẩn **APA** hoặc **IEEE**. Hạn chế trích dẫn các tài liệu không chính thống như Wikipedia.
