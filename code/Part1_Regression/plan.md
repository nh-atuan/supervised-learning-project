# Kế hoạch Thực hiện Phần 1: Bài toán Hồi quy (Regression)

## 1. Yêu cầu Xây dựng và Huấn luyện mô hình

### 1.1. Hồi quy Tuyến tính (Linear Regression)
* [ ] Cài đặt **Normal Equations** từ đầu (không sử dụng thư viện `sklearn.LinearRegression`) [1].
* [ ] Cài đặt **Mini-batch Gradient Descent** kết hợp learning rate schedule (step decay hoặc cosine annealing) và so sánh tốc độ hội tụ so với Normal Equations [1].
* [ ] Kiểm tra giả thiết **Gauss–Markov**: vẽ residual plot, QQ-plot, và thực hiện kiểm định Breusch–Pagan [1].
* [ ] Cài đặt **Weighted Least Squares (WLS)** trong trường hợp phát hiện hiện tượng heteroscedasticity [1].

### 1.2. Hồi quy có Regularization và Lựa chọn Đặc trưng
* [ ] Cài đặt **Ridge Regression** (L2) và **Lasso Regression** (L1) [2].
* [ ] Lựa chọn siêu tham số $\lambda$ tốt nhất bằng **$k$-fold CV** ($k$ = 10), sử dụng cả grid search lẫn warm start (Lasso path) [2].
* [ ] Vẽ **regularization path**: biểu diễn hệ số $w$ theo $\log\lambda$ cho cả Ridge và Lasso [2].
* [ ] Cài đặt **Elastic Net** và phân tích vùng tối ưu cho $(\lambda_1, \lambda_2)$ [2].
* [ ] Thực hiện **Lựa chọn đặc trưng (Feature Selection)**: so sánh các phương pháp Forward Stepwise Selection, Backward Elimination, và chọn đặc trưng dựa trên các hệ số khác 0 của Lasso [2].

### 1.3. Mô hình với Hàm Cơ sở Phi Tuyến và Ablation Study
* [ ] Áp dụng ít nhất **ba loại hàm cơ sở**: đa thức (polynomial), Gaussian RBF, và một loại tự chọn [3].
* [ ] Vẽ **validation curve**: thể hiện MSE theo bậc đa thức hoặc số lượng hàm cơ sở [3].
* [ ] Thực hiện **ablation study**: lần lượt loại bỏ từng nhóm đặc trưng hoặc từng loại hàm cơ sở, đo lường tác động đến hiệu năng để tìm ra yếu tố quan trọng nhất [3].
* [ ] Phân tích **hiệu ứng tương tác**: thêm các hạng tử $x_ix_j$ vào mô hình và đánh giá mức độ cải thiện [3].

---

## 2. Các yêu cầu Nâng cao (Bonus) [+5 điểm]
* [ ] **Hồi quy Bayesian đầy đủ**: Tính phân phối hậu nghiệm (posterior) $p(w|t)$ và vẽ phân phối dự đoán (predictive distribution) kèm vùng bất định $\bar{f}^* \pm 2\sigma_N$ trên tập test [4].
* [ ] **Evidence Maximization**: Tối ưu hóa $\alpha, \beta$ bằng re-estimation equations (EM-style) và so sánh chất lượng cũng như thời gian với Cross-Validation (CV) [4].
* [ ] **Kernel Ridge Regression**: Cài đặt ít nhất hai kernel (RBF, polynomial), chọn bandwidth bằng CV và so sánh với Ridge Regression tuyến tính [4].
* [ ] **Gaussian Process (GP) Regression**: Cài đặt GP với kernel RBF, tối ưu log-marginal-likelihood bằng gradient ascent để học tham số kernel và vẽ posterior predictive kèm error bars [4].
* [ ] **Robust Regression**: Cài đặt IRLS với phân phối Student-t hoặc Huber loss, so sánh độ nhạy với các giá trị ngoại lai (outliers) so với OLS [4].
* [ ] **Phân tích Bias–Variance thực nghiệm**: Thực hiện bootstrap 200 lần để ước lượng bias² và variance, sau đó vẽ trên cùng một trục với regularization path [4].

---

## 3. Cách Đánh giá mô hình
*Việc đánh giá phải được thực hiện trên tập test với các yêu cầu sau:*

### 3.1. Chỉ số đánh giá cơ bản
* [ ] Tính và báo cáo **MSE** (Sai số bình phương trung bình) [5].
* [ ] Tính và báo cáo **RMSE** (Căn bậc hai MSE) [5].
* [ ] Tính và báo cáo **MAE** (Sai số tuyệt đối trung bình) [5].
* [ ] Tính và báo cáo **$R^2$** (Hệ số xác định) [5].

### 3.2. Biểu đồ phân tích
* [ ] Vẽ **learning curves**: thể hiện train loss và validation loss thay đổi theo số lượng mẫu huấn luyện [5].
* [ ] Vẽ biểu đồ **phần dư (residuals)** để kiểm tra tính ngẫu nhiên của sai số [5].
* [ ] Vẽ biểu đồ so sánh **giá trị dự đoán vs. giá trị thật** (predicted vs. actual) [5].

### 3.3. So sánh và Kiểm định
* [ ] Thực hiện **$k$-fold cross-validation** ($k$ = 10) để tính và báo cáo mean $\pm$ std cho từng chỉ số [5].
* [ ] Trình bày và so sánh tất cả các mô hình trong **một bảng kết quả thống nhất** [5].
* [ ] Sử dụng kiểm định thống kê **paired t-test** hoặc **Wilcoxon signed-rank test** để xác nhận sự khác biệt giữa các mô hình có ý nghĩa thống kê hay không [6].