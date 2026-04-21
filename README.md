# Supervised Learning Project - Group 14

## 0) Mục đích của README này

README này là bản skeleton chi tiết cho toàn bộ repo hiện tại, phục vụ 4 mục tiêu:

1. Onboard nhanh thành viên mới vào đúng luồng chạy.
2. Chuẩn hóa cách chạy thực nghiệm để tái lập kết quả.
3. Mô tả rõ cấu trúc mã nguồn, notebook, dữ liệu, báo cáo LaTeX.
4. Làm tài liệu tham chiếu chính trước khi đóng gói bài nộp.

Lưu ý:

- Đây là skeleton đầy đủ theo trạng thái repo hiện tại.
- Có thể bổ sung hình minh họa, số liệu cuối cùng và kết luận ngắn gọn trước khi nộp.

---

## 1) Mục lục

- [0) Mục đích của README này](#0-mục-đích-của-readme-này)
- [1) Mục lục](#1-mục-lục)
- [2) Tổng quan đồ án](#2-tổng-quan-đồ-án)
- [3) Cấu trúc thư mục hiện tại](#3-cấu-trúc-thư-mục-hiện-tại)
- [4) Yêu cầu môi trường và cài đặt](#4-yêu-cầu-môi-trường-và-cài-đặt)
- [5) Luồng chạy nhanh (Quick Start)](#5-luồng-chạy-nhanh-quick-start)
- [6) Dữ liệu: nguồn, vị trí, quy ước](#6-dữ-liệu-nguồn-vị-trí-quy-ước)
- [7) Phần 1 - Regression](#7-phần-1---regression)
- [8) Phần 2 - Classification](#8-phần-2---classification)
- [9) Phần 3 - Comparison/Research](#9-phần-3---comparisonresearch)
- [10) Hệ thống báo cáo LaTeX](#10-hệ-thống-báo-cáo-latex)
- [11) Danh sách dependencies](#11-danh-sách-dependencies)
- [12) Tái lập thực nghiệm (Reproducibility)](#12-tái-lập-thực-nghiệm-reproducibility)
- [13) Quy ước làm việc nhóm](#13-quy-ước-làm-việc-nhóm)
- [14) Checklist trước khi nộp](#14-checklist-trước-khi-nộp)
- [15) Troubleshooting](#15-troubleshooting)
- [16) Tài liệu tham khảo nội bộ](#16-tài-liệu-tham-khảo-nội-bộ)

---

## 2) Tổng quan đồ án

### 2.1 Bối cảnh

Đây là đồ án môn Học có giám sát, gồm hai bài toán chính bắt buộc và một phần so sánh mở rộng:

- Part 1: Hồi quy (Regression) trên Bike Sharing.
- Part 2: Phân loại (Classification) trên Forest Cover Type (Covertype).
- Part 3: So sánh tổng hợp, phân tích độ bền vững, hội tụ và tái lập thực nghiệm.

### 2.2 Mục tiêu kỹ thuật

- Cài đặt mô hình từ đầu cho các thuật toán cốt lõi.
- Tổ chức pipeline dữ liệu rõ ràng, tránh data leakage.
- Đánh giá đa chiều: metric, trực quan hóa, kiểm định thống kê.
- Đảm bảo khả năng tái lập bằng seed cố định và artifact xuất file.

### 2.3 Trạng thái hiện tại của repo

Repo đã có đầy đủ:

- Notebook EDA, preprocessing, model training, evaluation cho Part 1 và Part 2.
- Mã Python module hóa cho Part 1 và Part 3.
- Artifacts kết quả dạng joblib/csv/jsonl/npz.
- Hệ thống báo cáo LaTeX với cấu trúc theo từng phần nội dung.

---

## 3) Cấu trúc thư mục hiện tại

### 3.1 Cây thư mục chính (rút gọn, theo trạng thái thực tế)

```text
supervised-learning-project/
├── README.md
├── requirements.txt
├── code/
│   ├── Part1_Regression/
│   │   ├── utils.py
│   │   ├── eda_preprocessing/
│   │   │   ├── 01_eda.ipynb
│   │   │   ├── 02_preprocessing.ipynb
│   │   │   └── load_data.py
│   │   ├── models/
│   │   │   ├── 03_linear_regression.ipynb
│   │   │   ├── 04_regularized_regression.ipynb
│   │   │   ├── 05_nonlinear_models.ipynb
│   │   │   ├── 06_advanced_regression.ipynb
│   │   │   ├── linear_regression.py
│   │   │   ├── regularized_regression.py
│   │   │   ├── nonlinear_models.py
│   │   │   ├── feature_selection.py
│   │   │   ├── bonus_models.py
│   │   │   └── results/
│   │   │       ├── trained_models_03.joblib
│   │   │       ├── trained_models_04.joblib
│   │   │       ├── trained_models_05.joblib
│   │   │       ├── predictions_03_linear_regression.joblib
│   │   │       ├── predictions_04_regularized_regression.joblib
│   │   │       ├── predictions_05_nonlinear_models.joblib
│   │   │       ├── metrics_03_linear_regression.joblib
│   │   │       ├── metrics_04_regularized_regression.joblib
│   │   │       ├── metrics_05_nonlinear_models.joblib
│   │   │       ├── best_params_04.joblib
│   │   │       └── best_params_05.joblib
│   │   └── evaluation/
│   │       ├── 07_evaluation.ipynb
│   │       ├── metrics.py
│   │       ├── plots.py
│   │       └── statistical_tests.py
│   ├── Part2_Classification/
│   │   ├── utils.py
│   │   ├── eda_preprocessing/
│   │   │   ├── 01_eda.ipynb
│   │   │   └── 02_preprocessing.ipynb
│   │   ├── models/
│   │   │   ├── 03_classification_model_training.ipynb
│   │   │   └── 04_advanced_classification.ipynb
│   │   ├── evaluation/
│   │   │   └── 05_evaluation.ipynb
│   │   └── results/
│   │       └── evaluation_artifacts_from_03.joblib
│   └── Part3_Comparison/
│       ├── 08_part3_comparison_research.ipynb
│       ├── part3_pipeline.py
│       └── results/
│           ├── part3_comprehensive_comparison_table.csv
│           ├── split_sensitivity_results.csv
│           ├── noise_robustness_results.csv
│           ├── corruption_robustness_results.csv
│           ├── convergence_summary.csv
│           ├── experiment_log.jsonl
│           ├── split_indices.npz
│           ├── environment_snapshot.txt
│           └── part3_manifest.json
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── README.md
│   │   ├── regression/
│   │   │   ├── README.md
│   │   │   ├── day.csv
│   │   │   └── hour.csv
│   │   └── classification/
│   │       ├── README.md
│   │       └── covtype.csv
│   └── processed/
│       ├── README.md
│       ├── regression/
│       │   ├── X_train.csv / X_val.csv / X_test.csv
│       │   ├── y_train_log.csv / y_val_log.csv / y_test_log.csv
│       │   ├── y_train_orig.csv / y_val_orig.csv / y_test_orig.csv
│       │   └── weights_train.csv / weights_val.csv / weights_test.csv
│       └── classification/
│           └── .gitkeep
├── docs/
│   ├── REQUIREMENT.md
│   ├── PLAN.md
│   ├── DATASET_SUMMARY.md
│   ├── REPORT_PLAN.md
│   └── REVIEW.md
└── report/
	├── main.tex
	├── hcmus-report-template.sty
	├── content/
	│   ├── title.tex
	│   ├── introduction.tex
	│   ├── conclusion.tex
	│   ├── part1/*.tex
	│   ├── part2/*.tex
	│   └── part3/*.tex
	├── appendix/appendix.tex
	├── img/
	│   ├── part1/
	│   ├── part2/
	│   └── part3/
	└── ref/
		├── ref.bib
		└── ref.tex
```

### 3.2 Mô tả nhanh các thư mục cấp cao

- code/: toàn bộ notebook và module Python cho 3 phần.
- data/: dữ liệu raw + dữ liệu processed sau preprocessing.
- docs/: tài liệu yêu cầu môn, kế hoạch, phân công, review.
- report/: mã nguồn LaTeX báo cáo, hình ảnh và tài liệu tham khảo.

---

## 4) Yêu cầu môi trường và cài đặt

### 4.1 Phiên bản Python khuyến nghị

- Bắt buộc: Python 3.11.x.
- Không nên chạy bằng Python 3.10 hoặc 3.12+ để tránh lệch dependency/kernel.

### 4.2 Tạo virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
```

macOS/Linux:

```bash
python3.11 -m venv .venv
```

### 4.3 Kích hoạt môi trường

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4.4 Cài dependencies

```bash
pip install -r requirements.txt
```

### 4.5 Đăng ký kernel Jupyter

```bash
python -m ipykernel install --user --name ml-project-py311
```

Khi mở notebook, chọn kernel:

```text
ml-project-py311
```

### 4.6 Kiểm tra nhanh môi trường

```bash
python --version
pip --version
```

---

## 5) Luồng chạy nhanh (Quick Start)

### 5.1 Trình tự khuyến nghị toàn dự án

1. Chạy Part 1 theo thứ tự notebook.
2. Chạy Part 2 theo thứ tự notebook.
3. Chạy Part 3 để tổng hợp so sánh.
4. Build báo cáo LaTeX.

### 5.2 Thứ tự chạy Part 1

```text
code/Part1_Regression/eda_preprocessing/01_eda.ipynb
-> code/Part1_Regression/eda_preprocessing/02_preprocessing.ipynb
-> code/Part1_Regression/models/03_linear_regression.ipynb
-> code/Part1_Regression/models/04_regularized_regression.ipynb
-> code/Part1_Regression/models/05_nonlinear_models.ipynb
-> code/Part1_Regression/models/06_advanced_regression.ipynb
-> code/Part1_Regression/evaluation/07_evaluation.ipynb
```

### 5.3 Thứ tự chạy Part 2

```text
code/Part2_Classification/eda_preprocessing/01_eda.ipynb
-> code/Part2_Classification/eda_preprocessing/02_preprocessing.ipynb
-> code/Part2_Classification/models/03_classification_model_training.ipynb
-> code/Part2_Classification/models/04_advanced_classification.ipynb
-> code/Part2_Classification/evaluation/05_evaluation.ipynb
```

### 5.4 Thứ tự chạy Part 3

```text
code/Part3_Comparison/08_part3_comparison_research.ipynb
```

Hoặc chạy script tổng hợp trực tiếp:

```bash
python code/Part3_Comparison/part3_pipeline.py
```

---

## 6) Dữ liệu: nguồn, vị trí, quy ước

### 6.1 Bộ dữ liệu sử dụng

Regression:

- Bike Sharing Dataset (UCI).
- File chính: data/raw/regression/hour.csv.

Classification:

- Forest Cover Type (UCI/Kaggle mirror).
- File chính: data/raw/classification/covtype.csv.

### 6.2 Quy ước dữ liệu raw

- Không chỉnh sửa file raw sau khi tải.
- Không đổi tên file đầu vào chuẩn.
- Mọi xử lý phải đi qua notebook/script preprocessing.

### 6.3 Quy ước dữ liệu processed

- Processed chỉ chứa đầu ra của pipeline.
- Với Regression, repo đã có sẵn đầy đủ file split và target dạng log/original.
- Với Classification, thư mục processed/classification hiện chỉ giữ chỗ bằng .gitkeep.

### 6.4 Lưu ý chống data leakage

- Mọi thống kê scaler/imputer/clip phải fit trên train.
- Val/Test chỉ transform theo tham số đã học từ train.
- Không để thông tin target rơi vào feature (ví dụ casual/registered trong bài toán cnt).

---

## 7) Phần 1 - Regression

### 7.1 Các notebook chính

- 01_eda.ipynb: thống kê, trực quan hóa, phát hiện ngoại lệ, kiểm định ban đầu.
- 02_preprocessing.ipynb: feature engineering, split, scaling, xử lý heteroscedasticity.
- 03_linear_regression.ipynb: OLS, Mini-batch GD, WLS.
- 04_regularized_regression.ipynb: Ridge/Lasso/Elastic Net + CV + regularization path.
- 05_nonlinear_models.ipynb: Polynomial, RBF, Fourier, ablation.
- 06_advanced_regression.ipynb: Bayesian, Evidence Maximization, KRR, GP, IRLS, bias-variance bootstrap.
- 07_evaluation.ipynb: tổng hợp metric, học đường cong, kiểm định thống kê.

### 7.2 Modules Python Part 1

code/Part1_Regression/utils.py

- set_seed, load_dataset, describe_dataset, split helper.

code/Part1_Regression/eda_preprocessing/load_data.py

- load_regression_data(): nạp dữ liệu processed đã chuẩn hóa/split.
- add_bias_column(): thêm cột bias.

code/Part1_Regression/models/linear_regression.py

- NormalEquationLR.
- MiniBatchGDLR (support schedule constant/step_decay/cosine).
- WeightedLeastSquares.
- gauss_markov_diagnostics (residual/QQ/Breusch-Pagan).

code/Part1_Regression/models/regularized_regression.py

- RidgeRegression.
- LassoRegression (coordinate descent).
- ElasticNetRegression.
- select_lambda_cv, compute_ridge_path, compute_lasso_path.

code/Part1_Regression/models/nonlinear_models.py

- PolynomialRegression.
- RBFRegression.
- FourierBasisRegression.
- add_interaction_terms.
- run_ablation_study.

code/Part1_Regression/models/feature_selection.py

- forward_stepwise_selection.
- backward_elimination.
- lasso_feature_selection.

code/Part1_Regression/models/bonus_models.py

- BayesianLinearRegression.
- EvidenceMaximization.
- KernelRidgeRegression.
- GaussianProcessRegression.
- RobustRegression (Huber IRLS).
- bias_variance_bootstrap.

code/Part1_Regression/evaluation/metrics.py

- mse, rmse, mae, medae, r_squared, adjusted_r_squared.
- k-fold split + cross_validate_model.

code/Part1_Regression/evaluation/plots.py

- learning curve, residual, predicted-vs-actual, QQ plot,
- validation curve, convergence, model comparison, CV boxplot, heatmap.

code/Part1_Regression/evaluation/statistical_tests.py

- paired_t_test.
- wilcoxon_test.
- friedman_test.
- pairwise_wilcoxon_posthoc.

### 7.3 Artifacts Part 1

Các file trong code/Part1_Regression/models/results chứa:

- trained_models_03/04/05.joblib: mô hình đã fit.
- predictions_03/04/05.joblib: dự đoán test theo mô hình.
- metrics_03/04/05.joblib: metric tổng hợp.
- best_params_04/05.joblib: siêu tham số tối ưu từ CV/search.

---

## 8) Phần 2 - Classification

### 8.1 Các notebook chính

- 01_eda.ipynb: khám phá Covertype, class imbalance, tương quan, outlier.
- 02_preprocessing.ipynb: feature engineering, split stratified, scaling, weighting.
- 03_classification_model_training.ipynb: Logistic (GD/Newton), OvR/OvO/Softmax, LDA/QDA, Perceptron, regularization.
- 04_advanced_classification.ipynb: Probit, Laplace approximation, kernel logistic, GNB/LDA diagnostics.
- 05_evaluation.ipynb: Accuracy/Precision/Recall/F1, confusion matrix, ROC/AUC, PR curve, calibration, McNemar, error analysis.

### 8.2 Module Python hiện có

code/Part2_Classification/utils.py

- set_seed, load_dataset, describe_dataset.
- train/val/test split tiện ích.
- log_experiment_params.

### 8.3 Artifacts Part 2

code/Part2_Classification/results/evaluation_artifacts_from_03.joblib gồm (theo pipeline hiện tại):

- Dữ liệu đã xử lý cho train/val/test.
- Dự đoán multiclass test của các biến thể Logistic/LDA/QDA.
- Lịch sử loss cho GD/Newton (nhị phân).
- Kết quả task nâng cao phục vụ phần evaluation và part3.

---

## 9) Phần 3 - Comparison/Research

### 9.1 Vai trò

Part 3 là cầu nối tổng hợp giữa Regression và Classification, gồm:

- Bảng so sánh toàn diện mô hình.
- Phân tích độ nhạy theo tỉ lệ split.
- Phân tích robustness với noise injection.
- Phân tích feature corruption + imputation.
- Phân tích hội tụ thuật toán.
- Snapshot môi trường + manifest tái lập.

### 9.2 Thành phần mã nguồn

code/Part3_Comparison/part3_pipeline.py

- resolve_repo_root(): định vị root repo.
- _load_part1_modules(): import module Part1 động.
- run_all(): chạy toàn bộ thí nghiệm Part3, xuất artifact.

Các nhóm thí nghiệm trong run_all():

1. Split sensitivity (60/40, 70/30, 80/20).
2. Noise robustness (sigma = 0.0, 0.1, 0.5, 1.0).
3. Feature corruption (rate = 0.1, 0.2, 0.3 + imputation mean/median/knn).
4. Convergence summary (GD regression, GD/Newton classification).
5. Ghi log + split indices + environment snapshot + manifest.

### 9.3 Ý nghĩa các file output

code/Part3_Comparison/results/part3_comprehensive_comparison_table.csv

- Bảng so sánh lý thuyết/thực nghiệm giữa nhóm mô hình chính.

code/Part3_Comparison/results/split_sensitivity_results.csv

- Kết quả theo các tỉ lệ chia dữ liệu, gồm score và chi phí tính toán.

code/Part3_Comparison/results/noise_robustness_results.csv

- Kết quả khi thêm nhiễu Gaussian vào feature test.

code/Part3_Comparison/results/corruption_robustness_results.csv

- Kết quả khi làm hỏng ngẫu nhiên feature + chiến lược bù dữ liệu.

code/Part3_Comparison/results/convergence_summary.csv

- Tóm tắt số vòng lặp, best loss, final loss theo thuật toán.

code/Part3_Comparison/results/experiment_log.jsonl

- Log từng bản ghi thí nghiệm (JSON Lines).

code/Part3_Comparison/results/split_indices.npz

- Chỉ số mẫu train/test/subset để tái lập đúng split.

code/Part3_Comparison/results/environment_snapshot.txt

- Snapshot phiên bản Python, package, platform tại thời điểm chạy.

code/Part3_Comparison/results/part3_manifest.json

- Manifest tổng hợp đường dẫn file và số lượng bản ghi thí nghiệm.

---

## 10) Hệ thống báo cáo LaTeX

### 10.1 Cấu trúc report

- report/main.tex: file gốc include toàn bộ nội dung.
- report/content/part1/*.tex: các phần lý thuyết, EDA, preprocessing, model, evaluation, discussion cho Regression.
- report/content/part2/*.tex: tương tự cho Classification.
- report/content/part3/*.tex: comparison, robustness, reproducibility.
- report/img/: toàn bộ hình minh họa dùng trong báo cáo.
- report/ref/ref.bib: tài liệu tham khảo.

### 10.2 Build PDF

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Nếu không dùng bibtex trong phiên bản hiện tại, có thể chỉ cần chạy pdflatex nhiều lần.

### 10.3 Trạng thái nội dung report

- Nhiều phần đã viết chi tiết đầy đủ (đặc biệt part1, part2 theory/model/eval, part3).
- Một số mục còn TODO trong introduction/conclusion/discussion part2 cần hoàn thiện trước khi nộp.

---

## 11) Danh sách dependencies

Theo requirements.txt hiện tại:

- numpy==1.26.4
- pandas==2.2.3
- matplotlib==3.10.0
- scikit-learn==1.4.2
- seaborn==0.13.2
- plotly==6.0.0
- ipykernel==7.2.0
- scipy==1.12.0
- statsmodels==0.14.2
- jupyter
- nbformat
- statsmodels

Ghi chú:

- File hiện có cả statsmodels pinned và statsmodels unpinned. Có thể dọn lại để nhất quán trước khi chốt bài nộp.

---

## 12) Tái lập thực nghiệm (Reproducibility)

### 12.1 Quy tắc bắt buộc

- Cố định seed trước mọi thao tác có ngẫu nhiên.
- Chạy notebook từ kernel sạch.
- Không trộn artifact cũ với kết quả mới khi so sánh.

Ví dụ set seed dùng chung:

```python
from code.Part1_Regression.utils import set_seed as set_seed_reg
from code.Part2_Classification.utils import set_seed as set_seed_cls

set_seed_reg(42)
set_seed_cls(42)
```

### 12.2 Khuyến nghị vận hành

- Mỗi lần chạy toàn bộ, lưu timestamp/ngữ cảnh môi trường.
- Kiểm tra các file output trọng yếu có được sinh ra đầy đủ.
- Giữ 1 bản snapshot kết quả ổn định để đối chiếu trước khi chỉnh sửa lớn.

---

## 13) Quy ước làm việc nhóm

### 13.1 Branch

Theo docs/PLAN.md:

```text
task/<phase>/<member>
```

Ví dụ:

- task/eda-preprocessing/memA
- task/models/memB
- task/evaluation/memD

### 13.2 Commit

Chuẩn đề xuất:

```text
type(scope): short description
```

Ví dụ:

- feat(regression): add ridge cv search and path plotting
- fix(classification): stabilize newton irls updates
- docs(report): update part3 robustness section

### 13.3 Merge notebook

- Chia section rõ ràng theo nhiệm vụ trước khi chỉnh notebook.
- Tránh chỉnh cell thuộc phần người khác nếu chưa thống nhất.
- Sau merge, luôn rerun notebook để kiểm tra tính nhất quán output.

---

## 14) Checklist trước khi nộp

### 14.1 Code và notebook

- [ ] Tất cả notebook chạy end-to-end không lỗi.
- [ ] Không còn đường dẫn cứng máy cá nhân.
- [ ] Seed được cố định nhất quán.
- [ ] Artifacts cần thiết được sinh đầy đủ.

### 14.2 Báo cáo

- [ ] Build report/main.tex ra PDF thành công.
- [ ] Hình/ bảng có caption và label đầy đủ.
- [ ] Kết quả trong báo cáo khớp artifact thực tế.
- [ ] Trích dẫn/bibliography đúng chuẩn.

### 14.3 Repo

- [ ] README cập nhật đúng trạng thái cuối.
- [ ] requirements.txt đồng bộ môi trường chạy cuối.
- [ ] Dọn file tạm không cần thiết.

---

## 15) Troubleshooting

### 15.1 Lỗi không chọn được kernel

Giải pháp:

1. Kích hoạt .venv.
2. Chạy lại lệnh cài ipykernel.
3. Reload VS Code và chọn lại kernel ml-project-py311.

### 15.2 Lỗi thiếu package

```bash
pip install -r requirements.txt
```

Nếu vẫn lỗi, kiểm tra:

- Đang đúng virtual environment chưa.
- Phiên bản Python có đúng 3.11.x chưa.

### 15.3 Notebook chạy khác kết quả mong đợi

Giải pháp:

1. Restart Kernel and Run All.
2. Xóa output cũ trong notebook (nếu cần).
3. Kiểm tra dữ liệu đầu vào và artifact có bị lẫn phiên bản cũ.

### 15.4 Lỗi khi build LaTeX

- Chạy pdflatex nhiều vòng.
- Nếu dùng bibtex, chạy đúng thứ tự pdflatex -> bibtex -> pdflatex -> pdflatex.
- Đảm bảo file ảnh tồn tại đúng thư mục report/img/.

---

## 16) Tài liệu tham khảo nội bộ

- docs/REQUIREMENT.md: đề bài và yêu cầu chấm.
- docs/PLAN.md: phân công, timeline, branch flow.
- docs/REPORT_PLAN.md: kế hoạch viết report chi tiết.
- docs/DATASET_SUMMARY.md: mô tả dữ liệu hai bài toán.
- docs/REVIEW.md: bảng theo dõi review chéo giữa thành viên.
- data/README.md: quy ước quản lý dữ liệu raw/processed.

---

## 17) Ghi chú hoàn thiện skeleton (TODO)

Các mục nên cập nhật thêm trước khi nộp chính thức:

- [TODO] Thêm bảng benchmark cuối cùng đã chốt (Part 1 + Part 2).
- [TODO] Thêm ảnh kiến trúc pipeline tổng quan của nhóm.
- [TODO] Đồng bộ số liệu README với số liệu final trong report PDF.
- [TODO] Dọn requirements.txt tránh package trùng dòng.
- [TODO] Chốt phần introduction/conclusion/discussion còn TODO trong report.

---

## 18) Thông tin nhóm

Theo docs/PLAN.md và report/main.tex:

- 23120099 - Lê Xuân Trí
- 23120208 - Dương Tuấn Anh
- 23120118 - Đàm Tiến Đạt
- 23120158 - Tống Thanh Phúc
- 23120185 - Nguyễn Hồ Anh Tuấn

Giảng viên phụ trách: Thầy Lê Nhựt Nam.

