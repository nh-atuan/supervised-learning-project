# Review Code Phần 2 - Tống Phúc

Người review: Tuấn Anh

Phạm vi review:
- `code/Part2_Classification/eda_preprocessing/01_eda.ipynb`
- `code/Part2_Classification/eda_preprocessing/02_preprocessing.ipynb`
- `code/Part2_Classification/evaluation/05_evaluation.ipynb`


## Findings

### 1. Dùng Pearson correlation trực tiếp với nhãn đa lớp là không phù hợp
- Vị trí: `code/Part2_Classification/eda_preprocessing/01_eda.ipynb:1134`
- Liên quan thêm: `code/Part2_Classification/eda_preprocessing/01_eda.ipynb:1183`
- Mô tả:
  Notebook đang tính:
  `corr_with_target = df[continuous_cols + [target_col]].corr()[target_col].drop(target_col)`
  rồi dùng kết quả đó để xếp hạng feature và chọn `top_scatter_features`.
- Vấn đề:
  `Cover_Type` là nhãn phân loại đa lớp, không phải biến số liên tục. Pearson correlation trên mã lớp 1..7 sẽ áp đặt quan hệ thứ tự giả giữa các lớp. Nếu chỉ đổi cách mã hóa nhãn, thứ hạng feature có thể thay đổi dù bản chất dữ liệu không đổi.
- Ảnh hưởng:
  Insight ở phần "tương quan với target" và pairplot theo "top biến" hiện không đủ cơ sở thống kê.
- Khuyến nghị:
  Thay bằng ANOVA/Fisher score, mutual information, Kruskal-Wallis, hoặc so sánh phân phối theo từng lớp.


## Điểm tốt
- Tách notebook EDA, preprocessing và evaluation rõ ràng.
- Pipeline preprocessing có `fit` trên train rồi mới `transform` val/test, đây là điểm đúng về mặt chống leakage.
- Phần evaluation có ý thức dùng nhiều nhóm metric thay vì chỉ bám vào accuracy.



