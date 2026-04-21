# Bảng tổng hợp kết quả kiểm duyệt

Tài liệu này tổng hợp kết quả kiểm duyệt của người kiểm tra (reviewer) đối với các phần đã được phân công của người thực hiện (performer), cùng với phản hồi chi tiết và các hành động cần thiết để cải thiện chất lượng công việc. Mục tiêu là đảm bảo rằng tất cả các phần của dự án đều đạt yêu cầu về chất lượng và hoàn thành đúng tiến độ.

## Xuân Trí (Người kiểm tra: Đàm Đạt)

| Yêu cầu | Kết quả kiểm duyệt | Phản hồi chi tiết | Hành động cần thiết | Phản hồi từ người thực hiện |
|---------|--------------------|-------------------|---------------------|-----------------------------|
| EDA Phần 1 (code) | | | | |
| Tiền xử lý Phần 1 (code)| | | | |
| Đánh giá mô hình Phần 1 (code) | | | | |
| Mô tả dữ liệu Phần 1 (báo cáo) | | | | |
| EDA Phần 1 (báo cáo) | | | | |
| Tiền xử lý Phần 1 (báo cáo) | | | | |
| Đánh giá mô hình Phần 1 (báo cáo) | | | | |

## Tuấn Anh (Người kiểm tra: Tống Phúc)

| Yêu cầu | Kết quả kiểm duyệt | Phản hồi chi tiết | Hành động cần thiết | Phản hồi từ người thực hiện |
|---------|--------------------|-------------------|---------------------|-----------------------------|
| Xây dựng & Huấn luyện mô hình Phần 1 (code) | Đạt nma cần chỉnh sửa nhỏ | Đã triển khai đầy đủ các nhóm mô hình trong code/Part1_Regression/models: (1) Linear Regression gồm Normal Equations tự cài đặt, Mini-batch GD có step decay và cosine annealing, có so sánh hội tụ/thời gian, có residual plot QQ-plot Breusch-Pagan và có WLS; (2) Regularization gồm Ridge, Lasso, Elastic Net, có 10-fold CV cho lambda của Ridge/Lasso, có regularization path và warm start cho Lasso path, có Forward Stepwise Backward Elimination và Lasso-based selection; (3) Nonlinear gồm Polynomial Gaussian RBF Fourier, có validation curve, có interaction terms và có ablation theo nhóm đặc trưng.Điểm chưa chặt theo đề: warm start mới dùng cho path (là ở cái regularization path á) chứ chưa dùng để chọn lambda tốt nhất (dòng 2 Yêu cầu 2, thấy ông sài grid search thui) ; ablation theo loại hàm cơ sở thì mới so sánh mô hình, chưa làm thiết kế bỏ từng loại nhưng mà yêu cầu thì tui thấy thầy để đặc trưng hay hàm cơ sở, nên làm đặc trưng rồi thì cơ sở làm vậy cũng ok rồi. | 1) Bổ sung quy trình chọn lambda cho Lasso bằng warm-start path có tiêu chí chọn ngưỡng rõ ràng rồi so sánh với grid-search CV (tui đề xuất v thui còn ông có thể tùy chỉnh). 2) (Optional) Bổ sung ablation theo loại basis trong cùng pipeline kết hợp (all basis rồi remove từng basis) và báo cáo delta MSE delta R2 | |
| Cơ sở lý thuyết của các mô hình Phần 1 (báo cáo)| Đạt | File theory.tex trình bày đầy đủ nền tảng toán cho 3 nhóm bắt buộc: Linear (Normal Equation, Mini-batch GD, Gauss-Markov, WLS), Regularization (Ridge/Lasso/Elastic Net, CV, regularization path, warm start), Nonlinear (Polynomial/RBF/Fourier, feature selection, bonus). Công thức và ký hiệu nhìn chung nhất quán với phần cài đặt. |(optional) Có thể bổ sung một bảng ký hiệu N, M, D ngay đầu phần lý thuyết để đọc mạch hơn . | |
| Xây dựng & Huấn luyện mô hình Phần 1 (báo cáo)| Đạt có lưu ý nhỏ | File model.tex đã mô tả pipeline, cách huấn luyện và bảng kết quả cho toàn bộ yêu cầu bắt buộc: so sánh Normal Equation vs Mini-batch GD; có Gauss-Markov và WLS; có Ridge/Lasso/Elastic Net, CV, regularization path, feature selection; có 3 hàm cơ sở phi tuyến, validation curve, ablation theo nhóm đặc trưng và phân tích interaction terms. Điểm lưu ý: trong báo cáo cần nói rõ warm start được dùng cho Lasso path (không phải quy trình chọn lambda tối ưu, nếu ông kh sửa code chỗ đó). | 1) Thêm 1-2 câu chốt rõ vai trò warm start trong phần chọn siêu tham số. 2)(optional) Ghi rõ baseline/metric nhất quán giữa các bảng để người chấm đối chiếu nhanh. | |
| Phân tích & Thảo luận Phần 1 (báo cáo) | Đạt nhma lưu ý nhỏ | discussion.tex đã được bổ sung đầy đủ các ý chính theo rubric: so sánh hiệu năng và chọn mô hình tốt nhất, phân tích bias-variance, overfitting/underfitting, hạn chế mô hình tuyến tính, đề xuất cải tiến, và thảo luận độ phức tạp tính toán. Phần learning curve đã có nhận xét định tính trong nội dung. Lưu ý nhỏ: mục độ phức tạp chưa tách rõ một dòng so sánh riêng cho Bayesian Regression như đề nêu; learning curve nên có hình/bảng dẫn chiếu trực tiếp để đối chiếu minh chứng rõ hơn. |  1) Bổ sung 1 tiểu mục ngắn so sánh O(.) riêng cho Normal Equation vs Mini-batch GD vs Bayesian Regression theo N,M. 2) (Optional) Chèn 1 hình hoặc bảng learning curve có label để liên kết trực tiếp với phần nhận xét hội tụ. | |

## Tống Phúc (Người kiểm tra: Tuấn Anh)

| Yêu cầu | Kết quả kiểm duyệt | Phản hồi chi tiết | Hành động cần thiết | Phản hồi từ người thực hiện |
|---------|--------------------|-------------------|---------------------|-----------------------------|
| EDA Phần 2 (code) | | | | |
| Tiền xử lý Phần 2 (code)| | | | |
| Đánh giá mô hình Phần 2 (code) | | | | |
| Mô tả dữ liệu Phần 2 (báo cáo) | | | | |
| EDA Phần 2 (báo cáo) | | | | |
| Tiền xử lý Phần 2 (báo cáo) | | | | |
| Đánh giá mô hình Phần 2 (báo cáo) | | | | |

## Đàm Đạt (Người kiểm tra: Xuân Trí)

| Yêu cầu | Kết quả kiểm duyệt | Phản hồi chi tiết | Hành động cần thiết | Phản hồi từ người thực hiện |
|---------|--------------------|-------------------|---------------------|-----------------------------|
| Xây dựng & Huấn luyện mô hình Phần 2 (code) | | | | |
| Cơ sở lý thuyết của các mô hình Phần 2 (báo cáo)| | | | |
| Xây dựng & Huấn luyện mô hình Phần 2 (báo cáo)| | | | |
| Phân tích & Thảo luận Phần 2 (báo cáo) | | | | |

## Anh Tuấn (Người kiểm tra:)

| Yêu cầu | Kết quả kiểm duyệt | Phản hồi chi tiết | Hành động cần thiết | Phản hồi từ người thực hiện |
|---------|--------------------|-------------------|---------------------|-----------------------------|
| Xây dựng & Huấn luyện các mô hình nâng cao (code) | | | | |
| Cơ sở lý thuyết của các mô hình nâng cao (báo cáo)| | | | |
| Xây dựng & Huấn luyện các mô hình nâng cao (báo cáo)| | | | |
| Phần 3 - Phân tích So sánh và Kỹ năng Nghiên cứu (báo cáo) | | | | |

