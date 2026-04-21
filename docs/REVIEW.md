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
| Xây dựng & Huấn luyện mô hình Phần 1 (code) | | | | |
| Cơ sở lý thuyết của các mô hình Phần 1 (báo cáo)| | | | |
| Xây dựng & Huấn luyện mô hình Phần 1 (báo cáo)| | | | |
| Phân tích & Thảo luận Phần 1 (báo cáo) | | | | |

## Tống Phúc (Người kiểm tra: Tuấn Anh)

| Yêu cầu | Kết quả kiểm duyệt | Phản hồi chi tiết | Hành động cần thiết | Phản hồi từ người thực hiện |
|---------|--------------------|-------------------|---------------------|-----------------------------|
| EDA Phần 2 (code) | Cần chỉnh sửa | Dùng Pearson correlation trực tiếp với nhãn đa lớp `Cover_Type` là không phù hợp, làm áp đặt quan hệ thứ tự giả giữa các lớp và sai lệch thứ hạng feature. | Thay bằng ANOVA/Fisher score, mutual information, hoặc so sánh phân phối theo từng lớp. | |
| Tiền xử lý Phần 2 (code)| Đạt | Code chuẩn mực, thiết lập Pipeline với `fit` trên train giúp tránh data leakage tốt. | Không có | |
| Đánh giá mô hình Phần 2 (code) | Cần chỉnh sửa | 1. PR curve vẽ theo micro-average nhưng gắn nhãn `AP macro`. 2. Biểu đồ `Decision Boundary Approx.` từ PCA 2 chiều nội suy không phản ánh decision boundary thực tế. | 1. Đổi nhãn thành `AP micro` hoặc tính theo `AP macro` chuẩn. 2. Đổi tên biểu đồ (minh họa vùng dự đoán) hoặc xóa bỏ. | |
| Mô tả dữ liệu Phần 2 (báo cáo) | Đạt | Mô tả chi tiết, đầy đủ. | Không có | |
| EDA Phần 2 (báo cáo) | Cần chỉnh sửa | Insight phần tương quan với target không đủ cơ sở thống kê do sai phương pháp tính. | Cập nhật lại diễn giải sau khi thay đổi phương pháp tính tương quan. | |
| Tiền xử lý Phần 2 (báo cáo) | Đạt | Giải thích hợp lý, rõ ràng chiến lược tiền xử lý. | Không có | |
| Đánh giá mô hình Phần 2 (báo cáo) | Cần chỉnh sửa | Các diễn giải liên quan đến PR/AP multiclass và decision boundary có thể dẫn đến hiểu sai sức mạnh của mô hình. | Cập nhật lại nhận xét sau khi code điều chỉnh. | |

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

