# Github: VitCoder - Đề tài(Nhóm 10 - 22CN2 - Khoa Công nghệ thông tin - Đại học Kiến Trúc Hà Nội):
## Ứng dụng Computer Vision nhận diện thủ ngữ(Ngôn ngữ ký hiệu) - cho người khiếm thị - Bảng ASL(Amiraca Sign Language) 

# QuocBao - Hướng dẫn chi tiết các sử dụng 
- Thu thập dữ liệu ảnh và lưu vào thư mục(Ảnh cần có nhiều góc độ, ánh sáng, trạng thái khác nhau)
- Chạy file cover_image_to_csv.py để covert ảnh sang csv, file sẽ biến đổi ảnh bằng mediapipe nhận dạng các tọa độ của 21 khớp để lưu vào file csv
- Khi đã chạy xong sẽ sinh ra 1 file csv, sau đó ta chạy file train_model.py để thực hiện train
- Sau file train sẽ sinh ra 1 file keras và 1 file pkl
- Sau đó chạy file detection.py để kiểm tra
- Lưu ý: Sửa lại đường dẫn trong các file

- Tìm kiếm mẫu trong Test result
- Nguồn ảnh: https://kaggle.com/ASL
             https://roboflow.com/ASL

- Các thư viện cần thiết có trong file lib.txt ở dự án
# Phát triển bởi Levanro
- Liên hệ: 0345591223