Đồ Án Trí Tuệ Nhân Tạo: Hệ Thống Phân Loại Rác Thải
streamlit app demo: https://waste-classification-app-hmlzeapya4vgbh2u9hy5rc.streamlit.app/

Cách Chạy Ứng Dụng Trên Máy Tính Cá Nhân
Bạn có thể chạy dự án này trên máy tính của mình (Windows, macOS, hoặc Linux) bằng cách làm theo các bước sau:

1. Yêu Cầu Cần Có

Đã cài đặt Python 3.9 trở lên (hãy đảm bảo bạn đã tick vào ô "Add Python to PATH" khi cài đặt trên Windows).

Đã cài đặt Git (công cụ quản lý mã nguồn).

2. Hướng Dẫn Cài Đặt Từng Bước

Bước 1: Lấy Mã Nguồn (Clone Repository)

Mở Terminal (hoặc Command Prompt) của bạn và chạy lệnh sau để tải mã nguồn về máy: git clone https://github.com/lmdat2708/waste-classification-app.git

Bước 2: Di Chuyển Vào Thư Mục Dự Án

Sau khi tải xong, hãy di chuyển vào thư mục dự án vừa được tạo: cd waste-classification-app

Bước 3: Cài Đặt Các Thư Viện Cần Thiết

Dự án này sử dụng các thư viện được liệt kê trong file requirements.txt. Chạy lệnh sau để tự động cài đặt tất cả chúng: pip install -r requirements.txt
(Nếu bạn dùng macOS/Linux, có thể bạn cần gõ pip3 thay vì pip)

Bước 4: Khởi Chạy Ứng Dụng!

Sau khi tất cả các thư viện đã được cài đặt, gõ lệnh sau để khởi chạy ứng dụng Streamlit: streamlit run app.py
(Nếu lệnh streamlit không được tìm thấy, hãy thử: python -m streamlit run app.py hoặc python3 -m streamlit run app.py)
