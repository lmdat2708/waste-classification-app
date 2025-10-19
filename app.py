# app.py

import streamlit as st
import tensorflow as tf
from PIL import Image # Thư viện xử lý ảnh
import numpy as np

# =================================================================
# PHẦN 1: TẢI LẠI MÔ HÌNH VÀ CÁC THIẾT LẬP
# =================================================================

# Tải mô hình đã được lưu
# Chú ý: File 'waste_classifier_model.h5' phải nằm cùng thư mục với file app.py này.
try:
    model = tf.keras.models.load_model('waste_classifier_model.h5')
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    # Dừng ứng dụng nếu không tải được mô hình
    st.stop()


# Định nghĩa tên các lớp theo đúng thứ tự lúc huấn luyện
# Em cần thay đổi danh sách này cho khớp với dự án của mình
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)

# =================================================================
# PHẦN 2: XÂY DỰNG GIAO DIỆN NGƯỜI DÙNG (UI) VỚI STREAMLIT
# =================================================================

st.set_page_config(page_title="Phân Loại Rác Thải", page_icon="♻️")

st.title("♻️ Ứng Dụng Nhận Diện & Phân Loại Rác Thải")
st.write("Tải lên một hình ảnh của rác thải, và mô hình AI sẽ cho bạn biết nó thuộc loại nào.")

# Tạo một vị trí để người dùng có thể kéo/thả hoặc chọn file ảnh
uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])

# =================================================================
# PHẦN 3: XỬ LÝ ẢNH VÀ DỰ ĐOÁN
# =================================================================

def preprocess_image(image):
    """Hàm này nhận một ảnh PIL và chuẩn bị nó cho mô hình."""
    # Resize ảnh về đúng kích thước mô hình yêu cầu
    image = image.resize(IMG_SIZE)
    # Chuyển ảnh thành một mảng numpy
    image_array = np.array(image)
    # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    image_array = image_array / 255.0
    # Thêm một chiều để tạo thành một "lô" (batch) chỉ có một ảnh
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if uploaded_file is not None:
    # Nếu người dùng đã tải ảnh lên...

    # 1. Hiển thị ảnh người dùng đã tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên.', use_column_width=True)
    st.write("")

    # 2. Xử lý ảnh và chuẩn bị cho mô hình
    st.write("Đang phân tích...")
    processed_image = preprocess_image(image)

    # 3. Đưa ảnh vào mô hình để dự đoán
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction) * 100

    # 4. Hiển thị kết quả
    st.success(f"**Kết quả:** {predicted_class_name}")
    st.info(f"**Độ tin cậy:** {confidence:.2f}%")
