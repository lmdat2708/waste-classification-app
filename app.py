# app.py (Phiên bản cuối cùng - Tải trọng số)

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# =================================================================
# PHẦN 1: XÂY DỰNG LẠI KIẾN TRÚC MÔ HÌNH
# =================================================================

# Cập nhật danh sách này cho chính xác với dự án của bạn
CLASS_NAMES = ['cardboard', 'ewaste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
IMG_SIZE = (224, 224)

def build_model(num_classes):
    """Hàm này xây dựng lại chính xác kiến trúc mô hình đã huấn luyện."""
    IMG_SHAPE = IMG_SIZE + (3,)

    # Tải mô hình gốc MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    # Xây dựng mô hình hoàn chỉnh
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Xây dựng "thể xác" rỗng
model = build_model(len(CLASS_NAMES))

# Nạp "linh hồn" (trọng số) đã học vào
try:
    model.load_weights('waste_classifier_weights.weights.h5')
except Exception as e:
    st.error(f"Lỗi khi tải trọng số của mô hình: {e}")
    st.stop()

# =================================================================
# PHẦN 2: GIAO DIỆN VÀ XỬ LÝ DỰ ĐOÁN (Giữ nguyên)
# =================================================================

st.set_page_config(page_title="Phân Loại Rác Thải", page_icon="♻️")
st.title("♻️ Ứng Dụng Nhận Diện & Phân Loại Rác Thải")
st.write("Tải lên một hình ảnh của rác thải, và mô hình AI sẽ cho bạn biết nó thuộc loại nào.")

uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    """Hàm chuẩn bị ảnh cho mô hình."""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    # Bỏ bước chuẩn hóa /255.0 vì mô hình MobileNetV2 tự xử lý việc này
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    st.image(image, caption='Ảnh đã tải lên.', use_column_width=True)
    st.write("")
    st.write("Đang phân tích...")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction) * 100

    st.success(f"**Kết quả:** {predicted_class_name}")
    st.info(f"**Độ tin cậy:** {confidence:.2f}%")
