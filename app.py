# app.py (phiên bản TFLite)

import streamlit as st
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

# =================================================================
# PHẦN 1: TẢI MÔ HÌNH TFLITE VÀ CÁC THIẾT LẬP
# =================================================================

# Tải mô hình TFLite
try:
    interpreter = tflite.Interpreter(model_path='waste_classifier.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình TFLite: {e}")
    st.stop()

# Cập nhật danh sách này cho chính xác với dự án của bạn
CLASS_NAMES = ['cardboard', 'ewaste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
IMG_HEIGHT = 224
IMG_WIDTH = 224

# =================================================================
# PHẦN 2: XÂY DỰNG GIAO DIỆN NGƯỜI DÙNG (UI)
# =================================================================

st.set_page_config(page_title="Phân Loại Rác Thải", page_icon="♻️")
st.title("♻️ Ứng Dụng Nhận Diện & Phân Loại Rác Thải (TFLite)")
st.write("Tải lên một hình ảnh của rác thải, mô hình AI siêu nhẹ sẽ phân loại nó.")

uploaded_file = st.file_uploader("Chọn một file ảnh...", type=["jpg", "jpeg", "png"])

# =================================================================
# PHẦN 3: XỬ LÝ ẢNH VÀ DỰ ĐOÁN VỚI TFLITE
# =================================================================

def preprocess_image(image):
    """Hàm chuẩn bị ảnh cho mô hình TFLite."""
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên.', use_column_width=True)
    st.write("")
    st.write("Đang phân tích...")

    processed_image = preprocess_image(image)

    # Đưa ảnh vào mô hình TFLite
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke() # Thực hiện dự đoán
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = np.max(prediction) * 100

    st.success(f"**Kết quả:** {predicted_class_name}")
    st.info(f"**Độ tin cậy:** {confidence:.2f}%")
