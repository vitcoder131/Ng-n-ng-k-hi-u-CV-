
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import joblib
import pyttsx3
import time

#ĐƯỜNG DẪN
MODEL_PATH = r"D:\Code\Top2\models\hand_gesture_csv.keras"
LABEL_PATH = r"D:\Code\Top2\models\label_encoder.pkl"
SCALER_PATH = r"D:\Code\Top2\models\scaler.pkl"

#LOAD MODEL
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
scaler = joblib.load(SCALER_PATH)

#MEDIAPIPE
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

#CAMERA
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Detection started! Nhấn 'q' để thoát.")

#BIẾN PHỤ TRỢ
recognized_text = ""
last_label = ""
last_time = 0
engine = pyttsx3.init()
engine.setProperty('rate', 170)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    num_hands = 0
    label = ""
    conf = 0

    #DỰ ĐOÁN
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        data = []

        if num_hands == 1:
            for lm in results.multi_hand_landmarks[0].landmark:
                data += [lm.x, lm.y, lm.z]
            data += [0] * 63
        elif num_hands == 2:
            for lm in results.multi_hand_landmarks[0].landmark:
                data += [lm.x, lm.y, lm.z]
            for lm in results.multi_hand_landmarks[1].landmark:
                data += [lm.x, lm.y, lm.z]

        data = np.array(data).reshape(1, -1)
        data = scaler.transform(data)
        prediction = model.predict(data, verbose=0)
        idx = np.argmax(prediction)
        conf = prediction[0][idx] * 100
        label = label_encoder.inverse_transform([idx])[0]

        # GHI NHÃN TRÊN CAMERA
        if conf > 60:
            cv2.putText(frame, f"{label} ({conf:.1f}%)", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #GỘP GIAO DIỆN
    white_panel = np.ones((720, 500, 3), dtype=np.uint8) * 255
   # cv2.line(frame, (780, 0), (780, 720), (0, 0, 0), 3)

    # Ghép 2 khung (video + vùng trắng)
    combined = np.hstack((frame, white_panel))

    #GHI CHỮ TRÊN VÙNG TRẮNG
    cv2.putText(combined, "Da nhan dien:", (1280, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    #THÊM KÝ TỰ MỚI
    if label and conf > 80:
        if label != last_label or (time.time() - last_time > 2.1):
            recognized_text += label
            last_label = label
            last_time = time.time()

    #HIỂN THỊ CHUỖI
    y_offset = 120
    text_lines = [recognized_text[i:i+15] for i in range(0, len(recognized_text), 15)]
    for line in text_lines:
        cv2.putText(combined, line, (1200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 3)
        y_offset += 50

    #HIỂN THỊ SỐ LƯỢNG TAY
    cv2.putText(combined, f"Hand: {num_hands}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

    cv2.imshow("Hand Gesture Detection", combined)

    # XỬ LÝ CLICK
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):  # phát âm thanh
        if recognized_text.strip():
            print("Phát:", recognized_text)
            engine.say(recognized_text)
            engine.runAndWait()
    elif key == ord('c'):  # xóa chuỗi
        recognized_text = ""

cap.release()
hands.close()
cv2.destroyAllWindows()
