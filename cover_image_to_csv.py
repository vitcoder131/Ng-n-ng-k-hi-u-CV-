import cv2
import mediapipe as mp
import os
import csv

#Đường dẫn dữ liệu
DATA_DIR = r"D:\Code\Top2\data\data_all"   # chứa cả 1 tay & 2 tay
CSV_PATH = r"D:\Code\Top2\data_csv\hand_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

#Tạo file CSV
with open(CSV_PATH, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    header = []
    for i in range(42):  # 21 điểm × 2 tay
        header += [f'x{i+1}', f'y{i+1}', f'z{i+1}']
    header.append('label')
    csv_writer.writerow(header)

    for label_dir in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, label_dir)
        if not os.path.isdir(folder_path):
            continue
        print(f"Đang xử lý thư mục: {label_dir}")

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            img = cv2.imread(file_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                row = []

                if len(results.multi_hand_landmarks) == 1:
                    # Chỉ 1 tay → thêm 0 cho tay còn lại
                    for lm in results.multi_hand_landmarks[0].landmark:
                        row += [lm.x, lm.y, lm.z]
                    row += [0] * 63
                else:
                    # 2 tay → ghi đủ cả 2 tay
                    for lm in results.multi_hand_landmarks[0].landmark:
                        row += [lm.x, lm.y, lm.z]
                    for lm in results.multi_hand_landmarks[1].landmark:
                        row += [lm.x, lm.y, lm.z]

                row.append(label_dir)
                csv_writer.writerow(row)

hands.close()
print(f"Đã lưu dữ liệu landmark vào: {CSV_PATH}")
