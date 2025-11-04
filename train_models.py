import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import joblib

#Cấu hình
CSV_PATH = r"D:\Code\Top2\data_csv\hand_data.csv"
MODEL_PATH = r"D:\Code\Top2\models\hand_gesture_csv.keras"
LABEL_PATH = r"D:\Code\Top2\models\label_encoder.pkl"
SCALER_PATH = r"D:\Code\Top2\models\scaler.pkl"
EPOCHS = 150
BATCH_SIZE = 32

#Kiểm tra GPU
gpus = tf.config.list_physical_devices('GPU')
print("GPU detected:" if gpus else "No GPU detected", gpus)

#Đọc dữ liệu
df = pd.read_csv(CSV_PATH)
print(f"Đọc {len(df)} mẫu - {df['label'].nunique()} nhãn.")

X = df.drop('label', axis=1).values
y = df['label'].values

#Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Mã hóa nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

#Tách train/val
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

#Mô hình MLP
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Callback
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

#Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

#Lưu label encoder và scaler
with open(LABEL_PATH, "wb") as f:
    pickle.dump(le, f)
joblib.dump(scaler, SCALER_PATH)

print(f"Mô hình đã lưu tại: {MODEL_PATH}")
print(f"Đã lưu label encoder và scaler.")
