import cv2
import numpy as np
import torch
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
import tkinter as tk
from tkinter import filedialog

# Kiểm tra CUDA
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print(f"CUDA available: {use_cuda}")

# Cấu hình MTCNN
mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

# Khởi tạo nhận diện cảm xúc
model_name = 'enet_b0_8_best_afew'
fer = HSEmotionRecognizer(model_name=model_name, device=device)

def load_encodings():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    pkl_file_path = filedialog.askopenfilename(title="Select Encodings File", filetypes=[("Pickle files", "*.pkl")])

    if not pkl_file_path:
        print("File selection was cancelled")
        return None, None

    with open(pkl_file_path, 'rb') as f:
        known_encodings, known_names = pickle.load(f)

    return known_encodings, known_names

def log_attendance(name, class_name, emotion, logged_names):
    # Bỏ qua nếu tên là "Unknown"
    if name == "Unknown" or name in logged_names:
        return

    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')
    attendance_data = {'Name': name, 'Emotion': emotion, 'Time': time}

    directory = os.path.join('Attendance', class_name)
    os.makedirs(directory, exist_ok=True)
    csv_file = os.path.join(directory, f'{date}.csv')

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Emotion', 'Time'])

    new_entry = pd.DataFrame([attendance_data])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(csv_file, index=False)
    logged_names.add(name)

def detect_faces_and_emotions(frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    results = []
    if bounding_boxes is not None:
        for box, prob in zip(bounding_boxes, probs):
            if prob > 0.9:  # Lọc theo độ tin cậy
                x1, y1, x2, y2 = box.astype(int)
                face_img = frame[y1:y2, x1:x2, :]
                if face_img.size > 0:  # Kiểm tra kích thước ảnh con
                    # Phát hiện cảm xúc
                    emotion, _ = fer.predict_emotions(face_img, logits=False)
                    results.append((box, emotion))
    return results

def main():
    known_encodings, known_names = load_encodings()
    if known_encodings is None or known_names is None:
        return

    class_name = input("Enter the class name: ")
    if not class_name:
        print("Class name input was cancelled")
        return

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Unable to open webcam.")
        return

    logged_names = set()

    while True:
        ret, frame_bgr = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detect_faces_and_emotions(frame)

        for box, emotion in results:
            x1, y1, x2, y2 = box.astype(int)
            face_img = frame[y1:y2, x1:x2]

            name = "Unknown"
            face_encoding = face_recognition.face_encodings(face_img)
            if face_encoding:
                matches = face_recognition.compare_faces(known_encodings, face_encoding[0])
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding[0]))
                    name = known_names[best_match_index]

            # Chỉ log nếu name != "Unknown"
            if name != "Unknown":
                log_attendance(name, class_name, emotion, logged_names)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"{name} - {emotion}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_bgr, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Attendance & Emotion Recognition', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
