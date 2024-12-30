import face_recognition
import os
import pickle
import tkinter as tk
from tkinter import filedialog

def encode_faces(dataset_path):
    known_encodings = []
    known_names = []
    unencodable_dirs = []

    for root, dirs, files in os.walk(dataset_path):
        for name in dirs:
            student_dir = os.path.join(root, name)
            has_images = False
            has_unencodable_images = False

            for file in os.listdir(student_dir):
                if file.endswith(('jpg', 'jpeg', 'png')):
                    has_images = True
                    image_path = os.path.join(student_dir, file)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(name)
                    else:
                        has_unencodable_images = True

            if has_images and has_unencodable_images:
                unencodable_dirs.append(student_dir)

    if unencodable_dirs:
        print("The following directories contain images that cannot be encoded:")
        for dir in unencodable_dirs:
            print(dir)

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select Folder to Save Encodings")

    if folder_path:
        save_path = os.path.join(folder_path, 'encodings.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump((known_encodings, known_names), f)
        print(f"Encodings saved to {save_path}")
    else:
        print("Save operation was cancelled")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    dataset_path = filedialog.askdirectory(title="Select Dataset Directory")

    if dataset_path:
        encode_faces(dataset_path)
    else:
        print("Dataset directory selection was cancelled")