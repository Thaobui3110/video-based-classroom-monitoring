import cv2
import os
from datetime import datetime

def record_video(output_file="captured_video.avi", duration=10, save_folder="uncutvideo"):
    os.makedirs(save_folder, exist_ok=True)
    output_path = os.path.join(save_folder, output_file)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to access the webcam.")
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Recording video for {duration} seconds...")
    frame_count = 0
    max_frames = fps * duration

    while frame_count < max_frames:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        out.write(frame)
        cv2.imshow('Recording', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped early by user.")
            break

        frame_count += 1

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

def extract_frames(video_path, output_folder, max_frames=10):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < max_frames:
        print(f"Warning: Video has fewer frames ({total_frames}) than requested ({max_frames}).")
        max_frames = total_frames

    interval = total_frames // max_frames

    print(f"Extracting {max_frames} evenly spaced frames from {video_path}...")
    frame_indices = [i * interval for i in range(max_frames)]
    saved_count = 0

    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame at index {frame_index}.")
            continue

        output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        saved_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_folder}")

def main():
    print("1. Record video and save to uncutvideo")
    print("2. Extract frames from video")
    choice = input("Choose an option (1 or 2): ")

    if choice == "1":
        save_folder = "uncutvideo"
        output_file = input("Enter output video file name (default: captured_video.avi): ") or "captured_video.avi"
        duration = int(input("Enter duration of recording in seconds (default: 10): ") or 10)
        record_video(output_file, duration, save_folder)
    elif choice == "2":
        video_folder = "uncutvideo"
        video_file = input(f"Enter the name of the video file in '{video_folder}': ")
        video_path = os.path.join(video_folder, video_file)

        folder_name = input("Enter the name for the folder to save frames: ")
        output_folder = os.path.join("frames", folder_name)

        max_frames = int(input("Enter the maximum number of frames to extract (default: 10): ") or 10)

        extract_frames(video_path, output_folder, max_frames)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
2