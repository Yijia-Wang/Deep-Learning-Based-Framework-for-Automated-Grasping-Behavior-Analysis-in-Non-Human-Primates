import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

input_folder = "right" 
output_folder = "right_output"  

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)
mp_drawing = mp.solutions.drawing_utils

for video_file in os.listdir(input_folder):
    if video_file.endswith(".mov"):
        video_path = os.path.join(input_folder, video_file)
        base_name = os.path.splitext(video_file)[0]

        output_annotated_video = os.path.join(output_folder, f"{base_name}-annotated.mp4")
        output_blank_video = os.path.join(output_folder, f"{base_name}-blank.mp4")
        output_csv = os.path.join(output_folder, f"{base_name}.csv")

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_annotated = cv2.VideoWriter(output_annotated_video, fourcc, fps, (width, height))
        out_blank = cv2.VideoWriter(output_blank_video, fourcc, fps, (width, height))

        landmarks_list = []

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            annotated_frame = frame.copy()
            blank_frame = 255 * np.ones((height, width, 3), dtype=np.uint8) 

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    mp_drawing.draw_landmarks(
                        blank_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    hand_data = {"frame": frame_index}
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        hand_data[f"x_{i}"] = landmark.x
                        hand_data[f"y_{i}"] = landmark.y
                        hand_data[f"z_{i}"] = landmark.z
                    landmarks_list.append(hand_data)

            out_annotated.write(annotated_frame)
            out_blank.write(blank_frame)

            frame_index += 1

        cap.release()
        out_annotated.release()
        out_blank.release()

        df = pd.DataFrame(landmarks_list)
        df.to_csv(output_csv, index=False)

        print(f"landmarks are saved as {output_csv}")

hands.close()

