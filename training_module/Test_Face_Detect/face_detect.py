import os
from datetime import datetime

import cv2 as cv


# Run face detection every 15 frames
FRAME_INTERVAL = 15

# How much extra space to add around the detected face bbox
# 0.45 means add 45% extra padding around the face
PADDING_RATIO = 0.45

# Folder to save detected face crops
FACE_IMG_DIR = "./Test_Face_Detect/Frame_IMGs"
os.makedirs(FACE_IMG_DIR, exist_ok=True)

# OpenCV face detector
FACE_CASCADE_PATH = cv.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")


def detect_faces(frame):
    """
    Detects faces in a webcam frame.

    Returns a list of bounding boxes:
    (x, y, w, h)
    """

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    return faces


def expand_bbox(x, y, w, h, frame_width, frame_height, padding_ratio=0.45, make_square=True):
    """
    Expands a face bounding box so the saved crop includes more context.

    This helps the anti-spoofing model because it can see more than just
    a tightly cropped face.
    """

    center_x = x + w / 2
    center_y = y + h / 2

    if make_square:
        box_size = max(w, h)
        box_w = box_size
        box_h = box_size
    else:
        box_w = w
        box_h = h

    # Add padding around the bbox
    box_w = box_w * (1 + padding_ratio * 2)
    box_h = box_h * (1 + padding_ratio * 2)

    x1 = int(center_x - box_w / 2)
    y1 = int(center_y - box_h / 2)
    x2 = int(center_x + box_w / 2)
    y2 = int(center_y + box_h / 2)

    # Keep coordinates inside the frame
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    return x1, y1, x2, y2


def save_face_images(frame, faces, frame_count):
    """
    Saves padded face crops from the frame.
    """

    if len(faces) == 0:
        return []

    frame_height, frame_width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    padded_faces = []

    for index, (x, y, w, h) in enumerate(faces, start=1):

        # Expand the bbox before cropping
        x1, y1, x2, y2 = expand_bbox(
            x=x,
            y=y,
            w=w,
            h=h,
            frame_width=frame_width,
            frame_height=frame_height,
            padding_ratio=PADDING_RATIO,
            make_square=True
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # Crop padded face from frame
        face_crop = frame[y1:y2, x1:x2]

        file_name = f"frame_{frame_count:06d}_{timestamp}_face_{index}.jpg"
        output_path = os.path.join(FACE_IMG_DIR, file_name)

        cv.imwrite(output_path, face_crop)

        print(f"Saved padded face image: {output_path}")

        padded_faces.append((x1, y1, x2 - x1, y2 - y1))

    return padded_faces


# Open webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_count = 0
current_padded_faces = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect and save faces every 15 frames
    if frame_count % FRAME_INTERVAL == 0:
        detected_faces = detect_faces(frame)

        current_padded_faces = save_face_images(
            frame=frame,
            faces=detected_faces,
            frame_count=frame_count
        )

    # Draw padded bounding boxes on the live webcam display
    for (x, y, w, h) in current_padded_faces:
        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv.putText(
            frame,
            "Padded Face",
            (x, y - 10 if y > 20 else y + 20),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv.imshow("Face Detection - Padded Crops", frame)

    frame_count += 1

    if cv.waitKey(1) == ord("q"):
        break


cap.release()
cv.destroyAllWindows()