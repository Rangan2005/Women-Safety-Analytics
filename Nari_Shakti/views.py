# views.py

from django.http import StreamingHttpResponse
from ultralytics import YOLO
import cv2
from django.shortcuts import render
import os
from django.conf import settings
from pathlib import Path

# Using pathlib for path construction
base_dir = Path(settings.BASE_DIR)
last_model_path = base_dir / 'Nari_Shakti' / 'models' / 'last.pt'
last1_model_path = base_dir / 'Nari_Shakti' / 'models' / 'last1.pt'

# Load pre-trained models
gender_classifier = YOLO(last1_model_path)
face_detector = YOLO(last_model_path)

def process_frame(frame):
    # Initialize gender counts
    male_count = 0
    female_count = 0

    # Face detection
    results = face_detector(frame)

    # Process the results and get bounding boxes
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates
            confidence = box.conf.item()

            # Crop the face region
            face_img = frame[y1:y2, x1:x2]

            # Resize the face image to fit the gender classifier input size
            face_img_resized = cv2.resize(face_img, (224, 224))

            # Gender classification
            gender_results = gender_classifier(face_img_resized)

            # Get the predicted class
            if gender_results and len(gender_results) > 0:
                predicted_class = gender_results[0].probs.top1
                confidence = gender_results[0].probs.top1conf
                gender_label = gender_results[0].names[predicted_class]

                # Increment gender count based on prediction
                if gender_label == 'male':
                    male_count += 1
                elif gender_label == 'female':
                    female_count += 1
            else:
                gender_label = 'Unknown'

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{gender_label} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display male and female counts on the frame
    cv2.putText(frame, f"Male: {male_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def generate_video_stream():
    # Replace '0' with the URL or ID of the CCTV camera stream
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        frame = process_frame(frame)

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the frame to the response stream
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

def live_feed(request):
    # Streaming HTTP response for live video feed
    return StreamingHttpResponse(generate_video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def display_live_feed(request):
    return render(request, 'live_feed.html')
