import os
import time

import cv2
import requests
import soundfile as sf
import torch
from ultralytics import YOLO
from omnivoice import OmniVoice

CAMERA_URL = "http://192.168.1.65:8080/video"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

OUTPUT_AUDIO = "scene_description.wav"
CAPTURE_SECONDS = 10


# ---------------- LOAD MODELS ----------------

print("Loading YOLO model...")
yolo = YOLO("yolo11n.pt")

print("Loading OmniVoice model...")

if torch.cuda.is_available():
    device = "cuda:0"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

tts = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=dtype,
)

print("Models loaded successfully!")


# ---------------- CAPTURE CAMERA FRAME ----------------


def capture_frame():
    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        raise RuntimeError("Cannot connect to Android IP Webcam.")

    print(f"Capturing camera for {CAPTURE_SECONDS} seconds...")

    start_time = time.time()
    last_frame = None

    while time.time() - start_time < CAPTURE_SECONDS:
        success, frame = cap.read()

        if success:
            last_frame = frame

    cap.release()

    if last_frame is None:
        raise RuntimeError("No camera frame received.")

    cv2.imwrite("captured_frame.jpg", last_frame)

    return last_frame


# ---------------- YOLO OBJECT DETECTION ----------------


def detect_objects(frame):
    results = yolo(
        frame,
        imgsz=640,
        conf=0.40,
        verbose=False,
    )

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())

        object_name = yolo.names[class_id]

        detection = {
            "object": object_name,
            "confidence": round(confidence, 3),
        }

        detections.append(detection)

        print(f"Object: {object_name}, " f"Confidence: {confidence:.2%}")

    return detections


# ---------------- SEND TEXT TO OLLAMA ----------------


def describe_objects(detections):
    if detections:
        detection_text = "\n".join(
            f"- {item['object']}: " f"{item['confidence']:.1%} confidence"
            for item in detections
        )
    else:
        detection_text = "No objects were detected."

    prompt = f"""
You are a helpful assistant describing objects detected
by a camera.

Here are the YOLO object detection results:

{detection_text}

Write a short, simple, natural English description
that sounds good when spoken aloud.

Instructions:
- Mention the detected objects.
- Use confidence scores only as supporting information.
- Combine repeated objects naturally.
- Do not invent locations, actions, colors, or relationships
  that are not provided by the detection results.
- If no objects were detected, say that no objects were detected.
- Use only 1 to 3 sentences.
- Return only the description, without headings or bullet points.
"""

    print("\nSending detection text to Ollama...")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )

    response.raise_for_status()

    description = response.json()["response"].strip()

    print("\nGemma's description:")
    print(description)

    return description


# ---------------- CONVERT TEXT TO SPEECH ----------------


def speak_description(description):
    print("\nGenerating speech with OmniVoice...")

    audio = tts.generate(
        text=description,
        instruct="female, american accent",
    )

    sf.write(
        OUTPUT_AUDIO,
        audio[0],
        24000,
    )

    print(f"Audio saved to: {os.path.abspath(OUTPUT_AUDIO)}")

    # Play the generated WAV file on Windows.
    os.startfile(os.path.abspath(OUTPUT_AUDIO))


# ---------------- RUN THE COMPLETE PIPELINE ----------------


def main():
    frame = capture_frame()

    detections = detect_objects(frame)

    description = describe_objects(detections)

    speak_description(description)


if __name__ == "__main__":
    main()
