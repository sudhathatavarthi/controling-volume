import cv2
import math
import threading
import os
from time import time

from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import webview

# ---------------- APP SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# ---------------- AUDIO ----------------
def init_volume():
    speakers = AudioUtilities.GetSpeakers()
    interface = speakers.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    return cast(interface, POINTER(IAudioEndpointVolume))

volume_interface = init_volume()

# ---------------- GLOBAL STATE ----------------
current_volume = 0
current_distance = 0
gesture_mode = "OPEN"

app_running = False
freeze_mode = False
app_status = "STOPPED"

# Manual calibration
DEFAULT_MIN = 20
DEFAULT_MAX = 200
min_dist = DEFAULT_MIN
max_dist = DEFAULT_MAX

start_time = None

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global current_volume, current_distance, gesture_mode

    while True:
        success, frame = cap.read()
        if not success:
            break

        if app_running and not freeze_mode:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    lm = hand.landmark
                    h, w, _ = frame.shape

                    x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
                    x2, y2 = int(lm[8].x * w), int(lm[8].y * h)
                    wx, wy = int(lm[0].x * w), int(lm[0].y * h)

                    dist = math.hypot(x2 - x1, y2 - y1)
                    open_dist = math.hypot(x2 - wx, y2 - wy)
                    current_distance = int(dist)

                    if dist < 30:
                        gesture_mode = "PINCH"
                    elif open_dist < 90:
                        gesture_mode = "CLOSED"
                        current_volume = 0
                        volume_interface.SetMasterVolumeLevelScalar(0.0, None)
                    else:
                        gesture_mode = "OPEN"
                        vol = int((dist - min_dist) * 100 / (max_dist - min_dist))
                        vol = max(0, min(100, vol))
                        current_volume = vol
                        volume_interface.SetMasterVolumeLevelScalar(vol / 100, None)

                    cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
                    cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame + b"\r\n")

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/metrics")
def metrics():
    return jsonify(
        volume=current_volume,
        distance=current_distance,
        gesture=gesture_mode,
        status=app_status
    )

@app.route("/start")
def start_app():
    global app_running, freeze_mode, app_status
    app_running = True
    freeze_mode = False
    app_status = "RUNNING"
    return jsonify(status=app_status)

@app.route("/stop")
def stop_app():
    global app_running, app_status
    app_running = False
    app_status = "STOPPED"
    return jsonify(status=app_status)

@app.route("/freeze")
def freeze_app():
    global freeze_mode, app_status
    freeze_mode = not freeze_mode
    app_status = "FROZEN" if freeze_mode else "RUNNING"
    return jsonify(status=app_status)

@app.route("/apply_calibration", methods=["POST"])
def apply_calibration():
    global min_dist, max_dist
    data = request.json
    min_dist = int(data["min"])
    max_dist = int(data["max"])
    return jsonify(min=min_dist, max=max_dist)

# ---------------- DESKTOP ----------------
def start_flask():
    app.run(port=5000, debug=False)

if __name__ == "__main__":
    threading.Thread(target=start_flask, daemon=True).start()
    webview.create_window(
        "Gesture Volume Control",
        "http://127.0.0.1:5000",
        width=1300,
        height=850
    )
    webview.start()
