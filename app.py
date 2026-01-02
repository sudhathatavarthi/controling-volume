import cv2
import math
import time
import numpy as np
import threading
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
import mediapipe as mp
import webview

# -------- SYSTEM VOLUME (Windows) --------
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None
    )
    return cast(interface, POINTER(IAudioEndpointVolume))

volume_interface = get_volume_interface()

def set_system_volume(volume_percent):
    vol = max(0, min(100, volume_percent))
    volume_interface.SetMasterVolumeLevelScalar(vol / 100, None)

# ---------------- Flask ----------------
app = Flask(__name__)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- Global State ----------------
app_running = False
freeze_mode = False

current_volume = 0
target_volume = 0
current_distance = 0
current_gesture = "Detection Stopped"

# Calibration
min_dist = 30
max_dist = 200
auto_calibration = False
calib_start = 0

# Smoothing
smooth_distance = 0
ALPHA = 0.25
VOLUME_STEP = 2

# Stability
distance_history = deque(maxlen=10)
stability = 100

# FPS
prev_time = time.time()
fps = 0

# ---------------- Helper Functions ----------------
def calculate_stability():
    if len(distance_history) < 5:
        return 100
    return int(max(0, 100 - np.std(distance_history) * 2))

def get_gesture(d):
    if d > 80:
        return "Open Hand"
    elif d > 30:
        return "Pinch"
    return "Closed"

def smooth_volume(curr, target):
    if abs(curr - target) <= VOLUME_STEP:
        return target
    return curr + VOLUME_STEP if target > curr else curr - VOLUME_STEP

# ---------------- Video Generator ----------------
def generate_frames():
    global current_distance, current_volume, target_volume
    global current_gesture, stability, fps, prev_time
    global smooth_distance, auto_calibration, calib_start
    global min_dist, max_dist

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # FPS
        now = time.time()
        fps = int(1 / max(0.001, now - prev_time))
        prev_time = now

        # ---------- STOP = NO DETECTION ----------
        if not app_running:
            current_gesture = "Detection Stopped"

            cv2.putText(frame, "DETECTION STOPPED",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

            cv2.putText(frame, f"FPS: {fps}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        # ---------- MediaPipe ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                lm = hand.landmark

                x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
                x2, y2 = int(lm[8].x * w), int(lm[8].y * h)

                raw_dist = int(math.hypot(x2 - x1, y2 - y1))

                # EMA smoothing
                smooth_distance = ALPHA * raw_dist + (1 - ALPHA) * smooth_distance
                current_distance = int(smooth_distance)

                distance_history.append(current_distance)
                stability = calculate_stability()

                # Auto calibration (3 seconds)
                if auto_calibration:
                    if time.time() - calib_start < 3:
                        min_dist = min(min_dist, current_distance)
                        max_dist = max(max_dist, current_distance)
                    else:
                        auto_calibration = False
                        min_dist = max(10, min_dist)
                        max_dist = max(min_dist + 40, max_dist)

                if not freeze_mode and stability > 60:
                    target_volume = int(
                        np.interp(current_distance, [min_dist, max_dist], [0, 100])
                    )

                current_volume = smooth_volume(current_volume, target_volume)

                #  APPLY TO REAL SYSTEM VOLUME
                set_system_volume(current_volume)

                current_gesture = get_gesture(current_distance)

                # Visual indicators
                cv2.circle(frame, (x1, y1), 8, (0,255,0), -1)
                cv2.circle(frame, (x2, y2), 8, (0,255,0), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 2)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        else:
            current_gesture = "Hand Not Detected"

        cv2.putText(frame, f"FPS: {fps}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    return jsonify({
        "volume": current_volume,
        "distance": current_distance,
        "gesture": current_gesture,
        "stability": stability,
        "fps": fps
    })

@app.route('/control', methods=['POST'])
def control():
    global app_running, freeze_mode, auto_calibration, calib_start
    action = request.json["action"]

    if action == "start":
        app_running = True
        freeze_mode = False
        auto_calibration = True
        calib_start = time.time()
    elif action == "stop":
        app_running = False
        freeze_mode = False
    elif action == "freeze":
        freeze_mode = not freeze_mode

    return jsonify(success=True)

@app.route('/calibrate', methods=['POST'])
def calibrate():
    global min_dist, max_dist, auto_calibration
    data = request.json
    min_dist = int(data["min"])
    max_dist = int(data["max"])
    auto_calibration = False
    return jsonify(success=True)

# ---------------- WebView ----------------
def start_flask():
    app.run(debug=False, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=start_flask, daemon=True).start()
    webview.create_window(
        "Gesture Volume Control",
        "http://127.0.0.1:5000",
        width=1100,
        height=720
    )
    webview.start()
