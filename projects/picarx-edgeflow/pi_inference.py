#Live surface classification on PiCar-X with keyboard driving.
#Drive with W A S D, see predictions in real-time.

import sys
import time
import threading
import termios
import tty
import numpy as np
import onnxruntime as rt
from collections import deque
from picarx import Picarx

# Load model and init car
sess = rt.InferenceSession("surface_classifier.onnx")
px = Picarx()

SPEED = 20
current_steering = 0
running = True
latest_prediction = "waiting..."
latest_confidence = 0.0
latest_gs = [0, 0, 0]
latest_distance = 0.0
latest_texture = 0.0
reading_count = 0

# Rolling window buffer
window_size = 10
gs_left_buf = deque(maxlen=window_size)
gs_center_buf = deque(maxlen=window_size)
gs_right_buf = deque(maxlen=window_size)


def get_key():
    # Read a single keypress without blocking print output.
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def drive_loop():
    # Keyboard thread: reads keys, controls car.
    global current_steering, running

    while running:
        key = get_key().lower()

        if key == "w":
            px.forward(SPEED)
        elif key == "s":
            px.backward(SPEED)
        elif key == "a":
            current_steering = max(current_steering - 10, -30)
            px.set_dir_servo_angle(current_steering)
        elif key == "d":
            current_steering = min(current_steering + 10, 30)
            px.set_dir_servo_angle(current_steering)
        elif key == " ":
            px.stop()
            current_steering = 0
            px.set_dir_servo_angle(0)
        elif key == "q":
            running = False


def predict_loop():
    # Sensor loop: reads sensors, computes features, predicts.
    global latest_prediction, latest_confidence, latest_gs
    global latest_distance, latest_texture, reading_count

    while running:
        gs = px.get_grayscale_data()
        distance = px.get_distance()
        latest_gs = gs
        latest_distance = distance

        gs_left_buf.append(gs[0])
        gs_center_buf.append(gs[1])
        gs_right_buf.append(gs[2])
        reading_count += 1

        if reading_count < window_size:
            latest_prediction = f"buffering ({reading_count}/{window_size})"
            time.sleep(0.5)
            continue

        # Compute features (same as dbt ml_surface_features)
        left = list(gs_left_buf)
        center = list(gs_center_buf)
        right = list(gs_right_buf)

        gs_mean = (left[-1] + center[-1] + right[-1]) / 3.0
        gs_spread = max(left[-1], center[-1], right[-1]) - min(left[-1], center[-1], right[-1])
        gs_left_rolling_std = np.std(left)
        gs_center_rolling_std = np.std(center)
        gs_right_rolling_std = np.std(right)
        gs_left_rolling_mean = np.mean(left)
        gs_center_rolling_mean = np.mean(center)
        gs_right_rolling_mean = np.mean(right)
        outer_avg = (left[-1] + right[-1]) / 2.0
        gs_center_to_outer_ratio = center[-1] / outer_avg if outer_avg > 0 else 0
        gs_texture_score = (gs_left_rolling_std + gs_center_rolling_std + gs_right_rolling_std) / 3.0
        latest_texture = gs_texture_score

        features = np.array([[
            gs_mean, gs_spread,
            gs_left_rolling_std, gs_center_rolling_std, gs_right_rolling_std,
            gs_left_rolling_mean, gs_center_rolling_mean, gs_right_rolling_mean,
            gs_center_to_outer_ratio, gs_texture_score,
        ]], dtype=np.float32)

        prediction = sess.run(None, {"features": features})
        latest_prediction = prediction[0][0]
        latest_confidence = max(prediction[1][0].values())

        time.sleep(0.5)


def display_loop():
    # Print predictions at regular intervals.
    while running:
        print(f"[{reading_count}] Surface: {latest_prediction:10s} ({latest_confidence:.0%}) | "
              f"dist={latest_distance:.1f}cm | gs={latest_gs} | "
              f"texture={latest_texture:.1f}")
        time.sleep(1.0)


# Start threads
print("Surface Classifier + Keyboard Driver")
print("=" * 50)
print("Controls: W=forward  S=backward  A=left  D=right  SPACE=stop  Q=quit\n")

predict_thread = threading.Thread(target=predict_loop, daemon=True)
predict_thread.start()

display_thread = threading.Thread(target=display_loop, daemon=True)
display_thread.start()

try:
    drive_loop()  # runs on main thread (handles keyboard input)
except KeyboardInterrupt:
    running = False

px.stop()
px.set_dir_servo_angle(0)
print("\nStopped.")
