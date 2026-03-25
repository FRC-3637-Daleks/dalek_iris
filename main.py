import json
import cv2
import numpy as np
import os
import glob
import re
from ultralytics import YOLO

from get_3d_coords import get_3d_coords

DEBUG = True  # Set to True to enable live 2D plotting & camera windows
CAMERA_INDEX = 1  # 0 is usually the built-in webcam, 1 or 2 for plugged in USB cams
EXPOSURE_VALUE = -9

model = YOLO('calibrate/model.pt')

def get_fuel_centers_ai(img, confidence=0.15):
    """
    Returns a list of dictionaries [{'x': cx, 'y': cy, 'area': a}]
    """
    # Run AI inference
    # conf=0.5 ignores anything the AI is less than 50% sure is a ball
    results = model.predict(img, conf=confidence, verbose=False)
    
    candidates = []
    
    for result in results:
        for box in result.boxes:
            # Get coordinates [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Calculate the center of the box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            w = x2 - x1
            h = y2 - y1
            area = int(w*h)
            
            candidates.append({'x': cx, 'y': cy, 'area': area})
            
    return candidates


def set_exposure(cap, value):

    # 1. Turn off Auto Exposure
    # On many cameras: 1 is manual mode, 3 is auto mode.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    
    # 2. Set the Exposure value
    cap.set(cv2.CAP_PROP_EXPOSURE, value)


def getImg(cap):
    """
    Captures a frame from the plugged-in camera.
    """
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to grab frame.")
        return None
    return frame


def draw_2d_map(fuel_offsets):
    """
    Helper function used in DEBUG mode.
    Creates a top-down radar view of the fuel relative to the robot.
    """
    # Create a blank 600x600 image
    map_size = 600
    map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)

    # Scale coordinates to fit on screen
    # X (Lateral): -10 to +10 feet
    # Z (Forward): 0 to 30 feet
    def feet_to_px(x_ft, z_ft):
        px_per_x = map_size / 20.0
        px_per_z = map_size / 30.0
        x_px = int((x_ft + 10) * px_per_x)
        z_px = int(map_size - (z_ft * px_per_z))
        return x_px, z_px

    # --- Draw Map Grid ---
    # Vertical lines (every 2 feet)
    for x in range(-10, 11, 2):
        px, py1 = feet_to_px(x, 0)
        _, py2 = feet_to_px(x, 30)
        cv2.line(map_img, (px, py1), (px, py2), (50, 50, 50), 1)
    # Horizontal lines (every 5 feet)
    for z in range(0, 31, 5):
        px1, py = feet_to_px(-10, z)
        px2, _ = feet_to_px(10, z)
        cv2.line(map_img, (px1, py), (px2, py), (50, 50, 50), 1)

    # --- Draw Robot ---
    robot_px = feet_to_px(0, 0)
    cv2.circle(map_img, robot_px, 8, (0, 0, 255), -1)  # Red dot
    cv2.putText(map_img, "Robot", (robot_px[0] - 20, robot_px[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- Draw Detected Fuel ---
    for x, z in fuel_offsets:
        px, py = feet_to_px(x, z)
        cv2.circle(map_img, (px, py), 6, (0, 255, 255), -1)  # Yellow dots

    return map_img
    

def processImg(img):
    results = model.predict(img, conf=0.25, verbose=False)

    fuel_offsets = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx_px = int((x1 + x2) / 2)
            cy_px = int((y1 + y2) / 2)
            print(cx_px, cy_px)
            wx, wz = get_3d_coords(cx_px, cy_px, 640, 480)

            if 0.5 < wz < 35 and -10 < wx < 10:
                fuel_offsets.append((wx, wz))

                if DEBUG:
                    # Draw Bounding Box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw Center Marker
                    cv2.drawMarker(img, (cx_px, cy_px), (0, 0, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                    
                    # Label with distance
                    cv2.putText(img, f"{wz:.1f}ft {wx:.1f}ftx", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if DEBUG:
        map_view = draw_2d_map(fuel_offsets)
        cv2.imshow("AI Detection View", img)
        cv2.imshow("Robot-Centric Map (Top-Down)", map_view)

    return fuel_offsets


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    set_exposure(cap, EXPOSURE_VALUE)

    print("Starting AI Vision Loop...")

    try:
        while True:
            # 1. Capture Camera Image
            frame = getImg(cap)
            if frame is None:
                continue

            # 2. Process image and get the current relative fuel offsets
            # This now uses the AI logic inside processImg
            current_fuel_coords = processImg(frame)

            # 3. Handle OpenCV Windows and breaks
            if DEBUG:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
