import json
import cv2
import numpy as np

# Import your existing 3D transform function
from get_3d_coords import get_3d_coords

# =========================================================
# CONFIGURATION
# =========================================================
DEBUG = True  # Set to True to enable live 2D plotting & camera windows
CAMERA_INDEX = 1  # 0 is usually the built-in webcam, 1 or 2 for plugged in USB cams


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
    """
    Finds yellow blobs, translates to 2D relative offset coordinates,
    and returns them as a list. Displays debug views if DEBUG is True.
    """
    # --- Yellow mask ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 90, 90]), np.array([35, 255, 255]))

    # --- Clean up noise ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Find connected blobs ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    MIN_BLOB_AREA = 200  # pixels² — filters out noise

    # This list will hold the (X, Z) relative coordinate offsets for this frame
    fuel_offsets = []

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA:
            continue

        cx_px, cy_px = centroids[i]

        # Calculate coordinate
        wx, wz = get_3d_coords(cx_px, cy_px)

        # Sanity filter — ignore points behind robot or impossibly far
        if 0.5 < wz < 35 and -10 < wx < 10:
            fuel_offsets.append((wx, wz))

            # Draw over the camera feed if debugging
            if DEBUG:
                radius = int(np.sqrt(area / np.pi))
                cv2.circle(img, (int(cx_px), int(cy_px)), radius, (0, 255, 255), 2)
                cv2.drawMarker(img, (int(cx_px), int(cy_px)), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # Render display if DEBUG mode is enabled
    if DEBUG:
        map_view = draw_2d_map(fuel_offsets)

        cv2.imshow("Camera View (Live)", img)
        cv2.imshow("Robot-Centric Map (Top-Down)", map_view)

    return fuel_offsets


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    print("Starting loop. " + ("Press 'q' in the window to quit." if DEBUG else "Press Ctrl+C in terminal to stop."))

    try:
        while True:
            # 1. Capture Camera Image
            frame = getImg(cap)
            if frame is None:
                continue

            # 2. Process image and get the current relative fuel offsets
            # This list is gathered, returned, and currently unimplemented to any robot logic
            current_fuel_coords = processImg(frame)

            # Example representation of the list currently populated:
            # current_fuel_coords =[(1.2, 5.5), (-3.4, 8.2), ...]

            # 3. Handle OpenCV Windows and breaks
            if DEBUG:
                # waitKey is required to make imshow windows update
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Clean up once the loop ends
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()