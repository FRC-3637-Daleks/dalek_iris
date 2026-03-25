import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data from previous step
DATA_PATH = "2_DataCleaned.json"
with open(DATA_PATH, 'r') as f:
    master_data = json.load(f)

# Collect all points into a flat list
# Format: [dist_ft, id, pixel_x, pixel_y]
all_points = []
for filename, entry in master_data.items():
    dist = entry['distance_ft']
    for b in entry['balls']:
        all_points.append([dist, b['index'], b['x'], b['y']])

all_points = np.array(all_points)

w = np.max(all_points[:, 2]) 
h = np.max(all_points[:, 3])

def rotate_points(pts, angle_deg, center_x, center_y):
    """ Rotates 2D points (x, y) around a center point. """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Shift to origin, rotate, shift back
    xy = pts[:, 2:4] - [center_x, center_y]
    rotated_xy = np.dot(xy, R.T) + [center_x, center_y]

    new_pts = pts.copy()
    new_pts[:, 2:4] = rotated_xy
    return new_pts


def draw_calibration_gui(angle):
    # Setup canvas (Assuming typical 1280x720 or 1920x1080)
    # Adjust canvas size if your source images are different
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # We rotate around the approximate center of the point cloud
    center_x = np.mean(all_points[:, 2])
    center_y = np.mean(all_points[:, 3])

    pts = all_points
    #rotate_points(all_points, angle, center_x, center_y)

    # Scale points for display
    display_scale = 0.5

    # Draw points connected by ID (the vertical lines)
    for i in range(11):  # IDs 0-10
        id_points = pts[pts[:, 1] == i]
        id_points = id_points[id_points[:, 0].argsort()]  # Sort by distance

        for j in range(len(id_points) - 1):
            p1 = (int(id_points[j, 2] * display_scale), int(id_points[j, 3] * display_scale))
            p2 = (int(id_points[j + 1, 2] * display_scale), int(id_points[j + 1, 3] * display_scale))
            cv2.line(canvas, p1, p2, (0, 255, 0), 1)
            cv2.circle(canvas, p1, 3, (0, 0, 255), -1)

    cv2.putText(canvas, f"Rotation: {angle:.2f} deg", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Adjust slider until ID lines are vertical/symmetric. Press ENTER to solve.",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas, pts


# Interactive Slider
cv2.namedWindow("Calibration")
cv2.createTrackbar("Roll Correction", "Calibration", 500, 1000, lambda x: None)

final_pts = None
while True:
    val = cv2.getTrackbarPos("Roll Correction", "Calibration")
    angle = (val - 500) / 50.0  # Range +/- 10 degrees

    display, current_pts = draw_calibration_gui(angle)
    cv2.imshow("Calibration", display)

    key = cv2.waitKey(1)
    if key == 13:  # Enter key
        final_pts = current_pts
        final_angle = angle
        break
    if key == 27:  # Esc key
        exit()

cv2.destroyAllWindows()

# --- REGRESSION PHASE ---

def inverse_func(y, a, b, c):
    return a / (y + b) + c

# 2. Extract your raw data
raw_y_vals = final_pts[:, 3]
z_vals = final_pts[:, 0]

norm_x_vals = (final_pts[:, 2] - (w / 2)) / (w / 2)
y_pixels = (final_pts[:, 3] * 1) / (h)

# 3. Fit Distance Z
# p0: [a, b, c]. For raw pixels, 'a' needs to be a large number (like 5000)
# and 'b' is a shift to prevent dividing by zero.
try:
    popt, _ = curve_fit(inverse_func, y_pixels, z_vals, p0=[5000, 100, 0])
    a_fit, b_fit, c_fit = popt
except:
    print("Error: Curve fit failed. Try adjusting the p0 initial guess.")
    a_fit, b_fit, c_fit = 0, 0, 0

# 4. Horizontal Regression: X_world = Z * (pixel_x * m + b)
x_world = final_pts[:, 1]  - 5 # Center is ID 5
print(x_world)
x_ratio = x_world / z_vals
x_coeffs = np.polyfit(norm_x_vals, x_ratio, 1)  # Linear fit
x_m, x_b = x_coeffs

# --- OUTPUT MATH ---

print("\n--- CALIBRATION RESULTS (RAW PIXELS) ---")
print(f"Rotation Correction: {final_angle:.2f} degrees")
print("\nPython function to use in your robot code:")
print("-" * 40)

# Calculate the center of the points for rotation
mean_x = np.mean(all_points[:, 2])
mean_y = np.mean(all_points[:, 3])

code = f"""
def get_3d_coords(pixel_x, pixel_y):
    # 1. Rotate point to correct camera tilt
    import math
    angle_rad = math.radians({final_angle:.4f})
    cx, cy = {mean_x:.1f}, {mean_y:.1f}

    # Shift to rotation center
    nx = pixel_x - cx
    ny = pixel_y - cy

    # Rotate
    rx = nx * math.cos(angle_rad) - ny * math.sin(angle_rad) + cx
    ry = nx * math.sin(angle_rad) + ny * math.cos(angle_rad) + cy

    # 2. Calculate Distance Z (feet) using Inverse Model
    # Formula: Z = a / (ry + b) + c
    z = {a_fit:.8f} / (ry + {b_fit:.8f}) + {c_fit:.8f}

    # 3. Calculate Horizontal X (feet)
    # Formula: X = Z * (rotated_x * m + b)
    x = z * (rx * {x_m:.8f} + ({x_b:.8f}))

    return x, z
"""
print(code)
print("-" * 40)

# --- VISUAL VERIFICATION ---

plt.figure(figsize=(12, 5))

# Plot 1: Distance Fit
plt.subplot(1, 2, 1)
plt.scatter(y_pixels, z_vals, label="Data (Raw Pixels)", alpha=0.6)
y_range = np.linspace(min(y_pixels), max(y_pixels), 100)
plt.plot(y_range, inverse_func(y_range, a_fit, b_fit, c_fit), color='red', linewidth=2, label="Inverse Fit")
plt.title("Z Distance vs Raw Pixel Y")
plt.xlabel("Pixel Y")
plt.ylabel("Feet")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Horizontal Fit
plt.subplot(1, 2, 2)
plt.scatter(norm_x_vals, x_ratio, label="Data (Raw Pixels)", color='green', alpha=0.6)
x_range = np.linspace(min(norm_x_vals), max(norm_x_vals), 100)
plt.plot(x_range, x_range * x_m + x_b, color='red', linewidth=2, label="Linear Fit")
plt.title("X/Z Ratio vs Raw Pixel X")
plt.xlabel("Pixel X")
plt.ylabel("X/Z Ratio")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()