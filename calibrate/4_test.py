print("Paste in the function")

# =========================================================
# PASTE YOUR FUNCTION FROM SCRIPT 3 HERE
# =========================================================
def get_3d_coords(pixel_x, pixel_y):
    # 1. Rotate point to correct camera tilt
    import math
    angle_rad = math.radians(0.0000)
    cx, cy = 998.6, 166.4

    # Shift to rotation center
    nx = pixel_x - cx
    ny = pixel_y - cy

    # Rotate
    rx = nx * math.cos(angle_rad) - ny * math.sin(angle_rad) + cx
    ry = nx * math.sin(angle_rad) + ny * math.cos(angle_rad) + cy

    # 2. Calculate Distance Z (feet)
    z = 0.00010625 * (ry**2) + -0.10210374 * ry + 26.51195823

    # 3. Calculate Horizontal X (feet)
    # Formula: X = Z * (pixel_x * m + b)
    x = z * (rx * 0.00068038 + (-0.68897339))

    return x, z
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import os

def run_test():
    images = glob.glob("./0_sourceImg/*.jpeg")
    if not images: return

    img_path = random.choice(images)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Yellow mask (same as before) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 90, 90]), np.array([35, 255, 255]))

    # --- Clean up noise ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # --- Find connected blobs ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    MIN_BLOB_AREA = 200  # pixels² — filters out noise

    world_x, world_z = [], []
    ball_pixel_centers = []  # for drawing on camera view

    for i in range(1, num_labels):  # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_AREA:
            continue

        cx_px, cy_px = centroids[i]

        # Optional: use the TOP of the blob instead of centroid for y,
        # since the ball bottom is on the ground and top is more "center height"
        # top_y = stats[i, cv2.CC_STAT_TOP]
        # cy_px = top_y + stats[i, cv2.CC_STAT_HEIGHT] * 0.5  # mid-height

        wx, wz = get_3d_coords(cx_px, cy_px)

        # Sanity filter — ignore points behind robot or impossibly far
        if 0.5 < wz < 35 and -10 < wx < 10:
            world_x.append(wx)
            world_z.append(wz)
            ball_pixel_centers.append((int(cx_px), int(cy_px), area))

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(img_rgb)
    # Draw detected blob centers on camera view
    for (bx, by, area) in ball_pixel_centers:
        radius = int(np.sqrt(area / np.pi))
        circle = plt.Circle((bx, by), radius, color='cyan', fill=False, linewidth=2)
        ax1.add_patch(circle)
        ax1.plot(bx, by, 'r+', markersize=10)
    ax1.set_title(f"Camera View: {os.path.basename(img_path)}\n{len(world_x)} blobs detected")
    ax1.axis('off')

    ax2.scatter(world_x, world_z, c='gold', s=100, edgecolors='black',
                linewidths=0.5, zorder=5, label=f"Fuel ({len(world_x)} blobs)")
    ax2.scatter([0], [0], c='red', marker='^', s=200, label="Robot")

    ax2.set_title("Robot-Centric Map (Top-Down)")
    ax2.set_xlabel("Lateral Distance (Feet) [Left <-> Right]")
    ax2.set_ylabel("Forward Distance (Feet)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim([-8, 8])
    ax2.set_ylim([0, 30])
    ax2.legend()

    plt.tight_layout()
    plt.savefig("4_testOutput.png")
    print(f"Saved. Detected {len(world_x)} fuel blobs.")

if __name__ == "__main__":
    run_test()