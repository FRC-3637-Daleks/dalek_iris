print("""
In ./0_sourceImg, put all images, going from 24-3ft, label images with there distance (ie 24.jpeg or 03.jpeg)
For right now this only supports fuel in a straight line with camera not doing any significant distortion
""")

import cv2
import numpy as np
import json
import os
import glob
import re


def process_fuel_images(input_folder, output_file, preview_folder):
    if not os.path.exists(preview_folder):
        os.makedirs(preview_folder)

    # Get all jpeg files and sort them (e.g., 24.jpeg down to 03.jpeg)
    image_paths = glob.glob(os.path.join(input_folder, "*.jpeg"))
    image_paths.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1)), reverse=True)

    master_data = {}

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        # Extract distance from filename (e.g. "24.jpeg" -> 24)
        dist_match = re.search(r'(\d+)', filename)
        distance = int(dist_match.group(1)) if dist_match else 0

        img = cv2.imread(img_path)
        if img is None: continue
        height, width, _ = img.shape

        # 1. Color Segmentation (Yellow)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Refined range for yellow game pieces under typical indoor lighting
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Cleanup mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 2. Find Blobs/Centers
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: continue  # Filter noise

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

        if not centers:
            print(f"Warning: No balls detected in {filename}")
            continue

        # Sort detections by X coordinate
        centers.sort(key=lambda c: c[0])

        # 3. Calculate common Y (linear line)
        all_y = [c[1] for c in centers]
        common_y = int(np.mean(all_y))

        # 4. Map to Grid Indices (0 to 10)
        # We calculate the average pixel distance between adjacent balls to find the 'foot' unit
        if len(centers) > 1:
            gaps = [centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)]
            # Use median to ignore gaps where balls are missing
            pixel_unit = np.median(gaps)

            # Anchor indices based on the assumption that the balls are centered in the image
            # or by relative distance if at least one end is known.
            # Heuristic: Find relative indices first
            relative_indices = [0]
            for i in range(len(centers) - 1):
                gap = centers[i + 1][0] - centers[i][0]
                num_steps = round(gap / pixel_unit)
                relative_indices.append(relative_indices[-1] + num_steps)

            # Offset relative indices so they sit within 0-10 based on horizontal center
            total_span = relative_indices[-1]
            offset = (10 - total_span) // 2
            final_indices = [idx + offset for idx in relative_indices]
        else:
            final_indices = [5]  # Default to center if only one ball

        # 5. Prepare Data and Visualization
        ball_entries = []
        preview_img = img.copy()

        # Draw the common Y line
        cv2.line(preview_img, (0, common_y), (width, common_y), (0, 255, 0), 2)

        for i, (cx, cy) in enumerate(centers):
            idx = int(final_indices[i])
            ball_entries.append({
                "index": idx,
                "x": cx,
                "y": cy
            })

            # Draw detections for verification
            cv2.circle(preview_img, (cx, cy), 12, (0, 0, 255), -1)
            cv2.putText(preview_img, f"ID:{idx}", (cx - 15, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        master_data[filename] = {
            "distance_ft": distance,
            "common_y": common_y,
            "balls": ball_entries
        }

        cv2.imwrite(os.path.join(preview_folder, f"out_{filename}"), preview_img)

    # Export to JSON
    with open(output_file, 'w') as f:
        json.dump(master_data, f, indent=4)

    print(f"Success! Processed {len(master_data)} images.")
    print(f"Data: {output_file} | Previews: {preview_folder}")


if __name__ == "__main__":
    process_fuel_images("./0_sourceImg", "1_DataOut.json", "./1_previews")
