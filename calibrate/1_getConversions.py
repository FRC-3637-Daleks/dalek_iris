import cv2
import numpy as np
import json
import os
import glob
import re


def process_fuel_images(input_folder, output_file, preview_folder):
    if not os.path.exists(preview_folder):
        os.makedirs(preview_folder)

    image_paths = glob.glob(os.path.join(input_folder, "*.jpeg"))
    image_paths.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1)), reverse=True)

    master_data = {}

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        dist_match = re.search(r'(\d+)', filename)
        distance = int(dist_match.group(1)) if dist_match else 0

        img = cv2.imread(img_path)
        if img is None: continue
        height, width, _ = img.shape

        # 1. Improved Color Segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 90, 80])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Close gaps inside balls (logos, marks) and open to remove small noise
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 2. Extract Detections
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100: continue
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                candidates.append({'x': cx, 'y': cy, 'area': area})

        if not candidates: continue

        # 3. Filter out reflections based on Y-density
        # Reflections are usually below the main row. We sort by Y (ascending = top to bottom)
        candidates.sort(key=lambda c: c['y'])

        # Heuristic: Find the most consistent Y-level (the real row)
        # We take the Y of the top-most few balls as the 'target'
        target_y = np.median([c['y'] for c in candidates[:min(len(candidates), 5)]])
        # Reject anything significantly lower than the top row (reflecting below)
        # We allow a small buffer for lens distortion
        valid_candidates = [c for c in candidates if abs(c['y'] - target_y) < (height * 0.1)]

        if not valid_candidates: continue
        valid_candidates.sort(key=lambda b: b['x'])

        # 4. Grid Mapping with Conflict Resolution
        # Calculate pixel-to-index unit
        if len(valid_candidates) > 1:
            gaps = [valid_candidates[i + 1]['x'] - valid_candidates[i]['x'] for i in range(len(valid_candidates) - 1)]
            pixel_unit = np.median(gaps)
            img_center_x = width / 2

            # Map every candidate to an ID
            # If multiple balls hit the same ID, we keep the one with MINIMUM Y (top-most)
            id_map = {}
            for cand in valid_candidates:
                # Find relative ID
                rel_gap = (cand['x'] - valid_candidates[0]['x']) / pixel_unit
                # Estimate starting index based on image center if possible, 
                # or just use first ball as anchor 0
                estimated_id = round(rel_gap)

                if estimated_id not in id_map or cand['y'] < id_map[estimated_id]['y']:
                    id_map[estimated_id] = cand

            # Normalize IDs to 0-10 range based on image centering
            all_ids = sorted(id_map.keys())
            span = all_ids[-1] - all_ids[0]
            offset = (10 - span) // 2 - all_ids[0]

            final_balls = []
            for raw_id, ball in id_map.items():
                final_id = max(0, min(10, raw_id + offset))
                final_balls.append({"index": final_id, "x": ball['x'], "y": ball['y']})
        else:
            final_balls = [{"index": 5, "x": valid_candidates[0]['x'], "y": valid_candidates[0]['y']}]

        # Final Common Y is the average of the "survivors"
        common_y = int(np.mean([b['y'] for b in final_balls]))

        # 5. Export and Preview
        master_data[filename] = {"distance_ft": distance, "common_y": common_y, "balls": final_balls}

        preview_img = img.copy()
        cv2.line(preview_img, (0, common_y), (width, common_y), (0, 255, 0), 2)
        for b in final_balls:
            cv2.circle(preview_img, (b['x'], b['y']), 8, (0, 0, 255), -1)
            cv2.putText(preview_img, f"ID:{b['index']}", (b['x'] - 15, b['y'] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(preview_folder, f"out_{filename}"), preview_img)

    with open(output_file, 'w') as f:
        json.dump(master_data, f, indent=4)
    print("Done. Reflections filtered.")


if __name__ == "__main__":
    process_fuel_images("./0_sourceImg", "1_DataOut.json", "./1_previews")