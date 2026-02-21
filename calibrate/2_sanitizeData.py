import json
import numpy as np
import cv2
import os
import re


def sanitize_data(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    # 1. Sort files by distance: DESCENDING (e.g., 24.jpeg -> 03.jpeg)
    # This starts processing from the "back" as requested.
    filenames = sorted(data.keys(),
                       key=lambda x: int(re.search(r'(\d+)', x).group(1)),
                       reverse=True)

    refined_data = {}
    prev_lanes = {}  # Stores {ID: last_known_x}

    # 2. Establish Anchor from the furthest frame
    # We assume the furthest frame is the most complete (closest to 11 balls)
    first_file = filenames[0]
    balls = sorted(data[first_file]['balls'], key=lambda b: b['x'])

    # Heuristic: If we have 11 balls, they are 0-10.
    # If we have fewer, we align the middle of the group to ID 5.
    center_offset = 5 - (len(balls) // 2)

    for i, b in enumerate(balls):
        new_id = int(i + center_offset)  # Cast to int to prevent JSON error
        b['index'] = new_id
        prev_lanes[new_id] = b['x']

    refined_data[first_file] = data[first_file]

    # 3. Iterative Lane Tracking (Moving from back to front)
    for i in range(1, len(filenames)):
        fname = filenames[i]
        curr_balls = sorted(data[fname]['balls'], key=lambda b: b['x'])

        # Find the best global ID shift to match previous lanes
        best_shift = 0
        min_error = float('inf')

        # Try shifting the whole row of detections across the 0-10 grid
        for shift in range(-5, 6):
            total_error = 0
            matches = 0
            for j, b in enumerate(curr_balls):
                target_id = j + shift + center_offset
                if target_id in prev_lanes:
                    total_error += abs(b['x'] - prev_lanes[target_id])
                    matches += 1

            if matches > 0:
                avg_error = total_error / matches
                if avg_error < min_error:
                    min_error = avg_error
                    best_shift = shift

        # Apply indices and update lane memory
        new_ball_list = []
        for j, b in enumerate(curr_balls):
            final_id = int(max(0, min(10, j + best_shift + center_offset)))
            b['index'] = final_id
            new_ball_list.append(b)
            prev_lanes[final_id] = b['x']

        data[fname]['balls'] = new_ball_list
        refined_data[fname] = data[fname]

    # 4. Interactive Review GUI
    print("\n--- DATA SANITY CHECK ---")
    print("Controls: 'A' shift Left, 'D' shift Right, 'SPACE' confirm row, 'ESC' quit.")
    cv2.namedWindow("Sanity Check")

    current_idx = 0
    while current_idx < len(filenames):
        fname = filenames[current_idx]
        # Black canvas for visualization
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Draw background "tracks" (already processed/confirmed data)
        for f_past in filenames[:current_idx]:
            for b in refined_data[f_past]['balls']:
                cv2.circle(canvas, (int(b['x']), int(b['y'])), 2, (60, 60, 60), -1)

        # Draw current frame
        entry = refined_data[fname]
        for b in entry['balls']:
            px, py = int(b['x']), int(b['y'])
            cv2.circle(canvas, (px, py), 8, (0, 0, 255), -1)
            cv2.putText(canvas, f"ID:{b['index']}", (px - 15, py - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        dist_label = entry.get('distance_ft', '??')
        cv2.putText(canvas, f"File: {fname} ({dist_label}ft) - SPACE to confirm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Sanity Check", canvas)
        key = cv2.waitKey(0)

        if key == ord('d'):  # Nudge IDs right
            for b in refined_data[fname]['balls']:
                b['index'] = int(min(10, b['index'] + 1))
        elif key == ord('a'):  # Nudge IDs left
            for b in refined_data[fname]['balls']:
                b['index'] = int(max(0, b['index'] - 1))
        elif key == 32:  # Space - Next frame
            current_idx += 1
        elif key == 27:  # Esc
            break

    # 5. Save to Cleaned JSON with casting fix
    # We use a custom lambda to ensure all numbers are standard Python ints
    def convert_to_builtin_type(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(refined_data, f, indent=4, default=convert_to_builtin_type)

    cv2.destroyAllWindows()
    print(f"\nSuccess! Cleaned data saved to {output_file}")


if __name__ == "__main__":
    sanitize_data("1_DataOut.json", "2_DataCleaned.json")