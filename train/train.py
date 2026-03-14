import cv2
import os
import time
import tkinter as tk
from PIL import Image, ImageTk

# =========================================================
# CONFIGURATION
# =========================================================
CAMERA_INDEX = 2  # Adjust this based on your Linux video nodes
SAVE_DIR = "toTrain"
DISPLAY_SIZE = (640, 480)  # Size of the video panels in the GUI


class DataCollectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Create save directory
        self.save_dir = SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)

        # Open Camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {CAMERA_INDEX}.")

        self.current_frame = None
        self.image_counter = 1

        # ==================== GUI LAYOUT ====================
        self.main_frame = tk.Frame(window)
        self.main_frame.pack(padx=10, pady=10)

        # Left Panel: Live Feed
        self.live_label = tk.Label(self.main_frame, text="Live Camera Feed", font=("Arial", 12, "bold"))
        self.live_label.grid(row=0, column=0, padx=10, pady=5)

        self.live_panel = tk.Label(self.main_frame, bg="black", width=64, height=30)
        self.live_panel.grid(row=1, column=0, padx=10)

        # Right Panel: Last Captured Photo
        self.prev_label = tk.Label(self.main_frame, text="Last Captured Photo", font=("Arial", 12, "bold"))
        self.prev_label.grid(row=0, column=1, padx=10, pady=5)

        self.prev_panel = tk.Label(self.main_frame, bg="gray", width=64, height=30, text="No image captured yet")
        self.prev_panel.grid(row=1, column=1, padx=10)

        # Status Bar
        self.status_label = tk.Label(window, text="Press [SPACEBAR] to capture  |  Close window to quit",
                                     font=("Arial", 14), fg="blue")
        self.status_label.pack(pady=10)

        # ==================== KEYBINDS ====================
        # Bind the spacebar to the capture function
        self.window.bind('<space>', self.capture_image)

        # Start the video loop
        self.delay = 15  # Update every 15 milliseconds (~60 fps)
        self.update_video()

    def update_video(self):
        """Pulls a frame from the camera and updates the Live Feed panel."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame  # Save the raw BGR frame for saving later

            # Convert frame for Tkinter (OpenCV uses BGR, Pillow uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, DISPLAY_SIZE)

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the live panel
            self.live_panel.imgtk = imgtk  # Keep a reference to prevent garbage collection
            self.live_panel.config(image=imgtk, width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1])

        # Schedule the next update
        self.window.after(self.delay, self.update_video)

    def capture_image(self, event=None):
        """Saves the current frame and updates the Last Captured panel."""
        if self.current_frame is not None:
            # 1. Save the FULL RESOLUTION frame to disk
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f"train_{timestamp}_{self.image_counter}.jpg")
            cv2.imwrite(filename, self.current_frame)

            self.status_label.config(text=f"Saved [{self.image_counter}]: {filename}", fg="green")
            self.image_counter += 1

            # 2. Update the "Previous Photo" panel on the right
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, DISPLAY_SIZE)

            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)

            self.prev_panel.imgtk = imgtk
            self.prev_panel.config(image=imgtk, text="", width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1])

    def on_close(self):
        """Safely release the camera and close the app."""
        print("Closing application...")
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    # Create the Tkinter root window
    root = tk.Tk()

    # Initialize the app
    app = DataCollectorApp(root, "Wayland Image Data Collector")

    # Handle the window close button (X) properly
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Start the GUI event loop
    root.mainloop()