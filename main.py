import cv2
import face_recognition
import numpy as np
import os
import time

# --- CONFIGURATION ---
IMAGES_FOLDER = "images"
THRESHOLD = 0.55  # Lower = Stricter, Higher = Looser

print("Loading training data from folder...")

known_face_encodings = []
known_face_names = []

if not os.path.exists(IMAGES_FOLDER):
    print(f"Error: Folder '{IMAGES_FOLDER}' not found.")
    exit()

# --- LOAD IMAGES ---
for filename in os.listdir(IMAGES_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGES_FOLDER, filename)

        try:
            cv_img = cv2.imread(path)

            if cv_img is None:
                print(f"Warning: Could not read {filename}. Is it corrupted?")
                continue

            # Convert to RGB and force 8-bit to fix "Unsupported Image" error
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            rgb_img = rgb_img.astype('uint8')

            encodings = face_recognition.face_encodings(rgb_img)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])

                # Extract clean name: "hafiz1.jpg" -> "hafiz"
                raw_name = os.path.splitext(filename)[0]
                name = ''.join([i for i in raw_name if not i.isdigit()]).strip("_")

                known_face_names.append(name)
                print(f"Loaded: {filename} -> Labelled as: {name}")
            else:
                print(f"Warning: No face found in {filename}. Skipping.")

        except Exception as e:
            print(f"ERROR processing {filename}: {e}")

print(f"\nSystem Ready. {len(known_face_names)} faces learned. Press 'q' to quit.")

# --- INITIALIZE WEBCAM ---
# Try 0, 1, or 2 if your camera doesn't open
video_capture = cv2.VideoCapture(1)

# Variables for Optimization
process_this_frame = True
prev_frame_time = 0
face_locations = []
face_names = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Webcam disconnected.")
        break

    # 1. Resize to 1/4 size (Huge speed boost)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 2. Convert to RGB and FORCE uint8 type (Crucial for stability)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

    # --- OPTIMIZATION: Only run AI on every OTHER frame ---
    if process_this_frame:
        # Find faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Distance calculation
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Threshold check
            if face_distances[best_match_index] < THRESHOLD:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            face_names.append(name)

    # Flip the switch so the next frame is SKIPPED
    process_this_frame = not process_this_frame

    # --- DISPLAY RESULTS ---
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up (since we resized by 1/4)
        top *= 4;
        right *= 4;
        bottom *= 4;
        left *= 4

        # Logic: GREEN for known names, RED for "Unknown"
        if name == "Unknown":
            color = (0, 0, 255)  # Red (BGR format)
        else:
            color = (0, 255, 0)  # Green (BGR format)

        # Draw box and Label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # --- FPS COUNTER ---
    new_frame_time = time.time()
    try:
        fps = 1 / (new_frame_time - prev_frame_time)
    except ZeroDivisionError:
        fps = 0
    prev_frame_time = new_frame_time

    # Display FPS in Yellow (Top Left)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Face Recognition System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

