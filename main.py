import cv2
import os
import random
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import torch
import time

# -------------------------------
# Model & Detection Configuration
# -------------------------------
EMBEDDING_MODEL = 'buffalo_s'
EMBEDDING_THRESHOLD = 0.8
YAW_THRESHOLD = 45
PITCH_THRESHOLD = 70
ROLL_THRESHOLD = 70
CAMERA_DEVICE = 0

# frame counts before removing a person
MISSING_FRAMES_THRESHOLD = 15

WIN_TITLE = "Face Recognition"
OUTPUT_FOLDER = "images"

UNIQUE_FACES = {}   # {person_id: {"embeddings": [...], "last_seen": frame_count}}
VISIBLE_FACES = {}  # {person_id: last_seen_frame_count}
ID_COLORS = {}      # {person_id: (B, G, R)}

# ---------------------------
# Create output folder if not exists
# ---------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------
# Initialize the face detector
# --------------------------------
detector = FaceAnalysis(name=EMBEDDING_MODEL)
detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(480, 480))

# --------------------------------
# Helper Functions
# --------------------------------
def get_similarity(e1, e2):
    """
    Returns cosine similarity between two embeddings (1 = identical, 0 = completely different).
    """
    return 1 - cosine(e1, e2)

def is_face_known(embedding):
    """
    Compare the current face embedding to all stored embeddings in UNIQUE_FACES.
    Return (True, person_id) if known, else (False, None).
    """
    best_match_id = None    # closest matching pid
    best_match_score = -1   # stores highest similarity

    for pid, data in UNIQUE_FACES.items():
        stored_embeddings = data["embeddings"]
        similarities = [get_similarity(embedding, se) for se in stored_embeddings]
        max_sim = max(similarities) if similarities else -1

        if max_sim > best_match_score:
            best_match_score = max_sim
            best_match_id = pid

    if best_match_score >= (1 - EMBEDDING_THRESHOLD):
        return True, best_match_id
    else:
        return False, None

def add_embedding_to_person(person_id, embedding):
    """
    Append the new embedding to the existing person's list.
    Limit to last 10 to avoid unbounded growth.
    """
    UNIQUE_FACES[person_id]["embeddings"].append(embedding)
    if len(UNIQUE_FACES[person_id]["embeddings"]) > 10:
        UNIQUE_FACES[person_id]["embeddings"].pop(0)

def save_unknown_face(frame, bbox, person_id, gender, age):
    """
    Save newly encountered face region to disk only if it's valid.
    """
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Ensure bounding box is within valid frame dimensions
    h, w, _ = frame.shape  # Get frame dimensions

    # Make sure the bounding box is within frame limits
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # If bounding box is too small, skip saving
    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        print(f"[WARN]: Skipping face save - Invalid bounding box {bbox}")
        return

    face_img = frame[y_min:y_max, x_min:x_max]

    # Ensure the cropped image is not empty
    if face_img is None or face_img.size == 0:
        print(f"[WARN]: Skipping face save - Empty image for {person_id}")
        return

    timestamp = int(time.time())
    filename = f"{OUTPUT_FOLDER}/{person_id}_{gender}_{age}_{timestamp}.jpg"

    success = cv2.imwrite(filename, face_img)
    if not success:
        print(f"[WARN]: Failed to save image {filename}")


def get_color_for_id(person_id):
    """
    Assign a (B, G, R) color to each person_id.
    If the ID is new, generate a random color. Return consistent color for that ID.
    """
    if person_id not in ID_COLORS:
        # Generate a random BGR color
        ID_COLORS[person_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return ID_COLORS[person_id]

# --------------------------------
# Main Function
# --------------------------------
def main():
    print('[INFO]: Starting webcam...')
    cv2.setUseOptimized(True)
    cap = cv2.VideoCapture(CAMERA_DEVICE)

    if not cap.isOpened():
        print('[ERR]: Unable to access webcam feed')
        return

    for i in range(2, 0, -1):
        print(f"[INFO]: Starting face detection in {i} seconds...")
        time.sleep(1)

    fps = 0
    frame_count = 0
    start_time = time.time()
    person_id = 1

    DETECTION_INTERVAL = 1  # run detection on every frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[ERR]: Unable to process frame from the webcam')
            break

        if frame_count % DETECTION_INTERVAL == 0:
            faces = detector.get(frame)
        else:
            faces = []

        for face in faces:
            x_min, y_min, x_max, y_max = map(int, face.bbox)
            face_embedding = face.embedding

            # Pose format is [yaw, pitch, roll]
            yaw, pitch, roll = face.pose

            # If the face is within acceptable angles, proceed
            if (abs(yaw) <= YAW_THRESHOLD and 
                abs(pitch) <= PITCH_THRESHOLD and 
                abs(roll) <= ROLL_THRESHOLD):

                # Retrieve gender/age from the face object
                gender = "Male" if face.gender == 1 else "Female"
                age = int(face.age)

                known, pid = is_face_known(face_embedding)
                if not known:
                    # New face
                    pid = person_id
                    UNIQUE_FACES[pid] = {"embeddings": [face_embedding]}
                    save_unknown_face(frame, face.bbox, pid, gender, age)
                    person_id += 1
                else:
                    # Already known face
                    add_embedding_to_person(pid, face_embedding)

                # Mark this person as seen in the current frame
                VISIBLE_FACES[pid] = frame_count

                # Retrieve color for bounding box + text
                color = get_color_for_id(pid)

                label = f"ID: {pid} | {gender}, {age}"

                # Draw bounding box
                thickness = 2
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

                # Draw text with outline for better visibility
                text_scale = 0.6
                text_thickness = 1
                x_text, y_text = x_min, y_min - 10

                # Outline (black)
                cv2.putText(
                    frame,
                    label,
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_COMPLEX,
                    text_scale,
                    (0, 0, 0),  # black
                    3,  # thicker text for shadow
                    cv2.LINE_AA
                )
                # Main colored text
                cv2.putText(
                    frame,
                    label,
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_COMPLEX,
                    text_scale,
                    color,
                    text_thickness,
                    cv2.LINE_AA
                )

        # Check if any person has been missing beyond the threshold
        inactive_people = [
            pid for pid, last_seen in VISIBLE_FACES.items()
            if frame_count - last_seen > MISSING_FRAMES_THRESHOLD
        ]
        for pid in inactive_people:
            del VISIBLE_FACES[pid]

        active_person_count = len(VISIBLE_FACES)
        tracked_faces = len(UNIQUE_FACES)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Show counters on screen
        # Possibly reorder them or style them however you like
        # Define text parameters
        text_scale = 0.8
        text_thickness = 2
        shadow_offset = 2  # Offset for shadow effect

        # Define text positions
        text_positions = {
            "Current Detected": (10, 100),
            "Total Detected": (10, 75),
            "FPS": (10, 50),
        }

        # Define dynamic colors for better visibility
        color_active = (0, 255, 0)    # Green for active people
        color_tracked = (255, 255, 0) # Yellow for tracked faces
        color_fps = (0, 165, 255)     # Orange for FPS

        # Create a shadow effect by first drawing a black offset version
        for label, (x, y) in text_positions.items():
            value = {
                "Current Detected": active_person_count,
                "Total Detected": tracked_faces,
                "FPS": f"{fps:.2f}"
            }[label]

            # Choose corresponding color
            color = color_active if label == "Current Detected" else (color_tracked if label == "Total Detected" else color_fps)

            # Shadow (black, slightly offset)
            cv2.putText(
                frame,
                f"{label}: {value}",
                (x + shadow_offset, y + shadow_offset),
                cv2.FONT_HERSHEY_COMPLEX,
                text_scale,
                (0, 0, 0),  # Black shadow
                text_thickness + 1,  # Slightly thicker for visibility
                cv2.LINE_AA
            )

            # Main text with selected color
            cv2.putText(
                frame,
                f"{label}: {value}",
                (x, y),
                cv2.FONT_HERSHEY_COMPLEX,
                text_scale,
                color,
                text_thickness,
                cv2.LINE_AA
            )


        cv2.imshow(WIN_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
