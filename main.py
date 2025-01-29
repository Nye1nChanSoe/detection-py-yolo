import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import torch
import time

# Model and detection settings
EMBEDDING_MODEL = 'buffalo_s'
EMBEDDING_THRESHOLD = 0.8
YAW_THRESHOLD = 15
PITCH_THRESHOLD = 20
ROLL_THRESHOLD = 20
CAMERA_DEVICE = 0
MISSING_FRAMES_THRESHOLD = 30  # Number of frames before removing a person

WIN_TITLE = "Face Recognition"
OUTPUT_FOLDER = "images"

UNIQUE_FACES = {}  # {person_id: {"embeddings": [embedding1, embedding2, ...], "last_seen": frame_count}}
VISIBLE_FACES = {} # {person_id: last_seen_frame_count}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

detector = FaceAnalysis(name=EMBEDDING_MODEL)
detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(480, 480))

def get_similarity(e1, e2):
    return 1 - cosine(e1, e2)

def is_face_known(embedding):
    best_match_id = None
    best_match_score = -1

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
    UNIQUE_FACES[person_id]["embeddings"].append(embedding)
    if len(UNIQUE_FACES[person_id]["embeddings"]) > 10:
        UNIQUE_FACES[person_id]["embeddings"].pop(0)

def save_unknown_face(frame, bbox, person_id):
    x_min, y_min, x_max, y_max = map(int, bbox)
    face_img = frame[y_min:y_max, x_min:x_max]
    timestamp = int(time.time())
    filename = f"{OUTPUT_FOLDER}/{person_id}_unknown_{timestamp}.jpg"
    cv2.imwrite(filename, face_img)

def main():
    cv2.setUseOptimized(True)
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print('[ERR]: Unable to access webcam feed')
        return

    print('[INFO]: Starting webcam...')
    time.sleep(2)

    fps = 0
    frame_count = 0
    start_time = time.time()
    person_id = 1

    DETECTION_INTERVAL = 1  

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[ERR]: Unable to process frame from the webcam')
            break

        if frame_count % DETECTION_INTERVAL == 0:
            faces = detector.get(frame)
        else:
            faces = []

        detected_ids = set()

        for face in faces:
            x_min, y_min, x_max, y_max = map(int, face.bbox)
            face_embedding = face.embedding

            yaw, pitch, roll = face.pose

            if (abs(yaw) <= YAW_THRESHOLD and 
                abs(pitch) <= PITCH_THRESHOLD and 
                abs(roll) <= ROLL_THRESHOLD):
                
                known, pid = is_face_known(face_embedding)
                if not known:
                    pid = person_id
                    UNIQUE_FACES[pid] = {"embeddings": [face_embedding]}
                    save_unknown_face(frame, face.bbox, pid)
                    person_id += 1
                else:
                    add_embedding_to_person(pid, face_embedding)

                VISIBLE_FACES[pid] = frame_count  # Update last seen frame

                gender = "Male" if face.gender == 1 else "Female"
                age = int(face.age)
                label = f"ID: {pid} | {gender}, {age}"

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, 
                            (0, 255, 255), 1, cv2.LINE_AA)

                detected_ids.add(pid)

        # Remove people who have not been seen in the last few frames
        inactive_people = [pid for pid, last_seen in VISIBLE_FACES.items() 
                           if frame_count - last_seen > MISSING_FRAMES_THRESHOLD]
        
        for pid in inactive_people:
            del VISIBLE_FACES[pid]  # Remove them from active tracking

        # Number of currently visible people
        active_person_count = len(VISIBLE_FACES)

        # Number of faces tracked
        tracked_faces = len(UNIQUE_FACES)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(frame, f"Current: {active_person_count}", (10, 100), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Detected Faces: {tracked_faces}", (10, 75), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(WIN_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
