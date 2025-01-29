from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine


# light-weight yolo nano pretrained model
pretrain_model = "yolo11n.pt"
model = YOLO(pretrain_model)

# https://github.com/deepinsight/insightface/tree/master/model_zoo

embedding_model = "buffalo_s"
app = FaceAnalysis(name=embedding_model)
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0: use gpu if available

camera = 0
confidence = 0.5
person_id = 0   # class id for person object
win_title = "Detection"

# store unique faces
output_folder = "detected_faces"
os.makedirs(output_folder, exist_ok=True)


# face embeddings of previously detected
# in memory
face_embeddings = []
face_labels = {}  # Dictionary {embedding_index: "Face_X"}
embedding_threshold = 0.5 # lower stricter


def extract_face(image, bbox, margin=20):
    """
    Extracts the face region from the bounding box.
    bbox: (x1, y1, x2, y2)
    margin: Extra pixels around the face to improve detection.
    """
    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox

    # Add margin and ensure the crop is within bounds
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return image[y1:y2, x1:x2]


def find_matching_face(new_embedding):
    """
    Checks if a new face is already detected and returns its label.
    """
    for idx, stored_embedding in enumerate(face_embeddings):
        similarity = 1 - cosine(new_embedding, stored_embedding)  # Cosine similarity
        if similarity > embedding_threshold:
            return face_labels[idx]  # Return existing label
    
    return None  # No match found


def save_new_face(face_crop, face_embedding):
    """
    Saves a new unique face and assigns it a label.
    """
    face_count = len(face_embeddings)  # Number of stored faces
    face_label = f"Face_{face_count + 1}"  # Assign new label
    
    # Save image file
    face_filename = os.path.join(output_folder, f"{face_label}.jpg")
    cv2.imwrite(face_filename, face_crop)
    
    # Store new face embedding
    face_embeddings.append(face_embedding)
    face_labels[face_count] = face_label

    print(f"New unique face saved: {face_filename} as {face_label}")
    return face_label


def main():
    print(f"PyTorch version: {torch.__version__}")

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Starting webcam feed. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # main detection from yolo
        results = model.predict(frame, conf=confidence)

        person_count = 0
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == person_id:
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # extract only the face area
                    face_crop = extract_face(frame, (x1, y1, x2, y2))

                    if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                        continue

                    # face embeddings from insightface
                    faces = app.get(face_crop)

                    if faces:
                        face_embedding = faces[0].embedding
                        print(f"Face detected. Embedding shape: {face_embedding.shape}")

                        # Check if face already exists
                        face_label = find_matching_face(face_embedding)

                        if not face_label:  # If unique, save and assign a new label
                            face_label = save_new_face(face_crop, face_embedding)

                        cv2.putText(frame, face_label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # window displays
        cv2.putText(frame, f"Persons: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(win_title, frame)

        # ascii 27 == escape
        # press one of these keys to quit
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == "__main__":
    main()
