import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
import torch
import time


# https://github.com/deepinsight/insightface/tree/master/model_zoo
EMBEDDING_MODEL = 'buffalo_s'
EMBEDDING_THRESHOLD = 0.3 # lower = stricter match

CAMERA_DEVICE = 0

WIN_TITLE = "Face Recognition"
OUTPUT_FOLDER = "images"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# detector = FaceAnalysis(name=EMBEDDING_MODEL)
# detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(480, 480))


def main():
    cv2.setUseOptimized(True)
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print('[ERR]: Unable to access webcam feed')
        return
    print('[INFO]: Starting webcam...')

    fps = 0
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[ERR]: Unable to process frame from the webcam')
            break
        # detect faces and extract embeddings

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (128,0,128), 1, cv2.LINE_AA)
        cv2.imshow(WIN_TITLE, frame)

        # exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()