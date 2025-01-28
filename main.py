from ultralytics import YOLO
import torch
import cv2

# light-weight yolo nano pretrained model
# trained on COCO dataset
pretrain_model = "yolo11n.pt"
camera = 0
confidence = 0.5
person_id = 0
win_title = "Detection"

def main():
    print(f"PyTorch version: {torch.__version__}")
    print("Loading YOLO model...")
    model = YOLO('yolo11n.pt')

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Starting webcam feed. Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the webcam.")
            break

        # Perform person detection
        results = model.predict(frame, conf=confidence)

        # Count persons and annotate frame
        person_count = 0
        for result in results:
            for box in result.boxes:
                # person class id = 0
                if int(box.cls[0]) == person_id:
                    person_count += 1
                    # x_min, y_min, x_max, y_max of bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # display person count
        cv2.putText(frame, f"Persons: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(win_title, frame)

        # ascii 27 = escape
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    # free resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")


if __name__ == "__main__":
    main()
