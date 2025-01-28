from ultralytics import YOLO
import torch
import cv2

# Pre-trained YOLO model
pretrain_model = "yolov8n.pt"  # YOLO nano model (lightweight and fast)

def main():
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Load the YOLO model
    print("Loading YOLO model...")
    model = YOLO(pretrain_model)

    # Open webcam (use 0 for the default camera)
    cap = cv2.VideoCapture(0)
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
        results = model.predict(frame, conf=0.5)  # Confidence threshold of 50%

        # Count persons and annotate frame
        person_count = 0
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Check if class is "person" (class ID 0)
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display person count
        cv2.putText(frame, f"Persons: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the annotated frame
        cv2.imshow("Person Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == "__main__":
    main()
