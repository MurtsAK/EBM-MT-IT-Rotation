from roboflow import Roboflow
import supervision as sv
import cv2
import time
import sys

# Initialize the Roboflow model
rf = Roboflow(api_key="TUJkSFnhqCo9n08uToo1")
project = rf.workspace().project("carton-count")
model = project.version(1).model

# Open a connection to the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the annotators
label_annotator = sv.LabelAnnotator()
bounding_box_annotator = sv.BoxAnnotator()

# Frame rate control
fps = 15  # Target FPS
frame_time = 1 / fps  # Time per frame

def process_frame(frame):
    # Resize frame to speed up processing (optional, adjust as needed)
    frame = cv2.resize(frame, (640, 640))  # Reduce size to 320x240

    # Make predictions using the Roboflow model
    result = model.predict(frame, confidence=80, overlap=30).json()

    # Convert the result to Detections for Supervision annotations
    detections = sv.Detections.from_inference(result)

    # Annotate the image with bounding boxes and labels
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=[item["class"] for item in result["predictions"]])

    return annotated_image

try:
    while True:
        start_time = time.time()  # Start time for frame processing

        # Capture frame-by-frame
        ret, frame = cap.read()

        # If the frame was not read correctly, exit the loop
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Process the frame
        annotated_image = process_frame(frame)

        # Display the annotated image
        cv2.imshow('Annotated Video', annotated_image)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Calculate elapsed time and wait if necessary
        elapsed_time = time.time() - start_time
        wait_time = max(1, int((frame_time - elapsed_time) * 1000))  # in milliseconds
        cv2.waitKey(wait_time)

except KeyboardInterrupt:
    # Handle the case when the user interrupts the program with Ctrl+C
    print("Program interrupted.")

finally:
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
