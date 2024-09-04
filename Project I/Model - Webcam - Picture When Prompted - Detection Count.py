from roboflow import Roboflow
import supervision as sv
import cv2
import keyboard

#  Roboflow model
rf = Roboflow(api_key="TUJkSFnhqCo9n08uToo1")
project = rf.workspace().project("carton-count")
model = project.version(1).model

#  webcam capture
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    raise IOError("Cannot open webcam")

print("Press 'p' to capture a frame or 'q' to quit.")
i = 0  #  a counter for the image filenames

while True:
    # Read a frame
    ret, frame = webcam.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Webcam Feed", frame)

    # Check if the 'p' key is pressed to capture a snapshot
    if keyboard.is_pressed('p'):
        print("Capturing frame...")

        # Save the frame as an image file
        original_image_path = f"./Testing pictures of Model - Webcam - Picture When Prompted/captured_frame_{i}.jpg"
        cv2.imwrite(original_image_path, frame)
        print(f"Original frame captured and saved to {original_image_path}")

        # Perform prediction on the captured image
        result = model.predict(original_image_path, confidence=90, overlap=30).json()

        # Extract labels and detections from the prediction result
        labels = [item["class"] for item in result["predictions"]]
        detections = sv.Detections.from_inference(result)

        # Annotate the captured frame with bounding boxes and labels
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()

        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Save the annotated image as well
        annotated_image_path = f"./Testing pictures of Model - Webcam - Picture When Prompted/captured_frame_annotated_{i}.jpg"
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Annotated frame captured and saved to {annotated_image_path}")

        # Display the original and annotated image
        carton_count = len(detections)
        sv.plot_images_grid([frame,annotated_image],(1,2),["Original","Annotated, counted:" + str(carton_count)])

        # Increment the counter for the next image filenames
        i += 1

    # Exit loop if 'q' is pressed
    if keyboard.is_pressed('q'):
        print("Terminating program...")
        break

    # Add a small delay to reduce CPU usage
    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()
