from dotenv import load_dotenv
load_dotenv('./API.env')
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.camera.entities import VideoFrame

# import opencv to display our annotated images
import cv2
# import supervision to help visualize our predictions
import supervision as sv
import keyboard

detection_count = 0

# create a bounding box annotator and label annotator to use in our custom sink
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoundingBoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    print(len(detections))
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = label_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    image = box_annotator.annotate(image, detections=detections)
    # display the annotated image
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

# Set video_reference to 0 to use the default webcam
pipeline = InferencePipeline.init(
    confidence=0.85,
    model_id="carton-count/1",
    video_reference=0,  # Use 0 to refer to the default webcam
    on_prediction=my_custom_sink,  # Function to run after each prediction
)


pipeline.start()
pipeline.join()

if(keyboard.is_pressed('q')):
    pipeline._stop()
    KeyboardInterrupt
