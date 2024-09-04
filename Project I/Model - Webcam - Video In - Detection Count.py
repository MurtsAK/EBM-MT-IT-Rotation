from dotenv import load_dotenv
load_dotenv('./API.env')
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv
import keyboard

while True:
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
        image = label_annotator.annotate(
            scene=video_frame.image.copy(), detections=detections, labels=labels
        )
        image = box_annotator.annotate(image, detections=detections)
        # display the annotated image
        cv2.imshow("Predictions", image)
        cv2.waitKey(1)

    pipeline = InferencePipeline.init(
        confidence=0.85,
        model_id="carton-count/1",
        video_reference=0,  # Use 0 to refer to the default webcam
        on_prediction=my_custom_sink,
    )


    pipeline.start()
    pipeline.join()

    if(keyboard.is_pressed('q')):
        pipeline._stop()
        break
    
cv2.destroyAllWindows()
