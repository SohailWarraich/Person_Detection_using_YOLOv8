import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

## Load YOLOv8-Nano pre-trained model ##
model = YOLO("yolov8n.pt")

# Initialize Video Capture
try:

    # For Test Sample Video
    cap = cv2.VideoCapture("input_video/video2.mp4")
except:
    print("error")
    raise ConnectionError

# Check to make sure reading video
assert cap.isOpened(), "Error reading video file"

# Parameters
class_id = 0  # Class_id 0 is for Person detection only

# While Loop to get frame-by-frame from video
while cap.isOpened():
    success, frame = cap.read()  # Read frames
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Make model predictions on each frame for specific class_id = 0: person
    results = model(frame, classes=[class_id])  # Run detection

    # Annotate frame
    annotator = Annotator(frame)
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()  # Bounding box
            class_id = int(box.cls[0])  # Class ID
            score = box.conf[0]  # Confidence score
            label = f'{model.names[class_id]} {score:.2f}'
            annotator.box_label(bbox, label, color=(0, 255, 0))  # Annotate box with label

    # Get annotated frame
    frame = annotator.result()

    # Display frame
    cv2.imshow("Frame", frame)
    
    # Break Window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
