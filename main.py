import cv2
import random
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

## Load YOLOv8-Nano pre-trained model ##
model = YOLO("yolov8n.pt")

# Initialize Video Capture
try:
    rtsp_url = 'rtsp://192.168.100.4:8080/h264_ulaw.sdp'
    # cap = cv2.VideoCapture(rtsp_url)
    # For Test Sample Video
    cap = cv2.VideoCapture("input_video/video2.mp4")
except:
    print("Retry to initiate your IP webcam")
    raise ConnectionError

# Check to make sure reading video
assert cap.isOpened(), "Error reading video file"

# Parameters
class_id = 0  # Class_id 0 is for Person detection only

def generate_random_color(exclude_red=True):
    """Generates a random BGR color, excluding shades of red if specified."""
    while True:
        color = [random.randint(0, 255) for _ in range(3)]
        if exclude_red:
            # Ensure the color is not predominantly red
            if color[2] < 100:  # BGR, so index 2 is Red channel
                break
        else:
            break
    return tuple(color)

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
            color = generate_random_color()  # Generate a random color excluding red
            annotator.box_label(bbox, label, color=color)  # Annotate box with label

    # Get annotated frame
    frame = annotator.result()

    # Display frame
    cv2.imshow("Frame", frame)
    
    # Break Window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
