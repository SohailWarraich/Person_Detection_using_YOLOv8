import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time

## Load YOLOv8-Nano pre-trained model ##
model = YOLO("yolov8n.pt")

## Initialize Video Capture ##
try:
    # For Test Sample Video
    cap = cv2.VideoCapture("input_video/video2.mp4")
except:
    print("Error")
    raise ConnectionError

## Check to make sure reading video ##
assert cap.isOpened(), "Error reading video file"

# Parameters
class_id = 0  ### Class_id 0 is for Person detection only
line_thickness = 2
exclude_red = True
grace_period = 5

colors = [
    (0, 0, 0),  # Black
    (150, 150, 100),
    (128, 0, 0),  # Navy
    (128, 128, 0),
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (128, 128, 0),  # Teal
    (255, 255, 0),  # Cyan/Aqua
    (0, 100, 0),  # Dark Green
    (128, 128, 0),  # Turquoise
    (205, 0, 0),  # Medium Blue
    (209, 206, 0),  # Dark Turquoise
    (170, 178, 32),  # Light Sea Green
    (127, 255, 0),  # Spring Green
    (87, 139, 46),  # Sea Green
    (255, 144, 30),  # Dodger Blue
    (255, 191, 0)  # Deep Sky Blue
]

start_time = None
selected_track_id = None
boxes = []
track_ids = []
last_seen_frame = {}  # Keep track of the last frame each track ID was seen
frame_count = 0

def mouse_callback(event, x, y, flags, param):
    global start_time, selected_track_id, boxes, track_ids, last_seen_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if int(x1) <= x <= int(x2) and int(y1) <= y <= int(y2):
                selected_track_id = track_ids[i]
                start_time = time.time()
                last_seen_frame[selected_track_id] = param['frame_count']
                print(f"Selected track ID: {selected_track_id}")
                break

def process_draw_Boundboxes(image, tracks, frame_count, classes_names=None):
    global boxes, track_ids, start_time, selected_track_id

    annotator = Annotator(image, line_width=line_thickness)

    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        # Update the last seen frame for each track ID
        for track_id in track_ids:
            last_seen_frame[track_id] = frame_count

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            if selected_track_id == track_id:
                color = (0, 0, 255)  # Red for selected box
                annotator.box_label(box, label=f"{classes_names[cls]}", color=color)
            else:
                try:
                    color = colors[int(track_id)]
                    annotator.box_label(box, label=f"{classes_names[cls]}", color=color)
                except:
                    color = colors[int(track_id) % len(colors)]
                    annotator.box_label(box, label=f"{classes_names[cls]}", color=color)

            # Display the timer on the top left if a box is selected
            if start_time is not None and selected_track_id == track_id:
                elapsed_time = int(time.time() - start_time)
                cv2.rectangle(image, (0, 0), (200, 40), (255, 144, 30), -1)  # Rectangle dimensions and color
                cv2.putText(image, f"Time-Tracker: {elapsed_time}sec", (15, 20),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (190, 215, 255), 1, cv2.LINE_AA)
    return image

def maintain_MisDetection(frame_count):
    global selected_track_id, start_time

    if selected_track_id is not None:
        last_seen = last_seen_frame.get(selected_track_id, None)
        if last_seen is None or (frame_count - last_seen) > grace_period:
            selected_track_id = None
            start_time = None

# Set the mouse callback function for the window
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback, {'frame_count': frame_count})

# While Loop to get frame-by-frame from video
while cap.isOpened():
    success, frame = cap.read()  # Read frames
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    # Increment frame count
    frame_count += 1

    # Make model predictions on each frame for specific class_id = 0: person
    tracks = model.track(frame, persist=True, show=False, classes=0)

    frame = process_draw_Boundboxes(image=frame, tracks=tracks, frame_count=frame_count, classes_names=model.names)

    # Maintain the selection for a few frames even if detection is skipped
    maintain_MisDetection(frame_count)

    """Display frame."""
    cv2.imshow("Frame", frame)
    # Break Window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
