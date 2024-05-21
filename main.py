import cv2

# Initialize Video Capture
try:
    # For Test Sample Video
    cap = cv2.VideoCapture("input_video/video2.mp4")
except:
    print("Retry to initiate your IP webcam")
    raise ConnectionError
# Check to make sure reading video
assert cap.isOpened(), "Error reading video file"

# While Loop to get frame-by-frame from video
while cap.isOpened():
    success, frame = cap.read()  # Read frames
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Display frame
    cv2.imshow("Frame", frame)
    
    # Break Window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
