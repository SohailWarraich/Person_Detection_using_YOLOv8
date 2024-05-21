Real-Time Person Detection and Tracking using YOLOv8
This project utilizes OpenCV to connect to an RTSP video stream, process object tracks, and interactively select bounding boxes with mouse clicks. A timer is displayed for the selected bounding box, even if detection is temporarily skipped for a few frames.

Features
Connect to an RTSP stream (e.g., from the IP Webcam app on Android).
Process object tracks and display bounding boxes.
Click on a bounding box to select it, changing its color to red, and starting a timer.
Display the timer on the selected bounding box.
Maintain the selected bounding box for a specified grace period if detection is temporarily skipped.
Prerequisites
Python
OpenCV
Ultralytics
IP Webcam app (or any RTSP video streaming camera)
Installation
Clone the repository:


git clone https://github.com/MYahya3/Real-Time_Person_Detection_and_Tracking_using_YOLOv8.git
cd Real-Time_Person_Detection_and_Tracking_using_YOLOv8
Install dependencies:

Copy code
pip install opencv-python-headless
pip install ultralytics
Install and configure the IP Webcam app:

Download and install the IP Webcam app from the Google Play Store.
Open the app, configure settings, and start the server.
Note the IP address and port displayed by the app.
Usage
Modify the RTSP URL:

In main.py, set the rtsp_url variable to match the IP address and port of your IP Webcam app:


rtsp_url = 'rtsp://<IP_ADDRESS>:<PORT>/h264_ulaw.sdp'
Run the script:

python main.py

File Structure
main.py: Main script to capture RTSP stream, process frames, and display the video.
Visualize Input and Output Result
