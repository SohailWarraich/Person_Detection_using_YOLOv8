# Real-Time Person Detection and Tracking using YOLOv8

Real-Time Person Detection and Tracking using YOLOv8 is a project that utilizes OpenCV to connect to an RTSP video stream, process object tracks, and interactively select bounding boxes with mouse clicks. It features the ability to connect to an RTSP stream, process object tracks, display bounding boxes, click on bounding boxes to select them, display a timer on the selected bounding box, and maintain the selected bounding box for a specified grace period if detection is temporarily skipped.

## Features

- **RTSP Stream Connection**: Connect to an RTSP stream (e.g., from the IP Webcam app on Android).
- **Object Tracking**: Process object tracks and display bounding boxes.
- **Interactive Bounding Boxes**: Click on a bounding box to select it, changing its color to red, and starting a timer.
- **Timer Display**: Display the timer on the selected bounding box.
- **Grace Period**: Maintain the selected bounding box for a specified grace period if detection is temporarily skipped.

## Prerequisites

- Python
- OpenCV
- Ultralytics
- IP Webcam app (or any RTSP video streaming camera)

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/MYahya3/Real-Time_Person_Detection_and_Tracking_using_YOLOv8.git
    cd Real-Time_Person_Detection_and_Tracking_using_YOLOv8
    ```

2. **Install dependencies**:

    ```bash
    pip install opencv-python-headless
    pip install ultralytics
    ```

3. **Install and configure the IP Webcam app**:

    - Download and install the IP Webcam app from the Google Play Store.
    - Open the app, configure settings, and start the server.
    - Note the IP address and port displayed by the app.

## Usage

1. **Modify the RTSP URL**:

    In `main.py`, set the `rtsp_url` variable to match the IP address and port of your IP Webcam app:

    ```python
    rtsp_url = 'rtsp://<IP_ADDRESS>:<PORT>/h264_ulaw.sdp'
    ```

2. **Run the script**:

    ```bash
    python main.py
    ```

## File Structure

- `main.py`: Main script to capture RTSP stream, process frames, and display the video.

## Visualize Input and Output Result

![image](https://github.com/SohailWarraich/SohailWarraich-Person_Detection_using_YOLOv8/assets/63116532/5c854d00-0346-4276-b4d2-b9bd3a566b6d)


