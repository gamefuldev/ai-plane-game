import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

def get_keypoint_position(keypoint_num, axis='x'):
    """ 
    Keypoint reference:
        0: nose          5: left_shoulder  10: right_wrist    15: left_ankle
        1: left_eye      6: right_shoulder 11: left_hip       16: right_ankle
        2: right_eye     7: left_elbow     12: right_hip
        3: left_ear		 8: right_elbow    13: left_knee
        4: right_ear	 9: left_wrist     14: right_knee
    """
    if not 0 <= keypoint_num <= 16:
        raise ValueError("Keypoint number must be between 0 and 16")
    if axis.lower() not in ['x', 'y']:
        raise ValueError("Axis must be 'x' or 'y'")
    
    # Get the keypoint data
    keypoint = results[0].keypoints.xyn[0][keypoint_num]
    
    # Return x or y coordinate based on axis parameter
    return keypoint[0].item() if axis.lower() == 'x' else keypoint[1].item()

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load our YOLO11 model
model = YOLO("yolo11n-pose_ncnn_model")

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLO model on the captured frame and store the results
    results = model.predict(frame, imgsz=320, verbose=False)

    try:
        # Get nose position (keypoint 0)
        nose_x = get_keypoint_position(0, 'x')
        nose_y = get_keypoint_position(0, 'y')
        print(f"Nose - X: {nose_x:.3f}, Y: {nose_y:.3f}")
        
    except (IndexError, AttributeError):
        print("No person detected in frame")
    
    # Output the visual detection data
    annotated_frame = results[0].plot()
    
    # Get inference time and calculate FPS
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    text = f'FPS: {fps:.1f}'
    
    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    
    # Draw the FPS text
    cv2.putText(annotated_frame, text, (text_x, text_y), 
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("Camera", annotated_frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()