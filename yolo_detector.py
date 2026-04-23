from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv

# Load YOLO model
model = YOLO("yolov8n.pt")

track_points = []

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        coordinator = box.xyxy[0]
        confidence = float(box.conf[0])
        
        if class_name == 'cat':
            # Draw bounding box (Blue)
            annotator.box_label(
                box=coordinator, 
                label=f'{class_name} {confidence:.2f}', 
                color=(255, 0, 0) 
            )
            
            # Tracking points
            x_center = int((coordinator[0] + coordinator[2]) / 2)
            y_center = int((coordinator[1] + coordinator[3]) / 2)
            track_points.append((x_center, y_center))

    # Draw tracking line
    for i in range(1, len(track_points)):
        cv.line(frame, track_points[i - 1], track_points[i], (0, 0, 255), 2)

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""
    # Detect object from image frame
    results = model.predict(frame, classes=[15], verbose=False)

    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            frame_result = detect_object(frame)
            
            # Watermark
            text = "Tanat Kunharee Clicknext-Internship"
            cv.putText(frame_result, text, (frame_result.shape[1] - 550, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", frame_result)
            
            if cv.waitKey(30) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv.destroyAllWindows()