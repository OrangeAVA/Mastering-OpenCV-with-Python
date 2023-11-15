import cv2


# Initialize our tracker object
tracker = cv2.TrackerMIL_create()

video = cv2.VideoCapture('car.mp4')
ret, frame = video.read()


# Custom bounding box
bbox = cv2.selectROI(frame)

# Initialize tracker with first frame and the drawn bounding box
tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    # Update the tracker to get the new bounding box
    success, bbox = tracker.update(frame)
    
    # Draw the bounding box
    if success:
        x, y, w, h = [int(val) for val in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(0):
        break

# Release video 
video.release()
cv2.destroyAllWindows()