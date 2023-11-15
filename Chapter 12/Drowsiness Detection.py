import cv2
import dlib
from scipy.spatial import distance as dist

# Load the face and eye detectors from Dlib
face_detector = dlib.get_frontal_face_detector()
eye_landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for drowsiness detection
VERTICAL_EYE_THRESHOLD_PIXELS = 13  # Vertical distance threshold in pixels to detect drowsiness
CONSECUTIVE_FRAMES_THRESHOLD = 20  # Number of consecutive frames with closed eyes to detect drowsiness

# Initialize variables
cap = cv2.VideoCapture(0)  # Open the default camera (you can replace 0 with your camera index)
consecutive_frames_closed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        # Detect eyes within the face
        landmarks = eye_landmark_detector(gray, face)

        # Calculate the vertical distance between specific landmarks
        left_eye_top = dist.euclidean((landmarks.part(37).x, landmarks.part(37).y), (landmarks.part(41).x, landmarks.part(41).y))
        left_eye_bottom = dist.euclidean((landmarks.part(40).x, landmarks.part(40).y), (landmarks.part(38).x, landmarks.part(38).y))
        right_eye_top = dist.euclidean((landmarks.part(43).x, landmarks.part(43).y), (landmarks.part(47).x, landmarks.part(47).y))
        right_eye_bottom = dist.euclidean((landmarks.part(46).x, landmarks.part(46).y), (landmarks.part(44).x, landmarks.part(44).y))

        # Calculate the average vertical distance for both eyes
        avg_vertical_distance = (left_eye_top + left_eye_bottom + right_eye_top + right_eye_bottom) / 4.0

        # Draw lines between points for the top and bottom of both eyes
        cv2.line(frame, (landmarks.part(37).x, landmarks.part(37).y), (landmarks.part(41).x, landmarks.part(41).y), (0, 0, 255), 1)
        cv2.line(frame, (landmarks.part(38).x, landmarks.part(38).y), (landmarks.part(40).x, landmarks.part(40).y), (0, 0, 255), 1)
        cv2.line(frame, (landmarks.part(43).x, landmarks.part(43).y), (landmarks.part(47).x, landmarks.part(47).y), (0, 0, 255), 1)
        cv2.line(frame, (landmarks.part(44).x, landmarks.part(44).y), (landmarks.part(46).x, landmarks.part(46).y), (0, 0, 255), 1)

        # Display the average vertical distance
        cv2.putText(frame, f"AVG Vertical Distance: {avg_vertical_distance:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the drowsiness threshold
        cv2.putText(frame, f"Drowsiness Threshold: {VERTICAL_EYE_THRESHOLD_PIXELS} pixels", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Detect drowsiness and display the result
        drowsiness_detected = avg_vertical_distance < VERTICAL_EYE_THRESHOLD_PIXELS
        if drowsiness_detected:
            consecutive_frames_closed += 1
            if consecutive_frames_closed >= CONSECUTIVE_FRAMES_THRESHOLD:
                cv2.putText(frame, "Alert: Drowsiness Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            consecutive_frames_closed = 0

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
