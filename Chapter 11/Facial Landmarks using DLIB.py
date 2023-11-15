import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

image = cv2.imread("face.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Loop over detected faces
for face in faces:
    # Predict the facial landmarks for each detected face
    landmarks = predictor(gray, face)

    cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
    
    # Draw landmarks on image
    for point in landmarks.parts():
        cv2.circle(image, (point.x, point.y), 1, (0, 0, 255), -1)

cv2.imshow("Facial Landmarks and Rectangles", image)
cv2.imwrite("output_landmarks_rectangles.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()