import cv2

# Loading pre-trained cat face cascade
cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

image_path = cat.png'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform cat face detection
cat_faces = cat_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in cat_faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Cat Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()