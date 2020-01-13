import numpy as np
import cv2
face_classsifier=cv2.CascadeClassifier('C:/Users/DELL/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face_classsifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
         return None
    for(x,y,w,h) in faces:
        cropped_face=img[y:y+h, x:x+w]

    return  cropped_face

cap=cv2.VideoCapture(0)
count =0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is None:
       print("face not found")
    else:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_path = 'C:/Users/DELL/Downloads/utkarsh/faces' + str(count) + '.jpg'
        cv2.imwrite(file_path, face)
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('crop', face)

    if cv2.waitKey(1) == 13 or count == 300:
        break
cap.release()
cv2.destroyAllWindows()
print("collected")
