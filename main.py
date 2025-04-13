import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(1)

# resolution and fps
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

#confirm what fps im getting
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {actual_fps}")

if not cap.isOpened():
    print("Cant open camera")
    exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("Error when capturing image from the camera")
        break

    img = cv2.flip(img, 1) #mirror the image
    
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)

    #Draw rectangle around the face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        #Extract the face ROI from the grayscale frame
        face_roi = gray_frame[y:y+h, x:x+w]

        #Detect smiles within the face ROI
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)

        if len(smiles) > 0:
            cv2.putText(img, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    

    cv2.imshow('Real-time Face and Smile Detection', img)
    cv2.waitKey(1)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()