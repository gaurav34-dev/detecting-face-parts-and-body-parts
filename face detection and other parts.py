import cv2
import numpy as np
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classifier1 = cv2.CascadeClassifier("haarcascade_eye.xml")
classifier2 = cv2.CascadeClassifier("haarcascade_smile.xml")
#classifier3 = cv2.CascadeClassifier("mouth.xml") 
#classifier4 = cv2.CascadeClassifier("nariz.xml")
#classifier5 = cv2.CascadeClassifier("haarcascade_righteye_2spilts.xml")

while True:
    ret, frame = cap.read()

    if ret:
        faces = classifier.detectMultiScale(frame)
        eyes  = classifier1.detectMultiScale(frame)
        smiles = classifier2.detectMultiScale(frame)
        #mouth = classifier3.detectMultiScale(frame)
        #nose = classifier4.detectMultiScale(frame)
        #eyes = classifier5.detectMultiScale(frame)
        
        font=cv2.FONT_ITALIC
        cv2.putText(frame,"OpenCV with Python!", (90,250), font, 1, (255,255,255), 3, cv2.LINE_AA)
        #cv2_imshow(img)
        
        for eye in eyes:
            x1,y1,w1,h1=eye
            frame = cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),4)

        #for face in faces:
             #x, y, w, h = face
             #(x,y),radius = cv2.minEnclosingCircle(face)
             #center = (int(x),int(y))
             #radius = int(radius)
             #frame = cv2.circle(face,(65,65),65,(0, 0, 255), 4)

        for face in faces:
            x,y,w,h = face
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
            #face = cv2.circle(face,(100, 0), 25, (0,255,0))

        for smile in smiles:
            x2,y2,w2,h2 = smile
            frame = cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,255,0),4)

        #for mouth in mouths:
             #x2,y2,w2,h2 = mouth
             #frame = cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,255,0),4)

        #for nose in noses:
             #x3,y3,w3,h3 = nose
             #frame = cv2.rectangle(frame,(x3,y3),(x3+w3,y3+h3),(0,255,0),4)

         
        cv2.imshow("My window", frame)

    key = cv2.waitKey(30)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

