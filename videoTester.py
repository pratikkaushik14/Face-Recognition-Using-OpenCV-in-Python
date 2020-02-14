import os
import cv2
import numpy as np
import faceRecognition as fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#Load saved training data

name = {0 : "Prateek",1 : "Kanchan", 2:"Narinder-Kamra", 3:"Jyoti", 4:"Emilia-Clarke", 5:"Kit-Harington", 6:"Nikolaj-Coster", 7:"Peter-Dinklage"}

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img=fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

    #resized_img = cv2.resize(test_img, (1000, 700))
    #cv2.imshow('face detection Tutorial ',resized_img)
    #cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence > 55:#If confidence less than 50 then don't print predicted face text on screen
           fr.put_text(test_img,predicted_name,x,y)


    resized_img = cv2.resize(test_img, (600, 570))
    cv2.imshow('FR',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows

