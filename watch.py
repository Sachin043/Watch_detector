
# -*- coding: utf-8 -*-

"""
Created on Sat Feb 24 10:11:30 2018
@author: prabhat
"""

import cv2
import numpy as np

#trained upto  2 parent nodes
smile_cascade=cv2.CascadeClassifier('myhaar7.xml')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)
while(True):
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(img,"Face", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        
    smile = smile_cascade.detectMultiScale(gray, 2, 5)
    for (sx,sy,sw,sh) in smile:
        cv2.rectangle(img,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
        cv2.putText(img,"Watch", (sx,sy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
        
       
    cv2.imshow('img',img)
    k=cv2.waitKey(1) & 0xff
    if k==27 :   
        break


    
cap.release()
cv2.destroyAllWindows()    
    
    
