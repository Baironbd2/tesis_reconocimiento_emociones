import cv2
import numpy as np

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

imagen = cv2.imread('WWW.jpg')
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(imagen,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(5,5),
    maxSize=(500,500)
)

for(x,y,w,h) in faces:
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

for (x,y,w,h) in faces:
    face = imagen[y:y+h, x:x+w]
    cv2.imwrite('rostro_{}.jpg'.format(1), face)


cv2.imshow('imagen',imagen)
cv2.waitKey(0)

cv2.destroyAllWindows()