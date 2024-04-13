import cv2
import os
import imutils

rostros = 'feliz'
dataPath = 'E:/Tesis/rostros'
rostroPath = dataPath+'/'+ rostros
if not os.path.exists(rostroPath):
    print('carpeta creada', rostroPath)
    os.makedirs(rostroPath)

cap = cv2.VideoCapture('feliz1.mp4')

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(100, 100),
        maxSize=(1024,1024)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(224,224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(rostroPath+ '/rostro_{}.jpg'.format(count),rostro)
        count = count+1
    cv2.imshow('frame',frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 5000:
        break

cap.release()
cv2.destroyAllWindows()