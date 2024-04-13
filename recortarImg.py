import cv2
import os

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rostros = 'feliz'
dataPathS = 'E:/Tesis/rostros'
rostroPath = os.path.join(dataPathS, rostros)

if not os.path.exists(rostroPath):
    print('Carpeta creada:', rostroPath)
    os.makedirs(rostroPath)

dataPath = 'rostro'

# Contador para nombres únicos de las imágenes de rostros
img_counter = 0

for img_name in os.listdir(dataPath):
    img_path = os.path.join(dataPath, img_name)
    imagen = cv2.imread(img_path)

    faces = faceClassif.detectMultiScale(imagen,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(5, 5),
        maxSize=(500, 500)
    )

    for i, (x, y, w, h) in enumerate(faces):
        rostro = imagen[y:y+h, x:x+w]
        rostro = cv2.resize(rostro,(224,224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(rostroPath, f'rostro_{img_counter}.png'), rostro)
        img_counter += 1

        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('imagen', imagen)
    cv2.waitKey(0)

cv2.destroyAllWindows()