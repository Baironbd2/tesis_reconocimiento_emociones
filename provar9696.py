#usar el modelo guardado para una imagen en local
import tensorflow as tf
import numpy as np
# Cargar el modelo guardado
model = tf.keras.models.load_model('modelo_3.h5')

# Cargar la imagen local
img = tf.keras.preprocessing.image.load_img('rostro_1.jpg', target_size=(96, 96))

# Convertir la imagen a un array de valores de píxeles
x = tf.keras.preprocessing.image.img_to_array(img)

# Expandir las dimensiones para que coincida con el formato de entrada del modelo
x = np.expand_dims(x, axis=0)

# Normalizar los valores de los píxeles
x = x / 255.0

# Realizar la predicción
predictions = model.predict(x)

# Obtener la clase con la mayor probabilidad
predicted_class = np.argmax(predictions[0])

# Imprimir la clase predicha
print('La imagen pertenece a la clase:', predicted_class)