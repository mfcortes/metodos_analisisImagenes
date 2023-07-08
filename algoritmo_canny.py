
# el ambiente de conda que se debe acivar es el de cienciadatospy
#Algoritmo de Canny es un operador desarrollado por John F. Canny en 1986 
#que utiliza un algoritmo de múltiples etapas para detectar una amplia gama 
#de bordes en imágenes.1​ Lo más importante es que Canny también desarrolló una 
#teoría computacional acerca de la detección de bordes que explica por qué la técnica funciona.

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Configura el tiempo de espera entre capturas de imagen
capture_interval = 4  # segundos
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    
    # Convierte a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplica desenfoque Gaussiano
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detección de bordes con Canny
    edges = cv2.Canny(blur, 20, 100)
    
    # Encuentra los contornos en los bordes
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibuja los contornos en la imagen original
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Si se detectan bordes y ha pasado el tiempo de captura, guarda la imagen
    if len(contours) > 0:
        current_time = time.time()
        if (current_time - last_capture_time) >= capture_interval:
            last_capture_time = current_time

            # Guarda la imagen capturada en un archivo con un nombre único
            capture_time_str = time.strftime("%Y%m%d-%H%M%S")
            capture_filename = f"captura-{capture_time_str}.jpg"
            cv2.imwrite(capture_filename, frame)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
