# el ambiente de conda que se debe acivar es el de cienciadatos
# Este algoritmo utiliza la biblioteca OpenCV para capturar video de una cámara, detectar movimiento en tiempo real 
# y guardar imágenes cuando se detecta movimiento. Se aplica un filtro gaussiano y umbralización para resaltar las 
# # diferencias entre cuadros consecutivos. Los contornos se encuentran en la imagen umbralizada y se dibujan en el 
# # cuadro original. Las imágenes capturadas se guardan en archivos con nombres únicos. El bucle principal se ejecuta " 
# hasta que se presiona la tecla 'q'. Al final, se libera la cámara y se cierran las ventanas de visualización.
import cv2
import numpy as np
import time

# Inicializa la cámara
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Captura el primer cuadro de video
ret, frame = cap.read()

# Convierte el primer cuadro a escala de grises
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Aplica un filtro gaussiano al primer cuadro
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Configura el tiempo de espera entre capturas de video
capture_interval = 0.5  # segundos
last_capture_time = time.time()

# Declara e inicializa la lista de colores
color = np.random.randint(0, 256, (100, 3), dtype=np.uint8)


# Bucle para capturar imágenes
while True:
    # Captura un cuadro de video
    ret, frame = cap.read()

    # Convierte el cuadro actual a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica un filtro gaussiano al cuadro actual
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    # Calcula la diferencia entre el cuadro actual y el anterior
    frame_diff = cv2.absdiff(gray, gray_frame)

    # Aplica un umbral a la diferencia
    thresh = cv2.threshold(frame_diff, 16, 255, cv2.THRESH_TOZERO)[1]

    # Dilata el umbral para llenar los agujeros
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Encuentra los contornos en el umbral
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja los contornos en el cuadro original y guarda las imágenes en las que se detecta movimiento
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 25:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), color[i].tolist(), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color[i % 100].tolist(), 2)


        # Captura la imagen si ha pasado el tiempo de espera
        current_time = time.time()
        if (current_time - last_capture_time) >= capture_interval:
            last_capture_time = current_time

            # Guarda la imagen capturada en un archivo con un nombre único
            capture_time_str = time.strftime("%Y%m%d-%H%M%S")
            capture_filename = f"captura-{capture_time_str}.jpg"
            cv2.imwrite(capture_filename, frame)

    # Muestra el cuadro de video en una ventana
    cv2.imshow('Captura de video', frame)

    # Actualiza el cuadro anterior
    gray = gray_frame

    # Espera a que el usuario presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
