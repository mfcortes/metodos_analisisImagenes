  # el ambiente de conda que se debe acivar es el de cienciadatos
  # este algoritmo se basa en calcular  Calcula la diferencia entre el cuadro actual y el anterior
  # aplicando una serie de filtros y umbralizaciones para detectar el movimiento

import cv2
import time
import numpy as np

# Inicializa la cámara
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Captura el primer cuadro de video
ret, frame = cap.read()

# Convierte el primer cuadro a escala de grises
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Aplica un filtro gaussiano al primer cuadro
gray = cv2.GaussianBlur(gray, (21, 21), 0)

# Configura el tiempo de espera entre capturas de video
capture_interval = 4  # segundos
last_capture_time = time.time()

# Inicializa los contornos previos
contornos_previos = {}
id_next = 0

# Bucle para capturar imágenes
while True:
    # Captura un cuadro de video
    ret, frame = cap.read()

    # Convierte el cuadro actual a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplica un filtro gaussiano al cuadro actual
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Calcula la diferencia entre el cuadro actual y el anterior
    frame_diff = cv2.absdiff(gray, gray_frame)

    # Aplica un umbral a la diferencia
    thresh = cv2.threshold(frame_diff, 9, 255, cv2.THRESH_BINARY)[1]

    # Dilata el umbral para llenar los agujeros
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Encuentra los contornos en el umbral
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja los contornos en el cuadro original y guarda las imágenes en las que se detecta movimiento
    nuevos_contornos_previos = {}
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calcula el centro del contorno
        centro = (x + w / 2, y + h / 2)
        id_cercano = None  # Inicializa id_cercano a None

        if contornos_previos:
            # Busca el contorno previo más cercano
            id_cercano, centro_cercano = min(contornos_previos.items(), key=lambda item: np.hypot(item[1][0] - centro[0], item[1][1] - centro[1]))

            # Si el contorno previo más cercano está lo suficientemente cerca, asume que es el mismo contorno y usa el mismo id
            if np.hypot(centro_cercano[0] - centro[0], centro_cercano[1] - centro[1]) < 20:
                nuevos_contornos_previos[id_cercano] = centro
            else:
                # Si no, crea un nuevo id para este contorno
                nuevos_contornos_previos[id_next] = centro
                id_next += 1
        else:
            # Si no hay contornos previos, crea un nuevo id para este contorno
            nuevos_contornos_previos[id_next] = centro
            id_next += 1

        # Muestra el id del contorno
        cv2.putText(frame, str(id_cercano), (int(centro[0]), int(centro[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

    # Actualiza el cuadro anterior y los contornos previos
    gray = gray_frame
    contornos_previos = nuevos_contornos_previos

    # Espera a que el usuario presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()