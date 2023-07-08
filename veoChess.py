import cv2
import time
from datetime import datetime

def capture_video():
    # Inicializa la cámara
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Verifica que la cámara esté disponible
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    # Configura el tiempo de espera entre capturas de video
    capture_interval = 0.5  # segundos
    last_capture_time = time.time()

    # Define la variable last_frame_gray para la primera captura en escala de grises
    _, last_frame = cap.read()
    last_frame_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

    # Bucle para capturar imágenes
    while True:
        # Captura un cuadro de video
        ret, frame = cap.read()

        # Verifica que el cuadro de video sea válido
        if not ret:
            print("No se puede recibir el cuadro de video (stream end?).")
            break

        # Convierte el cuadro de video a escala de grises
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calcula el porcentaje de cambio
        diff_percent = (cv2.absdiff(frame_gray, last_frame_gray) / 255.0).mean() * 100

        # Captura la imagen si ha pasado el tiempo de espera o si hay un cambio significativo en la imagen
        current_time = time.time()
        if (current_time - last_capture_time) >= capture_interval and diff_percent > 10:
            last_capture_time = current_time
            last_frame_gray = frame_gray.copy()

            # Guarda la imagen capturada en un archivo con un nombre único
            capture_time_str = time.strftime("%Y%m%d-%H%M%S")
            capture_filename = f"captura-{capture_time_str}.jpg"
            cv2.imwrite(capture_filename, frame)

        # Muestra la imagen capturada en una ventana si su tamaño es mayor que cero
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('Captura de video', frame)

        # Espera a que el usuario presione la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la cámara y cierra todas las ventanas
    cap.release()
    cv2.destroyAllWindows()

capture_video()
