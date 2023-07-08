import cv2
import time

# Inicializa la cámara
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Verifica que la cámara esté disponible
if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

# Configura el tiempo de espera entre capturas de video
capture_interval = 2  # segundos

# Define el tamaño del historial de imágenes y el tiempo de retención
history_size = 60  # imágenes
history_duration = 5 * 60  # segundos
history = []

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

    # Aplica un filtro gaussiano al cuadro de video
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

    # Verifica si hay un cambio significativo en la imagen
    current_time = time.time()
    diff_percent = None
    for history_frame, history_time in history:
        if current_time - history_time <= history_duration:
            diff = cv2.absdiff(frame_gray, history_frame)
            percent = (cv2.countNonZero(diff) / (diff.size / 255.0)) * 100
            if diff_percent is None or percent > diff_percent:
                diff_percent = percent
        else:
            history.pop(0)
    if diff_percent is None or diff_percent > 50:
        # Guarda la imagen capturada en un archivo con un nombre único
        capture_time_str = time.strftime("%Y%m%d-%H%M%S")
        capture_filename = f"captura-{capture_time_str}.jpg"
        cv2.imwrite(capture_filename, frame)

    # Agrega el cuadro de video al historial
    history.append((frame_gray.copy(), current_time))
    if len(history) > history_size:
        history.pop(0)

    # Muestra la imagen capturada en una ventana si su tamaño es mayor que cero
    if frame.shape[0] > 0 and frame.shape[1] > 0:
        cv2.imshow('Captura de video', frame)

    # Espera a que el usuario presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
