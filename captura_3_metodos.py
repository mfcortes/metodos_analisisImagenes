import cv2
import numpy as np
import time

# Inicializa la cámara
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Captura el primer cuadro de video
ret, frame1 = cap.read()

# Convierte el primer cuadro a escala de grises y aplica un filtro gaussiano
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
prvs = cv2.GaussianBlur(prvs, (21, 21), 0)

# Configura el tiempo de espera entre capturas de video
capture_interval = 1  # segundos
last_capture_time = time.time()

# Bucle para capturar imágenes
while True:
    ret, frame2 = cap.read()

    # Convierte el cuadro actual a escala de grises y aplica un filtro gaussiano
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next = cv2.GaussianBlur(next, (5, 5), 0)

    # Calcula el flujo óptico
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Encuentra los bordes en el cuadro actual
    edges = cv2.Canny(next, 8, 80)

    # Multiplica los bordes por el módulo del flujo óptico para obtener áreas con bordes que se mueven
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #print(f'edges {edges}')
    movement = np.multiply(edges, magnitude)
    #print(f'Cantidad de bordes que se mueven: {np.count_nonzero(movement)}')

    # Dibuja los bordes que se mueven en el cuadro original
    frame2[movement > 0] = [0, 0, 255]

    # Muestra el cuadro de video en una ventana
    cv2.imshow('Video', frame2)

    # Captura la imagen si ha pasado el tiempo de espera y si hay bordes que se mueven
    current_time = time.time()
    if (current_time - last_capture_time) >= capture_interval and np.any(movement > 0):
        last_capture_time = current_time

        # Guarda la imagen capturada en un archivo con un nombre único
        capture_time_str = time.strftime("%Y%m%d-%H%M%S")
        capture_filename = f"captura-{capture_time_str}.jpg"
        cv2.imwrite(capture_filename, frame2)

    # Actualiza el cuadro anterior
    prvs = np.copy(next)

    # Espera a que el usuario presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
