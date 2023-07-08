import cv2
import numpy as np

# Parámetros para la detección de esquinas de ShiTomasi
feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=4, blockSize=14)

# Parámetros para el flujo óptico de Lucas-Kanade
lk_params = dict(winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03))

# Selecciona los colores aleatorios
color = np.random.randint(0, 255, (200, 3))

# Toma el primer cuadro y encuentra las esquinas en él
cap = cv2.VideoCapture(3, cv2.CAP_DSHOW)
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Crea una máscara para la representación gráfica
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula el flujo óptico
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Selecciona los mejores puntos
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Dibuja las pistas
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    # Muestra el cuadro
    cv2.imshow('frame', img)

    # Espera a que el usuario presione la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Actualiza el cuadro y los puntos anteriores
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
