# el ambiente de conda que se debe acivar es el de cienciadatospy
#Ysamos Clustering COn DBSCAN
#Su nombre significa "barrido basado en densidad." Lo que hace DBSCAN es definir tres tipos de puntos en función de los otros puntos que lo rodean:
#Núcleos (puntos rojos): conjuntos de puntos que, entre sí, están dentro de un radio de distancia y, en total, son más que una cierta cantidad especificada como hiperparámetro.
#Puntos Alcanzables (puntos amarillos): puntos que no son núcleos, pero que están dentro del umbral de tolerancia de distancia a puntos núcleos.
#Ruido (punto azul): puntos que no son alcanzables desde los núcleos.


import cv2
import hdbscan
import numpy as np

# Variables para la detección de movimiento
frame_anterior = None
umbral_movimiento = 30

# Función para detectar agrupaciones de partículas
def detectar_agrupaciones(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Realizar detección de bordes con Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Realizar la detección de contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Obtener los centroides de los contornos
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    return centroids

# Inicializar la captura de video
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Variable para almacenar los centroides detectados en el primer frame con movimiento
primer_frame_movimiento = None

while True:
    # Leer el siguiente frame del video
    ret, frame = cap.read()

    # Salir del bucle si no se pudo obtener un frame
    if not ret:
        break

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Si es el primer frame, inicializar el marco de referencia
    if frame_anterior is None:
        frame_anterior = blurred
        continue

    # Calcular la diferencia absoluta entre el marco actual y el marco de referencia
    frame_diferencia = cv2.absdiff(frame_anterior, blurred)

    # Aplicar un umbral a la diferencia de cuadros
    _, frame_umbral = cv2.threshold(frame_diferencia, umbral_movimiento, 255, cv2.THRESH_BINARY)

    # Realizar operaciones de erosión y dilatación para eliminar el ruido
    frame_umbral = cv2.erode(frame_umbral, None, iterations=2)
    frame_umbral = cv2.dilate(frame_umbral, None, iterations=2)

    # Realizar detección de agrupaciones solo cuando hay movimiento
    if np.sum(frame_umbral) > 0:
        centroids = detectar_agrupaciones(frame)

        if primer_frame_movimiento is None:
            primer_frame_movimiento = centroids

        # Realizar agrupamiento con HDBSCAN en el primer frame con movimiento
        if len(centroids) > 0:
            X = np.array(primer_frame_movimiento)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            cluster_labels = clusterer.fit_predict(X)

            # Dibujar los centroides y etiquetas de los grupos en la imagen
            for centroid, label in zip(centroids, cluster_labels):
                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)  # Dibujar círculo verde en el centroide
                cv2.putText(frame, str(label), centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Etiqueta en rojo

        # Marcar los centroides con un punto rojo
        for centroid in centroids:
            cv2.circle(frame, centroid, 2, (0, 0, 255), -1)  # Dibujar punto rojo en el centroide

    # Mostrar el frame resultante
    cv2.imshow('Agrupaciones de partículas', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Actualizar el marco de referencia
    frame_anterior = blurred

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
