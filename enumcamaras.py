import cv2

# Enumeramos las cámaras disponibles
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.read()[0]:
        print(f"Cámara {i} detectada: {cap.get(cv2.CAP_PROP_BACKEND_NAME)}")
    else:
        print(f"Cámara {i} no detectada")
