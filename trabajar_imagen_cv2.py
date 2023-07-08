import cv2 as cv 

# Cargar la imagen

img = cv.imread(cv.samples.findFile("starry_night.jpg"))

# Convertir la imagen a escala de grises
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro gaussiano para suavizar la imagen
blurred_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

# Aplicar un umbral a la imagen
ret, thresh_img = cv2.threshold(blurred_img, 128, 255, cv2.THRESH_BINARY)

# Encuentra los contornos en la imagen umbralizada
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Muestra las imágenes
cv2.imshow('Imagen original', img)
cv2.imshow('Imagen en escala de grises', gray_img)
cv2.imshow('Imagen suavizada', blurred_img)
cv2.imshow('Imagen umbralizada', thresh_img)
cv2.imshow('Imagen con contornos', contour_img)

# Esperar a que el usuario presione una tecla y cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()

