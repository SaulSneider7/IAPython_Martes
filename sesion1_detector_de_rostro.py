# -------------------------------------------------------------------------
# Crack the Code
# Inteligencia Artificial con Python
# Sesion 1 - Detector de rostros
# -------------------------------------------------------------------------
# Importar bibliotecas que se utilizarán - no modifiques esta sección
import cv2
from camera import getcamera

# -------------------------------------------------------------------------
# Escribe tu código aquí:
#Abrir la camara Web
'''
cam = getcamera()
cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)

#Tomando una foto
ret, frame = cap.read()

#Mostrar la foto
cv2.imshow('TITULO', frame)

#Mantener imagen abierta
cv2.waitKey(0)
'''


#ACTIVIDAD 2
cam = getcamera()
cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier("rostros.xml")

while True:
    ret, frame = cap.read()
    #comprobar si existe imagen
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(120, 120),
                                         maxSize=(1000,1000))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #Mostrar en pantalla
    cv2.imshow('CAPTURA', frame)
    #Cerrar la ventana con "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# --------------------------------------------------------------------------
# Cierra la cámara y las ventanas - no borres estas lineas
# Deja estas lineas hasta abajo
cv2.destroyAllWindows()
cap.release()
