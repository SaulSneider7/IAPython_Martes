# -------------------------------------------------------------------------
# Crack the Code
# Inteligencia Artificial con Python
# -------------------------------------------------------------------------
# Importar bibliotecas que se utilizarán - no modifiques esta sección
import cv2
import os
import numpy as np

# Carpeta con fotos de entrenamiento
dataPath = './data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

# -------------------------------------------------------------------------
# Escribe tu código aquí:
labels = []
facesData = []
label = 0

#Creamos un buble For para poder leet todas las imagenes
print('Leyendo las imagenes ...')
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)

        # Agregamos label y la imagen del rostro a su arreglo correspondiente
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))

    # Incrementa label cuando se cambia de carpeta
    label = labels + 1

#Creamos un objeto para reconocer los rostros
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando Modelo ...")
face_recognizer.train(facesData, np.array(labels))

#Almacenamos el modelo que se obtuvo
face_recognizer.write('modelo.xml')
print('Modelo almacenado correctamente')