"""
Control de Asistencia - Reconocimiento Facial
"""

import os
from datetime import datetime
from cv2 import cv2
import face_recognition as fr
import numpy

# Crear Base de Datos
RUTA = "Empleados"
mis_imaganes = []
nombres_empleados = []
lista_empleados = os.listdir(RUTA)

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f"{RUTA}/{nombre}")
    mis_imaganes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

print(nombres_empleados)

# Codificar Imágenes
def codificar(imagenes):
    """Función para codificar imágenes"""
    # Crear una lista nueva
    lista_codificada = []
    # Pasar todas las imágenes a RGB
    for img in imagenes:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Codificar
        codificado = fr.face_encodings(img)[0]
        # Agregar a la lista
        lista_codificada.append(codificado)

    # Devolver lista codificada
    return lista_codificada

# Registrar los ingresos
def registrar_ingresos(persona):
    """Función para registrar ingresos"""
    file = open("registro.csv", "r+", encoding="utf-8")
    lista_datos = file.readlines()
    nombres_registro = []
    for linea in lista_datos:
        ingreso = linea.split(",")
        nombres_registro.append(ingreso[0])

    if persona not in nombres_registro:
        ahora = datetime.now()
        string_fecha = ahora.strftime("%d/%m/%Y")
        string_ahora = ahora.strftime("%H:%M:%S")
        file.writelines(f"\n{persona}, {string_fecha}, {string_ahora}")

lista_empleados_codificada = codificar(mis_imaganes)

# Tomar una imagen de la cámara web
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Leer la imágen de la cámara
exito, imagen = captura.read()

if not exito:
    print("No se ha podido tomar la captura")
else:
    # Reconocer cara en captura
    cara_captura = fr.face_locations(imagen)
    # Codificar la cara capturada
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)
    # Buscar coincidencias
    for caracodif, caraubic in zip(cara_captura_codificada, cara_captura):
        coincidencias = fr.compare_faces(lista_empleados_codificada, caracodif)
        distancias = fr.face_distance(lista_empleados_codificada, caracodif)
        print(distancias)
        indice_coincidencia = numpy.argmin(distancias)
        # Mostrar coincidencias
        if distancias[indice_coincidencia] > 0.6:
            no_encontrado = fr.load_image_file("no-encontrado.jpg")
            no_encontrado = cv2.cvtColor(no_encontrado, cv2.COLOR_BGR2RGB)
            cv2.putText(no_encontrado,
                       "NO ENCONTRADO",
                       (115, 100),
                       cv2.FONT_HERSHEY_COMPLEX,
                       1,
                       (0, 0, 255),
                       2)
            print("No coincide con ninguno de nuestros empleados")
            # Mostrar la imágen obtenida
            cv2.imshow("No Encontrado", no_encontrado)
            # Mantener ventana abierta
            cv2.waitKey(0) # Retardo en milisegundos. 0 significa "para siempre".
        else:
            # Buscar el nombre del Empleado encontrado
            nombre = nombres_empleados[indice_coincidencia]
            # Mostrar Rectángulos
            y1, x2, y2, x1 = caraubic
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imagen, (x1, y2 - 30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen,
                        nombre,
                        (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)
            # Registrar Ingresos
            registrar_ingresos(nombre)
            # Mostrar la imágen obtenida
            cv2.imshow("Imágen WebCam", imagen)
            # Mantener ventana abierta
            cv2.waitKey(0) # Retardo en milisegundos. 0 significa "para siempre".
