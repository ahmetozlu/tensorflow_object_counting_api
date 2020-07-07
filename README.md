
Conteo y reconocimiento de personas utilizando Computer Vision

Manual de Usuario

# Visión general de la solución

La aplicación de conteo y reconocimiento de personas ha sido desarrollada sobre la API proporcionada por Tensorflow la cual nos ofrece múltiples alternativas fiables para lograr el objetivo de reconocer, dar seguimiento y contar personas.

Se ha decidido utilizar un modelo previamente entrenado basado en SSD de MobileNet el cual nos provee un modelo confiable previamente entrenado, reduciendo la complejidad de cálculo computacional, esto es muy importante si tenemos en cuenta las características y especificaciones de los dispositivos en los cuales podremos correr la aplicación.

## Instalación y Prerrequisitos

Prerrequisitos:

-   Python 3.5 a 3.7 ([descargar del sitio oficial](https://www.python.org/downloads/))
    
-   Tensorflow ([guia de instalación](https://www.tensorflow.org/install/))
    

Proceso de instalación:

Para realizar la instalación se debe clonar el proyecto desde el repositorio del código fuente:

git clone [https://github.com/baesparza/tensorflow_object_counting_api](https://github.com/baesparza/tensorflow_object_counting_api)

Luego debemos instalar las dependencias del proyecto utilizando el comando:

pip install -r requirements.txt

Para correr el programa usamos el comando:

python pedestrian_counting.py

# Funcionamiento

Primero se debe configurar en el archivo de configuracion  pedestrians_counting.py:

En el configuramos la ruta del archivo de video esto para que la aplicación sepa de donde tomar el input para el procesamiento.

input_video = "./input_images_and_videos/pedestrian_survaillance.mp4"

Posteriormente se configuran las variables, para que el programa se ajuste al video de input:

-   is_color_recognition_enabled (feature de reconocimiento de color) 1 Si está activada, 0 si está desactivada.
    
-   roi (la posición del umbral de conteo) cualquier valor entero
    
-   deviation (el área de conteo del objeto) 1 o 0
    
-   axis (la orientación del umbral horizontal o vertical) x o y
    

Ejemplo de configuracion basica:

```
axis = 'x'
is_color_recognition_enabled = 0
roi = 385
deviation = 1
````

  

Una vez iniciada la aplicación empezará a realizar el procesamiento fotograma por fotograma hasta su terminación, dándonos como resultado el video con el reconocimiento y el conteo de las personas. este video se exportará en el directorio raíz del proyecto conjunto al log con los nombres the_output.avi y the_output.txt

**Video de entrada  Resultado**![](https://lh6.googleusercontent.com/8T7f9Io76nN_EdJeJtv9-dB7ZlUtL_AipzPO69ILnZdRZ0ZLXQfPuEn_pPQLyJejAIO0RTGmykn5b5EnekdNECby9iqUO88ZbTh0P6dTIY3MESdCtYHHshjQbuV7jRi6DdZPLSc9)![](https://lh4.googleusercontent.com/hl65jJG66_Cmnp_8cNxD5Cedowui7abgQanjYa-15_87gBrZ_pfBvmUObEhvRzuhFXJBU_W2UxTZv6wcu5I6ZVmCwyGyfLDCLb90E_u9qtKtfxPJC-Py8D4V3_4PuYeVvGiazDTr)

### Resultado:
|Conteo manual|Conteo automático|
|-----|--------|
|4 personas|3 personas|

  

### Conclusiones:

El resultado depende de la posición de las personas al cruzar el umbral, si estas se encuentran al mismo tiempo en el umbral solo contará como una sola por la superposición de las mismas, en esos casos una mejora podría ser tener videos de entrada de distintos ángulos para mejorar la precisión.
