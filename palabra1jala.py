import os
import warnings
import pickle
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from hmmlearn import hmm
from python_speech_features import mfcc, delta
from sklearn.metrics import accuracy_score
# pip install numpy scipy sounddevice hmmlearn python_speech-features scikit-learn

def construir_modelos(carpeta_entrada, archivo_modelos):
    """
    Crea modelos HMM para clasificar palabras o carga modelos existentes.
    
    Args:
        carpeta_entrada (str): Ruta del dataset con subcarpetas para cada etiqueta.
        archivo_modelos (str): Archivo donde se guardan los modelos entrenados.
    
    Returns:
        list: Lista de tuplas (modelo, etiqueta).
    """
    if os.path.exists(archivo_modelos):
        # Si los modelos ya existen, se cargan desde un archivo
        print("Cargando modelos desde archivo...")
        with open(archivo_modelos, 'rb') as f:
            modelos_voz = pickle.load(f)
    else:
        # Si no existen modelos, se entrenan desde el dataset
        print("Entrenando modelos desde el dataset...")
        modelos_voz = []

        # Iterar por cada subcarpeta en el dataset
        for nombre_directorio in os.listdir(carpeta_entrada):
            subcarpeta = os.path.join(carpeta_entrada, nombre_directorio)
            if not os.path.isdir(subcarpeta):
                continue

            etiqueta = nombre_directorio  # La etiqueta es el nombre de la carpeta
            X = np.array([])  # Array para almacenar características de los audios

            # Usar todos los archivos menos los últimos dos para entrenamiento
            archivos_entrenamiento = [x for x in os.listdir(subcarpeta) if x.endswith('.wav')][:-2]

            for nombre_archivo in archivos_entrenamiento:
                ruta_archivo = os.path.join(subcarpeta, nombre_archivo)

                # Leer el archivo de audio
                frecuencia_muestreo, señal = wavfile.read(ruta_archivo)
                señal = señal / np.max(np.abs(señal))  # Normalizar la señal

                # Extraer características MFCC y sus deltas
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
                    caracteristicas_delta = delta(caracteristicas_mfcc, 2)

                # Combinar MFCC y deltas en un solo array
                caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))

                # Agregar las características al conjunto de datos
                if len(X) == 0:
                    X = caracteristicas
                else:
                    X = np.append(X, caracteristicas, axis=0)

            # Entrenar un modelo HMM con los datos recopilados
            modelo = ModeloHMM()
            modelo.entrenar(X)
            modelos_voz.append((modelo, etiqueta))

        # Guardar los modelos entrenados en un archivo para uso futuro
        with open(archivo_modelos, 'wb') as f:
            pickle.dump(modelos_voz, f)
        print("Modelos entrenados y guardados en archivo.")

    return modelos_voz


def ejecutar_pruebas(carpeta_entrada, modelos_voz):
    """
    Evalúa la precisión de los modelos HMM utilizando datos de prueba.
    
    Args:
        carpeta_entrada (str): Ruta del dataset con subcarpetas para cada etiqueta.
        modelos_voz (list): Lista de tuplas (modelo, etiqueta).
    """
    etiquetas_originales = []  # Etiquetas reales
    etiquetas_predichas = []  # Etiquetas predichas por los modelos

    for nombre_directorio in os.listdir(carpeta_entrada):
        subcarpeta = os.path.join(carpeta_entrada, nombre_directorio)
        if not os.path.isdir(subcarpeta):
            continue

        etiqueta_original = nombre_directorio

        # Usar los últimos dos archivos de cada carpeta para prueba
        archivos_prueba = [x for x in os.listdir(subcarpeta) if x.endswith('.wav')][-2:]
        for archivo_prueba in archivos_prueba:
            ruta_prueba = os.path.join(subcarpeta, archivo_prueba)

            # Leer el archivo de audio
            frecuencia_muestreo, señal = wavfile.read(ruta_prueba)
            señal = señal / np.max(np.abs(señal))  # Normalizar la señal

            # Extraer características MFCC y sus deltas
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
                caracteristicas_delta = delta(caracteristicas_mfcc, 2)

            caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))

            # Clasificar el audio comparando con cada modelo
            mejor_puntuacion = -float('inf')
            etiqueta_predicha = None

            for modelo, etiqueta in modelos_voz:
                puntuacion = modelo.calcular_puntuacion(caracteristicas)
                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    etiqueta_predicha = etiqueta

            etiquetas_originales.append(etiqueta_original)
            etiquetas_predichas.append(etiqueta_predicha)

            print(f"Original: {etiqueta_original}, Predicha: {etiqueta_predicha}")

    # Calcular y mostrar la precisión global
    precision = accuracy_score(etiquetas_originales, etiquetas_predichas)
    print(f"\nPrecisión total: {precision * 100:.2f}%")


def clasificar_audio_microfono(modelos_voz, duracion=3, frecuencia_muestreo=16000):
    """
    Graba audio desde el micrófono y lo clasifica.
    
    Args:
        modelos_voz (list): Lista de tuplas (modelo, etiqueta).
        duracion (int): Duración de la grabación en segundos.
        frecuencia_muestreo (int): Frecuencia de muestreo en Hz.
    """
    print("Grabando desde el micrófono...")
    # Grabar audio
    señal = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='float32')
    sd.wait()
    señal = señal.flatten()  # Aplanar señal en 1D

    señal = señal / np.max(np.abs(señal))  # Normalizar la señal

    # Extraer características MFCC y sus deltas
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
        caracteristicas_delta = delta(caracteristicas_mfcc, 2)

    caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))

    # Clasificar la señal grabada
    mejor_puntuacion = -float('inf')
    etiqueta_predicha = None

    for modelo, etiqueta in modelos_voz:
        puntuacion = modelo.calcular_puntuacion(caracteristicas)
        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            etiqueta_predicha = etiqueta

    print(f"Predicción para el audio grabado: {etiqueta_predicha}")


class ModeloHMM:
    """
    Clase para entrenar y usar Modelos Ocultos de Markov (HMM).
    """

    def __init__(self, num_componentes=8, num_iteraciones=2000):
        """
        Inicializa un modelo HMM con parámetros predeterminados.
        
        Args:
            num_componentes (int): Número de estados ocultos.
            num_iteraciones (int): Número máximo de iteraciones para el entrenamiento.
        """
        self.num_componentes = num_componentes
        self.num_iteraciones = num_iteraciones
        self.tipo_covarianza = 'diag'
        self.modelo = hmm.GaussianHMM(n_components=self.num_componentes, covariance_type=self.tipo_covarianza, n_iter=self.num_iteraciones)

    def entrenar(self, datos_entrenamiento):
        """
        Entrena el modelo HMM con los datos proporcionados.
        
        Args:
            datos_entrenamiento (np.array): Características MFCC y deltas para entrenamiento.
        """
        np.seterr(all='ignore')  # Ignorar advertencias numéricas
        self.modelo.fit(datos_entrenamiento)

    def calcular_puntuacion(self, datos_entrada):
        """
        Calcula la puntuación del modelo para un conjunto de datos.
        
        Args:
            datos_entrada (np.array): Características MFCC y deltas de entrada.
        
        Returns:
            float: Puntuación del modelo para los datos.
        """
        return self.modelo.score(datos_entrada)

def clasificar_audio_archivo(modelos_voz, ruta_archivo):
    """Clasifica un archivo de audio en formato .wav"""
    if not os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} no existe.")
        return

    print(f"Clasificando el archivo de audio: {ruta_archivo}")

    # Leer archivo de audio
    frecuencia_muestreo, señal = wavfile.read(ruta_archivo)
    señal = señal / np.max(np.abs(señal))  # Normalizar la señal

    # Extraer características MFCC
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo, nfft=2048)
        caracteristicas_delta = delta(caracteristicas_mfcc, 2)


    caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))

    # Clasificar la señal del archivo
    mejor_puntuacion = -float('inf')
    etiqueta_predicha = None

    for modelo, etiqueta in modelos_voz:
        puntuacion = modelo.calcular_puntuacion(caracteristicas)
        if puntuacion > mejor_puntuacion:
            mejor_puntuacion = puntuacion
            etiqueta_predicha = etiqueta

    print(f"Predicción para el archivo de audio: {etiqueta_predicha}")

def clasificar_audio_microfono_multiple(modelos_voz, veces=3, duracion=3, frecuencia_muestreo=16000):
    """
    Graba audio desde el micrófono varias veces y clasifica cada grabación.
    
    Args:
        modelos_voz (list): Lista de tuplas (modelo, etiqueta).
        veces (int): Número de veces que se grabará y clasificará.
        duracion (int): Duración de cada grabación en segundos.
        frecuencia_muestreo (int): Frecuencia de muestreo en Hz.
    """
    predicciones = []  # Lista para almacenar las predicciones
    
    for i in range(veces):
        print(f"\nGrabación {i + 1} de {veces}:")
        # Grabar audio
        señal = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='float32')
        sd.wait()
        señal = señal.flatten()  # Aplanar señal en 1D

        señal = señal / np.max(np.abs(señal))  # Normalizar la señal

        # Extraer características MFCC y sus deltas
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
            caracteristicas_delta = delta(caracteristicas_mfcc, 2)

        caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))

        # Clasificar la señal grabada
        mejor_puntuacion = -float('inf')
        etiqueta_predicha = None

        for modelo, etiqueta in modelos_voz:
            puntuacion = modelo.calcular_puntuacion(caracteristicas)
            if puntuacion > mejor_puntuacion:
                mejor_puntuacion = puntuacion
                etiqueta_predicha = etiqueta

        # print(f"Predicción para la grabación {i + 1}: {etiqueta_predicha}")
        predicciones.append(etiqueta_predicha)
    
    print("\nResultados Finales:")
    for idx, palabra in enumerate(predicciones, start=1):
        print(f"Grabación {idx}: {palabra}")

# Código principal
if __name__ == '__main__':
    carpeta_entrada = 'filtered_speech_commands'  # Cambia esto por la ruta de tu dataset
    archivo_modelos = 'modelos_hmm.pkl'

    modelos_voz = construir_modelos(carpeta_entrada, archivo_modelos)

    # Pruebas con el dataset
    ejecutar_pruebas(carpeta_entrada, modelos_voz)

    # Clasificar un audio desde el micrófono
    clasificar_audio_microfono_multiple(modelos_voz, veces=3)

    # Clasificar un archivo de audio .wav
    # ruta_audio_prueba = 'sixale.wav'  # Cambia esto por la ruta del archivo .wav
    # clasificar_audio_archivo(modelos_voz, ruta_audio_prueba)
    # ruta_audio_prueba = 'sevenale.wav'  # Cambia esto por la ruta del archivo .wav
    # clasificar_audio_archivo(modelos_voz, ruta_audio_prueba)
    # ruta_audio_prueba = 'twoalfredo.wav'  # Cambia esto por la ruta del archivo .wav
    # clasificar_audio_archivo(modelos_voz, ruta_audio_prueba)
    # ruta_audio_prueba = 'learnalfredo.wav'  # Cambia esto por la ruta del archivo .wav
    # clasificar_audio_archivo(modelos_voz, ruta_audio_prueba)

