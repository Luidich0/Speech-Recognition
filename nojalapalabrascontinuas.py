import os
import warnings
import pickle
import numpy as np
import itertools
from scipy.io import wavfile
from python_speech_features import mfcc, delta
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
import sounddevice as sd


# Modelo HMM
class ModeloHMM:
    def __init__(self, num_componentes=8, num_iteraciones=2000):
        self.num_componentes = num_componentes
        self.num_iteraciones = num_iteraciones
        self.tipo_covarianza = 'diag'
        self.modelo = hmm.GaussianHMM(n_components=self.num_componentes, covariance_type=self.tipo_covarianza, n_iter=self.num_iteraciones)

    def entrenar(self, datos_entrenamiento):
        np.seterr(all='ignore')  # Ignorar advertencias numéricas
        self.modelo.fit(datos_entrenamiento)

    def calcular_puntuacion(self, datos_entrada):
        return self.modelo.score(datos_entrada)

# Construir modelos acústicos (HMM)
def construir_modelos(carpeta_entrada, archivo_modelos):
    if os.path.exists(archivo_modelos):
        print("Cargando modelos desde archivo...")
        with open(archivo_modelos, 'rb') as f:
            modelos_voz = pickle.load(f)
    else:
        print("Entrenando modelos desde el dataset...")
        modelos_voz = []
        for nombre_directorio in os.listdir(carpeta_entrada):
            subcarpeta = os.path.join(carpeta_entrada, nombre_directorio)
            if not os.path.isdir(subcarpeta):
                continue

            etiqueta = nombre_directorio
            X = np.array([])

            archivos_entrenamiento = [x for x in os.listdir(subcarpeta) if x.endswith('.wav')][:-2]
            for nombre_archivo in archivos_entrenamiento:
                ruta_archivo = os.path.join(subcarpeta, nombre_archivo)
                frecuencia_muestreo, señal = wavfile.read(ruta_archivo)
                señal = señal / np.max(np.abs(señal))

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
                    caracteristicas_delta = delta(caracteristicas_mfcc, 2)

                caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))
                X = caracteristicas if len(X) == 0 else np.append(X, caracteristicas, axis=0)

            modelo = ModeloHMM()
            modelo.entrenar(X)
            modelos_voz.append((modelo, etiqueta))

        with open(archivo_modelos, 'wb') as f:
            pickle.dump(modelos_voz, f)
        print("Modelos entrenados y guardados en archivo.")

    return modelos_voz

# Construir modelo de lenguaje (n-gramas) o cargarlo
def construir_o_cargar_modelo_lenguaje(archivo_modelo_lenguaje, corpus_texto=None, n=2):
    if os.path.exists(archivo_modelo_lenguaje):
        print("Cargando modelo de lenguaje desde archivo...")
        with open(archivo_modelo_lenguaje, 'rb') as f:
            modelo_lenguaje = pickle.load(f)
    else:
        print("Creando modelo de lenguaje...")
        train_data, vocab = padded_everygram_pipeline(n, corpus_texto)
        modelo_lenguaje = MLE(n)
        modelo_lenguaje.fit(train_data, vocab)

        with open(archivo_modelo_lenguaje, 'wb') as f:
            pickle.dump(modelo_lenguaje, f)
        print("Modelo de lenguaje creado y guardado en archivo.")
    
    return modelo_lenguaje

# Decodificador para clasificar con modelo acústico y de lenguaje
def decodificar(modelos_voz, modelo_lenguaje, caracteristicas, palabras_lexico):
    puntuaciones = []
    for palabra in palabras_lexico:
        for modelo, etiqueta in modelos_voz:
            if etiqueta == palabra:
                puntuacion = modelo.calcular_puntuacion(caracteristicas)
                puntuaciones.append((palabra, puntuacion))

    mejores_palabras = sorted(puntuaciones, key=lambda x: x[1], reverse=True)[:len(palabras_lexico)]
    secuencia_palabras = [palabra for palabra, _ in mejores_palabras]

    mejor_secuencia = max(itertools.permutations(secuencia_palabras, len(secuencia_palabras)), 
                          key=lambda seq: modelo_lenguaje.score(' '.join(seq)))

    return ' '.join(mejor_secuencia)

# Evaluar precisión de los modelos HMM
def ejecutar_pruebas(carpeta_entrada, modelos_voz, modelo_lenguaje):
    etiquetas_originales = []
    etiquetas_predichas = []

    for nombre_directorio in os.listdir(carpeta_entrada):
        subcarpeta = os.path.join(carpeta_entrada, nombre_directorio)
        if not os.path.isdir(subcarpeta):
            continue

        etiqueta_original = nombre_directorio
        archivos_prueba = [x for x in os.listdir(subcarpeta) if x.endswith('.wav')][-2:]
        for archivo_prueba in archivos_prueba:
            ruta_prueba = os.path.join(subcarpeta, archivo_prueba)
            frecuencia_muestreo, señal = wavfile.read(ruta_prueba)
            señal = señal / np.max(np.abs(señal))

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
                caracteristicas_delta = delta(caracteristicas_mfcc, 2)

            caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))
            palabras_lexico = [etiqueta for _, etiqueta in modelos_voz]
            etiqueta_predicha = decodificar(modelos_voz, modelo_lenguaje, caracteristicas, palabras_lexico)

            etiquetas_originales.append(etiqueta_original)
            etiquetas_predichas.append(etiqueta_predicha)

            print(f"Original: {etiqueta_original}, Predicha: {etiqueta_predicha}")

    precision = accuracy_score(etiquetas_originales, etiquetas_predichas)
    print(f"\nPrecisión total: {precision * 100:.2f}%")

# Clasificar audio desde un archivo
def clasificar_audio_archivo(modelos_voz, modelo_lenguaje, ruta_archivo):
    if not os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} no existe.")
        return

    frecuencia_muestreo, señal = wavfile.read(ruta_archivo)
    señal = señal / np.max(np.abs(señal))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
        caracteristicas_delta = delta(caracteristicas_mfcc, 2)

    caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))
    palabras_lexico = [etiqueta for _, etiqueta in modelos_voz]
    etiqueta_predicha = decodificar(modelos_voz, modelo_lenguaje, caracteristicas, palabras_lexico)

    print(f"Predicción para el archivo de audio: {etiqueta_predicha}")

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

if __name__ == '__main__':
    carpeta_entrada = 'filtered_speech_commands'
    archivo_modelos = 'modelos_hmm.pkl'
    archivo_modelo_lenguaje = 'modelo_lenguaje.pkl'

    # Crear o cargar modelos acústicos
    modelos_voz = construir_modelos(carpeta_entrada, archivo_modelos)

    # Crear o cargar modelo de lenguaje
    corpus_ejemplo = [
        ['six', 'two'],
        ['aprende', 'a', 'programar'],
        ['python', 'es', 'increíble'],
        ['este', 'es', 'un', 'ejemplo']
    ]
    modelo_lenguaje = construir_o_cargar_modelo_lenguaje(archivo_modelo_lenguaje, corpus_texto=corpus_ejemplo, n=3)

    # Evaluar precisión
    # ejecutar_pruebas(carpeta_entrada, modelos_voz, modelo_lenguaje)

    # Clasificar un archivo de audio
    # ruta_audio_prueba = 'sixale.wav'  # Cambia por la ruta del archivo
    # clasificar_audio_archivo(modelos_voz, modelo_lenguaje, ruta_audio_prueba)

    clasificar_audio_microfono(modelos_voz)
