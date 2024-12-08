import os
import warnings
import pickle
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from python_speech_features import mfcc, delta
from hmmlearn import hmm
import webrtcvad
from nltk.util import ngrams
from collections import Counter
from scipy.io import wavfile


# Diccionario léxico
lexico = {
    "yes": ["y", "e", "s"],
    "two": ["t", "u"],
    "stop": ["s", "t", "o", "p"],
    "six": ["s", "i", "k", "s"],
    "right": ["r", "aɪ", "t"],
    "off": ["ɔ", "f"],
    "no": ["n", "oʊ"],
    "learn": ["l", "ɜ", "n"],
    "go": ["g", "oʊ"],
    "down": ["d", "aʊ", "n"],
    "up": ["ʌ", "p"],
    "seven": ["s", "ɛ", "v", "ə", "n"],
    "on": ["ɒ", "n"],
    "left": ["l", "ɛ", "f", "t"],
    "house": ["h", "aʊ", "s"]
}


# Clase HMM
class ModeloHMM:
    def __init__(self, num_componentes=8, num_iteraciones=2000):
        self.modelo = hmm.GaussianHMM(n_components=num_componentes, covariance_type='diag', n_iter=num_iteraciones)

    def entrenar(self, datos_entrenamiento):
        np.seterr(all='ignore')
        self.modelo.fit(datos_entrenamiento)

    def calcular_puntuacion(self, datos_entrada):
        return self.modelo.score(datos_entrada)


# Construcción del modelo de trigramas
def construir_ngramas(corpus, n=3):
    tokens = []
    for frase in corpus:
        tokens.extend(frase.split())
    return Counter(ngrams(tokens, n))


# Calcular probabilidad de una frase con el modelo de n-gramas
def calcular_probabilidad_ngramas(frase, modelo_ngramas, n=3):
    tokens = frase.split()
    probabilidad = 1.0
    for ngrama in ngrams(tokens, n):
        frecuencia = modelo_ngramas.get(ngrama, 1e-6)  # Usa una frecuencia mínima para evitar probabilidades cero
        probabilidad *= frecuencia
    return probabilidad


# Función para construir modelos HMM
def construir_modelos(carpeta_entrada, archivo_modelos):
    if os.path.exists(archivo_modelos):
        print("Cargando modelos desde archivo...")
        with open(archivo_modelos, 'rb') as f:
            return pickle.load(f)

    print("Entrenando modelos desde el dataset...")
    modelos_voz = []
    for etiqueta in os.listdir(carpeta_entrada):
        subcarpeta = os.path.join(carpeta_entrada, etiqueta)
        if not os.path.isdir(subcarpeta):
            continue

        X = np.array([])

        for archivo in os.listdir(subcarpeta):
            if not archivo.endswith('.wav'):
                continue
            frecuencia_muestreo, señal = wavfile.read(os.path.join(subcarpeta, archivo))
            señal = señal / np.max(np.abs(señal))
            caracteristicas_mfcc = mfcc(señal, frecuencia_muestreo)
            caracteristicas_delta = delta(caracteristicas_mfcc, 2)
            caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))
            X = np.append(X, caracteristicas, axis=0) if X.size else caracteristicas

        modelo = ModeloHMM()
        modelo.entrenar(X)
        modelos_voz.append((modelo, etiqueta))

    with open(archivo_modelos, 'wb') as f:
        pickle.dump(modelos_voz, f)
    return modelos_voz


# Función para segmentar audio usando VAD
def detectar_segmentos_voz(señal, frecuencia_muestreo, ventana_ms=30):
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    ventana_muestras = int(ventana_ms * frecuencia_muestreo / 1000)
    segmentos = []
    inicio = None

    for i in range(0, len(señal), ventana_muestras):
        ventana = señal[i:i + ventana_muestras]
        if len(ventana) < ventana_muestras:
            break
        if vad.is_speech(ventana.tobytes(), frecuencia_muestreo):
            if inicio is None:
                inicio = i
        elif inicio is not None:
            segmentos.append((inicio, i))
            inicio = None
    if inicio is not None:
        segmentos.append((inicio, len(señal)))
    return segmentos


# Función de decodificación con Viterbi
def decodificar_viterbi(segmentos, señal, modelos_hmm):
    palabras = []
    for inicio, fin in segmentos:
        mejor_puntuacion = -float('inf')
        mejor_etiqueta = None

        for modelo, etiqueta in modelos_hmm:
            try:
                puntuacion = modelo.calcular_puntuacion(mfcc(señal[inicio:fin], 16000))
                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    mejor_etiqueta = etiqueta
            except Exception:
                continue
        if mejor_etiqueta:
            palabras.append(mejor_etiqueta)
    return palabras

# Grabar audio desde el micrófono
def grabar_audio(duracion, frecuencia_muestreo=16000):
    print("Grabando... Hable ahora.")
    señal = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='int16')
    sd.wait()
    print("Grabación completa.")
    return frecuencia_muestreo, señal.flatten()

# Preprocesar audio para VAD
def preparar_audio_para_vad(señal, frecuencia_muestreo):
    ventana_muestras = int(30 * frecuencia_muestreo / 1000)  # Ventana de 30 ms
    if len(señal) % ventana_muestras != 0:
        # Ajusta el tamaño de la señal para que sea múltiplo de la ventana
        exceso = len(señal) % ventana_muestras
        señal = np.pad(señal, (0, ventana_muestras - exceso), mode='constant')
    return señal

# Reconocimiento continuo con trigramas
def reconocimiento_continuo(desde_microfono, modelos_hmm, modelo_trigramas, duracion=5):
    if desde_microfono:
        frecuencia_muestreo, señal = grabar_audio(duracion)
    else:
        frecuencia_muestreo, señal = wavfile.read('sixale.wav')

    señal = señal / np.max(np.abs(señal))  # Normalizar la señal
    print(f"Duración de la señal: {len(señal) / frecuencia_muestreo:.2f} segundos")

    señal = preparar_audio_para_vad(señal, frecuencia_muestreo)  # Prepara la señal para VAD
    segmentos = detectar_segmentos_voz(señal, frecuencia_muestreo)
    print(f"Segmentos detectados: {segmentos}")

    palabras = decodificar_viterbi(segmentos, señal, modelos_hmm)
    print(f"Palabras detectadas: {palabras}")

    frase = " ".join(palabras)
    print(f"Frase construida: '{frase}'")

    probabilidad = calcular_probabilidad_ngramas(frase, modelo_trigramas, 3)
    if probabilidad > 0:
        print(f"Transcripción: '{frase}'")
        print(f"Probabilidad basada en trigramas: {probabilidad}")
    else:
        print("Frase desconocida o baja probabilidad.")

# Código principal
if __name__ == '__main__':
    # Corpus para trigramas
    corpus = [
        "yes go left",
        "no stop right",
        "go down left",
        "go up right",
        "house on left",
        "off learn right",
        "go down up",
        "stop up left",
        "right go down"
    ]
    modelo_trigramas = construir_ngramas(corpus, 3)

    # Configuración HMM
    carpeta_entrada = 'filtered_speech_commands'  # Ruta del dataset
    archivo_modelos = 'modelos_hmm_continuo.pkl'
    modelos_hmm = construir_modelos(carpeta_entrada, archivo_modelos)

    # Reconocimiento desde micrófono
    reconocimiento_continuo(True, modelos_hmm, modelo_trigramas, duracion=5)
