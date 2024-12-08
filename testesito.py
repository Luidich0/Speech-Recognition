import os
import pickle
import numpy as np
import sounddevice as sd
from python_speech_features import mfcc, delta
from hmmlearn import hmm
from nltk.util import ngrams
from collections import Counter
from scipy.io import wavfile
import matplotlib.pyplot as plt

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

# Decodificar fonemas usando el léxico
def decodificar_fonemas(fonemas, lexico):
    for palabra, transcripcion in lexico.items():
        if fonemas == transcripcion:
            return palabra
    return None

# Extraer características MFCC y Delta
def extraer_caracteristicas(señal, frecuencia_muestreo):
    mfcc_features = mfcc(señal, frecuencia_muestreo)
    delta_features = delta(mfcc_features, 2)
    caracteristicas = np.hstack((mfcc_features, delta_features))
    return caracteristicas

# Construir modelos HMM
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
            caracteristicas = extraer_caracteristicas(señal, frecuencia_muestreo)
            X = np.vstack((X, caracteristicas)) if X.size else caracteristicas

        modelo = ModeloHMM()
        modelo.entrenar(X)
        modelos_voz.append((modelo, etiqueta))

    with open(archivo_modelos, 'wb') as f:
        pickle.dump(modelos_voz, f)
    return modelos_voz

# Función para grabar audio
def grabar_audio(duracion, frecuencia_muestreo=16000):
    print("Grabando... Hable ahora.")
    señal = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='int16')
    sd.wait()
    print("Grabación completa.")
    wavfile.write("audio_prueba.wav", frecuencia_muestreo, señal)  # Guarda la grabación para inspección
    return frecuencia_muestreo, señal.flatten()

# Decodificar con Viterbi
def decodificar_viterbi(señal, frecuencia_muestreo, modelos_hmm, lexico):
    palabras = []
    caracteristicas = extraer_caracteristicas(señal, frecuencia_muestreo)
    plt.imshow(caracteristicas.T, aspect='auto', origin='lower')
    plt.title("MFCC Features")
    #plt.show()  # Visualiza las características MFCC
    for modelo, etiqueta in modelos_hmm:
        try:
            puntuacion = modelo.calcular_puntuacion(caracteristicas)
            palabras.append((puntuacion, etiqueta))
            print(f"Puntuación para '{etiqueta}': {puntuacion}")
        except Exception as e:
            print(f"Error con modelo {etiqueta}: {e}")
    palabras.sort(reverse=True, key=lambda x: x[0])
    if palabras:
        mejor_etiqueta = palabras[0][1]
        return decodificar_fonemas(lexico.get(mejor_etiqueta, []), lexico)
    return None

# Lista acumuladora de palabras reconocidas
frase_acumulada = []

# Reconocimiento continuo actualizado
def reconocimiento_continuo(duracion, modelos_hmm, modelo_trigramas, lexico):
    global frase_acumulada  # Usamos una variable global para acumular palabras
    frecuencia_muestreo, señal = grabar_audio(duracion)
    if np.max(np.abs(señal)) > 0:
        señal = señal / np.max(np.abs(señal))  # Normaliza el audio
    else:
        print("Señal vacía o inválida.")
        return
    
    palabra = decodificar_viterbi(señal, frecuencia_muestreo, modelos_hmm, lexico)
    if palabra:
        frase_acumulada.append(palabra)
        print(f"Palabra reconocida: {palabra}")
    else:
        print("No se reconoció ninguna palabra.")
        return

    # Solo calculamos trigramas si hay al menos 3 palabras acumuladas
    if len(frase_acumulada) >= 3:
        frase = " ".join(frase_acumulada[-3:])  # Usamos las últimas 3 palabras
        probabilidad = calcular_probabilidad_ngramas(frase, modelo_trigramas, 3)
        print(f"Transcripción acumulada: '{frase}'")
        print(f"Probabilidad basada en trigramas: {probabilidad}")
    else:
        print(f"Frase acumulada hasta ahora: {' '.join(frase_acumulada)}")


# Función para procesar un archivo .wav y realizar el reconocimiento
def reconocimiento_por_archivo(ruta_archivo, modelos_hmm, modelo_trigramas, lexico):
    if not os.path.exists(ruta_archivo):
        print(f"Archivo no encontrado: {ruta_archivo}")
        return

    # Leer el archivo .wav
    frecuencia_muestreo, señal = wavfile.read(ruta_archivo)
    if np.max(np.abs(señal)) > 0:
        señal = señal / np.max(np.abs(señal))  # Normaliza el audio
    else:
        print("Archivo de audio vacío o inválido.")
        return

    # Decodificar usando Viterbi y el léxico
    palabra = decodificar_viterbi_archivo(señal, frecuencia_muestreo, modelos_hmm, lexico)
    frase = palabra if palabra else ""
    
    # Evaluar la probabilidad de la frase en el modelo de trigramas
    if frase:
        probabilidad = calcular_probabilidad_ngramas(frase, modelo_trigramas, 3)
        print(f"Transcripción desde archivo: '{frase}'")
        print(f"Probabilidad basada en trigramas: {probabilidad}")
    else:
        print("Frase desconocida o vacía en el archivo.")

# Extraer características MFCC y Delta con ajustes
def extraer_caracteristicas_archivo(señal, frecuencia_muestreo):
    mfcc_features = mfcc(señal, frecuencia_muestreo, winlen=0.02, nfft=2048)
    delta_features = delta(mfcc_features, 2)
    caracteristicas = np.hstack((mfcc_features, delta_features))
    return caracteristicas

# Decodificar con Viterbi (con normalización de puntuaciones)
def decodificar_viterbi_archivo(señal, frecuencia_muestreo, modelos_hmm, lexico):
    palabras = []
    caracteristicas = extraer_caracteristicas_archivo(señal, frecuencia_muestreo)
    for modelo, etiqueta in modelos_hmm:
        try:
            puntuacion = modelo.calcular_puntuacion(caracteristicas)
            palabras.append((puntuacion, etiqueta))
            print(f"Puntuación para '{etiqueta}': {puntuacion}")
        except Exception as e:
            print(f"Error con modelo {etiqueta}: {e}")
    
    # Normalizar puntuaciones
    puntuaciones_normalizadas = [(puntuacion / len(caracteristicas), etiqueta) for puntuacion, etiqueta in palabras]
    puntuaciones_normalizadas.sort(reverse=True, key=lambda x: x[0])
    if puntuaciones_normalizadas:
        mejor_etiqueta = puntuaciones_normalizadas[0][1]
        return decodificar_fonemas(lexico.get(mejor_etiqueta, []), lexico)
    return None


# Código principal
if __name__ == '__main__':
    # Corpus para trigramas
    corpus = [
        "two two two",
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
    reconocimiento_continuo(duracion=5, modelos_hmm=modelos_hmm, modelo_trigramas=modelo_trigramas, lexico=lexico)

    archivo_prueba = 'sixale.wav'  # Cambia esto por la ruta de tu archivo .wav
    #reconocimiento_por_archivo(archivo_prueba, modelos_hmm, modelo_trigramas, lexico)