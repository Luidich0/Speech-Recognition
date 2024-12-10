import os
import pickle
import numpy as np
from python_speech_features import mfcc, delta
from hmmlearn import hmm
from collections import Counter
from nltk.util import ngrams
from scipy.io import wavfile
import sounddevice as sd

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
        frecuencia = modelo_ngramas.get(ngrama, 1e-3)  # Probabilidad mínima ajustada
        probabilidad *= frecuencia
    return probabilidad

# Clase HMM
class ModeloHMM:
    def __init__(self, num_componentes=8, num_iteraciones=2000):
        self.modelo = hmm.GaussianHMM(n_components=num_componentes, covariance_type='diag', n_iter=num_iteraciones)

    def entrenar(self, datos_entrenamiento):
        np.seterr(all='ignore')
        self.modelo.fit(datos_entrenamiento)

    def calcular_puntuacion(self, datos_entrada):
        return self.modelo.score(datos_entrada)

# Clase Decodificador
class DecodificadorHMM:
    def __init__(self, modelos_hmm, modelo_trigramas, lexico, umbral_puntuacion=-100):
        self.modelos_hmm = modelos_hmm
        self.modelo_trigramas = modelo_trigramas
        self.lexico = lexico
        self.umbral_puntuacion = umbral_puntuacion

    def calcular_probabilidad_transicion(self, prev, curr, next_):
        trigrama = (prev, curr, next_)
        return self.modelo_trigramas.get(trigrama, 1e-3)

    def viterbi_decodificacion(self, caracteristicas):
        estados = [etiqueta for _, etiqueta in self.modelos_hmm]
        n = len(caracteristicas)
        dp = np.full((n, len(estados)), float('-inf'))
        backpointer = np.zeros((n, len(estados)), dtype=int)

        # Inicialización
        for j, (modelo, estado) in enumerate(self.modelos_hmm):
            try:
                dp[0][j] = modelo.calcular_puntuacion(caracteristicas[0].reshape(1, -1))
                if dp[0][j] < self.umbral_puntuacion:
                    dp[0][j] = float('-inf')
            except:
                dp[0][j] = float('-inf')

        # Recursión
        for t in range(1, n):
            for j, (modelo, estado) in enumerate(self.modelos_hmm):
                max_prob, max_state = max(
                    (dp[t-1][k] + np.log(self.calcular_probabilidad_transicion(None, estados[k], estado)), k)
                    for k in range(len(estados))
                )
                penalizacion = -0.5 if estados[max_state] == estado else 0
                dp[t][j] = max_prob + modelo.calcular_puntuacion(caracteristicas[t].reshape(1, -1)) + penalizacion
                if dp[t][j] < self.umbral_puntuacion:
                    dp[t][j] = float('-inf')
                backpointer[t][j] = max_state

        # Backtracking
        mejor_estado = np.argmax(dp[-1])
        secuencia = []
        for t in range(n-1, -1, -1):
            secuencia.insert(0, estados[mejor_estado])
            mejor_estado = backpointer[t][mejor_estado]

        # Eliminar repeticiones consecutivas
        secuencia = [key for i, key in enumerate(secuencia) if i == 0 or key != secuencia[i-1]]

        return secuencia

def preparar_conjunto_prueba(carpeta_prueba):
    conjunto_prueba = []
    for etiqueta in os.listdir(carpeta_prueba):
        subcarpeta = os.path.join(carpeta_prueba, etiqueta)
        if not os.path.isdir(subcarpeta):
            continue

        for archivo in os.listdir(subcarpeta):
            if not archivo.endswith('.wav'):
                continue
            frecuencia_muestreo, señal = wavfile.read(os.path.join(subcarpeta, archivo))
            señal = señal / np.max(np.abs(señal))  # Normalización
            caracteristicas = extraer_caracteristicas(señal, frecuencia_muestreo)
            conjunto_prueba.append((caracteristicas, etiqueta))
    
    return conjunto_prueba

def evaluar_modelo(modelos_hmm, conjunto_prueba):
    correctos = 0
    total = 0

    for caracteristicas, etiqueta_real in conjunto_prueba:
        mejor_etiqueta = None
        mejor_puntuacion = float('-inf')

        for modelo, etiqueta in modelos_hmm:
            try:
                puntuacion = modelo.calcular_puntuacion(caracteristicas)
                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    mejor_etiqueta = etiqueta
            except:
                continue

        if mejor_etiqueta == etiqueta_real:
            correctos += 1
        
        total += 1

    if total == 0:
        return 0.0  # Evitar división por cero
    
    return correctos / total

# Extraer características MFCC con normalización
def extraer_caracteristicas(señal, frecuencia_muestreo):
    # Eliminar silencio basado en un umbral de amplitud
    umbral_amplitud = 0.01
    señal = señal[np.abs(señal) > umbral_amplitud]

    if len(señal) == 0:
        return np.empty((0, 13))

    mfcc_features = mfcc(señal, frecuencia_muestreo, winlen=0.025, nfft=2048)
    delta_features = delta(mfcc_features, 2)
    caracteristicas = np.hstack((mfcc_features, delta_features))
    return (caracteristicas - np.mean(caracteristicas, axis=0)) / (np.std(caracteristicas, axis=0) + 1e-6)

# Construir modelos HMM para cada palabra
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
            señal = señal / np.max(np.abs(señal))  # Normalización
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
    return frecuencia_muestreo, señal.flatten()

# Reconocimiento desde el micrófono
def reconocimiento_por_microfono(duracion, decodificador):
    frecuencia_muestreo, señal = grabar_audio(duracion)
    señal = señal / np.max(np.abs(señal))  # Normalización
    caracteristicas = extraer_caracteristicas(señal, frecuencia_muestreo)
    if caracteristicas.size == 0:
        print("Audio vacío o sin contenido útil.")
        return
    secuencia = decodificador.viterbi_decodificacion(caracteristicas)
    print(f"Secuencia reconocida desde micrófono: {' '.join(secuencia)}")

# Código principal
if __name__ == '__main__':
    corpus = [
        "six six six",
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

    carpeta_entrada = 'filtered_speech_commands'
    archivo_modelos = 'modelos_hmm_continuo.pkl'
    
    modelos_hmm = construir_modelos(carpeta_entrada, archivo_modelos)
    #conjunto_prueba = preparar_conjunto_prueba(carpeta_entrada)

    # Evaluar la precisión
    # if conjunto_prueba:
    #     precision = evaluar_modelo(modelos_hmm, conjunto_prueba)
    #     print(f"Precisión del modelo en el conjunto de prueba: {precision:.2f}")
    # else:
    #     print("No se encontraron datos de prueba.")
    decodificador = DecodificadorHMM(modelos_hmm, modelo_trigramas, lexico)

    # Prueba con reconocimiento por micrófono
    print("Reconociendo desde micrófono...")

    reconocimiento_por_microfono(duracion=3, decodificador=decodificador)
