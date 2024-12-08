import os
import warnings
import pickle
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from python_speech_features import mfcc, delta
from hmmlearn import hmm

# Helper function: Segment audio based on silence
def segment_audio(señal, frecuencia_muestreo, threshold=0.02, min_silence_duration=0.2):
    """
    Segments audio into chunks based on energy levels.
    
    Args:
        señal (np.array): Audio signal (1D).
        frecuencia_muestreo (int): Sampling frequency.
        threshold (float): Energy threshold for silence detection.
        min_silence_duration (float): Minimum silence duration in seconds.
    
    Returns:
        list: List of segmented audio signals (numpy arrays).
    """
    energy = np.abs(señal)
    silence_frames = energy < threshold
    silence_samples = int(min_silence_duration * frecuencia_muestreo)
    silence_indices = np.where(silence_frames)[0]
    split_indices = find_peaks(np.diff(silence_indices), height=silence_samples)[0]
    
    segments = []
    start_idx = 0
    for idx in split_indices:
        end_idx = silence_indices[idx]
        segments.append(señal[start_idx:end_idx])
        start_idx = end_idx
    segments.append(señal[start_idx:])  # Append the last segment
    
    return [seg for seg in segments if len(seg) > 0]

# Class for HMM model
class ModeloHMM:
    def __init__(self, num_componentes=8, num_iteraciones=2000):
        self.modelo = hmm.GaussianHMM(n_components=num_componentes, covariance_type='diag', n_iter=num_iteraciones)

    def entrenar(self, datos_entrenamiento):
        np.seterr(all='ignore')  # Ignore numerical warnings
        self.modelo.fit(datos_entrenamiento)

    def calcular_puntuacion(self, datos_entrada):
        return self.modelo.score(datos_entrada)

# Function to build or load HMM models
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
                X = np.vstack((X, caracteristicas)) if X.size else caracteristicas
            modelo = ModeloHMM()
            modelo.entrenar(X)
            modelos_voz.append((modelo, etiqueta))

        with open(archivo_modelos, 'wb') as f:
            pickle.dump(modelos_voz, f)
        print("Modelos entrenados y guardados en archivo.")
    return modelos_voz

def clasificar_secuencia_audio(modelos_voz, señal, frecuencia_muestreo):
    """
    Classifies a sequence of words from an audio signal.

    Args:
        modelos_voz (list): List of tuples (HMM model, label).
        señal (np.array): Audio signal (1D).
        frecuencia_muestreo (int): Sampling frequency.

    Returns:
        list: Predicted sequence of words.
    """
    segmentos = segment_audio(señal, frecuencia_muestreo)
    print(f"Number of segments detected: {len(segmentos)}")

    secuencia_predicha = []
    for i, segmento in enumerate(segmentos):
        print(f"Processing segment {i+1}...")
        if len(segmento) == 0:
            print(f"Segment {i+1} is empty, skipping.")
            continue

        # Normalize the segment
        segmento = segmento / np.max(np.abs(segmento))

        # Extract features
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            caracteristicas_mfcc = mfcc(segmento, frecuencia_muestreo)
            caracteristicas_delta = delta(caracteristicas_mfcc, 2)

        caracteristicas = np.hstack((caracteristicas_mfcc, caracteristicas_delta))
        print(f"Features shape for segment {i+1}: {caracteristicas.shape}")

        if caracteristicas.shape[0] == 0:
            print(f"Warning: No features extracted for segment {i+1}, skipping.")
            continue

        # Classify the segment
        mejor_puntuacion = -float('inf')
        etiqueta_predicha = None
        for modelo, etiqueta in modelos_voz:
            try:
                puntuacion = modelo.calcular_puntuacion(caracteristicas)
                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    etiqueta_predicha = etiqueta
            except Exception as e:
                print(f"Error while scoring segment {i+1}: {e}")
                continue

        if etiqueta_predicha:
            print(f"Segment {i+1} classified as: {etiqueta_predicha}")
            secuencia_predicha.append(etiqueta_predicha)
        else:
            print(f"Segment {i+1} could not be classified.")

    return secuencia_predicha


# Function to classify audio from microphone
def clasificar_audio_microfono_secuencia(modelos_voz, duracion=3, frecuencia_muestreo=16000):
    print("Grabando desde el micrófono...")
    señal = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='float32')
    sd.wait()
    señal = señal.flatten()
    secuencia_predicha = clasificar_secuencia_audio(modelos_voz, señal, frecuencia_muestreo)
    print(f"Predicción para el audio grabado: {' '.join(secuencia_predicha)}")

# Function to classify audio from file
def clasificar_audio_archivo_secuencia(modelos_voz, ruta_archivo):
    if not os.path.exists(ruta_archivo):
        print(f"El archivo {ruta_archivo} no existe.")
        return
    frecuencia_muestreo, señal = wavfile.read(ruta_archivo)
    señal = señal / np.max(np.abs(señal))
    secuencia_predicha = clasificar_secuencia_audio(modelos_voz, señal, frecuencia_muestreo)
    print(f"Predicción para el archivo de audio: {' '.join(secuencia_predicha)}")

# Main code
if __name__ == '__main__':
    carpeta_entrada = 'filtered_speech_commands'  # Replace with your dataset path
    archivo_modelos = 'modelos_hmm.pkl'

    # Build or load models
    modelos_voz = construir_modelos(carpeta_entrada, archivo_modelos)

    # Test with microphone
    clasificar_audio_microfono_secuencia(modelos_voz)

    # Test with audio file
    clasificar_audio_archivo_secuencia(modelos_voz, 'sevenale.wav')
