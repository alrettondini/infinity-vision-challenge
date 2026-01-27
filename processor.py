import cv2
import numpy as np


def transform(img: np.ndarray):
    """
    Transformações de imagem:
        - Converte imagem para grayscale
        - Redimensiona imagem para 256x256
    """
    # Converte a imagem colorida para grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Redimensiona imagem para 256x256
    grayscale_resized = cv2.resize(grayscale, (256, 256), interpolation=cv2.INTER_AREA)
    
    return grayscale_resized

def euclidian_distance(img1: np.ndarray, img2: np.ndarray):
    """
    Calcula a distância euclidiana entre duas imagens
    """

    # Converte imagens para vetores
    img1_vector = img1.astype(np.float32).ravel()
    img2_vector = img2.astype(np.float32).ravel()

    # Calcula a norma da diferença entre os vetores
    return float(np.linalg.norm(img1_vector - img2_vector))