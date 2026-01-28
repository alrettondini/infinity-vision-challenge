import json
from pathlib import Path
import sys

import cv2
import numpy as np
from processor import euclidian_distance, align_images

IMAGE_DIMENSIONS = (256, 256)

def read_config(path: str):
    """
    Lê arquivo de configuração
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    
    return cfg

def main():
    # Lê caminho passado pelo terminal
    cfg = read_config(sys.argv[1])

    # Lê imagens fornecidas pela config
    img1 = cv2.imread(cfg["image_1_path"])
    img2 = cv2.imread(cfg["image_2_path"])

    # Aplica grayscale nas imagens
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Alinha as imagens com ORB + Homografia
    img1_aligned = align_images(img1, img2)

    # Aplica resize nas imagens para 256x256
    img1 = cv2.resize(img1_aligned, IMAGE_DIMENSIONS, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, IMAGE_DIMENSIONS, interpolation=cv2.INTER_AREA)

    # Blur para redução de ruído
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    # Calcula distância euclidiana
    distance = euclidian_distance(img1, img2)

    print(f"Distância: {distance:.4f}")

    threshold = cfg.get("threshold", 0.6)
    print(f"Threshold Definido:  {threshold:.4f}")

    if distance < threshold:
        print("\n>>> RESULTADO: MESMO PRODUTO <<<")
    else:
        print("\n>>> RESULTADO: PRODUTOS DIFERENTES <<<")

    # Caso não exista, cria diretório de saída
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Caminho para imagem final
    out_path = out_dir / cfg["output_filename"]

    # Concatena imagens transformadas
    concat = np.concatenate([img1, img2], axis=1)

    # Guarda imagens no caminho final
    cv2.imwrite(str(out_path), concat)

    print(f"Concatenação de imagens salva em: {out_path}")

if __name__ == "__main__":
    main()
