import json
from pathlib import Path
import sys

import cv2
import numpy as np
from processor import transform, euclidian_distance

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

    # Aplica transformações
    img1 = transform(img1)
    img2 = transform(img2)

    # Calcula distância euclidiana
    distance = euclidian_distance(img1, img2)

    print(f"Distância: {distance:.4f}")

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
