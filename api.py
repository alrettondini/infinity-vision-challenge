from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from processor import align_images, euclidian_distance, IMAGE_DIMENSIONS

PRODUCTS_DIR = Path("products")

app = FastAPI(title="Product Compare API", version="1.0.0")


class ProductItem(BaseModel):
    id: str
    path: str


class CompareRequest(BaseModel):
    product_1: str
    product_2: str
    threshold: float = 0.8


class CompareResponse(BaseModel):
    product_1: str
    product_2: str
    distance: float
    threshold: float
    same_product: bool


def _safe_product_path(filename: str):
    """
    Garante que o usuário só consegue apontar para arquivos dentro de ./products.
    """
    candidate = (PRODUCTS_DIR / filename).resolve()
    base = PRODUCTS_DIR.resolve()
    if base not in candidate.parents and candidate != base:
        raise HTTPException(status_code=400, detail="Nome de arquivo inválido.")
    return candidate


def _load_bgr_image(filename: str):
    path = _safe_product_path(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Produto '{filename}' não encontrado em ./products.")
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(status_code=400, detail=f"Falha ao ler a imagem '{filename}'.")
    return img


def _preprocess(img1_bgr: np.ndarray, img2_bgr: np.ndarray):
    """
    Pré-processamento de imagens
    """
    # Aplica grayscale nas imagens
    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    # Alinha as imagens com ORB + Homografia
    img1_aligned = align_images(img1, img2)

    # Aplica resize nas imagens para 256x256
    img1 = cv2.resize(img1_aligned, IMAGE_DIMENSIONS, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, IMAGE_DIMENSIONS, interpolation=cv2.INTER_AREA)

    # Blur para redução de ruído
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

    return img1, img2


@app.get("/products", response_model=List[ProductItem])
def list_products():
    if not PRODUCTS_DIR.exists():
        return []

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    items: list[ProductItem] = []

    for p in sorted(PRODUCTS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in valid_ext:
            items.append(ProductItem(id=p.name, path=str(p)))

    return items


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    # Carrega imagens
    img1_bgr = _load_bgr_image(req.product_1)
    img2_bgr = _load_bgr_image(req.product_2)

    # Realiza pré-processamento nas imagens
    img1, img2 = _preprocess(img1_bgr, img2_bgr)

    # Calcula distância e verifica similaridade com base no threshold
    distance = euclidian_distance(img1, img2)
    same = distance < req.threshold

    return CompareResponse(
        product_1=req.product_1,
        product_2=req.product_2,
        distance=distance,
        threshold=req.threshold,
        same_product=same,
    )
