import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from processor import align_images, euclidian_distance, IMAGE_DIMENSIONS
from db import CompareResult, get_db 

PRODUCTS_DIR = Path("products")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

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
    output_image: str


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
def compare(req: CompareRequest, db: Session = Depends(get_db)):
    # Carrega imagens
    img1_bgr = _load_bgr_image(req.product_1)
    img2_bgr = _load_bgr_image(req.product_2)

    # Realiza pré-processamento nas imagens
    img1, img2 = _preprocess(img1_bgr, img2_bgr)

    # Calcula distância e verifica similaridade com base no threshold
    distance = euclidian_distance(img1, img2)
    same = distance < req.threshold

    # Gera Imagem Concatenada
    concat_img = np.concatenate([img1, img2], axis=1)
    
    # Gera um nome único para não sobrescrever
    output_filename = f"compare_{uuid.uuid4().hex}.png"
    output_path = OUTPUTS_DIR / output_filename
    
    # Salva no disco
    cv2.imwrite(str(output_path), concat_img)

    # Salva no Banco de Dados
    db_record = CompareResult(
        product_1_path=str(_safe_product_path(req.product_1)),
        product_2_path=str(_safe_product_path(req.product_2)),
        output_image_path=str(output_path),
        distance=distance,
        is_same=same
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return CompareResponse(
        product_1=req.product_1,
        product_2=req.product_2,
        distance=distance,
        threshold=req.threshold,
        same_product=same,
        output_image=str(output_path)
    )

@app.get("/history")
def list_history(db: Session = Depends(get_db)):
    """
    Lista todas as imagens salvas anteriormente no banco de dados.
    """
    records = db.query(CompareResult).all()
    return records