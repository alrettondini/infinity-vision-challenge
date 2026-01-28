import cv2
import numpy as np

IMAGE_DIMENSIONS = (256, 256)

def hog_feature(img: np.ndarray):
    """
    Extrai um vetor de características baseado em bordas e formas (HOG)
    """
    # Configura o HOG com dimensões da imagem e valores padrão
    hog = cv2.HOGDescriptor(_winSize=IMAGE_DIMENSIONS, _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)

    # Gera vetor de características
    feat = hog.compute(img).flatten().astype(np.float32)

    # Normalização do vetor
    feat /= (np.linalg.norm(feat) + 1e-9)

    # Devolve vetor de características de forma linear (1 dimensão)
    return feat.flatten()

def align_images(img_src, img_ref):
    """
    Tenta alinhar a img_src com a img_ref usando ORB e Matriz de Homografia.
    """    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_s = clahe.apply(img_src)
    gray_r = clahe.apply(img_ref)

    # Detector ORB 
    orb = cv2.ORB_create(5000)
    kp_s, des_s = orb.detectAndCompute(gray_s, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)

    if des_s is None or des_r is None:
        return img_src

    # Matching de features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_s, des_r)
    matches = sorted(matches, key=lambda x: x.distance)

    # Cálculo da Homografia (mínimo 10 pontos para ser confiável)
    if len(matches) > 10:
        src_pts = np.float32([kp_s[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            # Verifica quantidade de inliers
            inliers = np.sum(mask)
            
            # Verifica se a matriz M não é "extrema"
            det = np.linalg.det(M[0:2, 0:2])
            
            # Aplica o warp caso tenham muitos pontos e a matriz for saudável
            if inliers > 25 and abs(det) > 0.1: 
                h_ref, w_ref = img_ref.shape[:2]
                img_aligned = cv2.warpPerspective(img_src, M, (w_ref, h_ref))
                return img_aligned
            else:
                print("Alinhamento descartado por baixa confiança (evitando distorção).")

    return img_src # Usa a original se não for confiável


def euclidian_distance(img1: np.ndarray, img2: np.ndarray):
    """
    Calcula a distância euclidiana entre duas imagens
    """

    # Converte imagens para vetores
    img1_vector = hog_feature(img1)
    img2_vector = hog_feature(img2)

    # Calcula a norma da diferença entre os vetores
    return float(np.linalg.norm(img1_vector - img2_vector))