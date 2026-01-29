# Desafio Técnico - Visão Computacional

Este projeto é a solução para o case técnico de visão computacional, focado na identificação e comparação de similaridade entre produtos de supermercado.

O sistema compara duas imagens, calcula uma distância baseada em características visuais e determina se representam o mesmo produto com base em um threshold configurável.


## Decisões de Arquitetura e Motivação Técnica

A especificação original sugeria o uso de **Distância Euclidiana** sobre os pixels brutos, no entanto, durante a análise exploratória dos dados, identificou-se uma limitação crítica:

* **O Problema (Par E vs F):** O produto "Filtro de Café" apresentava rotação entre suas imagens, o que causava distâncias elevadas quando comparadas usando comparação pixel-a-pixel, classificando erroneamente o mesmo produto como "Diferente" (Falso Negativo).

### Solução Adotada:

Para resolver isso mantendo o requisito matemático da Distância Euclidiana, foi alterado o espaço de representação dos dados. O pipeline final consiste em:

1.  **Alinhamento Geométrico (ORB + Homografia):**
    * Utilização do algoritmo ORB para detectar pontos de interesse (keypoints).
    * Cálculo da Matriz de Homografia (com RANSAC) para "desentortar" a imagem de entrada, alinhando-a com a referência, tentando neutralizar a rotação.
2.  **Extração de Características (HOG):**
    * Em vez de comparar a cor do pixel, extraio vetores de HOG (Histogram of Oriented Gradients).
    * O HOG foca na forma e estrutura (bordas) do objeto, tornando a comparação robusta a pequenas variações de iluminação e cor de fundo.
3.  **Métrica (Distância Euclidiana):**
    * A Distância Euclidiana solicitada é calculada sobre os vetores HOG normalizados, garantindo uma decisão baseada na estrutura do objeto.

## Tecnologias utilizadas

* **Linguagem:** Python 3.11
* **Visão Computacional:** OpenCV (cv2)
* **API Web:** FastAPI (Alta performance e documentação automática)
* **Banco de Dados:** SQLite + SQLAlchemy
* **Containerização:** Docker & Docker Compose

## Como Rodar

### Opção 1: Via Docker (Recomendado)
Para garantir que o ambiente seja idêntico ao de desenvolvimento, basta ter o Docker instalado e rodar:

```bash
docker-compose up --build
```

A API estará disponível em: http://localhost:8000/docs

### Opção 2: Instalação Manual
Caso prefira rodar localmente sem Docker:

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

Para rodar o Script de Terminal (CLI):

```bash
python main.py config.json
```

Para rodar a API Web:

```bash
uvicorn api:app --reload
```

## Utilizando a API
A aplicação possui documentação interativa (Swagger UI) acessível em /docs.

### Principais Endpoints:

``GET /products``: Lista as imagens disponíveis na pasta products/.

``POST /compare``: Compara dois produtos e salva o resultado.

#### Body:

```JSON
{
  "product_1": "product_a.jpg",
  "product_2": "product_b.jpg",
  "threshold": 0.8
}
```
Retorno: JSON com a distância calculada, decisão boolean (same_product) e caminho da imagem de evidência gerada.
A saída é uma imagem concatenada das versões preprocessadas em grayscale, salva em ``outputs/``.

``GET /history``: Consulta o banco de dados SQLite para listar todas as comparações já realizadas.

## Persistência

Cada execução do endpoint `/compare` registra no SQLite (`app.db`) os caminhos das duas imagens originais
e da imagem final concatenada (`output_image_path`), permitindo consulta posterior via `/history`.