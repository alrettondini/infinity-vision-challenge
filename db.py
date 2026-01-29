from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Cada linha representa uma comparação executada:
# - paths das 2 imagens originais
# - path da imagem final (concatenação em P&B)
# - métricas auxiliares (distância/mesmo) para auditoria e debug
class CompareResult(Base):
    __tablename__ = "compare_results"

    id = Column(Integer, primary_key=True, index=True)
    product_1_path = Column(String)
    product_2_path = Column(String)
    output_image_path = Column(String)
    distance = Column(Float)
    is_same = Column(Boolean)
    created_at = Column(DateTime, default=datetime.now(datetime.timezone.utc))

# Dependência do FastAPI: fornece uma sessão por request e garante fechamento.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Cria as tabelas automaticamente ao iniciar a aplicação
Base.metadata.create_all(bind=engine)