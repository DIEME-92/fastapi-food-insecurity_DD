from sqlalchemy import create_engine, Column, Integer, String, Float, TIMESTAMP
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# ✅ Render utilise directement les variables d'environnement
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
    raise ValueError("La variable DATABASE_URL est introuvable. Configure-la dans Render Settings.")

# ✅ Créer l'engine SQLAlchemy
engine = create_engine(DATABASE_URL)

# ✅ Créer une session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ✅ Déclarer le modèle
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(TIMESTAMP, nullable=False)
    niveau = Column(String(50), nullable=False)
    score = Column(Float, nullable=False)

# ✅ Créer les tables si elles n'existent pas
def init_db():
    Base.metadata.create_all(bind=engine)
