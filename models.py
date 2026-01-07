from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    niveau = Column(String)
    profil = Column(String)
    score = Column(Float)
    # ❌ supprime created_at si tu ne veux pas l’utiliser

