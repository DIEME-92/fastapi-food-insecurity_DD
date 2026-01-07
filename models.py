from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    niveau = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    profil = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
