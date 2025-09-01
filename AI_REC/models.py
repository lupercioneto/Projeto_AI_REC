from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from database import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    nome = Column(String, index=True, nullable=False)
    tags = Column(JSONB) 
    generos = Column(JSONB)
    nota_media = Column(Float, nullable=True) 
    metacritic = Column(Integer, nullable=True)
    ano_lancamento = Column(Integer, nullable=True)
    screenshots = Column(JSONB, nullable=True)
    # cover_image = Column(String, nullable=True)

class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    game_name = Column(String, index=True, nullable=False)
    top_n = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
