from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from fuzzywuzzy import process
from sqlalchemy.orm import Session
from database import SessionLocal
from models import SearchHistory, Game
from training_test_KNN import GameRecommenderKNN
import pandas as pd

app = FastAPI(title="Game Recommender API")

_model_instance = None

class RecommendRequest(BaseModel):
    game_name: str
    top_n: int = 10
    auto_correct: bool = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = GameRecommenderKNN.load_model("game_recommender_knn")
    return _model_instance


def find_closest_game(name, games_list, threshold=80):
    match, score = process.extractOne(name, games_list)
    return match if score >= threshold else None


@app.get("/")
def root():
    return {"message": "API rodando!"}

@app.post("/recommend")
def recommend(request: RecommendRequest, db: Session = Depends(get_db)):

    recommender = get_model()

    game_name = request.game_name
    top_n = request.top_n

    if not 1 <= top_n <= 50:
        raise HTTPException(status_code=400, detail="top_n deve estar entre 1 e 50")

    # Verifica se o jogo existe
    if game_name not in recommender.data["nome"].values:
        closest = find_closest_game(game_name, recommender.data["nome"].values)
        if closest and request.auto_correct:
            raise HTTPException(404, detail=f"Jogo não encontrado. Você quis dizer '{closest}'?")
        raise HTTPException(status_code=404, detail=f"Jogo '{game_name}' não encontrado.")

    # Salva histórico
    history = SearchHistory(game_name=game_name, top_n=top_n)
    db.add(history)
    db.commit()

    # Gera recomendações
    recs_df = recommender.recommend(game_name, top_n=top_n)
    recs_df["similaridade"] = recs_df["similaridade"].apply(lambda x: round(x, 2))

    recs_df = recs_df.where(pd.notnull(recs_df), None)

    # Pega screenshots de todas as recomendações de uma vez
    screenshots_df = pd.read_sql(
        db.query(Game.nome, Game.screenshots).filter(Game.nome.in_(recs_df["nome"].tolist())).statement,
        db.bind
    )

    # Merge direto no DataFrame de recomendações
    recs_df = recs_df.merge(screenshots_df, how="left", left_on="nome", right_on="nome")

    # Seleciona colunas finais
    recs_json = recs_df[["nome", "similaridade", "nota_media", "metacritic", "ano_lancamento", "screenshots"]].to_dict(orient="records")

    return {"recommendations": recs_json}