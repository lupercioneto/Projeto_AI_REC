import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Game
import ast  # para converter strings de listas em listas Python

# 1️⃣ Cria as tabelas (se ainda não existirem)
Base.metadata.create_all(bind=engine)

# 2️⃣ Lê o CSV dos jogos
games_df = pd.read_csv("jogo_s.csv")

# Converte colunas de listas (tags e generos) de string para lista Python
def parse_list_column(value):
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except:
            return []
    return value if isinstance(value, list) else []

games_df["tags"] = games_df["tags"].apply(parse_list_column)
games_df["generos"] = games_df["generos"].apply(parse_list_column)

# 3️⃣ Lê o CSV das screenshots
screenshots_df = pd.read_csv("games_with_screenshots.csv")
screenshots_df["screenshot_urls"] = screenshots_df["screenshot_urls"].apply(parse_list_column)

# Agrupa screenshots por game_id
screenshots_grouped = screenshots_df.groupby("id")["screenshot_urls"].apply(lambda x: [url for sublist in x for url in sublist]).to_dict()

# 4️⃣ Cria objetos Game e usa bulk_save_objects
db: Session = SessionLocal()
games_to_add = []

for _, row in games_df.iterrows():
    game_id = row["id"]

    game = Game(
        id=game_id,
        nome=row["nome"],
        tags=row["tags"],
        generos=row["generos"],
        nota_media=float(row["nota_media"]) if pd.notna(row["nota_media"]) else None,
        metacritic=int(row["metacritic"]) if pd.notna(row["metacritic"]) else None,
        ano_lancamento=int(row["ano_lancamento"]) if pd.notna(row["ano_lancamento"]) else None,
        screenshots=screenshots_grouped.get(game_id, [])
    )
    
    games_to_add.append(game)

# Inserção em bloco (muito mais rápido que linha a linha)
db.bulk_save_objects(games_to_add)
db.commit()
db.close()

print("Banco populado com sucesso!")
