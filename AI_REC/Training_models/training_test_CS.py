import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import os

class GameRecommender:
    def __init__(self, csv_path, tags_excluir=None, generic_tags=None,
                 embeddings_file="embeddings.npy", model_name="all-MiniLM-L6-v2",
                 n_clusters=30):
        self.data = pd.read_csv(csv_path)
        self.tags_excluir = tags_excluir if tags_excluir else set()
        self.generic_tags = generic_tags if generic_tags else set([
            "adventure", "action", "open world", "rpg", "multiplayer"
        ])
        self.embeddings_file = embeddings_file

        # Normaliza notas e Metacritic
        scaler = MinMaxScaler()
        self.data[["nota_media_norm", "metacritic_norm"]] = scaler.fit_transform(
            self.data[["nota_media", "metacritic"]].fillna(0)
        )

        # Processa texto separado
        self.data["tags_text"] = self.data.apply(
            lambda row: self._process_tags(row["tags"]), axis=1
        )
        self.data["genres_text"] = self.data.apply(
            lambda row: self._process_genres(row["generos"]), axis=1
        )

        # Inicializa embeddings
        print("🎯 Carregando modelo de embeddings...")
        self.model = SentenceTransformer(model_name)
        
        if os.path.exists(self.embeddings_file):
            print("📦 Carregando embeddings salvos...")
            self.embeddings_tags = np.load(self.embeddings_file + "_tags.npy")
            self.embeddings_genres = np.load(self.embeddings_file + "_genres.npy")
        
        else:
            print("⚡ Gerando embeddings...")
            self.embeddings_tags = self.model.encode(self.data["tags_text"].tolist(), convert_to_numpy=True)
            self.embeddings_genres = self.model.encode(self.data["genres_text"].tolist(), convert_to_numpy=True)
            np.save(self.embeddings_file + "_tags.npy", self.embeddings_tags)
            np.save(self.embeddings_file + "_genres.npy", self.embeddings_genres)
            print("✅ Embeddings prontos e salvos!")

        # Clustering sobre embeddings de tags (estilo/jogabilidade)
        print("🌀 Aplicando clustering sobre tags...")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data["cluster"] = self.kmeans.fit_predict(self.embeddings_tags)

    def _process_tags(self, tags_str):
        tags = [t.strip().lower() for t in str(tags_str).split(",") if t.strip()]
        processed = []
        for tag in tags:
            if tag in self.tags_excluir:
                processed.extend([tag] * 1)
            elif tag in self.generic_tags:
                processed.extend([tag] * 2)
            else:
                processed.extend([tag] * 3)
        return " ".join(processed)

    def _process_genres(self, genres_str):
        genres = [g.strip().lower() for g in str(genres_str).split(",") if g.strip()]
        processed = [g for g in genres for _ in range(4)]
        return " ".join(processed)

    def _name_similarity(self, name1, name2):
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()

    def recommend(self, game_name, top_n=10,
                  weight_tags=0.5, weight_genres=0.3,
                  weight_ratings=0.05, weight_year=0.05,
                  weight_name_bonus=0.1, weight_cluster_bonus=0.05,
                  filter_same_genre=True, show_plot=True):

        if game_name not in self.data["nome"].values:
            print(f"❌ Jogo '{game_name}' não encontrado!")
            return None

        idx = self.data.index[self.data["nome"] == game_name][0]
        target_genres = set([g.strip().lower() for g in str(self.data.loc[idx, "generos"]).split(",")])
        target_name = self.data.loc[idx, "nome"]
        target_cluster = self.data.loc[idx, "cluster"]

        # Similaridade embeddings
        sim_tags = cosine_similarity([self.embeddings_tags[idx]], self.embeddings_tags).flatten()
        sim_genres = cosine_similarity([self.embeddings_genres[idx]], self.embeddings_genres).flatten()

        # Similaridade notas/metacritic
        target_ratings = self.data.loc[idx, ["nota_media_norm", "metacritic_norm"]].values.astype(float)
        ratings_matrix = self.data[["nota_media_norm", "metacritic_norm"]].values.astype(float)
        distances = np.linalg.norm(ratings_matrix - target_ratings, axis=1)
        sim_ratings = 1 / (1 + distances)

        # Similaridade ano
        target_year = self.data.loc[idx, "ano_lancamento"]
        year_diff = np.abs(self.data["ano_lancamento"] - target_year)
        sim_year = 1 / (1 + year_diff)

        # Bônus de nome não-linear (>0.5)
        sim_name_bonus = np.array([
            self._name_similarity(target_name, candidate_name) if self._name_similarity(target_name, candidate_name) > 0.5 else 0
            for candidate_name in self.data["nome"]
        ])

        # Bônus de cluster (mesmo cluster ganha boost)
        sim_cluster_bonus = np.array([
            1 if cluster == target_cluster else 0
            for cluster in self.data["cluster"]
        ])

        # Combina similaridades
        combined_sim = (weight_tags * sim_tags +
                        weight_genres * sim_genres +
                        weight_ratings * sim_ratings +
                        weight_year * sim_year +
                        weight_name_bonus * sim_name_bonus +
                        weight_cluster_bonus * sim_cluster_bonus)

        self.data["similaridade"] = combined_sim

        # Filtra por gênero
        if filter_same_genre:
            mask = self.data["generos"].apply(
                lambda g: len(target_genres.intersection(set([x.strip().lower() for x in str(g).split(",")]))) > 0
            )
            recs = self.data[mask].drop(idx).sort_values("similaridade", ascending=False).head(top_n)
        else:
            recs = self.data.drop(idx).sort_values("similaridade", ascending=False).head(top_n)

        # Exibição
        print(f"\n🎮 Recomendações para: {game_name}\n")
        for i, row in enumerate(recs.itertuples(), 1):
            print(f"{i}. {row.nome:<25} | Similaridade: {row.similaridade:.2f} "
                  f"| Nota Média: {row.nota_media:.1f} | Metacritic: {row.metacritic} | Ano: {row.ano_lancamento}")

        # Gráfico
        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.barh(recs["nome"], recs["similaridade"], color="skyblue")
            plt.xlabel("Similaridade combinada")
            plt.title(f"Top {top_n} recomendações para {game_name}")
            plt.gca().invert_yaxis()
            plt.show()

        return recs


from tags_exclude import TAGS_EXCLUIR

recommender = GameRecommender(
    "jogos.csv",
    tags_excluir=TAGS_EXCLUIR
)

recs = recommender.recommend("Grand Theft Auto V", top_n=10)
