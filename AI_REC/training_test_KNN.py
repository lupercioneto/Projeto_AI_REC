import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from joblib import dump, load

from tag_relevant import get_tags_weight
from tags_exclude import TAGS_EXCLUIR

TAGS_PESOS = dict(get_tags_weight())

class GameRecommenderKNN:
    def __init__(self, data_frame=None, csv_path=None,
                 model_name="all-MiniLM-L6-v2",
                 embeddings_prefix="embeddings",
                 n_neighbors=500,
                 pca_dim=128):

        # -------------------- Carregar dados --------------------
        if data_frame is not None:
            self.data = data_frame.copy()
        elif csv_path is not None:
            self.data = pd.read_csv(csv_path)
        else:
            raise ValueError("Forneça data_frame ou csv_path")

        self.embeddings_prefix = embeddings_prefix
        self.pca_dim = pca_dim
        self.n_neighbors = n_neighbors

        # -------------------- Normalizar ratings --------------------
        self._normalize_ratings()

        # -------------------- Processar tags e gêneros --------------------
        self.data["tags_text"] = self.data["tags"].apply(self._process_tags)
        self.data["genres_text"] = self.data["generos"].apply(self._process_genres)

        # -------------------- Embeddings + PCA --------------------
        self._load_or_generate_embeddings(model_name)

        # -------------------- KNN --------------------
        self.knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(self.data)), metric="cosine")
        self.knn.fit(self.base_features)

    # -------------------- Normalização ratings --------------------
    def _normalize_ratings(self):
        self.data[["nota_media", "metacritic"]] = self.data[["nota_media", "metacritic"]].fillna(0)
        scaler = MinMaxScaler()
        self.data[["nota_media_norm", "metacritic_norm"]] = scaler.fit_transform(
            self.data[["nota_media", "metacritic"]]
        )

    def _normalize_array(self, arr):
        arr = np.array(arr)
        if np.all(arr == arr[0]):
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # -------------------- Processamento de tags e gêneros --------------------
    def _process_tags(self, tags_str):
        tags = [t.strip().lower() for t in str(tags_str).split(",") if t.strip()]
        tags = [tag for tag in tags if tag not in TAGS_EXCLUIR]
        # aplicar pesos de tags
        weighted_tags = [tag for tag in tags for _ in range(int(TAGS_PESOS.get(tag, 1) * 2))]
        return " ".join(weighted_tags)

    def _process_genres(self, genres_str):
        genres = [g.strip().lower() for g in str(genres_str).split(",") if g.strip()]
        return " ".join(genres)

    # -------------------- Similaridade de nomes --------------------
    def _name_similarity(self, name1, name2):
        return SequenceMatcher(None, str(name1).lower(), str(name2).lower()).ratio()

    # -------------------- Embeddings + PCA --------------------
    def _load_or_generate_embeddings(self, model_name):
        path = "game_recommender_knn"
        os.makedirs(path, exist_ok=True)

        base_features_file = os.path.join(path, f"{self.embeddings_prefix}_base_features.npy")
        emb_tags_file = os.path.join(path, f"{self.embeddings_prefix}_tags.npy")
        emb_genres_file = os.path.join(path, f"{self.embeddings_prefix}_genres.npy")
        pca_tags_file = os.path.join(path, f"{self.embeddings_prefix}_pca_tags.joblib")
        pca_genres_file = os.path.join(path, f"{self.embeddings_prefix}_pca_genres.joblib")

        if os.path.exists(base_features_file):
            self.base_features = np.load(base_features_file)
            return

        model = SentenceTransformer(model_name)

        # Embeddings
        if os.path.exists(emb_tags_file) and os.path.exists(emb_genres_file):
            emb_tags = np.load(emb_tags_file)
            emb_genres = np.load(emb_genres_file)
        else:
            emb_tags = model.encode(self.data["tags_text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
            emb_genres = model.encode(self.data["genres_text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
            np.save(emb_tags_file, emb_tags)
            np.save(emb_genres_file, emb_genres)

        # PCA
        if os.path.exists(pca_tags_file) and os.path.exists(pca_genres_file):
            pca_tags = load(pca_tags_file)
            pca_genres = load(pca_genres_file)
        else:
            pca_tags = PCA(n_components=self.pca_dim).fit(emb_tags)
            pca_genres = PCA(n_components=self.pca_dim).fit(emb_genres)
            dump(pca_tags, pca_tags_file)
            dump(pca_genres, pca_genres_file)

        emb_tags = pca_tags.transform(emb_tags)
        emb_genres = pca_genres.transform(emb_genres)

        # Combinar com ratings e ano
        additional_features = np.column_stack([
            self.data["nota_media_norm"].values,
            self.data["metacritic_norm"].values,
            self.data["ano_lancamento"].fillna(0).values / 2025  # normalizado
        ])
        self.base_features = np.hstack([emb_tags, emb_genres, additional_features]).astype(np.float32)
        np.save(base_features_file, self.base_features)

    # -------------------- Função de recomendação --------------------
    def recommend(self, game_name, top_n=10,
                  weight_tags=0.36, weight_genres=0.36,
                  weight_ratings=0.05, weight_year=0.02,
                  weight_name_bonus=0.21, filter_same_genre=False):

        # Busca case-insensitive
        matches = self.data[self.data["nome"].str.lower() == game_name.lower()]
        if len(matches) == 0:
            print(f"❌ Jogo '{game_name}' não encontrado!")
            return None

        idx = matches.index[0]

        # KNN para pré-filtrar candidatos
        distances, indices = self.knn.kneighbors(self.base_features[idx:idx+1], n_neighbors=min(self.n_neighbors, len(self.data)))
        candidates = indices.flatten()

        # Similaridade tags + gêneros
        sim_tags = cosine_similarity(
            self.base_features[idx, :self.pca_dim].reshape(1, -1),
            self.base_features[candidates, :self.pca_dim]
        )[0]
        sim_genres = cosine_similarity(
            self.base_features[idx, self.pca_dim:2*self.pca_dim].reshape(1, -1),
            self.base_features[candidates, self.pca_dim:2*self.pca_dim]
        )[0]

        # Similaridade ratings
        target_ratings = self.data.loc[idx, ["nota_media_norm", "metacritic_norm"]].values.astype(np.float32)
        candidate_ratings = self.data.iloc[candidates][["nota_media_norm", "metacritic_norm"]].values.astype(np.float32)
        sim_ratings = 1 / (1 + np.linalg.norm(candidate_ratings - target_ratings, axis=1))

        # Similaridade ano
        target_year = self.data.loc[idx, "ano_lancamento"]
        candidate_years = self.data.iloc[candidates]["ano_lancamento"].fillna(target_year).values
        sigma = 5
        sim_year = np.exp(-((candidate_years - target_year)**2)/(2*sigma**2))

        # Bônus nome
        sim_name_bonus = np.array([
            self._name_similarity(self.data.loc[idx, "nome"], self.data.iloc[i]["nome"])
            for i in candidates
        ])
        sim_name_bonus[sim_name_bonus < 0.6] = 0

        # Normalizar
        sim_tags = self._normalize_array(sim_tags)
        sim_genres = self._normalize_array(sim_genres)
        sim_ratings = self._normalize_array(sim_ratings)
        sim_year = self._normalize_array(sim_year)
        sim_name_bonus = self._normalize_array(sim_name_bonus)

        # Combinar
        combined_sim = (
            weight_tags*sim_tags +
            weight_genres*sim_genres +
            weight_ratings*sim_ratings +
            weight_year*sim_year +
            weight_name_bonus*sim_name_bonus
        )
        combined_sim = self._normalize_array(combined_sim)

        recs = self.data.iloc[candidates].copy()
        recs["similaridade"] = combined_sim
        recs = recs.drop(idx)

        if filter_same_genre:
            target_genres_set = set(g.strip().lower() for g in str(self.data.loc[idx, "generos"]).split(","))
            recs = recs[recs["generos"].apply(lambda genres: any(g.strip().lower() in target_genres_set for g in str(genres).split(",")))]

        return recs.sort_values("similaridade", ascending=False).head(top_n)

    # -------------------- Salvamento e carregamento --------------------
    def save_model(self, path="game_model"):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "base_features.npy"), self.base_features)
        dump(self.knn, os.path.join(path, "knn.joblib"))
        dump(self.data, os.path.join(path, "data.joblib"))
        dump({"pca_dim": self.pca_dim, "n_neighbors": self.n_neighbors, "embeddings_prefix": self.embeddings_prefix},
             os.path.join(path, "attrs.joblib"))

    @classmethod
    def load_model(cls, path="game_model"):
        self = cls.__new__(cls)
        self.base_features = np.load(os.path.join(path, "base_features.npy"))
        self.knn = load(os.path.join(path, "knn.joblib"))
        self.data = load(os.path.join(path, "data.joblib"))
        attrs = load(os.path.join(path, "attrs.joblib"))
        self.pca_dim = attrs["pca_dim"]
        self.n_neighbors = attrs["n_neighbors"]
        self.embeddings_prefix = attrs["embeddings_prefix"]
        return self
