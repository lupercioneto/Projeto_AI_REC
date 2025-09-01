import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Carregar e processar os dados
df = pd.read_csv('jogos.csv')

# Pré-processamento
df['tags'] = df['tags'].apply(ast.literal_eval)
df['generos'] = df['generos'].apply(ast.literal_eval)
df['combinado'] = df['tags'] + df['generos']
df['combinado_str'] = df['combinado'].apply(lambda x: ' '.join(x))

# Criar features usando TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = tfidf.fit_transform(df['combinado_str'])
tfidf_features = pd.DataFrame(tfidf_matrix.todense(), columns=tfidf.get_feature_names_out())

# Adicionar outras features numéricas
features = tfidf_features.copy()
features['nota_media'] = df['nota_media'].values
features['metacritic'] = df['metacritic'].values
features['ano_lancamento'] = df['ano_lancamento'].values

# Identificar e tratar valores NaN
print("Valores NaN antes do tratamento:")
print(features.isnull().sum())

# Usar SimpleImputer para preencher valores NaN
imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features)

# Normalizar as features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_imputed)

print(f"\nShape das features: {features_normalized.shape}")

# Treinar MLP para aprender representações latentes
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),  # Arquitetura da rede
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)

print("Treinando MLP...")
# Usar as próprias features como target (autoencoder-like)
X_train, X_test = train_test_split(features_normalized, test_size=0.2, random_state=42)
mlp.fit(X_train, X_train)

print("Treinamento concluído!")

# Obter representações latentes (saída da última camada oculta)
def get_latent_representations(model, data):
    # Forward pass até a última camada oculta
    hidden_output = data
    for i in range(len(model.coefs_) - 1):
        hidden_output = np.maximum(0, np.dot(hidden_output, model.coefs_[i]) + model.intercepts_[i])
    return hidden_output

# Calcular representações latentes para todos os jogos
latent_representations = get_latent_representations(mlp, features_normalized)

# Calcular matriz de similaridade baseada nas representações latentes
latent_similarity = cosine_similarity(latent_representations)

# Mapear índices para títulos
indices = pd.Series(df.index, index=df['nome']).drop_duplicates()

# Função de recomendação com MLP
def recomendar_jogos_mlp(nome_jogo, latent_similarity=latent_similarity, df=df, indices=indices, top_n=10):
    try:
        idx = indices[nome_jogo]
        
        # Obter similaridades
        sim_scores = list(enumerate(latent_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        
        jogo_indices = [i[0] for i in sim_scores]
        
        resultado = df[['nome', 'generos', 'nota_media', 'metacritic', 'ano_lancamento']].iloc[jogo_indices]
        resultado['similaridade'] = [i[1] for i in sim_scores]
        
        return resultado
    
    except KeyError:
        print(f"Jogo '{nome_jogo}' não encontrado. Verifique o nome.")
        jogos_similares = df[df['nome'].str.contains(nome_jogo, case=False)]['nome'].head(5)
        if len(jogos_similares) > 0:
            print("\nSugestões de jogos com nomes similares:")
            for jogo in jogos_similares:
                print(f"  - {jogo}")
        return None

# Testar o sistema
print("\n=== SISTEMA DE RECOMENDAÇÃO COM MLP ===\n")

# Exemplos de teste
jogos_teste = [
    "The Witcher 3: Wild Hunt",
    "Grand Theft Auto V", 
    "Portal 2",
    "Red Dead Redemption 2"
]

for jogo in jogos_teste:
    print(f"🔍 Recomendações para: {jogo}")
    recomendacoes = recomendar_jogos_mlp(jogo)
    
    if recomendacoes is not None:
        print(f"Top 5 jogos similares:")
        for i, (_, row) in enumerate(recomendacoes.head().iterrows(), 1):
            generos = ast.literal_eval(str(row['generos']))
            print(f"  {i}. {row['nome']}")
            print(f"     Gêneros: {', '.join(generos[:3])}")
            print(f"     Nota: {row['nota_media']:.2f} | Metacritic: {row['metacritic']} | Similaridade: {row['similaridade']:.3f}")
        print("-" * 80)

# Função para interface interativa
def buscar_recomendacoes():
    print("\n" + "="*60)
    nome_jogo = input("🎮 Digite o nome de um jogo para receber recomendações: ")
    
    if nome_jogo.strip() == "":
        print("❌ Nome não pode estar vazio!")
        return
    
    recomendacoes = recomendar_jogos_mlp(nome_jogo)
    
    if recomendacoes is not None:
        print(f"\n✅ Recomendações para '{nome_jogo}':")
        print("="*80)
        for i, (_, row) in enumerate(recomendacoes.iterrows(), 1):
            generos = ast.literal_eval(str(row['generos']))
            print(f"{i}. {row['nome']}")
            print(f"   📊 Nota: {row['nota_media']:.2f} | Metacritic: {row['metacritic']} | Ano: {row['ano_lancamento']}")
            print(f"   🎯 Gêneros: {', '.join(generos[:3])}")
            print(f"   🔗 Similaridade: {row['similaridade']:.3f}")
            print()

# Executar interface interativa
print("\n💡 Dica: Experimente com jogos como 'The Witcher 3', 'GTA V', 'Portal 2', etc.")
while True:
    try:
        buscar_recomendacoes()
        continuar = input("🔄 Deseja buscar outra recomendação? (s/n): ")
        if continuar.lower() != 's':
            print("👋 Até logo!")
            break
    except KeyboardInterrupt:
        print("\n👋 Saindo...")
        break