# Arquivo: analise_jogos.py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def data_game_analysis(csv_path="jogos.csv"):
    
    # Carregar os dados
    df = pd.read_csv(csv_path)
    
    # Converter strings para listas
    df["generos"] = df["generos"].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df["tags"] = df["tags"].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    # Estatísticas básicas
    print(f"✅ Total de jogos coletados: {len(df):,}")
    print(f"\n📊 Média de avaliação: {df['nota_media'].mean():.2f}")
    print(f"\n⭐ Jogos com rating >= 4.0: {(df['nota_media'] >= 4.0).sum():,}")
    print(f"⭐ Jogos com rating >= 3.5: {(df['nota_media'] >= 3.5).sum():,}")
    print(f"⭐ Jogos com rating >= 3.0: {(df['nota_media'] >= 3.0).sum():,}")
    
    # Análise de anos 
    anos_validos = df[df['ano_lancamento'].notna()]['ano_lancamento']
    if not anos_validos.empty:
        print(f"\n📅 Ano mais recente: {int(anos_validos.max())}")
        print(f"📅 Ano mais antigo: {int(anos_validos.min())}")
        print(f"📅 Ano médio: {int(anos_validos.mean())}")
        print(f"📅 Jogos sem ano: {df['ano_lancamento'].isna().sum():,}")
    
    # Análise de Metacritic 
    if 'metacritic' in df.columns:
        metacritic_validos = df[df['metacritic'].notna() & (df['metacritic'] != '')]
        if not metacritic_validos.empty:
    
            # Converter para numérico, tratando erros
            metacritic_validos = pd.to_numeric(metacritic_validos['metacritic'], errors='coerce')
            metacritic_validos = metacritic_validos.dropna()
    
            if len(metacritic_validos) > 0:
                print(f"\n🎯 Jogos com Metacritic: {len(metacritic_validos):,}")
                print(f"🏆 Metacritic médio: {metacritic_validos.mean():.1f}")
    
    # Análise de Gêneros e Tags
    todos_generos = [genero for sublist in df['generos'] for genero in sublist]
    todos_tags = [tag for sublist in df['tags'] for tag in sublist]
    
    print(f"\n🎭 Total de gêneros únicos: {len(set(todos_generos))}")
    print(f"🏷️ Total de tags únicas: {len(set(todos_tags))}")
    
    print(f"\n📊 TOP 10 Gêneros:")
    for genero, count in Counter(todos_generos).most_common(10):
        print(f" >  {genero}: {count:,} jogos")
    
    print(f"\n📊 TOP 10 Tags:")
    for tag, count in Counter(todos_tags).most_common(10):
        print(f" >  {tag}: {count:,} jogos")
    
    # Top 10 melhores jogos 
    print(f"\n🏆 TOP 10 Melhores Jogos (por nota):")
    top_10 = df.nlargest(10, 'nota_media')[['nome', 'nota_media', 'ano_lancamento']]
    
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        ano = int(row['ano_lancamento']) if not pd.isna(row['ano_lancamento']) else 'N/A'
        print(f"{i:2d}. {row['nome']} - ⭐ {row['nota_media']:.1f} ({ano})")
    
    # Distribuição por década 
    print(f"\n📈 Distribuição por Década:")

    # Filtrar apenas anos válidos e calcular década
    df_anos_validos = df[df['ano_lancamento'].notna()].copy()
    df_anos_validos['decada'] = (df_anos_validos['ano_lancamento'] // 10 * 10).astype(int)
    
    decadas = df_anos_validos['decada'].value_counts().sort_index()
    for decada, count in decadas.items():
        print(f" >  {decada}s: {count:,} jogos")
    
    return df

def criar_graficos(df):
    """Cria gráficos visuais da análise"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Gráfico 1: Distribuição de notas
        plt.subplot(2, 2, 1)
        df['nota_media'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribuição de Notas dos Jogos')
        plt.xlabel('Nota')
        plt.ylabel('Quantidade de Jogos')
        plt.grid(axis='y', alpha=0.3)
        
        # Gráfico 2: Distribuição por década (apenas anos válidos)
        plt.subplot(2, 2, 2)
        df_anos_validos = df[df['ano_lancamento'].notna()].copy()
        df_anos_validos['decada'] = (df_anos_validos['ano_lancamento'] // 10 * 10).astype(int)
        decadas_count = df_anos_validos['decada'].value_counts().sort_index()
        
        # Filtrar apenas décadas relevantes (1970+)
        decadas_count = decadas_count[decadas_count.index >= 1970]
        
        decadas_count.plot(kind='bar', color='lightgreen', alpha=0.7, edgecolor='black')
        plt.title('Jogos por Década')
        plt.xlabel('Década')
        plt.ylabel('Quantidade de Jogos')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Gráfico 3: Top 10 gêneros
        plt.subplot(2, 2, 3)
        todos_generos = [genero for sublist in df['generos'] for genero in sublist]
        top_generos = Counter(todos_generos).most_common(10)
        generos_names = [gen[0] for gen in top_generos]
        generos_counts = [gen[1] for gen in top_generos]
        
        plt.barh(generos_names, generos_counts, color='lightcoral', alpha=0.7)
        plt.title('Top 10 Gêneros')
        plt.xlabel('Quantidade de Jogos')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        # Gráfico 4: Top 10 tags
        plt.subplot(2, 2, 4)
        todos_tags = [tag for sublist in df['tags'] for tag in sublist]
        top_tags = Counter(todos_tags).most_common(10)
        tags_nomes = [tag[0] for tag in top_tags]
        tags_counts = [tag[1] for tag in top_tags]
        
        plt.barh(tags_nomes, tags_counts, color='gold', alpha=0.7)
        plt.title('Top 10 Tags')
        plt.xlabel('Quantidade de Jogos')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_game_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Gráficos salvos como 'data_game_analysis.png'")
        
    except Exception as e:
        print(f"⚠️ Erro ao criar gráficos: {e}")


# Arquivo: analisar_tags.py
from collections import Counter
import pandas as pd

def analise_completa_tags(csv_path="jogos.csv"):
    df = pd.read_csv(csv_path)
    df["tags"] = df["tags"].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    todas_tags = [tag for sublist in df['tags'] for tag in sublist]
    contador = Counter(todas_tags)
    
    print("🏷️ ANÁLISE COMPLETA DE TAGS:")
    print("=" * 50)
    
    print("\n📊 TOP 50 TAGS MAIS COMUNS:")
    for tag, count in contador.most_common(50):
        print(f"{tag:25}: {count:4} jogos")
    
    # Sugerir tags para excluir
    TAGS_SUGERIDAS_EXCLUIR = {
        tag for tag, count in contador.items() 
        if count < 20 or  # Muito raras
        any(word in tag for word in ['steam', 'achievement', 'cloud', 'dlc'])  # Irrelevantes
    }
    
    print(f"\n🚫 SUGESTÃO: Excluir {len(TAGS_SUGERIDAS_EXCLUIR)} tags irrelevantes")
    print("Exemplo:", list(TAGS_SUGERIDAS_EXCLUIR)[:10])

if __name__ == "__main__":
    # Análise completa
    # df = data_game_analysis("jogos.csv")
    
    # # Criar gráficos (opcional)
    # criar_graficos(df)
    
    # print("\n" + "=" * 60)
    # print("✅ Análise concluída! Os dados estão prontos para o sistema de recomendação!")

    analise_completa_tags()