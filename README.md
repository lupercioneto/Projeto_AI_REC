# AI_REC - Recomendador de Jogos com IA

## Descrição do Projeto

- **AI_REC** é uma aplicação de recomendação de jogos que utiliza **Inteligência Artificial** para sugerir jogos semelhantes a partir de um título de entrada.

- Ainda está em fase de desenvolvimento, com uma parte do segmento *Front-end* ainda em análise. Aqui, consta apenas o segmento *Back-end*.

## Detalhes de Desenvolvimento
- Linguamgem usada: **Python**

- Banco de Dados: **PostegreSQL**
   
- Bibliotecas principais: 
  - Construção do Modelo: `scikit-learn`, `joblib`;
  - Busca de dados: `pandas`, `aoihttp`, `pandas`, `numpy`;
  - Ambiente da API: `fastapi`;
  - Armazenamento de dados: `sqlalchemy`

- Dados utilizados para a construção do modelo foram retirados da API pública [RAWG](https://rawg.io/apidocs);