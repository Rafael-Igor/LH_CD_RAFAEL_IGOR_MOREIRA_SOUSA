
Este repositório contém a solução do desafio de análise e modelagem sobre filmes.

Estrutura do projeto
- `model_training.py` → Script para treinar e salvar o modelo (`imdb_rating_model.pkl`).
- `analyse_recommend.py` → Script para análise exploratória e recomendação de gênero mais promissor.
- `imdb_rating_model.pkl` → Modelo salvo, treinado com variáveis numéricas e TF-IDF da sinopse (`Overview`).
- `analysis_notebook.ipynb` → Notebook com análise exploratória (EDA).
- `report.txt` → Relatório com hipóteses, análises e recomendação de gênero.
- `requirements.txt` → Dependências para rodar os códigos.
- `README.md` → Este arquivo.

Como rodar o projeto
Crie e ative um ambiente virtual, depois instale as dependências:
```bash
pip install -r requirements.txt
```

Para treinar e salvar o modelo:
```bash
python model_training.py --data desafio_indicium_imdb.csv --output imdb_rating_model.pkl
```

Para rodar a análise exploratória e obter recomendação do gênero mais promissor:
```bash
python analyse_recommend.py
```

O arquivo `shaw_prediction.txt` contém a previsão da nota para *The Shawshank Redemption* usando o modelo treinado.

---

Com base nas análises, filmes de **Action/Adventure** com alto orçamento, bom elenco e histórias originais têm maior probabilidade de sucesso.
