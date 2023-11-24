import pandas as pd
import joblib

def make_prediction(data):
    # Carregar o modelo treinado
    model = joblib.load('model.pkl')

    # Construir um DataFrame com os dados recebidos na requisição
    new_data = pd.DataFrame([data])

    # Selecionar as mesmas características usadas no treinamento
    features = ['Pclass', 'SibSp', 'Parch']
    X_new = new_data[features]

    # Fazer previsões com o modelo carregado
    prediction_proba = model.predict_proba(X_new)

    # Mapear a predição para 'Sobreviveu' ou 'Não Sobreviveu'
    result = {"Sobreviveu": prediction_proba[0][0], "Não_Sobreviveu": prediction_proba[0][1]}
    print(result)

    return result
