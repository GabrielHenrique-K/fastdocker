import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Carregar o DataFrame
df = pd.read_csv("dataset\\train.csv")

# Preencher valores ausentes com a média
mean_age = df['Age'].mean()
mean_fare = df['Fare'].mean()

df['Age'].fillna(mean_age, inplace=True)
df['Fare'].fillna(mean_fare, inplace=True)

# Selecionar características relevantes e o alvo (Survived), excluindo "Sex" e "Embarked"
features = ['Pclass', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.dropna()
y_train = y_train[X_train.index]  # Atualize os rótulos correspondentes após a remoção


# Criar um pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('classifier', RandomForestClassifier(random_state=42)) 
])

# Treina o modelo no conjunto de treinamento
pipeline.fit(X_train, y_train)

# Salva o modelo treinado em um arquivo
joblib.dump(pipeline, 'model.pkl')

# Salva as médias de 'Age' e 'Fare'
feature_means = {'Age': mean_age, 'Fare': mean_fare}
joblib.dump(feature_means, 'feature_means.pkl')

print("Modelo treinado e estatísticas salvas com sucesso.")
