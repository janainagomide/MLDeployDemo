from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib

# Carregar o dataset Iris
data = load_iris()
X = data.data
y = data.target

# Treinar um modelo simples
modelo = RandomForestClassifier()
modelo.fit(X, y)

# Salvar o modelo
joblib.dump(modelo, 'iris_model.pkl')
