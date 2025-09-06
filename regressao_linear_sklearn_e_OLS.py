import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('base_funcionarios.csv')
print(df)

print(df.info())

(df['Cargo'].value_counts())

df['Cargo'] = df['Cargo'].map({'Assistente': 0, 'Gerente': 1, 'Coordenador': 2, 'Analista': 3})

(df['Genero'].value_counts())

df['Genero'] = df['Genero'].map({'Masculino': 0, 'Feminino': 1, 'Outro': 2})

(df['Departamento'].value_counts())

df['Departamento'] = df['Departamento'].map({'Financeiro': 0, 'Marketing': 1, 'RH': 2, 'Operações': 3, 'TI': 4,'Vendas': 5})

(df['Escolaridade'].value_counts())

df['Escolaridade'] = df['Escolaridade'].map({'Mestrado': 0, 'Ensino Médio': 1, 'Graduação': 2, 'Pós-graduação': 3, 'Doutorado': 4})

print(df['EstadoCivil'].value_counts())

df['EstadoCivil'] = df['EstadoCivil'].map({'Viúvo': 0, 'Divorciado': 1, 'Solteiro': 2, 'Casado': 3})

corr = df.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(30, 15))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', linewidths=0.5, linecolor='white')
plt.show()

x = df[['Cargo', 'TempoEmpresa', 'BonusAnual']]
y = df[['SalarioBase']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0 )

modelo_skl = LinearRegression()
modelo_skl.fit(x, y)

y_pred = modelo_skl.predict(x_test)

rmse = metric.root_mean_squared_error(y_test, y_pred)
print(f'RMSE: {rmse}')

r2 = metric.r2_score(y_test, y_pred)
print(f'R²: {r2}')

import statsmodels.api as sm
import numpy as np

constante = sm.add_constant(x)
ols = sm.OLS(y, constante).fit()
print(ols.summary())