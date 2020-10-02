# Importando as bibliotecas que irei usar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando o dataset que sera utilizado

dataset = pd.read_csv('C:/Users/allan/OneDrive/Área de Trabalho/Udemy - Curso/Position_Salaries.csv')

# Definindo qual sera a variavel dependente e independente

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Treinando a decision Tree Regression model

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

# Prevendo um valor que nao esta na taberla

regressor.predict([[6.5]])

# Vizualizando em um grafico (em alta resolucao)

# com essas duas linhas, colocamos o grafico em alta resolucao
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='black')
plt.title('Tree Decision Regressor predict')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()
