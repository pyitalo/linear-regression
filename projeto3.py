import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression

# Configurações do gráfico
plt.style.use('dark_background')

# Criação de dados fictícios
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Dados de entrada
y = 2.5 * X + np.random.randn(100, 1) * 2  # Dados de saída com ruído

# Configuração do modelo de regressão
model = LinearRegression()

# Criação da figura e eixos
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(-5, 30)
ax.set_xlabel('X', fontsize=12, color='white')
ax.set_ylabel('y', fontsize=12, color='white')
ax.set_title('Regressão Linear em Tempo Real', fontsize=14, color='white')

# Criar a linha de regressão e os pontos de dados
line, = ax.plot([], [], 'r-', lw=2)  # Linha da regressão
scat = ax.scatter([], [], c='cyan', edgecolor='white')  # Pontos de dados

def init():
    line.set_data([], [])
    scat.set_offsets(np.zeros((0, 2)))  # Inicializa os dados dos pontos
    return line, scat

def update(frame):
    # Geração de novos dados fictícios
    X_new = np.random.rand(20, 1) * 10
    y_new = 2.5 * X_new + np.random.randn(20, 1) * 2
    
    # Adicionar novos dados aos dados existentes
    global X, y
    X = np.vstack((X, X_new))
    y = np.vstack((y, y_new))
    
    # Treinamento do modelo
    model.fit(X, y)
    
    # Atualizar a linha de regressão
    x_range = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    line.set_data(x_range, y_pred)
    
    # Atualizar os pontos de dados
    scat.set_offsets(np.hstack((X, y)))
    
    return line, scat

# Criação da animação
ani = FuncAnimation(fig, update, init_func=init, frames=200, interval=500, blit=True)

# Mostrar o gráfico
plt.show()
