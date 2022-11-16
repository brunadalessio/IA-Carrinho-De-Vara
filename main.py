import gym
import random
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Criar as variavies do ambiente da A.I
ambiente = gym.make('CartPole-v1')
estados = ambiente.observation_space.shape[0]
acoes = ambiente.action_space.n

# Construir as redes neurais do modelo da A.I
def construir_modelo(estados, acoes):
    modelo = Sequential()
    modelo.add(Flatten(input_shape=(1, estados)))
    modelo.add(Dense(24, activation='relu'))
    modelo.add(Dense(24, activation='relu'))
    modelo.add(Dense(acoes, activation='linear'))
    return modelo

modelo = construir_modelo(estados, acoes)

# modelo.summary()

# Construir o treinamento da A.I
def construir_agente(modelo, acoes):
    politica = BoltzmannQPolicy()
    memoria = SequentialMemory(limit=10000, window_length=1)
    dqn = DQNAgent(model=modelo, memory=memoria, policy=politica, nb_actions=acoes, nb_steps_warmup=100, target_model_update=1e-2)

    return dqn

# Treinar A.I (ensinar 10000 passos)
dqn = construir_agente(modelo, acoes)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(ambiente, nb_steps=10000, visualize= False, verbose=1)

# Mostrar os resultados e o histórico do treinamento (100 etapas)
pontuacoes = dqn.test(ambiente, nb_episodes=100, visualize=True)
print(np.mean(pontuacoes.history['episode_reward']))

# _ = dqn.test(ambiente, nb_episodes=15, visualize= True)


