import gym
import random

# Criar as variavies do ambiente da A.I
ambiente = gym.make('CartPole-v1')
estados = ambiente.observation_space.shape[0]
acoes = ambiente.action_space.n

# Percorra 10 etapas
etapas = 10
for etapa in range(1, etapas+1):
    estado = ambiente.reset()
    vitoria = False
    pontuacao = 0

    while not vitoria:
        ambiente.render()
        acao = random.choice([0, 1])
        n_estado, recompensa, vitoria, info = ambiente.step(acao)
        pontuacao+=recompensa
    print('Etapa: {} Pontuação: {}'.format(etapa, pontuacao))

# Fecha o render do ambiente
ambiente.close()