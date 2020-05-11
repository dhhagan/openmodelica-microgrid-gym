import gym
import numpy as np

if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   model_output=dict(lc1=['inductor1.i', 'inductor2.i', 'inductor3.i']),
                   model_path='../fmu/grid.network.fmu')

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(np.random.random(3))  # take a random action
    env.close()