import gym
import numpy as np
import pandas as pd
from openmodelica_microgrid_gym import Agent, Runner


class RndAgent(Agent):
    def act(self, obs: pd.Series) -> np.ndarray:
        return self.env.action_space.sample()


if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   max_episode_steps=1000,
                   net='../net/net.yaml',
                   model_path='../fmu/grid.network.fmu')

    agent = RndAgent()
    runner = Runner(agent, env)

    runner.run(1)
