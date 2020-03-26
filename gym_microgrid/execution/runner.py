"""
This class will handle the execution
"""

from gym import Env
from tqdm import tqdm

from gym_microgrid.agents import Agent


class Runner:
    """
    This class will execute an agent on the environment.
    It handles communication between agent and environment and handles the execution of multiple epochs
    """

    def __init__(self, agent: Agent, env: Env):
        self.agent = agent
        self.env = env

    def run(self, n_episodes=10, visualize=False):
        # TODO pass action space when resetting agent
        self.agent.reset()

        for _ in tqdm(range(n_episodes), desc='episodes', unit='epoch'):
            obs = self.env.reset()

            done, r = False, None
            while not done:
                self.agent.observe(r, done)
                act = self.agent.act(obs)
                obs, r, done = self.env.step(act)
                if visualize:
                    self.env.render()
            self.agent.observe(r, done)
            self.env.close()