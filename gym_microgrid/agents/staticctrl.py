from typing import Dict, Union, List

from gym_microgrid.agents import Agent
from gym_microgrid.common.itertools_ import fill_params, nested_map
from gym_microgrid.auxiliaries import Controller

from gym_microgrid.env import EmptyHistory

import pandas as pd
import numpy as np


class StaticControlAgent(Agent):
    def __init__(self, ctrls: Dict[str, Controller], observation_action_mapping: Dict, history=EmptyHistory()):
        super().__init__(history)
        self.episode_reward = 0
        self.controllers = ctrls
        self.obs_template = observation_action_mapping

    def act(self, state: pd.Series):
        """

        :param state: the agent is stateless. the state is stored in the controllers.
        Therefore we simply pass the observation from the environment into the controllers.
        """
        obs = fill_params(self.obs_template, state)
        controls = list()
        for key, params in obs.items():
            controls.append(self.controllers[key].step(*params))

        return np.append(*controls)

    def observe(self, reward, terminated):
        self.episode_reward += reward or 0
        if terminated:
            # reset episode reward
            self.prepare_episode()
        # on other steps we don't need to do anything

    @property
    def measurement(self) -> Union[pd.Series, List]:
        measurements = []
        for name, ctrl in self.controllers.items():
            def prepend(col): return '.'.join([name, col])

            measurements.append((nested_map(prepend, ctrl.history.structured_cols(None)),
                                 ctrl.history.df.tail(1).rename(columns=prepend).squeeze()))

        return measurements

    def prepare_episode(self):
        """
        Prepares the next episode; resets all controllers and filters (inital value of integrators...)
        """
        for ctrl in self.controllers.values():
            ctrl.reset()
        self.episode_reward = 0
