# base.py: Basic Wrapper class
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, Tuple, Union, cast

import gym
import numpy as np

from project.util.typing import Action, Observations, ObsTuple, Reward


class Wrapper(gym.Wrapper):
    def step(self, action: Action) -> ObsTuple:
        return cast(ObsTuple, self.env.step(action))

    def reset(self) -> Observations:
        return cast(Observations, self.env.reset())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
