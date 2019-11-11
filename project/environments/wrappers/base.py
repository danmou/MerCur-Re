# base.py: Basic Wrapper class
#
# (C) 2019, Daniel Mouritzen

from typing import Any, Dict, Tuple, Union, cast

import gym
import numpy as np

Action = Union[int, np.ndarray]
Observations = Union[np.ndarray, Dict[str, np.ndarray]]
Reward = Union[float, np.ndarray]
ObsTuple = Tuple[Observations, Reward, bool, Dict[str, Any]]  # obs, reward, done, info


class Wrapper(gym.Wrapper):
    def step(self, action: Action) -> ObsTuple:
        return cast(ObsTuple, self.env.step(action))

    def reset(self) -> Observations:
        return cast(Observations, self.env.reset())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
