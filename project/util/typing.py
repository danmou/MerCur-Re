# typing.py: Useful type definitions
#
# (C) 2019, Daniel Mouritzen

from typing import *

import numpy as np

T = TypeVar('T')
Nested = Union[T, Sequence[T], Mapping[Any, T]]

Action = Union[int, np.ndarray]
Observations = Union[np.ndarray, Dict[str, np.ndarray]]
Reward = Union[float, np.ndarray]
ObsTuple = Tuple[Observations, Reward, bool, Dict[str, Any]]  # obs, reward, done, info
