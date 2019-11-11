# planet_interface.py: Functionality for interfacing with PlaNet
#
# (C) 2019, Daniel Mouritzen

from typing import Any

import gin

from .planet import AttrDict


@gin.configurable('planet')
class PlanetParams(AttrDict):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not self.get('tasks'):
            self.tasks = ['habitat']
