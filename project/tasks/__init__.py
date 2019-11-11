# __init__.py
#
# (C) 2019, Daniel Mouritzen

from .habitat import habitat_task as habitat
from .planet_tasks import (cartpole_balance,
                           cartpole_swingup,
                           cheetah_run,
                           cup_catch,
                           finger_spin,
                           gym_cheetah,
                           gym_racecar,
                           reacher_easy,
                           walker_walk)

__all__ = ['habitat',
           'cartpole_balance',
           'cartpole_swingup',
           'cheetah_run',
           'cup_catch',
           'finger_spin',
           'gym_cheetah',
           'gym_racecar',
           'walker_walk',
           'reacher_easy']
