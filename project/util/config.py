# config.py: Config utilities
#
# (C) 2019, Daniel Mouritzen

from pathlib import Path


def get_config_dir() -> str:
    return str(Path(__file__).parent.parent / 'configs')
