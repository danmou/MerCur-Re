# files.py: File utils
#
# (C) 2019, Daniel Mouritzen

import os.path
from pathlib import Path
from typing import Optional


def link_directory_contents(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src_file in source.iterdir():
        dest_file = dest / src_file.name
        dest_file.symlink_to(os.path.relpath(src_file, dest))


def get_latest_checkpoint(checkpoint: Path, base_dir: Optional[Path] = None) -> Path:
    """
    Get the full path to a checkpoint. If checkpoint is not absolute, it is considered relative to base_dir. If the
    resulting path is a directory, the latest file matching 'checkpoint*.h5' inside it is returned instead.
    Raises: ValueError if the checkpoint could not be found.
    """
    try:
        checkpoint = checkpoint.expanduser()
        if not checkpoint.is_absolute():
            assert base_dir is not None
            checkpoint = base_dir / checkpoint
        checkpoint = checkpoint.absolute()
        if not checkpoint.is_file():
            # Check if there is a 'checkpoint_latest' file
            latest = checkpoint / 'checkpoint_latest'
            if latest.is_file():
                with open(latest) as f:
                    checkpoint = checkpoint / f.read().strip()
            else:
                # Get latest modified checkpoint file
                checkpoint = max(checkpoint.glob('checkpoint*.h5'), key=lambda p: p.stat().st_mtime)
        assert checkpoint.is_file()
    except (AssertionError, ValueError):
        raise ValueError(f'Checkpoint for {checkpoint} not found.')
    return checkpoint
