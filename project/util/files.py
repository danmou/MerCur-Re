# files.py: File utils
#
# (C) 2019, Daniel Mouritzen

from pathlib import Path


def link_directory_contents(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src_file in source.iterdir():
        dest_file = dest / src_file.name
        dest_file.symlink_to(src_file)
