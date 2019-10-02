#!/usr/bin/env python
# run-sweep.py: Script to run hyperparameter sweep
#
# (C) 2019, Daniel Mouritzen

import os
import sys
import time
from multiprocessing import Process

import click
import wandb
import yaml

sys.path.append('planet')
from project.main import main_configure


def run_agent(sweep_id: str, gpu: str, config: str, verbosity: str = 'INFO') -> None:
    def train() -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        os.environ['WANDB_SILENT'] = 'true'
        wandb.init()
        extra_options = tuple(f'{name}={val}' for name, val in wandb.config.user_items())
        main_configure(config, extra_options, verbosity, name=gpu)

    wandb.agent(sweep_id, function=train)


@click.command()
@click.option('--sweep-config', type=click.Path(dir_okay=False), default='configs/sweep.yaml', help='sweep config')
@click.option('--base-config', type=click.Path(dir_okay=False), default='configs/default.gin', help='gin config')
@click.option('--gpus', envvar='CUDA_VISIBLE_DEVICES')
@click.option('--verbosity', default='WARNING')
def main(sweep_config: str, base_config: str, gpus: str, verbosity: str) -> None:
    assert gpus, 'No GPUs specified, specify with --gpus or by setting the CUDA_VISIBLE_DEVICES env var.'
    with open(sweep_config) as conf:
        config_dict = yaml.load(conf, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(config_dict)
    processes = {}
    try:
        for gpu in gpus.split(','):
            assert gpu not in processes.keys(), f'GPU {gpu} was specified twice!'
            print(f'Starting agent on GPU {gpu}.')
            processes[gpu] = Process(target=run_agent, args=(sweep_id, gpu, base_config, verbosity))
            processes[gpu].start()
            time.sleep(1)  # Otherwise they receive the same parameters on the first run
        print('Waiting for agents.')
        while processes:
            for gpu, p in processes.items():
                if not p.is_alive():
                    print(f'Agent on GPU {gpu} finished.')
                    processes.pop(gpu)
                    break
        print('Sweep done.')
    finally:
        for gpu, p in processes.items():
            print(f'Killing agent on gpu {gpu}')
            p.terminate()


if __name__ == '__main__':
    main()
