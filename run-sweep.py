#!/usr/bin/env python
# run-sweep.py: Script to run hyperparameter sweep
#
# (C) 2019, Daniel Mouritzen

import os
import random
import string
import sys
import time
from multiprocessing import Process
from typing import Any, Dict, List, Optional

import click
import wandb
import yaml

sys.path.append('planet')
from project.main import main_configure


def run_agent(sweep_id: str, gpu: str, config: str, verbosity: str = 'INFO') -> None:
    wandb.wandb_agent.logger.setLevel = lambda _: None  # make wandb_agent quiet

    def train() -> None:
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
            os.environ['WANDB_SILENT'] = 'true'
            wandb.init()
            extra_options = tuple(f'{name}={val}' for name, val in wandb.config.user_items())
            print(f'Job on GPU {gpu} starting with options:\n' + '\n'.join(extra_options))
            main_configure(config, extra_options, verbosity, catch_exceptions=False, name=gpu)
        except Exception as e:
            # An exception in this function would cause an infinite hang
            print(f'Job on GPU {gpu} failed with exception of type {type(e).__name__}')

    wandb.agent(sweep_id, function=train)
    print('Agent finished')


class Config:
    def __init__(self, file: str) -> None:
        self._file = file
        self._modified_time = None
        self.dict = None
        self.update()

    def update(self) -> bool:
        mtime = os.stat(self._file).st_mtime
        if mtime != self._modified_time:
            with open(self._file) as conf:
                self.dict = yaml.load(conf, Loader=yaml.FullLoader)
            self.dict['controller'] = {'type': 'local'}
            self._modified_time = mtime
            return True
        return False

    def check_params(self, params: Dict[str, Any]) -> bool:
        for condition in self.dict.get('conditions', []):
            if not eval(condition, {}, {name.split('.')[-1]: val['value'] for name, val in params.items()}):
                return False
        return True


class SweepController:
    def __init__(self, config_file: str, base_config: str, id: Optional[str], gpus: str) -> None:
        assert gpus, 'No GPUs specified, specify with --gpus or by setting the CUDA_VISIBLE_DEVICES env var.'
        self.gpus = gpus.split(',')
        self.config = Config(config_file)
        self.base_config = base_config
        self.sweep_id = id or wandb.sweep(self.config.dict)
        self.tuner = wandb.controller(self.sweep_id)
        self.tuner.configure(self.config.dict)
        self.processes: Dict[str, Process] = {}

    def run(self, verbosity: str) -> None:
        try:
            self.start_agents(verbosity)
            print('Running sweep.')
            while self.processes or not self.tuner.done():
                self.check_agents()
                if not self.tuner.done():
                    if self.config.update():
                        print('Updated config.')
                    self.step()
                    time.sleep(1)
            print('Sweep done.')
        finally:
            self.kill_agents()

    def start_agents(self, verbosity: str) -> None:
        for gpu in self.gpus:
            assert gpu not in self.processes.keys(), f'GPU {gpu} was specified twice!'
            print(f'Starting agent on GPU {gpu}.')
            self.processes[gpu] = Process(target=run_agent, args=(self.sweep_id, gpu, self.base_config, verbosity))
            self.processes[gpu].start()
            time.sleep(1)  # Otherwise they receive the same parameters on the first run

    def check_agents(self) -> None:
        for gpu, p in self.processes.items():
            if not p.is_alive():
                print(f'Agent on GPU {gpu} finished.')
                self.processes.pop(gpu)
                self.check_agents()
                break

    def kill_agents(self) -> None:
        for gpu, p in self.processes.items():
            print(f'Killing agent on gpu {gpu}')
            p.terminate()

    def step(self) -> None:
        self.tuner._sweep_config = self.config.dict
        self.tuner._step()

        if self.can_schedule():
            params = self.search_params()
            self.schedule(params)

        stop_runs = self.tuner.stopping()
        if stop_runs:
            self.tuner.stop_runs(stop_runs)

        self.print_status()

    def search_params(self) -> Dict[str, Any]:
        params = self.tuner.search()
        while not self.config.check_params(params):
            params = self.tuner.search()
        return params

    def can_schedule(self) -> bool:
        schedule = [p['id'] for p in self.tuner._controller.get('schedule', [])]
        num_scheduled = len(set(schedule).union({p['id'] for p in self.get_running()}))
        if not schedule or num_scheduled < len(self.gpus):
            return True
        latest = schedule[-1]
        scheduled = self.tuner._scheduler.get('scheduled') or []
        return latest in {p['id'] for p in scheduled}

    def schedule(self, params: Dict[str, Any]) -> None:
        schedule_list = self.get_running()
        schedule_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        schedule_list.append({'id': schedule_id, 'data': {'args': params}})
        self.tuner._controller["schedule"] = schedule_list
        self.tuner._sweep_object_sync_to_backend()

    def get_running(self) -> List[Dict[str, Any]]:
        self.tuner._sweep_object_read_from_backend()
        scheduled = self.tuner._scheduler.get('scheduled') or []
        return [p for p in scheduled
                if self.tuner._sweep_runs_map[p['runid']].state == 'running']

    def print_status(self) -> None:
        self.tuner._sweep_object_read_from_backend()
        controller_schedule = [p['id'] for p in self.tuner._controller.get('schedule', [])]
        scheduled = self.tuner._scheduler.get('scheduled') or []
        scheduled_ids = {p['id'] for p in scheduled}
        scheduled_runids = {p['id']: p['runid'] for p in scheduled}
        scheduled_agents = {p['id']: p['agent'] for p in scheduled}
        states = {id: self.tuner._sweep_runs_map[runid].state for id, runid in scheduled_runids.items()}
        print()
        print('Jobs')
        print('ID:        ' + ', '.join(f'{id:>8s}' for id in controller_schedule))
        print('Scheduled: ' + ', '.join(f'{str(id in scheduled_ids):>8s}' for id in controller_schedule))
        print('Run ID:    ' + ', '.join(f'{scheduled_runids.get(id, "-"):>8s}' for id in controller_schedule))
        print('Agent ID:  ' + ', '.join(f'{scheduled_agents.get(id, "-"):>8s}' for id in controller_schedule))
        print('State:     ' + ', '.join(f'{states.get(id, "-"):>8s}' for id in controller_schedule))
        print()
        self.tuner.print_status()


@click.command()
@click.option('-c', '--sweep-config', type=click.Path(dir_okay=False), default='configs/sweep.yaml', help='sweep config')
@click.option('-b', '--base-config', type=click.Path(dir_okay=False), default='configs/default.gin', help='gin config')
@click.option('--id', help='existing sweep id to use (if not specified, a new sweep will be created)')
@click.option('--gpus', envvar='CUDA_VISIBLE_DEVICES')
@click.option('--verbosity', default='WARNING')
def main(sweep_config: str, base_config: str, id: Optional[str], gpus: str, verbosity: str) -> None:
    SweepController(sweep_config, base_config, id, gpus).run(verbosity)


if __name__ == '__main__':
    main()
