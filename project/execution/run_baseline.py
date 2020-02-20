# train_baseline.py: Train model from project.habitat_baselines
#
# (C) 2019, Daniel Mouritzen

import random
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from project.environments.habitat import get_config as get_habitat_config
from project.habitat_baselines.common.baseline_registry import baseline_registry  # type: ignore[import]
from project.habitat_baselines.config.default import get_config as get_baseline_config
from project.util.timing import measure_time


@measure_time
def run_baseline(logdir: Path,
                 run_type: str,
                 exp_config: Path,
                 checkpoint: Optional[Path] = None,
                 num_processes: Optional[int] = None,
                 ) -> None:
    config = get_baseline_config(str(exp_config))

    config.defrost()
    config.TENSORBOARD_DIR = str(logdir / config.TENSORBOARD_DIR)
    config.VIDEO_DIR = str(logdir / config.VIDEO_DIR)
    config.CHECKPOINT_FOLDER = str(logdir / config.CHECKPOINT_FOLDER)
    if run_type == 'eval' and checkpoint is not None:
        config.EVAL_CKPT_PATH_DIR = str(checkpoint)
    config.TASK_CONFIG = get_habitat_config(training=run_type == 'train', max_steps=450)['config']
    if num_processes is not None:
        config.NUM_PROCESSES = num_processes
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        logger.info(f'Training baseline {config.TRAINER_NAME}')
        trainer.train(str(checkpoint) if checkpoint is not None else None)
    elif run_type == "eval":
        logger.info(f'Evaluating baseline {config.TRAINER_NAME}')
        trainer.eval()
    else:
        logger.error(f'Unknown run type {run_type}')
