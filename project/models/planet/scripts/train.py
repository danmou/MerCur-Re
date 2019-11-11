# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# Need offline backend to render summaries from within tf.py_func.
import matplotlib
import tensorflow as tf

from project.models.planet import tools, training
from project.models.planet.scripts import configs

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


matplotlib.use('Agg')


def process(logdir, args):
    with args.params.unlocked:
        args.params.logdir = logdir
    config = tools.AttrDict()
    with config.unlocked:
        config = getattr(configs, args.config)(config, args.params)
    envs = {name: task.env_ctor() for name, task in config.tasks.items()}
    training.utility.collect_initial_episodes(config, envs)
    tf.compat.v1.reset_default_graph()
    dataset = tools.numpy_episodes.numpy_episodes(
        config.train_dir, config.test_dir, config.batch_shape,
        reader=config.data_reader,
        loader=config.data_loader,
        num_chunks=config.num_chunks,
        preprocess_fn=config.preprocess_fn)
    for score in training.utility.train(
            training.define_model, dataset, logdir, config, envs):
        yield score
