# model.py: Provides Model class
#
# (C) 2019, Daniel Mouritzen

import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gin
import tensorflow as tf
from loguru import logger

from project import networks
from project.util import losses
from project.util.files import get_latest_checkpoint
from project.util.system import is_debugging
from project.util.tf import auto_shape
from project.util.timing import measure_time


@gin.configurable(whitelist=['predictor', 'disable_tf_optimization'])
class Model(auto_shape.Model):
    """This class defines the top-level model structure and losses"""
    def __init__(self,
                 observation_components: Iterable[str],
                 data_shapes: Dict[str, tf.TensorShape],
                 data_dtypes: Dict[str, tf.DType],
                 predictor: networks.predictors.Predictor = gin.REQUIRED,
                 disable_tf_optimization: bool = False,
                 ) -> None:
        super().__init__(batch_dims=2)
        self._observation_components = list(observation_components)
        self._data_shapes = data_shapes
        self._data_dtypes = data_dtypes
        self._batch_size = next(iter(data_shapes.values()))[0]

        if disable_tf_optimization or is_debugging():
            logger.warning('Running without tf.function optimization.')
            tf.config.experimental_run_functions_eagerly(True)
            self.run_eagerly = True

        additional_observations = set(observation_components) - {'image'}
        self.encoder = networks.SelectItems(
            networks.ExtraBatchDim(networks.Encoder(image_input='image',
                                                    vector_inputs=additional_observations,
                                                    name='image_encoder')),
            keys=list(observation_components)
        )
        self.decoders = {'image': networks.ExtraBatchDim(networks.Decoder(), name='image_decoder')}
        for key in sorted(additional_observations | {'reward'}):
            data_shape = data_shapes[key][2:].as_list()
            self.decoders[key] = self._get_vector_decoder(data_shape,
                                                          name=f'{key}_decoder')
        # Layers in a dict are not automatically tracked, so we add them manually
        for layer in self.decoders.values():
            self._layers.append(layer)
        self._track_layers(self._layers)
        self.predictor = predictor
        self.rnn = auto_shape.RNN(self.predictor, return_sequences=True, name='rnn')

    @staticmethod
    @gin.configurable('Model.decoders', whitelist=['num_units', 'num_layers', 'activation'])
    def _get_vector_decoder(output_shape: Sequence[int],
                            num_units: int = gin.REQUIRED,
                            num_layers: int = gin.REQUIRED,
                            activation: str = 'relu',
                            name: str = 'vector_encoder'
                            ) -> auto_shape.Layer:
        return networks.ExtraBatchDim(auto_shape.Sequential([networks.SequentialBlock(num_units=num_units,
                                                                                      num_layers=num_layers,
                                                                                      activation=activation,
                                                                                      name=f'{name}_block'),
                                                             networks.ShapedDense(output_shape,
                                                                                  activation=None,
                                                                                  name=f'{name}_shaped_dense')],
                                                            name=f'{name}_sequential'),
                                      name=name)

    @staticmethod
    def _get_mask(data: Dict[str, tf.Tensor]) -> tf.Tensor:
        return tf.sequence_mask(data['length'], tf.shape(data['reward'])[1])

    @property
    def dummy_data(self) -> Dict[str, tf.Tensor]:
        """Create dummy data suitable for initializing the model's weights"""
        data = {}
        for key in self._data_shapes.keys():
            if key != 'length':
                data[key] = tf.zeros([2, 2] + self._data_shapes[key][2:], self._data_dtypes[key])
        data['length'] = tf.constant([2, 2], self._data_dtypes['length'])
        return data

    def closed_loop(self, data: Dict[str, tf.Tensor]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        embedded = self.encoder(data)
        use_obs = tf.ones(tf.shape(embedded[:, :, :1])[:3], tf.bool)
        prior, posterior = self.rnn((embedded, data['action'], use_obs), mask=self._get_mask(data))
        return prior, posterior

    @gin.configurable(whitelist=['context'])
    def open_loop(self, data: Dict[str, tf.Tensor], context: int = 5) -> List[tf.Tensor]:
        embedded = self.encoder(data)
        mask = self._get_mask(data)
        context = min(mask.shape[1] - 1, context)
        use_obs = tf.ones(tf.shape(embedded[:, :context, :1])[:3], tf.bool)
        _, closed_posterior = self.rnn((embedded[:, :context], data['action'][:, :context], use_obs),
                                       mask=mask[:, :context])
        use_obs = tf.zeros(tf.shape(embedded[:, context:, :1])[:3], tf.bool)
        last_posterior = tf.nest.map_structure(lambda x: x[:, -1], closed_posterior)
        open_prior, _ = self.rnn((tf.zeros_like(embedded[:, context:]), data['action'][:, context:], use_obs),
                                 initial_state=last_posterior,
                                 mask=mask[:, context:])
        return tf.nest.map_structure(lambda x, y: tf.concat([x, y], 1), closed_posterior, open_prior)

    def decode(self, state_features: tf.Tensor) -> Dict[str, tf.Tensor]:
        reconstructions = {}
        for name, decoder in self.decoders.items():
            reconstructions[name] = decoder(state_features)
        return reconstructions

    # def __call__(self, inputs: Dict[str, tf.Tensor]) -> Tuple[List[tf.Tensor],
    #                                                       List[tf.Tensor],
    #                                                       List[tf.Tensor],
    #                                                       Dict[str, tf.Tensor]]:  # type: ignore[override]
    #     return super().__call__(inputs)

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:  # type: ignore[override]
        inputs = inputs.copy()  # Shallow copy input dict so we can modify it safely
        if self._batch_size and tf.nest.flatten(inputs)[0].shape[0] is None:
            # Workaround for keras making the batch dimension undefined
            tf.nest.map_structure(lambda x: x.set_shape([self._batch_size] + x.shape[1:]), inputs)
        if inputs['length'].shape.ndims > 1:
            inputs['length'] = inputs['length'][:, 0]
        prior, posterior = self.closed_loop(inputs)
        reconstructions = self.decode(self.predictor.state_to_features(posterior))
        mask = self._get_mask(inputs)
        losses = {'divergence': self.divergence_loss(prior, posterior, mask)}
        losses.update(self.reconstruction_log_probs(inputs, reconstructions, mask))
        combined_loss = self.combine_losses(losses)
        self.add_loss(combined_loss, inputs=True)
        self.add_metric(combined_loss, aggregation='mean', name='loss')
        for name, loss in losses.items():
            self.add_metric(loss, aggregation='mean', name=name)
        # return prior, posterior, open_loop, reconstructions
        return tf.constant(0.0)

    @gin.configurable(whitelist=['free_nats'])
    @tf.function(experimental_relax_shapes=True)
    def divergence_loss(self,
                        prior: List[tf.Tensor],
                        posterior: List[tf.Tensor],
                        mask: Optional[tf.Tensor] = None,
                        free_nats: float = 3.0,
                        ) -> tf.Tensor:
        # TODO: Shouldn't we use tf.stop_gradient here? In principle we only want this loss to affect the prior
        divergence_loss = self.predictor.state_divergence(posterior, prior, mask)
        if free_nats:
            divergence_loss = tf.maximum(0.0, divergence_loss - float(free_nats))
        if mask is not None:
            divergence_loss = tf.boolean_mask(divergence_loss, mask)
        if divergence_loss.shape[0] == 0:
            divergence_loss = tf.constant(0.0)
        return tf.reduce_mean(divergence_loss, name='divergence_loss')

    @tf.function(experimental_relax_shapes=True)
    def reconstruction_log_probs(self,
                                 targets: Dict[str, tf.Tensor],
                                 reconstructions: Dict[str, tf.Tensor],
                                 mask: Optional[tf.Tensor] = None,
                                 ) -> Dict[str, tf.Tensor]:
        log_probs = {}
        for name, reconstruction in reconstructions.items():
            target = targets[name]
            log_prob = losses.log_prob(reconstruction, target, batch_dims=2)
            if mask is not None:
                log_prob = tf.boolean_mask(log_prob, mask)
            if log_prob.shape[0] == 0:
                log_prob = tf.constant(0.0)
            log_probs[f'{name}_reconstruction'] = tf.reduce_mean(log_prob, name=f'{name}_reconstruction_loss')
        return log_probs

    @gin.configurable(whitelist=['scales'])
    def combine_losses(self, all_losses: Dict[str, tf.Tensor], scales: Dict[str, float] = gin.REQUIRED) -> tf.Tensor:
        total = 0.0
        for name, loss in all_losses.items():
            scale = scales.get(name, 0.0)
            if not scale:
                continue
            total += scale * loss
        return tf.identity(total, name='total_loss')

    def save_weights(self, filepath: str, **kwargs: Any) -> None:
        super().save_weights(filepath, **kwargs)
        path = Path(filepath)
        additional_data_file = path.parent / 'checkpoint_additional_data.pickle'
        additional_data = {'observation_components': self._observation_components,
                           'data_shapes': self._data_shapes,
                           'data_dtypes': self._data_dtypes}
        with open(additional_data_file, 'wb') as f:
            pickle.dump(additional_data, f, pickle.HIGHEST_PROTOCOL)
        latest_checkpoint_file = path.parent / 'checkpoint_latest'
        with open(latest_checkpoint_file, 'w') as f:
            f.write(path.name)


@measure_time
@gin.configurable(whitelist=['optimizer'])
def get_model(observation_components: Iterable[str],
              data_shapes: Dict[str, tf.TensorShape],
              data_dtypes: Dict[str, tf.DType],
              optimizer: tf.keras.optimizers.Optimizer,
              ) -> Model:
    """Returns a built model with random weights"""
    logger.info(f'Building model...')
    model = Model(observation_components, data_shapes, data_dtypes)
    model.compile(optimizer=optimizer, loss=None)
    model.build_with_input(model.dummy_data)  # Initialize weights
    model.reset_metrics()
    return model


def restore_model(checkpoint: Path, base_dir: Optional[Path] = None) -> Tuple[Model, int]:
    """Returns a fully initialized model and the index of the next epoch"""
    checkpoint = get_latest_checkpoint(checkpoint, base_dir)
    try:
        epoch = int(re.findall(r'epoch_(\d+)', checkpoint.name)[0]) + 1
    except (IndexError, ValueError):
        logger.warning(f"Could't read epoch from checkpoint name {checkpoint.name}")
        epoch = 0
    additional_data_file = checkpoint.parent / 'checkpoint_additional_data.pickle'
    if not additional_data_file.is_file():
        raise ValueError(f'Additional data not found, should be {additional_data_file}')
    with open(additional_data_file, 'rb') as f:
        additional_data = pickle.load(f)
    model = get_model(**additional_data)
    logger.info(f'Restoring weights from {checkpoint}...')
    measure_time(model.load_weights)(str(checkpoint))
    return model, epoch
