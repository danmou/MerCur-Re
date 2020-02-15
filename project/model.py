# model.py: Provides Model class
#
# (C) 2019, Daniel Mouritzen

import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import gin
import numpy as np
import tensorflow as tf
from loguru import logger

from project import networks
from project.util.files import get_latest_checkpoint
from project.util.system import is_debugging
from project.util.tf import auto_shape, combine_dims, swap_dims
from project.util.tf.discounting import lambda_return
from project.util.tf.losses import binary_crossentropy, mse
from project.util.timing import measure_time


@gin.configurable(whitelist=['predictor_class', 'rnn_class', 'dreamer', 'disable_tf_optimization'])
class Model(auto_shape.Model):
    """This class defines the top-level model structure and losses"""
    def __init__(self,
                 observation_components: Iterable[str],
                 data_spec: Mapping[str, tf.TensorSpec],
                 predictor_class: Type[networks.predictors.Predictor] = gin.REQUIRED,
                 rnn_class: Type[networks.rnns.RNN] = gin.REQUIRED,
                 dreamer: bool = False,
                 disable_tf_optimization: bool = False,
                 ) -> None:
        super().__init__(batch_dims=2, min_batch_shape=[1, 2])
        self._observation_components = list(observation_components)
        self._data_spec = data_spec
        self._batch_size = next(iter(data_spec.values())).shape[0]

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
        self.loss_fns: Dict[str, Any] = {'image': mse}
        for key in sorted(additional_observations | {'reward', 'done'}):
            data_shape = data_spec[key].shape[2:].as_list()
            if key == 'done':
                self.loss_fns[key] = binary_crossentropy
                kwargs = {'output_activation': 'sigmoid'}
            else:
                self.loss_fns[key] = mse
                kwargs = {}
            self.decoders[key] = self._get_vector_decoder(data_shape, **kwargs, name=f'{key}_decoder')
        # Layers in a dict are not automatically tracked, so we add them manually
        for layer in self.decoders.values():
            self._layers.append(layer)
        self._track_layers(self._layers)
        self.rnn = rnn_class(predictor_class)

        self._dreamer = dreamer
        if dreamer:
            self.action_network = self._get_action_network(data_spec['action'].shape[2:].as_list())
            self.value_network = self._get_value_network()
        else:
            self.action_network = None
            self.value_network = None

    @gin.configurable('Model.action_network', whitelist=['num_units', 'num_layers', 'activation', 'batch_norm'])
    def _get_action_network(self,
                            action_shape: Sequence[int],
                            num_units: int = gin.REQUIRED,
                            num_layers: int = gin.REQUIRED,
                            activation: Union[None, str, Type[tf.keras.layers.Layer]] = auto_shape.ReLU,
                            batch_norm: bool = False,
                            name: str = 'action_network'
                            ) -> auto_shape.Layer:
        return auto_shape.Sequential([networks.ExtraBatchDim(networks.SequentialBlock(num_units=num_units,
                                                                                      num_layers=num_layers,
                                                                                      activation=activation,
                                                                                      batch_norm=batch_norm,
                                                                                      name=f'{name}_block'),
                                                             name=f'{name}_block_ebd'),
                                      networks.TanhNormalTanh(action_shape,
                                                              extra_batch_dim=True,
                                                              name=f'{name}_dist')],
                                     batch_dims=2,
                                     name=name)

    @gin.configurable('Model.value_network', whitelist=['num_units', 'num_layers', 'activation', 'batch_norm'])
    def _get_value_network(self,
                           num_units: int = gin.REQUIRED,
                           num_layers: int = gin.REQUIRED,
                           activation: Union[None, str, Type[tf.keras.layers.Layer]] = auto_shape.ReLU,
                           batch_norm: bool = False,
                           name: str = 'value_network'
                           ) -> auto_shape.Layer:
        return cast(auto_shape.Layer, self._get_vector_decoder(output_shape=[],
                                                               num_units=num_units,
                                                               num_layers=num_layers,
                                                               activation=activation,
                                                               batch_norm=batch_norm,
                                                               name=name))

    @staticmethod
    @gin.configurable('Model.decoders', whitelist=['num_units', 'num_layers', 'activation', 'batch_norm'])
    def _get_vector_decoder(output_shape: Sequence[int],
                            num_units: int = gin.REQUIRED,
                            num_layers: int = gin.REQUIRED,
                            activation: Union[None, str, Type[tf.keras.layers.Layer]] = auto_shape.ReLU,
                            output_activation: Union[None, str, Type[tf.keras.layers.Layer]] = None,
                            batch_norm: bool = False,
                            name: str = 'vector_encoder'
                            ) -> auto_shape.Layer:
        return networks.ExtraBatchDim(auto_shape.Sequential([networks.SequentialBlock(num_units=num_units,
                                                                                      num_layers=num_layers,
                                                                                      activation=activation,
                                                                                      batch_norm=batch_norm,
                                                                                      name=f'{name}_block'),
                                                             networks.ShapedDense(output_shape,
                                                                                  activation=output_activation,
                                                                                  name=f'{name}_shaped_dense')],
                                                            name=f'{name}_sequential'),
                                      name=name)

    @staticmethod
    def _get_mask(data: Mapping[str, tf.Tensor]) -> tf.Tensor:
        return tf.sequence_mask(data['length'], tf.shape(data['reward'])[1])

    @property
    def dummy_data(self) -> Dict[str, tf.Tensor]:
        """Create dummy data suitable for initializing the model's weights"""
        data = {}
        for key in self._data_spec.keys():
            if key != 'length':
                data[key] = tf.zeros(self._data_spec[key].shape, self._data_spec[key].dtype)
        batch_shape = data['image'].shape[:2]
        data['length'] = tf.constant([[batch_shape[1]]] * batch_shape[0], self._data_spec['length'].dtype)
        return data

    def closed_loop(self,
                    data: Mapping[str, tf.Tensor],
                    **kwargs: Any,
                    ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        embedded = self.encoder(data, **kwargs)
        prior, posterior = self.rnn.closed_loop(embedded, data['action'], mask=self._get_mask(data), **kwargs)
        return prior, posterior

    @gin.configurable(whitelist=['context'])
    def open_loop(self, data: Mapping[str, tf.Tensor], context: int = 5, **kwargs: Any) -> Tuple[tf.Tensor, ...]:
        embedded = self.encoder(data, **kwargs)
        mask = self._get_mask(data)
        context = min(mask.shape[1] - 1, context)
        _, closed_loop = self.rnn.closed_loop(embedded[:, :context],
                                              data['action'][:, :context],
                                              mask=mask[:, :context],
                                              **kwargs)
        last_posterior = tf.nest.map_structure(lambda x: x[:, -1], closed_loop)
        open_loop = self.rnn.open_loop(data['action'][:, context:],
                                       initial_state=last_posterior,
                                       mask=mask[:, context:],
                                       **kwargs)
        return cast(Tuple[tf.Tensor, ...],
                    tf.nest.map_structure(lambda x, y: tf.concat([x, y], 1), closed_loop, open_loop))

    def decode(self, state_features: tf.Tensor, **kwargs: Any) -> Dict[str, tf.Tensor]:
        reconstructions = {}
        for name, decoder in self.decoders.items():
            reconstructions[name] = decoder(state_features, **kwargs)
        return reconstructions

    def call(self, inputs: Mapping[str, tf.Tensor], **kwargs: Any) -> tf.Tensor:
        inputs = dict(inputs)  # Shallow copy input dict so we can modify it safely
        if self._batch_size and tf.nest.flatten(inputs)[0].shape[0] is None:
            # Workaround for keras making the batch dimension undefined
            tf.nest.map_structure(lambda x: x.set_shape([self._batch_size] + x.shape[1:]), inputs)
        if inputs['length'].shape.ndims > 1:
            inputs['length'] = inputs['length'][:, 0]
        mask = self._get_mask(inputs)
        prior, posterior = self.closed_loop(inputs, **kwargs)
        features = self.rnn.state_to_features(posterior)
        reconstructions = self.decode(features, **kwargs)
        reconstruction_losses = self.reconstruction_loss(inputs, reconstructions, mask)
        for name, (loss, scale) in reconstruction_losses.items():
            self.add_named_loss(loss, name=f'{name}_recon', scaling=scale)

        if self._dreamer:
            imagined_states = self.imagine_forward(posterior)
            imagined_features = self.rnn.state_to_features(imagined_states)
            values = self.value_network(imagined_features, **kwargs)
            rewards = self.decoders['reward'](imagined_features, **kwargs)
            done_probs = self.decoders['done'](imagined_features, **kwargs)
            action_return = self.compute_action_return(values, rewards, done_probs)
            value_loss = self.compute_value_loss(values, rewards, done_probs)
            self.add_named_loss(action_return, name='action_return', scaling=-1.0, layer=self.action_network)
            # TODO: See if removing the layer constraint on value loss helps
            self.add_named_loss(value_loss, name='value_loss', scaling=1.0, layer=self.value_network)

        return tf.constant(0.0)

    @gin.configurable(whitelist=['horizon'])
    def imagine_forward(self, initial_states: Tuple[tf.Tensor, ...], horizon: int = 15, **kwargs: Any) -> Tuple[tf.Tensor, ...]:
        initial_states = tf.nest.map_structure(lambda x: tf.stop_gradient(x[:, :-1]), initial_states)
        initial_states = combine_dims(initial_states, [0, 1])  # type: ignore[assignment]

        def step_fn(prev: Tuple[tf.Tensor, ...], index: tf.Tensor) -> Tuple[tf.Tensor, ...]:
            features = tf.stop_gradient(self.rnn.state_to_features(prev))
            action = self.action_network(features[tf.newaxis, :], **kwargs).sample()[0, :]
            _, state = self.rnn.predictor.open_loop_predictor(action, prev)
            return cast(Tuple[tf.Tensor, ...], state)

        states = tf.scan(step_fn, tf.range(horizon), initial_states, back_prop=True)
        states = swap_dims(states, 0, 1)
        return cast(Tuple[tf.Tensor, ...], states)

    @gin.configurable(whitelist=['lambda_'])
    def compute_action_return(self,
                              values: tf.Tensor,
                              rewards: tf.Tensor,
                              done_probs: tf.Tensor,
                              lambda_: float = 1.0,
                              ) -> tf.Tensor:
        rewards = rewards[:, :-1]
        final_value = values[:, -1]
        values = values[:, :-1]
        discounts = 1 - done_probs[:, :-1]
        return_ = lambda_return(rewards, values, discounts, lambda_, final_value, axis=1, stop_gradient=False)

        # TODO: Check if this helps
        return_ *= self.discount_factors(discounts)
        return tf.reduce_mean(return_)

    @gin.configurable(whitelist=['lambda_'])
    def compute_value_loss(self,
                           values: tf.Tensor,
                           rewards: tf.Tensor,
                           done_probs: tf.Tensor,
                           lambda_: float = 1.0,
                           ) -> tf.Tensor:
        rewards = rewards[:, :-1]
        final_value = values[:, -1]
        values = values[:, :-1]
        discounts = tf.stop_gradient(1 - done_probs[:, :-1])
        return_ = lambda_return(rewards, values, discounts, lambda_, final_value, axis=1, stop_gradient=True)
        loss = mse(values, return_, reduce=False)

        # TODO: Check if this helps
        loss *= self.discount_factors(discounts)
        return tf.reduce_mean(loss)

    @staticmethod
    def discount_factors(discounts: tf.Tensor) -> tf.Tensor:
        """[1, gamma, gamma**2, ...]"""
        return tf.stop_gradient(tf.math.cumprod(tf.concat([tf.ones_like(discounts[:, :1]), discounts[:, :-1]], 1), 1))

    @gin.configurable(whitelist=['scales'])
    @tf.function(experimental_relax_shapes=True)
    def reconstruction_loss(self,
                            targets: Mapping[str, tf.Tensor],
                            reconstructions: Mapping[str, tf.Tensor],
                            mask: Optional[tf.Tensor] = None,
                            scales: Mapping[str, float] = gin.REQUIRED,
                            ) -> Dict[str, Tuple[tf.Tensor, float]]:
        losses = {}
        for name, reconstruction in reconstructions.items():
            assert name in scales, f'No reconstruction loss scale specified for {name!r}'
            target = targets[name]
            scale = scales[name]
            losses[name] = (self.loss_fns[name](reconstruction, target, mask, name=f'{name}_recon_loss'), scale)
        return losses

    def save_weights(self, filepath: str, **kwargs: Any) -> None:
        super().save_weights(filepath, **kwargs)
        path = Path(filepath)
        additional_data_file = path.parent / 'checkpoint_additional_data.pickle'
        additional_data = {'observation_components': self._observation_components,
                           'data_spec': self._data_spec}
        with open(additional_data_file, 'wb') as f1:
            pickle.dump(additional_data, f1, pickle.HIGHEST_PROTOCOL)
        latest_checkpoint_file = path.parent / 'checkpoint_latest'
        with open(latest_checkpoint_file, 'w') as f2:
            f2.write(path.name)


@measure_time
@gin.configurable(whitelist=['optimizers'])
def get_model(observation_components: Iterable[str],
              data_spec: Mapping[str, tf.TensorSpec],
              optimizers: Mapping[str, tf.keras.optimizers.Optimizer],
              ) -> Model:
    """Returns a built model with random weights"""
    assert 'model' in optimizers.keys(), 'No optimizer specified for base model'
    logger.info(f'Building model...')
    model = Model(observation_components, data_spec)
    model.compile(optimizer=list(optimizers.values()), loss=None)
    model.optimizer_targets = list(optimizers.keys())
    model.build_with_input(model.dummy_data)  # Initialize weights
    model.reset_metrics()
    return model


@measure_time
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
    # Model.load_weights with by_name=True will silently ignore if any weights are missing from the checkpoint,
    # so we have to save the current weights so we can see if they changed.
    initial_weights = [layer.get_weights() for layer in model.layers]
    measure_time(model.load_weights)(str(checkpoint), by_name=True)
    # Check if weights were loaded for all layers
    for layer, initial in zip(model.layers, initial_weights):
        weights = layer.get_weights()
        if weights and all(tf.nest.flatten(tf.nest.map_structure(np.array_equal, weights, initial))):
            logger.warning(f'Checkpoint contained no weights for layer {layer.name}!')
    return model, epoch
