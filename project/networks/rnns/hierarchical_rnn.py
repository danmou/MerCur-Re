# hierarchical_rnn.py: Hierarchical RNN class for use with Predictor cells
#
# (C) 2019, Daniel Mouritzen

from typing import List, Optional, Tuple, Type, Union

import gin
import tensorflow as tf

from project.networks.dense_vae import DenseVAE
from project.networks.predictors.base import OpenLoopPredictor, Predictor
from project.networks.predictors.rssm import OpenLoopRSSMPredictor, RSSMPredictor
from project.util.tf import sliding_window

from .base import RNN, SimpleRNN


@gin.configurable(module='rnns', whitelist=['open_loop_predictor_class', 'time_scales', 'divergence_loss_scales',
                                            'divergence_loss_free_nats', 'action_embedding_sizes'])
class HierarchicalRNN(RNN):
    def __init__(self,
                 predictor_class: Type[Predictor] = RSSMPredictor,
                 *,
                 open_loop_predictor_class: Type[OpenLoopPredictor] = OpenLoopRSSMPredictor,
                 time_scales: List[int] = gin.REQUIRED,
                 divergence_loss_scales: List[float] = gin.REQUIRED,
                 divergence_loss_free_nats: float = 3.0,
                 action_embedding_sizes: List[int] = gin.REQUIRED,
                 name: str = 'hierarchical_rnn',
                 ) -> None:
        assert time_scales[0] == 1, 'First time scale must be 1.'
        assert len(time_scales) == len(action_embedding_sizes), 'There must be exactly one action embedding size for each time scale.'
        assert len(time_scales) == len(divergence_loss_scales), 'There must be exactly one divergence loss scale for each time scale.'
        self.time_scales = time_scales
        self.divergence_loss_scales = divergence_loss_scales
        self.divergence_loss_free_nats = divergence_loss_free_nats
        self.action_embedding_sizes = action_embedding_sizes
        self.base_rnn = SimpleRNN(predictor_class,
                                  divergence_loss_scale=divergence_loss_scales[0],
                                  divergence_loss_free_nats=divergence_loss_free_nats,
                                  name=f'{name}_base')
        self.predictors = list(open_loop_predictor_class(name=f'{name}_predictor_{s}') for s in time_scales[1:])
        self.action_vaes: List[DenseVAE] = []
        for scale_i, scale_j, embed_size_i, embed_size_j in zip(time_scales,
                                                                time_scales[1:],
                                                                action_embedding_sizes,
                                                                action_embedding_sizes[1:]):
            assert scale_i < scale_j, f'Time scales must be strictly increasing! {time_scales}'
            assert scale_j % scale_i == 0, f'Each time scale must be a multiple of the previous one! {time_scales}'
            factor = scale_j // scale_i
            self.action_vaes.append(DenseVAE(input_shape=(factor, embed_size_i),
                                             latent_shape=(embed_size_j,),
                                             name=f'action_vae_{scale_i}_{scale_j}'))
        super().__init__(predictor_class=predictor_class, name=name, min_batch_shape=[1, time_scales[-1] + 1])

    @property
    def predictor(self) -> Predictor:
        return self.base_rnn.predictor

    def state_to_features(self, state: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        return self.predictor_class.state_to_features(state)

    def state_divergence(self,
                         state1: Tuple[tf.Tensor, ...],
                         state2: Tuple[tf.Tensor, ...],
                         mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        return self.predictor_class.state_divergence(state1, state2, mask)

    def closed_loop(self,
                    observations: tf.Tensor,
                    actions: tf.Tensor,
                    initial_state: Optional[tf.Tensor] = None,
                    mask: Optional[tf.Tensor] = None,
                    training: Optional[Union[tf.Tensor, bool]] = None,
                    ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        # This is the first point where we know the size of the action space, so we perform this check here
        assert actions.shape[-1] == self.action_embedding_sizes[0], 'First action embedding size must match action space.'
        return super().closed_loop(observations, actions, initial_state, mask, training)

    def _apply_time_scales(self,
                           actions: tf.Tensor,
                           mask: Optional[tf.Tensor] = None,
                           training: Optional[Union[tf.Tensor, bool]] = None,
                           ) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        action_sequences = [actions]
        masks = [mask if mask is not None else tf.ones(actions.shape[:2], tf.bool)]
        for prev_scale, scale, vae in zip(self.time_scales, self.time_scales[1:], self.action_vaes):
            factor = scale // prev_scale
            prev_actions = action_sequences[-1]
            prev_mask = masks[-1]
            if actions.shape[1] < scale + 1:
                break
            # TODO: This is basically equivalent to 1D convolution, perhaps we should just use that
            actions_windowed = sliding_window(prev_actions, factor, axis=1)
            action_sequences.append(vae(actions_windowed, training=training))
            mask_windowed = sliding_window(prev_mask, factor, axis=1)
            masks.append(tf.reduce_all(mask_windowed, axis=-1))
        return action_sequences[1:], masks[1:]

    def call(self,
             inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
             initial_state: Optional[tf.Tensor] = None,
             mask: Optional[tf.Tensor] = None,
             training: Optional[Union[tf.Tensor, bool]] = None,
             ) -> Tuple[Tuple[tf.Tensor, ...], Tuple[tf.Tensor, ...]]:
        prior: Tuple[tf.Tensor, ...]
        posterior: Tuple[tf.Tensor, ...]
        prior, posterior = self.base_rnn(inputs, initial_state=initial_state, mask=mask, training=training)
        obs, base_actions, use_obs = inputs
        action_sequences, masks = self._apply_time_scales(base_actions, mask, training)
        for actions, mask, time_scale, loss_scale, predictor in zip(action_sequences,
                                                                    masks,
                                                                    self.time_scales[1:],
                                                                    self.divergence_loss_scales[1:],
                                                                    self.predictors):
            # TODO: Investigate whether using the prior or posterior as target works better
            # Note: prior is only open-loop for the last step; if this works best we should see if it's even better to
            # use `time_scale`-step open-loop predictions
            target = tf.nest.map_structure(lambda x: x[:, time_scale:], posterior)
            post, actions, mask = tf.nest.map_structure(lambda x: x[:, :target[0].shape[1]], (posterior, actions, mask))
            post, actions = tf.nest.map_structure(lambda x: tf.reshape(x, [-1] + x.shape[2:].as_list()), (post, actions))
            _, predictions = predictor(actions, post, training=training)
            predictions = tf.nest.map_structure(lambda x: tf.reshape(x, mask.shape[:2] + x.shape[1:]), predictions)
            divergence_loss = self.divergence_loss(predictions, target, mask=mask, free_nats=self.divergence_loss_free_nats)
            self.add_loss(divergence_loss * loss_scale, inputs=True)
            self.add_metric(divergence_loss, aggregation='mean', name=f'divergence_scale_{time_scale}')
        return prior, posterior
