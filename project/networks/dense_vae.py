# dense_vae.py: Variational autoencoder with fully-connected layers
#
# (C) 2019, Daniel Mouritzen

from typing import Dict, Optional, Sequence

import gin
import numpy as np
import tensorflow as tf

from project.util.tf import auto_shape
from project.util.tf.losses import kl_divergence, mse

from .basic import SequentialBlock, ShapedDense
from .wrappers import ExtraBatchDim


class DenseVAEEncoder(auto_shape.Layer):
    def __init__(self,
                 latent_shape: Sequence[int],
                 num_units: int = 100,
                 num_layers: int = 1,
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 name: str = 'vae_encoder',
                 ) -> None:
        super().__init__()
        self._encoder = SequentialBlock(num_units,
                                        num_layers,
                                        activation,
                                        batch_norm,
                                        initial_layers=[auto_shape.Flatten(name=f'{name}_flatten')],
                                        name=f'{name}_sequential')
        self._encoder.add(ShapedDense([2] + list(latent_shape), name=f'{name}_shaped_dense'))

    def call(self, input: tf.Tensor) -> Dict[str, tf.Tensor]:
        output = self._encoder(input)
        assert output.shape[1] == 2
        mean, log_var = output[:, 0], output[:, 1]
        sample = tf.random.normal(mean.shape) * tf.exp(log_var * .5) + mean
        return {'mean': mean, 'log_var': log_var, 'sample': sample}


class DenseVAEDecoder(auto_shape.Layer):
    def __init__(self,
                 output_shape: Sequence[int],
                 num_units: int = 100,
                 num_layers: int = 1,
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 name: str = 'vae_decoder',
                 ) -> None:
        super().__init__()
        self._decoder = SequentialBlock(num_units,
                                        num_layers,
                                        activation,
                                        batch_norm,
                                        initial_layers=[auto_shape.Flatten(name=f'{name}_flatten')],
                                        name=f'{name}_sequential')
        self._decoder.add(ShapedDense(output_shape, name=f'{name}_shaped_dense'))

    def call(self, input: tf.Tensor) -> tf.Tensor:
        return self._decoder(input)


@gin.configurable(whitelist=['num_units', 'num_layers', 'activation'])
class DenseVAE(auto_shape.Layer):
    def __init__(self,
                 input_shape: Sequence[int],
                 latent_shape: Sequence[int],
                 num_units: int = 100,
                 num_layers: int = 1,
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 name: str = 'vae',
                 ) -> None:
        self.encoder = ExtraBatchDim(DenseVAEEncoder(latent_shape,
                                                     num_units,
                                                     num_layers,
                                                     activation,
                                                     batch_norm,
                                                     name=f'{name}_encoder'))
        self.decoder = ExtraBatchDim(DenseVAEDecoder(input_shape,
                                                     num_units,
                                                     num_layers,
                                                     activation,
                                                     batch_norm,
                                                     name=f'{name}_decoder'))
        super().__init__(batch_dims=2, name=name)

    def call(self, input: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        z = self.encoder(input)
        input_recon = self.decoder(z['sample'])
        rec_loss = mse(input_recon, input, mask) * np.prod(input.shape[2:])
        kl_loss = kl_divergence(z['mean'], z['log_var'], mask)
        self.add_loss(rec_loss + kl_loss, inputs=True)
        self.add_metric(rec_loss, aggregation='mean', name=f'{self.name}_recon')
        self.add_metric(kl_loss, aggregation='mean', name=f'{self.name}_kl')
        return z['sample']
