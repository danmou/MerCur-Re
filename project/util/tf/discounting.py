# discounting.py: Discounted return functions
#
# (C) 2020, Daniel Mouritzen

from typing import Optional, Tuple, Union

import tensorflow as tf

from .general import move_dim, scan


def discounted_return(rewards: tf.Tensor,
                      discount: Union[tf.Tensor, float],
                      final_value: Optional[tf.Tensor] = None,
                      axis: int = 1,
                      stop_gradient: bool = True,
                      ) -> tf.Tensor:
    """
    Calculate discounted return as per this formula:
        V[t] = sum(discount**n * rewards[t+n] for n in range(len(rewards)-t)) + discount**(len(rewards)-t) * final_value

    For numerical stability, it is implemented recursively:
        V[last+1] = final_value
        V[t] = rewards[t] + discount * V[t + 1]
    """
    if isinstance(discount, (float, int)) or discount.shape.num_elements() == 1:
        if discount == 1:
            return_ = tf.reduce_sum(rewards, axis)
            if final_value is not None:
                return_ += final_value
            return return_
        discount = discount * tf.ones_like(rewards)
    else:
        assert rewards.shape == discount.shape, (rewards.shape, discount.shape)
    if final_value is None:
        final_value = tf.zeros_like(rewards[-1])
    return_ = scan(fn=lambda accumulated, current: current[0] + current[1] * accumulated,
                   elems=(rewards, discount),
                   initializer=final_value,
                   back_prop=not stop_gradient,
                   axis=axis,
                   reverse=True)
    if stop_gradient:
        return_ = tf.stop_gradient(return_)
    return return_


def lambda_return(rewards: tf.Tensor,
                  values: tf.Tensor,
                  discount: Union[tf.Tensor, float],
                  lambda_: float,
                  final_value: Optional[tf.Tensor] = None,
                  axis: int = 1,
                  stop_gradient: bool = True,
                  ) -> tf.Tensor:
    """
    Calculate lambda return as per this formula:
        dr(t, n) = discounted_return(reward[t:t + n], discount[t:t + n], values[t + n])[0]
        V[t] = ((1 - lambda_) * sum(lambda_**(n - 1) * dr(t, n) for n in range(1, T - t))
                + lambda_**(T - t - 1) * dr(t, T - t))

    For numerical stability, it is implemented recursively:
        V[last+1] = final_value
        V[t] = rewards[t] + discount * ((1 - lambda_) * values[t + 1] + lambda * V[t + 1])

    Setting lambda=1 gives a discounted Monte Carlo return.
    Setting lambda=0 gives a fixed 1-step return.
    """
    if isinstance(discount, (int, float)) or discount.shape.num_elements() == 1:
        discount = discount * tf.ones_like(rewards)
    assert rewards.shape == values.shape == discount.shape, 'Incompatible shapes!'
    rewards, values, discount = move_dim((rewards, values, discount), axis, 0)
    if final_value is None:
        final_value = tf.zeros_like(values[-1])
    next_values = tf.concat([values[1:], final_value[tf.newaxis]], 0)

    def fn(accumulated: tf.Tensor, current: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        reward, next_value, d = current
        return reward + d * ((1 - lambda_) * next_value + lambda_ * accumulated)

    return_ = scan(fn=fn,
                   elems=(rewards, next_values, discount),
                   initializer=final_value,
                   back_prop=not stop_gradient,
                   axis=0,
                   reverse=True)
    if stop_gradient:
        return_ = tf.stop_gradient(return_)
    return move_dim(return_, 0, axis)
