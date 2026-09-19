# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:45:07 2020
@author: sifan
Updated on September 3 2026 - Tensorflow 2 version
@GamuceroJP

TensorFlow 2 replacement for the TF1-style jacobian() helper
(originally built on tf.gradients / pfor / tensorflow.python internals).

This version uses only public TF2 APIs (tf.GradientTape), works in both
eager mode and inside tf.function, and supports a nested structure of
inputs just like the original.
"""

import tensorflow as tf


def jacobian(f, inputs, parallel_iterations=None):
    """Computes the Jacobian of f(inputs) w.r.t. inputs.

    Args:
        f: A callable that takes `inputs` and returns a tensor `output`.
           (In TF2, GradientTape needs to watch the *computation*, so we
           pass a function instead of a precomputed `output` tensor.)
        inputs: A tensor, or a nested structure (list/tuple/dict) of tensors.
        parallel_iterations: Passed through to tape.jacobian's internal
            pfor call to control memory usage during the vectorized loop.

    Returns:
        A tensor or nested structure of tensors matching the structure of
        `inputs`. If output has shape [y_1, ..., y_n] and inputs_i has shape
        [x_1, ..., x_m], the corresponding jacobian has shape
        [y_1, ..., y_n, x_1, ..., x_m].
    """
    flat_inputs = tf.nest.flatten(inputs)

    with tf.GradientTape(persistent=True) as tape:
        for x in flat_inputs:
            tape.watch(x)
        output = f(inputs)

    flat_jacobians = [
        tape.jacobian(
            output,
            x,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
            parallel_iterations=parallel_iterations,
        )
        for x in flat_inputs
    ]

    del tape  # release the persistent tape's resources explicitly

    return tf.nest.pack_sequence_as(inputs, flat_jacobians)


if __name__ == "__main__":
    # Quick sanity check: f(x) = x^2 elementwise, jacobian should be diag(2x)
    x = tf.constant([1.0, 2.0, 3.0])

    def f(x):
        return x ** 2

    J = jacobian(f, x)
    print("x:", x.numpy())
    print("Jacobian:\n", J.numpy())
    # Expected: diagonal matrix with entries [2, 4, 6]

    # Nested-input example, mirroring PINN-style usage with (x, t) inputs
    x2 = tf.constant([[1.0], [2.0]])
    t2 = tf.constant([[0.5], [1.5]])

    def g(inputs):
        x_, t_ = inputs
        return x_ * t_ + x_ ** 2  # shape (2, 1)

    J2 = jacobian(g, (x2, t2))
    print("\nNested-input Jacobian (dg/dx, dg/dt):")
    for j in J2:
        print(j.numpy())