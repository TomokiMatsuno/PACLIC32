
#===============================================================
def symmetry_matrix_bilinear(inputs1, inputs2, output_size, n_splits=1, add_bias2=True, add_bias1=True, add_bias=True, initializer=None, moving_params=None):
  """"""

  with tf.variable_scope('Bilinear'):
    # Reformat the inputs
    print("this is real")
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims - 2]
    inputs1_size = inputs1.get_shape().as_list()[-1]

    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims - 2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    output_shape = []
    batch_size = 1
    for i in xrange(ndims - 2):
      batch_size *= inputs1_shape[i]
      output_shape.append(inputs1_shape[i])
    output_shape.append(inputs1_bucket_size)
    output_shape.append(output_size)
    output_shape.append(inputs2_bucket_size)
    output_shape = tf.stack(output_shape)

    inputs_reshaped1 = tf.reshape(inputs1, tf.stack([batch_size * inputs1_bucket_size, inputs1_size]))
    inputs_reshaped2 = tf.reshape(inputs2, tf.stack([batch_size * inputs2_bucket_size, inputs2_size]))

    # Get the matrix
    if initializer is None and moving_params is None:
      mat = orthonormal_initializer(inputs1_size, 1)[:, None, :]
      mat = np.concatenate([mat] * output_size, axis=1)
      initializer = tf.constant_initializer(mat)

    weights_re = tf.get_variable('Weights_re', [1, 1, output_size, inputs1_size],
                                 initializer=initializer)

    if moving_params is not None:
      weights_re = moving_params.average(weights_re)

    else:
      tf.add_to_collection('Weights_re', weights_re)


    X = tf.reshape(inputs1, tf.stack([batch_size, inputs1_bucket_size, 1, inputs1_size]))
    Y = tf.reshape(inputs2, tf.stack([batch_size, inputs1_bucket_size, inputs1_size]))
    W = weights_re

    lin = tf.multiply(X, W)
    bilin = tf.matmul(tf.reshape(lin, tf.stack([batch_size, -1, inputs2_size])), Y, transpose_b=True)

    bilin = tf.reshape(bilin, tf.stack([-1, output_size, inputs2_bucket_size]))
    
    bilin = tf.reshape(bilin, output_shape)

    if add_bias:
      bias_x = tf.get_variable('Biases_x', initializer=tf.zeros([1, inputs1_size, output_size]))
      bias_y = tf.get_variable('Biases_y', initializer=tf.zeros([1, inputs1_size, output_size]))

      if moving_params is not None:
        bias_x = moving_params.average(bias_x)
        bias_y = moving_params.average(bias_y)
      else:
        tf.add_to_collection('Biases_x', bias_x)
        tf.add_to_collection('Biases_y', bias_y)

      inputs1_summed = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_reshaped1, 2), bias_x), 1)
      inputs2_summed = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_reshaped2, 2), bias_y), 1)
      inputs1_summed = tf.reshape(inputs1_summed, tf.stack([batch_size, inputs1_bucket_size, output_size, 1]))
      inputs2_summed = tf.reshape(inputs2_summed, tf.stack([batch_size, inputs2_bucket_size, output_size, 1]))

      tmp = tf.add(bilin, inputs1_summed)
      inputs2_summed = tf.transpose(inputs2_summed, [0, 3, 2, 1])

      bilin = tf.add(tmp, inputs2_summed)

    if n_splits > 1:
      return tf.split(bilin, n_splits, n_dims-2)
    else:
      return bilin



def circulant_matrix_bilinear(inputs1, inputs2, output_size, n_splits=1, add_bias2=True, add_bias1=True, add_bias=True, initializer=None, moving_params=None):
  """"""

  with tf.variable_scope('Bilinear'):
    # Reformat the inputs
    ndims = len(inputs1.get_shape().as_list())
    inputs1_shape = tf.shape(inputs1)
    inputs1_bucket_size = inputs1_shape[ndims - 2]
    inputs1_size = inputs1.get_shape().as_list()[-1]

    inputs2_shape = tf.shape(inputs2)
    inputs2_bucket_size = inputs2_shape[ndims - 2]
    inputs2_size = inputs2.get_shape().as_list()[-1]
    output_shape = []
    batch_size = 1
    for i in xrange(ndims - 2):
      batch_size *= inputs1_shape[i]
      output_shape.append(inputs1_shape[i])
    output_shape.append(inputs1_bucket_size)
    output_shape.append(output_size)
    output_shape.append(inputs2_bucket_size)
    output_shape = tf.stack(output_shape)

    inputs_reshaped1 = tf.reshape(inputs1, tf.stack([batch_size * inputs1_bucket_size, inputs1_size]))
    inputs_reshaped2 = tf.reshape(inputs2, tf.stack([batch_size * inputs2_bucket_size, inputs2_size]))

    # Get the matrix
    if moving_params is None:
      print("this is circular")
      mat = np.random.rand(1, inputs1_size)[:, None, :]
      mat = np.concatenate([mat] * output_size, axis=1)

      mat_fft = np.fft.fft(mat)
      initializer_re = tf.constant_initializer(np.real(np.transpose(mat_fft, [2, 1, 0])))
      initializer_im = tf.constant_initializer(np.imag(np.transpose(mat_fft, [2, 1, 0])))
    else:
      initializer_re = initializer
      initializer_im = initializer

    weights_re = tf.get_variable('Weights_re', [1, 1, output_size, inputs1_size],
                                 initializer=initializer_re)
    weights_im = tf.get_variable('Weights_im', [1, 1, output_size, inputs2_size],
                                 initializer=initializer_im)

    if moving_params is not None:
      weights_re = moving_params.average(weights_re)
      weights_im = moving_params.average(weights_im)
    else:
      tf.add_to_collection('Weights_re', weights_re)
      tf.add_to_collection('Weights_im', weights_im)

    W = tf.concat([weights_re, weights_re, weights_im, tf.negative(weights_im)], axis=3)

    freq_v1 = tf.fft(tf.cast(inputs_reshaped1, tf.complex64))
    freq_v2 = tf.fft(tf.cast(inputs_reshaped2, tf.complex64))
    x_r, x_i = tf.real(freq_v1), tf.imag(freq_v1)
    y_r, y_i = tf.real(freq_v2), tf.imag(freq_v2)

    x_r = tf.reshape(x_r, tf.stack([batch_size, inputs1_bucket_size, 1, inputs1_size]))
    x_i = tf.reshape(x_i, tf.stack([batch_size, inputs1_bucket_size, 1, inputs1_size]))
    y_r = tf.reshape(y_r, tf.stack([batch_size, inputs1_bucket_size, inputs1_size]))
    y_i = tf.reshape(y_i, tf.stack([batch_size, inputs1_bucket_size, inputs1_size]))

    X = tf.concat([x_r, x_i, x_r, tf.negative(x_i)], axis=3)
    Y = tf.concat([y_r, y_i, y_i, tf.negative(y_r)], axis=2)

    lin = tf.multiply(X, W)
    bilin = tf.matmul(tf.reshape(lin, tf.stack([batch_size, -1, inputs2_size * 4])), Y, transpose_b=True)

    bilin = tf.reshape(bilin, tf.stack([-1, output_size, inputs2_bucket_size]))

    bilin = tf.reshape(bilin, output_shape)

    if add_bias:
      bias_x = tf.get_variable('Biases_x', initializer=tf.zeros([1, inputs1_size, output_size]))
      bias_y = tf.get_variable('Biases_y', initializer=tf.zeros([1, inputs1_size, output_size]))

      if moving_params is not None:
        bias_x = moving_params.average(bias_x)
        bias_y = moving_params.average(bias_y)
      else:
        tf.add_to_collection('Biases_x', bias_x)
        tf.add_to_collection('Biases_y', bias_y)

      inputs1_summed = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_reshaped1, 2), bias_x), 1)
      inputs2_summed = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_reshaped2, 2), bias_y), 1)
      inputs1_summed = tf.reshape(inputs1_summed, tf.stack([batch_size, inputs1_bucket_size, output_size, 1]))
      inputs2_summed = tf.reshape(inputs2_summed, tf.stack([batch_size, inputs2_bucket_size, output_size, 1]))
      
      tmp = tf.add(bilin, inputs1_summed)
      inputs2_summed = tf.transpose(inputs2_summed, [0, 3, 2, 1])
      
      bilin = tf.add(tmp, inputs2_summed)

    if n_splits > 1:
      return tf.split(bilin, n_splits, n_dims-2)
    else:
      return bilin

