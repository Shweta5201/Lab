from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import tf.layers.InputSpec 

class RRNNCell(tf.contrib.rnn.RNNCell):
  """The most basic RNN cell in NADE setting.
  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self, num_units, activation=None, reuse=None, 
    name=None, debug=False):
    #super(RRNNCell, self).__init__(_reuse=reuse, name=name)

    self._num_units = num_units
    if activation=="tanh":
      self._activation = tf.tanh
    elif activation == "relu":
      self._activation = tf.nn.relu
    else:
      self._activation = activation
    self.is_debug = debug

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def _debug(self):
    return self.is_debug

  def __call__(self, inputs, state, scope=None):
    """ RNN: output = new_state = act( W * state + input )."""

    inputs_u, inputs_i = tf.split(value=inputs, num_or_size_splits=2, axis=1)
    
    with tf.variable_scope(scope or "basic_rnn_cell"):
      recc_kernel = tf.get_variable("recurrent_kernel",
        [self.state_size, self.state_size], dtype=tf.float32)
      recc_bias = tf.get_variable("recurrent_bias",
        [self.state_size], dtype = tf.float32)
      wh = tf.add(tf.matmul(state,recc_kernel),recc_bias)
      #wh = state
        
      u_bias = tf.get_variable("user_bias",
        [self.state_size], dtype = tf.float32)
      i_bias = tf.get_variable("item_bias",
        [self.state_size], dtype = tf.float32)
      inputs_u = tf.add(inputs_u,u_bias)
      inputs_i = tf.add(inputs_i,i_bias)

#      wh = tf.Print(wh,[tf.shape(wh)],"state shape")
#      inputs_i = tf.Print(inputs_i,[tf.shape(inputs_i)],
#        "Inputs shape for cell")

      #new_state = tf.add(tf.add(wh,inputs_u),inputs_i)
      output = self._activation(tf.add(tf.add(wh,inputs_u),inputs_i))
      #output = self._activation(tf.add(new_state,recc_bias))

      if self._debug:
        output = tf.Print(output, [output],
          summarize=10, message="Recurrent cell calculations")

    return output, output

class CGRUCell(tf.contrib.rnn.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(GRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable(
        "gates/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, 2 * self._num_units],
        initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable(
        "gates/%s" % _BIAS_VARIABLE_NAME,
        shape=[2 * self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable(
        "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable(
        "candidate/%s" % _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=(
            self._bias_initializer
            if self._bias_initializer is not None
            else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells.
    Gate equations:
	r, z = sig( W * state + U * inputs + bias)
        new_state = r * state
        c = act( W_c * new_state + U_c * inputs + bias_c )
        output = state = z * state + (1 - z) * c
    """

    inputs_u, inputs_i = tf.split(value=inputs, num_or_size_splits=2, axis=1)
    gate_inputs = math_ops.matmul(
        array_ops.concat([inputs, state], 1), self._gate_kernel)
    gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

    value = math_ops.sigmoid(gate_inputs)
    r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

    r_state = r * state

    candidate = math_ops.matmul(
        array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
    candidate = nn_ops.bias_add(candidate, self._candidate_bias)

    c = self._activation(candidate)
    new_h = u * state + (1 - u) * c
    return new_h, new_h

