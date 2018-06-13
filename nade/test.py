import tensorflow as tf
import numpy as np

_a = np.arange(1,25).reshape([2,3,4,1])
_b = np.ones([2,3,4,2])
_d = np.ones([2*3,1])

# Arbitrarity, we'll use placeholders and allow batch size to vary,
# but fix vector dimensions.
# You can change this as you see fit
a = tf.placeholder(tf.float32, shape=(None,3,4,1))
b = tf.placeholder(tf.float32, shape=(None,3,4,2))
d = tf.placeholder(tf.int32, shape=(None,1))

#c = tf.tensordot(a,b,2)
#c = tf.reshape(tf.multiply(a,b),[-1,1])
c = tf.reduce_sum( tf.multiply( a, b ), 2)#, keep_dims=True )
#c = tf.reshape(c,[-1,2])
#depth = tf.shape(d)
#depth = tf.Print(depth,[depth],"depth")
#x = tf.map_fn(lambda x:x,d)
#x = tf.reshape(tf.range(depth[0]),[-1,1])
#x = tf.concat([x[:d.shape[0]],d],axis=1)

#c = tf.gather_nd(c,x)
#c = tf.Print(c , [c],summarize=5)
#:c = tf.reshape(c,[2,3])
with tf.Session() as session:
    print( c.eval(
        feed_dict={ a: _a, b: _b , d:_d}
    ))
