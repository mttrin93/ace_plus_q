import dataclasses
import tensorflow as tf

DOUBLE_FLOAT_TYPE: tf.dtypes = tf.float64
SINGLE_FLOAT_TYPE: tf.dtypes = tf.float32

DOUBLE_INT_TYPE: tf.dtypes = tf.int64
SINGLE_INT_TYPE: tf.dtypes = tf.int32

DEFAULT_FLOAT_TYPE: tf.dtypes = DOUBLE_FLOAT_TYPE
DEFAULT_INT_TYPE: tf.dtypes = SINGLE_INT_TYPE


@dataclasses.dataclass
class ModelDataTypes:
    float: tf.dtypes = DEFAULT_FLOAT_TYPE
    int: tf.dtypes = DEFAULT_INT_TYPE

