import tensorflow as tf
print('Tensorflow Version:{}'.format(tf.__version__))
print('Tensorflow Path:{}'.format(tf.__file__))
tf.config.list_physical_devices('GPU')

