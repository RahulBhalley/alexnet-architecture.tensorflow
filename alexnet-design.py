'''
AlexNet using TensorFlow library.

References:
- Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton.
  ImageNet Classification with Deep Convolutional Neural Networks. NIPS, 2012.

Link:
- https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
'''

__author__ = 'Rahul Bhalley'

import tensorflow as tf

n_inputs = 227 * 227 # number of input vector elements i.e. pixels per training example
n_classes = 1000 # number of classes to be classified

# input and output vector placeholders
x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# fully connected layer
fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)

def alex_net(img, weights, biases):

    # reshape the input image vector to 227 x 227 x 3 dimensions
    img = tf.reshape(img, [-1, 227, 227, 3])

    # 1st convolutional layer
    conv1 = tf.nn.conv2d(img, weights["wc1"], strides=[1, 4, 4, 1], padding="SAME", name="conv1")
    conv1 = tf.nn.bias_add(conv1, biases["bc1"])
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 2nd convolutional layer
    conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 1, 1, 1], padding="SAME", name="conv2")
    conv2 = tf.nn.bias_add(conv2, biases["bc2"])
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # 3rd convolutional layer
    conv3 = tf.nn.conv2d(conv2, weights["wc3"], strides=[1, 1, 1, 1], padding="SAME", name="conv3")
    conv3 = tf.nn.bias_add(conv3, biases["bc3"])
    conv3 = tf.nn.relu(conv3)

    # 4th convolutional layer
    conv4 = tf.nn.conv2d(conv3, weights["wc4"], strides=[1, 1, 1, 1], padding="SAME", name="conv4")
    conv4 = tf.nn.bias_add(conv4, biases["bc4"])
    conv4 = tf.nn.relu(conv4)

    # 5th convolutional layer
    conv5 = tf.nn.conv2d(conv4, weights["wc5"], strides=[1, 1, 1, 1], padding="SAME", name="conv5")
    conv5 = tf.nn.bias_add(conv5, biases["bc5"])
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # stretching out the 5th convolutional layer into a long n-dimensional tensor
    shape = [-1, weights['wf1'].get_shape().as_list()[0]]
    flatten = tf.reshape(conv5, shape)

    # 1st fully connected layer
    fc1 = fc_layer(flatten, weights["wf1"], biases["bf1"], name="fc1")
    fc1 = tf.nn.tanh(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    # 2nd fully connected layer
    fc2 = fc_layer(fc1, weights["wf2"], biases["bf2"], name="fc2")
    fc2 = tf.nn.tanh(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

    # 3rd fully connected layer
    fc3 = fc_layer(fc2, weights["wf3"], biases["bf3"], name="fc3")
    fc3 = tf.nn.softmax(fc3)

    # Return the complete AlexNet model
    return fc3

# Weight parameters as devised in the original research paper
weights = {
    "wc1": tf.Variable(tf.truncated_normal([11, 11, 3, 96],     stddev=0.01), name="wc1"),
    "wc2": tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name="wc2"),
    "wc3": tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name="wc3"),
    "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name="wc4"),
    "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name="wc5"),
    "wf1": tf.Variable(tf.truncated_normal([28*28*256, 4096],   stddev=0.01), name="wf1"),
    "wf2": tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name="wf2"),
    "wf3": tf.Variable(tf.truncated_normal([4096, n_classes],   stddev=0.01), name="wf3")
}

# Bias parameters as devised in the original research paper
biases = {
    "bc1": tf.Variable(tf.constant(0.0, shape=[96]), name="bc1"),
    "bc2": tf.Variable(tf.constant(1.0, shape=[256]), name="bc2"),
    "bc3": tf.Variable(tf.constant(0.0, shape=[384]), name="bc3"),
    "bc4": tf.Variable(tf.constant(1.0, shape=[384]), name="bc4"),
    "bc5": tf.Variable(tf.constant(1.0, shape=[256]), name="bc5"),
    "bf1": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf1"),
    "bf2": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf2"),
    "bf3": tf.Variable(tf.constant(1.0, shape=[n_classes]), name="bf3")
}
