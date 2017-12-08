import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class CNN4Rec:
    def __init__(self, args):
        self.args = args
        if not args.is_training:
            self.args.batch_size = 1
        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        self.predict = False

###########################################ACTIVATION FUNCTION#########################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def softmax(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)
############################################Layer########################################
    def conv_layer(self, input_tensor, name, kh, kw, n_out, dh=1, dw=1):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
            biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
            activation = self.relu(tf.nn.bias_add(conv, biases))
            return activation
    def fc_layer(self, input_tensor, name, n_out):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
            biases = tf.get_variable('bias', [n_out], tf.float32, tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
            return self.relu(logits)
    def max_pool(self, input_tensor, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_tensor, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='VALID', name=name)
    def loss(self, logits, onehot_labels):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=onehot_labels, name='xentropy')      
        loss = tf.reduce_mean(xentropy, name='loss')
        return loss
#########################################################################################
    def build_model(self):
        self.imgs = tf.placeholder(tf.float32, [self.args.batch_size, 100, 256, 1], name='input')
        self.labels = tf.placeholder(tf.float32, [self.args.batch_size, self.args.n_classes], name='output')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.conv1 = self.conv_layer(self.imgs, 'conv1', kh=3, kw=3, n_out=64)
        self.pool1 = self.max_pool(self.conv1, 'pool1', kh=2, kw=2, dh=2, dw=2)
        self.conv2 = self.conv_layer(self.pool1, 'conv2', kh=3, kw=3, n_out=128)
        self.pool2 = self.max_pool(self.conv2, 'pool2', kh=2, kw=2, dh=2, dw=2)
        self.conv3 = self.conv_layer(self.pool2, 'conv3', kh=3, kw=3, n_out=256)
        self.pool3 = self.max_pool(self.conv3, 'pool3', kh=2, kw=2, dh=2, dw=2)
        self.conv4 = self.conv_layer(self.pool3, 'conv4', kh=3, kw=3, n_out=512)
        self.pool4 = self.max_pool(self.conv4, 'pool4', kh=2, kw=2, dh=2, dw=2)
        self.flattened_shape = np.prod([s.value for s in self.pool4.get_shape()[1:]])
        self.flatten = tf.reshape(self.pool4, [-1, self.flattened_shape], name='flatten')
        self.fc1 = self.fc_layer(self.flatten, 'fc1', n_out=128)
        self.drop = tf.nn.dropout(self.fc1, self.args.keep_prob)
        self.fc2 = self.fc_layer(self.drop, 'fc2', n_out=self.args.n_classes)
        #self.probs = tf.nn.softmax(self.fc2, name='prob')
        self.probs = self.fc2
        self.prob = self.sigmoid(self.probs)
        print 'build model finished'
        if self.args.is_training:
            self.cost = self.loss(self.probs, self.labels)
        else:
            self.preds = self.sigmoid(self.probs)


        if not self.args.is_training:
            return

        self.lr = tf.maximum(1e-5, tf.train.exponential_decay(self.args.learning_rate, self.global_step, self.args.decay_steps, self.args.decay, staircase=True))
        optimizer = tf.train.AdamOptimizer(self.lr)
        
        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.args.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.args.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
    
    def predict_label(self, sess, in_img):
        if in_img.shape[0] != 1:
            raise Exception('Predict batch size must be one!')
        fetches = self.preds
        feed_dict = {self.imgs: in_img}
        preds = sess.run(fetches, feed_dict)
        return preds.T
            
        
    

