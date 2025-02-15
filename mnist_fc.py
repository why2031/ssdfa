
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="glorot_uniform")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="mnist_conv")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import tensorflow as tf
import keras
import numpy as np

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.Activation import Relu
from lib.Activation import Tanh

##############################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_examples = 60000
test_examples = 10000

assert(np.shape(x_train) == (train_examples, 28, 28))
x_train = np.reshape(x_train, [train_examples, 28, 28, 1])
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (test_examples, 28, 28))
x_test = np.reshape(x_test, [test_examples, 28, 28, 1])
y_test = keras.utils.to_categorical(y_test, 10)

##############################################

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
lr = tf.placeholder(tf.float32, shape=())

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)
Y = tf.placeholder(tf.float32, [None, 10])

l0 = ConvToFullyConnected(input_shape=[28, 28, 1])

l1 = FullyConnected(input_shape=784, size=400, init=args.init, activation=act, bias=args.bias, name='fc1')
l2 = Dropout(rate=dropout_rate)
l3 = FeedbackFC(size=[784, 400], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc1_fb')

l4 = FullyConnected(input_shape=400, size=10, init=args.init, bias=args.bias, name='fc2')

##############################################

model = Model(layers=[l0, l1, l2, l3, l4])
predict = model.predict(X=X)
weights = model.get_weights()

if args.dfa:
    grads_and_vars = model.dfa_gvs(X=X, Y=Y)
else:
    grads_and_vars = model.gvs(X=X, Y=Y)
        
train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

##############################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

train_accs = []
test_accs = []

for ii in range(args.epochs):

    #############################
    
    _total_correct = 0
    for jj in range(0, train_examples, args.batch_size):
        s = jj
        e = min(jj + args.batch_size, train_examples)
        b = e - s
        
        xs = x_train[s:e]
        ys = y_train[s:e]
        
        _correct, _ = sess.run([total_correct, train], feed_dict={batch_size: b, dropout_rate: args.dropout, lr: args.lr, X: xs, Y: ys})
        _total_correct += _correct

    train_acc = 1.0 * _total_correct / (train_examples - (train_examples % args.batch_size))
    train_accs.append(train_acc)

    #############################

    _total_correct = 0
    for jj in range(0, test_examples, args.batch_size):
        s = jj
        e = min(jj + args.batch_size, test_examples)
        b = e - s
        
        xs = x_test[s:e]
        ys = y_test[s:e]
        
        _correct = sess.run(total_correct, feed_dict={batch_size: b, dropout_rate: 0.0, lr: 0.0, X: xs, Y: ys})
        _total_correct += _correct
        
    test_acc = 1.0 * _total_correct / (test_examples - (test_examples % args.batch_size))
    test_accs.append(test_acc)
    
    #############################
            
    p = "%d | train acc: %f | test acc: %f" % (ii, train_acc, test_acc)
    print (p)
    f = open(filename, "a")
    f.write(p + "\n")
    f.close()

##############################################

if args.save:
    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_accs
    w['test_acc'] = test_accs
    np.save(args.name, w)
    
##############################################

import matplotlib.pyplot as plt
import numpy as np

# 随机选择几个测试样本进行展示
sample_indices = np.random.choice(len(x_test), 5, replace=False)
sample_images = x_test[sample_indices]
sample_labels = y_test[sample_indices]

# 获取这些样本的预测结果
pred_probs = sess.run(predict, feed_dict={
    X: sample_images,
    dropout_rate: 0.0,
    lr: 0.0,
    batch_size: len(sample_images)
})
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(sample_labels, axis=1)

for i in range(len(sample_images)):
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title("True: {}, Pred: {}".format(true_labels[i], pred_labels[i]))
    plt.axis('off')
    plt.show()



