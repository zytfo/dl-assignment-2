# Part 1: Deep Learning 

import tensorflow as tf
from functools import partial
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from DNNClassifier import DNNClassifier
from sklearn.model_selection import RandomizedSearchCV

def shuffle_split(X, y, n_batches):
    np.random.seed(seed=42)
    rnd_idx = np.random.permutation(len(X))
    for i_idx in np.array_split(rnd_idx, n_batches):
        X_batch = X[i_idx]
        y_batch = y[i_idx]
        yield X_batch, y_batch

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

'''
1. Build a DNN with five hidden layers of 100 neurons each, He initialization, and the ELU
activation function.
'''

print('Task 1 start')
time.sleep(1)

n_inputs = 28 * 28
n_hidden1 = 100
n_hidden2 = 100
n_hidden3 = 100
n_hidden4 = 100
n_hidden5 = 100
n_outputs = 5

he_init = tf.contrib.layers.variance_scaling_initializer()
dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init)
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    hidden1 = dense_layer(X, n_hidden1, name='hidden1')
    hidden2 = dense_layer(hidden1, n_hidden2, name='hidden2')
    hidden3 = dense_layer(hidden2, n_hidden3, name='hidden3')
    hidden4 = dense_layer(hidden3, n_hidden4, name='hidden4')
    hidden5 = dense_layer(hidden4, n_hidden5, name='hidden5')
    logits = dense_layer(hidden5, n_outputs, activation=None, name='outputs')
    
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

'''
2. Using Adam optimization and early stopping, try training it on MNIST but only on digits 0
to 4, as we will use transfer learning for digits 5 to 9 in the next exercise. You will need a
softmax output layer with five neurons.
'''

print('Task 2 start')
time.sleep(1)

learning_rate = 0.001

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
mnist = input_data.read_data_sets('/tmp/data/')
X_train = mnist.train.images[mnist.train.labels < 5]
y_train = mnist.train.labels[mnist.train.labels < 5]
X_test = mnist.test.images[mnist.test.labels < 5]
y_test = mnist.test.labels[mnist.test.labels < 5]
X_valid = mnist.validation.images[mnist.validation.labels < 5]
y_valid = mnist.validation.labels[mnist.validation.labels < 5]


n_epochs = 50
batch_size = 50
n_batches = len(X_train) // batch_size
best_loss = float('inf')
patience = 2
cnt_patience = 0
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_split(X_train, y_train, n_batches):
            sess.run([training_op, loss], feed_dict={X: X_batch, y: y_batch})
        accuracy_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        accuracy_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        loss_test = loss.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, 'train_acc:', accuracy_train, 'test_acc:', accuracy_test, 'loss', loss_test,)
        if loss_test < best_loss:
            best_loss = loss_test
        else:
            cnt_patience += 1
            if cnt_patience > patience:
                'Early stopping!'
                break

'''
3. Tune the hyperparameters using cross-validation and see what precision you can
achieve.
'''

print('Task 3 start')
time.sleep(1)

param_distribs = {
    "n_neurons": [10, 100, 150],
    "batch_size": [10, 50],
    "learning_rate": [0.01, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu],
    "n_hidden_layers": [0, 1, 3],
    "optimizer_class": [tf.train.AdamOptimizer, tf.train.AdagradOptimizer]
}

random_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,fit_params={"X_valid": X_valid, "y_valid": y_valid, "n_epochs": 10},
                                random_state=42, verbose=2)

random_search.fit(X_train, y_train)
y_pred = rnd_search.predict(X_test)
accuracy_score(y_test, y_pred)

'''
4. Now try adding Batch Normalization and compare the learning curves: is it converging
faster than before? Does it produce a better model?
'''

print('Task 4 start')
time.sleep(1)

dnn_clf_bn = DNNClassifier(activation=leaky_relu(alpha=0.1), batch_size=500, learning_rate=0.01,n_neurons=90, random_state=42,
                           batch_norm_momentum=0.95)
dnn_clf_bn.fit(X_train, y_train, n_epochs=10, X_valid=X_valid, y_valid=y_valid)


'''
5. Is the model overfitting the training set? Try adding dropout to every layer and try
again. Does it help?
'''

param_distribs = {
    "n_neurons": [10, 100, 150],
    "batch_size": [10, 50],
    "learning_rate": [0.01, 0.1],
    "activation": [tf.nn.relu, tf.nn.elu],
    "n_hidden_layers": [0, 1, 3],
    "optimizer_class": [tf.train.AdamOptimizer, tf.train.AdagradOptimizer],
    "dropout_rate": [0.2, 0.4],
    "batch_norm_momentum": [0.9, 0.95, 0.98]
}


rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50, fit_params={"X_valid": X_valid, "y_valid": y_valid, "n_epochs": 10},
                                random_state=42, verbose=2)
rnd_search.fit(X_train, y_train)
y_pred = rnd_search.predict(X_test)
accuracy_score(y_test, y_pred)
rnd_search.best_estimator_.save("./model")