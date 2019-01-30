# Part 2: Transfer learning

import tensorflow as tf
import numpy as np
import os, sys, time
sys.path.append(os.getcwd())
from DNNClassifier import DNNClassifier
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score

he_init = tf.contrib.layers.variance_scaling_initializer()
mnist = input_data.read_data_sets("/tmp/data/")

X_train_full = mnist.train.images[mnist.train.labels >= 5]
y_train_full = mnist.train.labels[mnist.train.labels >= 5] - 5
X_valid_full = mnist.validation.images[mnist.validation.labels >= 5]
y_valid_full = mnist.validation.labels[mnist.validation.labels >= 5] - 5
X_test = mnist.test.images[mnist.test.labels >= 5]
y_test = mnist.test.labels[mnist.test.labels >= 5] - 5

def sample_n_instances_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)

X_train, y_train = sample_n_instances_per_class(X_train_full, y_train_full, n=100)
X_valid, y_valid = sample_n_instances_per_class(X_valid_full, y_valid_full, n=30)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

n_epochs = 1000
batch_size = 20

'''
1. Create a new DNN that reuses all the pretrained hidden layers of the previous model,
freezes them, and replaces the softmax output layer with a fresh new one.
'''

print('Task 1 start')
time.sleep(1)

reset_graph()

restore_saver = tf.train.import_meta_graph("./model.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
loss = tf.get_default_graph().get_tensor_by_name("loss:0")
Y_proba = tf.get_default_graph().get_tensor_by_name("Y_proba:0")
logits = Y_proba.op.inputs[0]
accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")
learning_rate = 0.01


output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam2")
training_op = optimizer.minimize(loss, var_list=output_layer_vars)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
five_frozen_saver = tf.train.Saver()


'''
2. Train this new DNN on digits 5 to 9, using only 100 images per digit, and time how long
it takes. Despite this small number of examples, can you achieve high precision?
'''

print('Task 2 start')
time.sleep(1)

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./model")
    for var in output_layer_vars:
        var.initializer.run()
    t0 = time.time()
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train))
        for rnd_indices in np.array_split(rnd_idx, len(X_train) // batch_size):
            X_batch, y_batch = X_train[rnd_indices], y_train[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid, y: y_valid})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))
    t1 = time.time()
    print("Total training time: {:.1f}s".format(t1 - t0))

with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

'''
3. Try caching the frozen layers, and train the model again: how much faster is it now?
'''

print('Task 3 start')
time.sleep(1)

hidden5_out = tf.get_default_graph().get_tensor_by_name("hidden5_out:0")

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./model")
    for var in output_layer_vars:
        var.initializer.run()
    t0 = time.time()
    hidden5_train = hidden5_out.eval(feed_dict={X: X_train, y: y_train})
    hidden5_valid = hidden5_out.eval(feed_dict={X: X_valid, y: y_valid})
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train))
        for rnd_indices in np.array_split(rnd_idx, len(X_train) // batch_size):
            h5_batch, y_batch = hidden5_train[rnd_indices], y_train[rnd_indices]
            sess.run(training_op, feed_dict={hidden5_out: h5_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={hidden5_out: hidden5_valid, y: y_valid})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, "./my_mnist_model_5_to_9_five_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))
    t1 = time.time()
    print("Total training time: {:.1f}s".format(t1 - t0))

with tf.Session() as sess:
    five_frozen_saver.restore(sess, "./my_mnist_model_5_to_9_five_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

'''
4. Try again reusing just four hidden layers instead of five. Can you achieve a higher
precision?
'''

print('Task 4 start')
time.sleep(1)

reset_graph()

n_outputs = 5
learning_rate = 0.01

restore_saver = tf.train.import_meta_graph("./bestDNNDropout_batch_model.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden4_out = tf.get_default_graph().get_tensor_by_name("hidden4_out:0")
logits = tf.layers.dense(hidden4_out, n_outputs, kernel_initializer=he_init, name="new_logits")
Y_proba = tf.nn.softmax(logits)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="new_logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam2")
training_op = optimizer.minimize(loss, var_list=output_layer_vars)

init = tf.global_variables_initializer()
four_frozen_saver = tf.train.Saver()
n_epochs = 1000
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, "./model")
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train))
        for rnd_indices in np.array_split(rnd_idx, len(X_train) // batch_size):
            X_batch, y_batch = X_train[rnd_indices], y_train[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid, y: y_valid})
        if loss_val < best_loss:
            save_path = four_frozen_saver.save(sess, "./mnistModel_5_to_9_four_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    four_frozen_saver.restore(sess, "./mnistModel_5_to_9_four_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

'''
5. Now unfreeze the top two hidden layers and continue training: can you get the model
to perform even better?
'''

print('Task 5 start')
time.sleep(1)

learning_rate = 0.01
unfrozen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|new_logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam23")
training_op = optimizer.minimize(loss, var_list=unfrozen_vars)

init = tf.global_variables_initializer()
two_frozen_saver = tf.train.Saver()

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    four_frozen_saver.restore(sess, "./mnistModel_5_to_9_four_frozen")

    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train))
        for rnd_indices in np.array_split(rnd_idx, len(X_train) // batch_size):
            X_batch, y_batch = X_train[rnd_indices], y_train[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid, y: y_valid})
        if loss_val < best_loss:
            save_path = two_frozen_saver.save(sess, "./mnistModel_5_to_9_two_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    two_frozen_saver.restore(sess, "./mnistModel_5_to_9_two_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

learning_rate = 0.01

optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam24")
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
no_frozen_saver = tf.train.Saver()

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    two_frozen_saver.restore(sess, "./mnistModel_5_to_9_two_frozen")
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train))
        for rnd_indices in np.array_split(rnd_idx, len(X_train) // batch_size):
            X_batch, y_batch = X_train[rnd_indices], y_train[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid, y: y_valid})
        if loss_val < best_loss:
            save_path = no_frozen_saver.save(sess, "./mnistModel_5_to_9_no_frozen")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
            epoch, loss_val, best_loss, acc_val * 100))

with tf.Session() as sess:
    no_frozen_saver.restore(sess, "./mnistModel_5_to_9_no_frozen")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))

dnn_clf_5_to_9 = DNNClassifier(n_hidden_layers=4, random_state=42)
dnn_clf_5_to_9.fit(X_train, y_train, n_epochs=1000, X_valid=X_valid, y_valid=y_valid)
y_pred = dnn_clf_5_to_9.predict(X_test)
accuracy_score(y_test, y_pred)