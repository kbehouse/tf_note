import sys
import tensorflow as tf
import threading
BATCH_SIZE = 4

# Features are length-100 vectors of floats
feature_input = tf.placeholder(tf.float32, shape=[])
# Labels are scalar integers.
label_input = tf.placeholder(tf.float32, shape=[])

# Alternatively, could do:
# feature_batch_input = tf.placeholder(tf.float32, shape=[None, 100])
# label_batch_input = tf.placeholder(tf.int32, shape=[None])

q = tf.FIFOQueue(100, [tf.float32, tf.float32], shapes=[[], []])
enqueue_op = q.enqueue([feature_input, label_input])

# For batch input, do:
# enqueue_op = q.enqueue_many([feature_batch_input, label_batch_input])

feature_batch, label_batch = q.dequeue_many(BATCH_SIZE)
# Build rest of model taking label_batch, feature_batch as input.
# [...]
train_op = feature_batch + label_batch

sess = tf.Session()

a = 10.0
b = 1000.0


def load_and_enqueue():
    count = 1.0

    while True:
    #   feature_array = numpy.fromfile(feature_file, numpy.float32, 100)
    #   if not feature_array:
    #     return
    #   label_value = numpy.fromfile(feature_file, numpy.int32, 1)[0]
        count += 1.0
        i_a = 1.0 * count
        i_b = 100.0 * count
        print('count = ' + str(count))
        sys.stdout.flush()

        print('queue size=  {}'.format( q.size()) )
        
        sess.run(enqueue_op, feed_dict={feature_input: i_a,
                                      label_input: i_b})

# Start a thread to enqueue data asynchronously, and hide I/O latency.
t = threading.Thread(target=load_and_enqueue)
t.start()


TRAINING_EPOCHS = 100

for _ in range(TRAINING_EPOCHS):
    print sess.run(train_op)
    sys.stdout.flush()
