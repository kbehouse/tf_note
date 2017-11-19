
import tensorflow as tf
import numpy as np 


class Alpha:
    def __init__(self, multi, prefix):
        self.prefix = prefix
        self.multi = multi
        scope_name = prefix + "_scope"

        self._build_graph(scope_name)
        # self.sess = sess

        self.num = 0 
        
    def _build_graph(self, scope_name):
        with tf.variable_scope(scope_name) as scope:
            self.a = tf.placeholder("float", shape=[])
            m = tf.constant(self.multi)
            
            self.y = self.a * m 

    def initialize(self,sess):
        self.sess=sess
        

    def enqueue_op(self,queue):
        def _func():

            # print('func_i = ' + str(func_i) )
            self.num += 1.0
            print("Run {} with num = {}".format(self.prefix, self.num) )
            # return self.prefix + '_' + str(self.num)
            

            return self.sess.run(self.y, feed_dict={self.a : self.num})

        data = tf.py_func(_func,[], [tf.float32],stateful=True)
        return queue.enqueue(data)



RANDOM_SEED= 33
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.reset_default_graph()


A = Alpha(  1.0     ,"A")
B = Alpha(  0.000001,"B")


queue = tf.FIFOQueue(capacity=10, dtypes=[tf.float32])
qr = tf.train.QueueRunner(queue, [A.enqueue_op(queue), B.enqueue_op(queue)])
# qr = tf.train.QueueRunner(queue, [A.enqueue_op(queue)])
tf.train.queue_runner.add_queue_runner(qr)
y = queue.dequeue()


init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
# sess = tf.Session()
sess = tf.Session()
sess.graph.finalize()
sess.run(init_op)



# We start the session as usual ...
with tf.Session() as sess:
    # But now we build our coordinator to coordinate our child threads with
    # the main thread
    A.initialize(sess)
    B.initialize(sess)
    coord = tf.train.Coordinator()
    # Beware, if you don't start all your queues before runnig anything
    # The main threads will wait for them to start and you will hang again
    # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
    threads = tf.train.start_queue_runners(coord=coord)

    # import time
    # time.sleep(1)
    # The QueueRunner will automatically call the enqueue operation
    # asynchronously in its own thread ensuring that the queue is always full
    # No more hanging for the main process, no more waiting for the GPU
    print sess.run(y)
    print sess.run(y) 
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)
    print sess.run(y)

    # We request our child threads to stop ...
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)