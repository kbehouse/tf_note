
import tensorflow as tf



class Alpha:
    def __init__(self, multi, prefix, sess):
        self.prefix = prefix
        self.multi = multi

        self._build_graph()
        self.sess = sess

        self.num = 0 
        

    def _build_graph(self):

        self.a = tf.placeholder("float", shape=[])
        m = tf.constant(self.multi)
        
        self.y = self.a * m 
        

    def enqueue_op(self,queue):
        def _func(func_i):

            # print('func_i = ' + str(func_i) )
            self.num += 1.0
            print("Run {} with num = {}".format(self.prefix, self.num) )
            # return self.prefix + '_' + str(self.num)
            

            return self.sess.run(self.y, feed_dict={self.a : self.num})

        data = tf.py_func(_func,[tf.float32],stateful=True)
        return queue.enqueue(data)


sess = tf.Session()
A = Alpha(  1.0, "A", sess)
B = Alpha(100.0, "B", sess)



queue = tf.FIFOQueue(capacity=10, dtypes=[tf.float32])
# qr = tf.train.QueueRunner(queue, [A.enqueue_op(queue), B.enqueue_op(queue)])
qr = tf.train.QueueRunner(queue, [A.enqueue_op(queue)])
tf.train.queue_runner.add_queue_runner(qr)
y = queue.dequeue()


"""
# This time, let's start with 6 samples of 1 data point
x_input_data = tf.random_normal([4], mean=-1, stddev=4)

# Note that the FIFO queue has still a capacity of 3
q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)

# To check what is happening in this case:
# we will print a message each time "x_input_data" is actually computed
# to be used in the "enqueue_many" operation
x_input_data = tf.Print(x_input_data, data=[x_input_data], message="Raw inputs data generated:", summarize=6)
enqueue_op = q.enqueue_many(x_input_data)

# To leverage multi-threading we create a "QueueRunner"
# that will handle the "enqueue_op" outside of the main thread
# We don't need much parallelism here, so we will use only 1 thread
numberOfThreads = 2 
qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
# Don't forget to add your "QueueRunner" to the QUEUE_RUNNERS collection
tf.train.add_queue_runner(qr) 

input = q.dequeue() 
input = tf.Print(input, data=[q.size(), input], message="Nb elements left, input:")

# fake graph: START
y = input + 1
# fake graph: END 
"""



# We start the session as usual ...
with tf.Session() as sess:
    # But now we build our coordinator to coordinate our child threads with
    # the main thread
    coord = tf.train.Coordinator()
    # Beware, if you don't start all your queues before runnig anything
    # The main threads will wait for them to start and you will hang again
    # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
    threads = tf.train.start_queue_runners(coord=coord)

    # The QueueRunner will automatically call the enqueue operation
    # asynchronously in its own thread ensuring that the queue is always full
    # No more hanging for the main process, no more waiting for the GPU
    print sess.run(y)
    print sess.run(y) 
    print sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)
    sess.run(y)

    # We request our child threads to stop ...
    coord.request_stop()
    # ... and we wait for them to do so before releasing the main thread
    coord.join(threads)