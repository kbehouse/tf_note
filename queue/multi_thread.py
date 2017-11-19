import tensorflow as tf
 
class MultithreadedTensorProvider():
 
    """ A class designed to provide tensors input in a
    separate threads. """
 
    def __init__(self, capacity, sess, dtypes, shuffle_queue=False,
        number_of_threads=1):
 
        """Initialize a class to provide a tensors with input data.
 
        Args:
            capacity: maximum queue size measured in examples. 
            sess: a tensorflow session.
            dtypes: list of data types
            shuffle_queue: either to use RandomShuffleQueue or FIFOQueue
        """
 
        self.dtypes = dtypes
        self.sess = sess
        self.number_of_threads = number_of_threads
 
        if shuffle_queue:
            self.queue = tf.RandomShuffleQueue(
                               dtypes=dtypes,
                               capacity=capacity)
        else:
            self.queue = tf.FIFOQueue(
                capacity=capacity,
                dtypes=dtypes)
 
        self.q_size = self.queue.size()
 
 
    def get_input(self):
        """ Return input tensor """
        self.batch = self.queue.dequeue()
        return self.batch
 
    def get_queue_size(self):
        """ Return how many batch left in the queue """
        return self.sess.run(self.q_size)
 
    def set_data_provider(self, data_provider):
        """ Set data provider to generate input tensor
 
        Args:
            data_provider: a callable to produce a tuple of inputs to be
            placed into a queue.
        Raises:
            TypeError: if data provider is not a callable
        """
 
        if not callable(data_provider):
            raise TypeError('Data provider should be a callable.')
         
        data = tf.py_func(data_provider, [], self.dtypes)
        enqueue_op = self.queue.enqueue(data)
        qr = tf.train.QueueRunner(self.queue, [enqueue_op]*self.number_of_threads)
        tf.train.add_queue_runner(qr)
 
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)