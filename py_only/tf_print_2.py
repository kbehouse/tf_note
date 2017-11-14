import tensorflow as tf

x_input_data = tf.random_normal([4], mean=-1, stddev=4)

x_input_data = tf.Print(x_input_data, data=[ x_input_data], message="x_input_data:")

final = x_input_data + 1

final = tf.Print(final, data=[ final], message="final:")


# Method 1 
# with tf.Session() as sess:
#     print sess.run(final)

# Method 2
# Cannot use sess = tf.Session()
sess = tf.InteractiveSession()
final.eval()