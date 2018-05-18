# import tensorflow as tf

# # creates nodes in a graph
# # "construction phase"
# x1 = tf.constant(5)
# x2 = tf.constant(6)

# #print (dir(tf));

# result = tf.multiply(x1,x2)

# sess = tf.Session()

# print(sess.run(result))

# sess.close()


# Fields such as image recognition, speech and natural language processing etc.
#machine learning with tensorflow pdf



# # import tensorflow
# import tensorflow as tf

# # build computational graph
# a = tf.placeholder(tf.int16)
# b = tf.placeholder(tf.int16)

# addition = tf.add(a, b)

# # initialize variables
# init = tf.global_variables_initializer()

# # create session and run the graph
# with tf.Session() as sess:
#     sess.run(init)
#     print ("Addition: %i" % sess.run(addition, feed_dict={a: 2, b: 3}))

# # close session
# sess.close()


# import numpy as np
# import tensorflow as tf

# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# # In this example, we limit mnist data
# Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
# Xte, Yte = mnist.test.next_batch(200) #200 for testing

# print (Xte, Yte)

# fffffff

# # tf Graph Input
# xtr = tf.placeholder("float", [None, 784])
# xte = tf.placeholder("float", [784])

# # Nearest Neighbor calculation using L1 Distance
# # Calculate L1 Distance
# distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# # Prediction: Get min distance index (Nearest neighbor)
# pred = tf.arg_min(distance, 0)

# accuracy = 0.

# # Initializing the variables
# init = tf.global_variables_initializer()

# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)

#     # loop over test data
#     for i in range(len(Xte)):
#         # Get nearest neighbor
#         nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
#         # Get nearest neighbor class label and compare it to its true label
#         print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
#             "True Class:", np.argmax(Yte[i]))
#         # Calculate accuracy
#         if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
#             accuracy += 1./len(Xte)
#     print("Done!")
#     print("Accuracy:", accuracy)




import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)    