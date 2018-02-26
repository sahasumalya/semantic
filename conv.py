	import tensorflow as tf
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
	
	n_nodes_hl1 = 200
	n_nodes_hl2 = 200
	
	
	n_classes = 1
	batch_size = 100
	
	x = tf.placeholder('float', [None, 784])
	y = tf.placeholder('float')
	
	def neural_network_model(data):
	    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
	                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	
	    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	
	    
	
	    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
	                    'biases':tf.Variable(tf.random_normal([n_classes])),}
	
	
	    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	    l1 = tf.nn.relu(l1)
	
	    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	    l2 = tf.nn.relu(l2)
	
	    
	
	    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
	    output=tf.reshape(output,[-1,784])
	    return output
	
	def train_neural_network(x):
	    prediction = neural_network_model(x)
	    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,label=x) )
	    optimizer = tf.train.AdamOptimizer().minimize(cost)
	    
	    hm_epochs = 2
	    with tf.Session() as sess:
	        sess.run(tf.initialize_all_variables())
	
	        for epoch in range(hm_epochs):
			    
	            epoch_loss = 0
				
	            for _ in range(int(mnist.train.num_examples/batch_size)):
	                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
	                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_x})
	                epoch_loss += c
	                
	        #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
			print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
			#print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
			h1layer=hidden_1_layer.eval()   
	        
			
			saver = tf.train.Saver(h1layer)
			saver.save(sess, 'my-model')
	
	train_neural_network(x)

