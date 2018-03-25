import tensorflow as tf 
import preprocessing as pre
from tqdm import tqdm


x = tf.placeholder(tf.float32,shape=[None,784],name="x")
y = tf.placeholder(tf.float32,name="y")

n_hidden_1 = 500
n_hidden_2 = 500
n_classes = 62
n_epoch = 100
batch_size = 100


def model():

	layer_1 = {'weights':tf.Variable(tf.random_normal([784,n_hidden_1])),
				'biases':tf.Variable(tf.random_normal([n_hidden_1]))}
	layer_2 = {'weights':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
				'biases':tf.Variable(tf.random_normal([n_hidden_2]))}
	layer_3 = {'weights':tf.Variable(tf.random_normal([n_hidden_2,n_classes])),
				'biases':tf.Variable(tf.random_normal([n_classes]))}


	out_layer_1 = tf.add(tf.matmul(x,layer_1['weights']),layer_1['biases'])
	out_layer_1 = tf.nn.relu(out_layer_1)

	out_layer_2 = tf.add(tf.matmul(out_layer_1,layer_2['weights']),layer_2['biases'])
	out_layer_2 = tf.nn.relu(out_layer_2)

	out = tf.add(tf.matmul(out_layer_2,layer_3['weights']),layer_3['biases'])

	return out 


images,labels = pre.load_data('E:\Projects\Main Project\learning machine learning\Training')

def train():
	prediction = model()

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	for epoch in tqdm(range(n_epoch)):
		for batch in range(int(images.shape[0]/batch_size)):
			limit = (batch+1)*batch_size
			x_train = images[limit-batch_size:limit]
			y_train = labels[limit-batch_size:limit]

			sess.run(optimizer,feed_dict={x:x_train,y:y_train})

	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
	print('Accuracy:',sess.run(accuracy,feed_dict={x:images,y:labels}))

	print(sess.run(prediction,feed_dict={x:[images[20]],y:[labels[20]]}))
	


train()
