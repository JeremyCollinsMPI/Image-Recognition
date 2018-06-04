#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import scipy
tf.logging.set_verbosity(tf.logging.INFO)

random.seed(13)

def copy(x):
	new=[]
	for member in x:
		new.append(member)
	return new	
	
def rep(x,y):
	new=[]
	for m in xrange(y):
		new.append(x)
	return new
def replist(x,y):
	new=[]
	for m in xrange(y):
		new.append(copy(x))
	return new
	
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  tf.print(input_layer)
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



	

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  
  n = 24

  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#   m = train_data[sevens[0]]
#   for n in range(1, 4):
#   	print(n)
#   	m = np.add(m, train_data[sevens[n]])
#   print(m)
#   m = np.divide(m, len(sevens))
#   print(m)
# 
#  
 
  first = train_data[n]
#   first = [1 - x for x in first]
  img = Image.new('1',(28,28))
  img.putdata(first)
  
  
  def show(number):
  	x = train_data[number]
  	x = [1-y for y in x] 
  	img = Image.new('1',(28,28))
  	img.putdata(x)
  	img.show()

#   img.show()
  
#   def findError(imageNumber1, imageNumber2):
#   	first = train_data[imageNumber1]
#   	second = train_data[imageNumber2]
#   	result = np.absolute(np.subtract(first, second))
#   	result = [1 - (0.95*x) for x in result]
#   	logResult = [np.log(m) for m in result]
#   	print(logResult)
#   	print(np.sum(logResult))
#   def prepareImage(imageNumber):
#   
#   	first = train_data[imageNumber]
#   	first = np.ndarray.flatten(first)
#   	img = Image.new('1',(28,28))
#   	img.putdata(first)
#   	first = np.reshape(img, 784)
#   	return img

  def findError(imageNumber1, imageNumber2, a, b):
  	first = train_data[imageNumber1]
  	first = np.reshape(first, (28,28))
  	second = train_data[imageNumber2]
  	second = np.reshape(second, (28,28))
  	second = np.roll(second, a, axis = 1)
  	second = np.roll(second, b, axis = 0)
  	result = np.absolute(np.subtract(first, second))
  	result = [1 - (0.95*x)**0.2 for x in result]
  	logResult = [np.log(m) for m in result]
  	return np.sum(logResult)
  
  def findError2(imageNumber1, imageNumber2, a, b, c, d):
  	first = train_data[imageNumber1]
  	first = np.reshape(first, (28,28))
  	second = train_data[imageNumber2]
  	second = np.reshape(second, (28,28))
  	second = np.split(second, 2)
  	second[0] = np.roll(second[0], a, axis = 1)
  	second[0] = np.roll(second[0], b, axis = 0)
  	second[1] = np.roll(second[1], c, axis = 1)
  	second[1] = np.roll(second[1], d, axis = 0)
  	second = np.concatenate((second[0], second[1]))
  	result = np.absolute(np.subtract(first, second))
  	result = [1 - (0.95*x)**0.2 for x in result]
  	logResult = [np.log(m) for m in result]
  	return np.sum(logResult)  	

  def findError3(imageNumber1, imageNumber2, a, b, c, d, e, f, g, h):
  	first = train_data[imageNumber1]
  	first = np.reshape(first, (28,28))
  	second = train_data[imageNumber2]
  	second = np.reshape(second, (28,28))
  	second = np.split(second, 2)
  	second[0] = np.split(second[0], 2, axis = 1)
  	second[1] = np.split(second[1], 2, axis = 1)
  	second[0][0] = np.roll(second[0][0], a, axis = 1)
  	second[0][0] = np.roll(second[0][0], b, axis = 0)
  	second[0][1] = np.roll(second[0][1], c, axis = 1)
  	second[0][1] = np.roll(second[0][1], d, axis = 0)
  	second[1][0] = np.roll(second[1][0], e, axis = 1)
  	second[1][1] = np.roll(second[1][1], f, axis = 0)
  	second[1][0] = np.roll(second[1][0], g, axis = 1)
  	second[1][1] = np.roll(second[1][1], g, axis = 0)
  	second[0] = np.concatenate((second[0][0],second[0][1]), axis = 1)
  	second[1] = np.concatenate((second[1][0],second[1][1]),axis=1)
  	second = np.concatenate((second[0],second[1]),axis=0)
  	result = np.absolute(np.subtract(first, second))
  	result = [1 - (0.95*x)**0.2 for x in result]
  	logResult = [np.log(m) for m in result]
  	return np.sum(logResult)
  	
  def tryOut(imageNumber1, imageNumber2):	
	  result = []
	  for a in range(-10,10):
	  	for b in range(-10,10):
	  		result.append([findError(imageNumber1, imageNumber2, a, b), a, b])
	  result = sorted(result, key = lambda x: x[0])
	  return result

  def tryOut2(imageNumber1, imageNumber2):	
	  result = []
	  for a in range(-3,3):
	  	for b in range(-3,3):
	  		for c in range(-3, 3):
	  			for d in range(-3, 3):
					result.append([findError2(imageNumber1, imageNumber2, a, b, c, d), a, b, c, d])
	  result = sorted(result, key = lambda x: x[0])
	  return result

  def tryOut3(imageNumber1, imageNumber2):	
	  result = []
	  for a in range(-1,1):
	  	for b in range(-1,1):
	  		for c in range(-1, 1):
	  			for d in range(-1, 1):
	  				for e in range(-1, 1):
	  					for f in range(-1, 1):
	  						for g in range(-1, 1):
	  							for h in range(-1, 1):
									result.append([findError3(imageNumber1, imageNumber2, a, b, c, d, e, f, g, h), a, b, c, d, e, f, g, h])
	  result = sorted(result, key = lambda x: x[0])
	  return result

  def showClass(number):
  	 return([x for x in xrange(len(train_labels)) if train_labels[x] == number])
  def tryAgainstRandomExample(imageNumber1, x):
  	imageNumber2 = random.sample(showClass(x),1)[0]
  	temp = tryOut(imageNumber1, imageNumber2)
  	return temp[-1][0]
  def tryAgainstRandomExample2(imageNumber1, x):
  	imageNumber2 = random.sample(showClass(x),1)[0]
  	temp = tryOut2(imageNumber1, imageNumber2)
  	return temp[-1][0]
  def tryAgainstRandomExample3(imageNumber1, x):
  	imageNumber2 = random.sample(showClass(x),1)[0]
  	temp = tryOut3(imageNumber1, imageNumber2)
  	return temp[-1][0]
  def classify(imageNumber1, trainingExamples = 1):
  	show(imageNumber1)
  	results = []
  	numericalStability = 200
  	for x in range(0,10):
  		total = 0
  		for i in xrange(trainingExamples):
  			y = tryAgainstRandomExample(imageNumber1, x)
  			y = y + numericalStability
  			y = np.exp(y)
			total = total + y
		total = total / trainingExamples
		total = np.log(total)
		total = total - numericalStability
  		results.append(total)
  	return results.index(max(results)), results	
  def classify2(imageNumber1, trainingExamples = 1):
  	show(imageNumber1)
  	results = []
  	numericalStability = 200
  	for x in range(0,10):
  		total = 0
  		for i in xrange(trainingExamples):
  			y = tryAgainstRandomExample2(imageNumber1, x)
  			y = y + numericalStability
  			y = np.exp(y)
			total = total + y
		total = total / trainingExamples
		total = np.log(total)
		total = total - numericalStability
  		results.append(total)
  	return results.index(max(results)), results
  def classify1and2(imageNumber1, trainingExamples = 1):
	show(imageNumber1)
	results = []
	numericalStability = 200
	for x in range(0,10):
		intermediateResults = []
		total = 0
		for i in xrange(trainingExamples):
			y = tryAgainstRandomExample(imageNumber1, x)
			y2 = tryAgainstRandomExample2(imageNumber1, x)
			y = max(y, y2)
# 			y = y + numericalStability
# 			y = np.exp(y)
# 			total = total + y
# 		total = total / trainingExamples
# 		total = np.log(total)
# 		total = total - numericalStability
			intermediateResults.append(y)
		results.append(max(intermediateResults))
	return results.index(max(results)), results
  def classify3(imageNumber1, trainingExamples = 1):
  	show(imageNumber1)
  	results = []
  	numericalStability = 200
  	for x in range(0,10):
  		total = 0
  		for i in xrange(trainingExamples):
  			y = tryAgainstRandomExample3(imageNumber1, x)
  			y = y + numericalStability
  			y = np.exp(y)
			total = total + y
		total = total / trainingExamples
		total = np.log(total)
		total = total - numericalStability
  		results.append(total)
  	return results.index(max(results)), results
  	
  print(classify1and2(252,3)[0])
  	
#   result = []
#   for i in range(350,370):
#   	result.append(train_labels[i]==classify1and2(i, 10)[0])
#   	print(len(result))
#   	print(len([x for x in result if x == True])/len(result))
#   print(len([x for x in result if x == True])/len(result))

  def produce(array, list):
  	for member in list:
  		x = member[0]
  		y = member[1]
  		xContinue = member[2]
  		yContinue = member[3]
  		for m in xrange(xContinue):
  			array[y][x+m] = 1
  		for m in xrange(yContinue):
  			array[y+m][x] = 1
  	return array
  
#   def model(array, numberOfComponents):
  	
#   print(produce(replist(rep(0,28), 28), [[0,1,2,1]]))
  		

	  
	  
	  
	  
# 	  def showClass(number):
# 		sevens = [x for x in xrange(len(train_labels)) if train_labels[x] == number]
# 		print(sevens)
# 	showClass(4)

  
  
#   findError(24,44, 3, 3)
  	

  
#   print(train_labels[n])
#   print(len(train_labels))
  
#   img.save('my.png')
  
#   eval_data = mnist.test.images  # Returns np.array
#   eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
#   mnist_classifier = tf.estimator.Estimator(
#       model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
# 
#   # Set up logging for predictions
#   # Log the values in the "Softmax" tensor with label "probabilities"
#   tensors_to_log = {"probabilities": "softmax_tensor"}
#   logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=50)
# 
#   # Train the model
#   train_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={"x": train_data},
#       y=train_labels,
#       batch_size=100,
#       num_epochs=None,
#       shuffle=True)
#   mnist_classifier.train(
#       input_fn=train_input_fn,
#       steps=20000,
#       hooks=[logging_hook])
# 
#   # Evaluate the model and print results
#   eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#       x={"x": eval_data},
#       y=eval_labels,
#       num_epochs=1,
#       shuffle=False)
#   eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
#   print(eval_results)


if __name__ == "__main__":
  tf.app.run()