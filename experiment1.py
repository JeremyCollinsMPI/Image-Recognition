# Bayesian image classifier, which classifies images from MNIST with 10 images per class as training data 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import numpy as np
import tensorflow as tf
import random
import scipy
tf.logging.set_verbosity(tf.logging.INFO)

# set random seed
random.seed(13)

# set training data size
training_data_size = 10

# set chain length
chain_length = 700

# Load data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32) 

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

def show(number):
	x = train_data[number]
	x = [1-y for y in x] 
	img = Image.new('1',(28,28))
	img.putdata(x)
	img.show()

def returnImage(imageNumber1):
	first = train_data[imageNumber1]
	first = np.reshape(first, (28,28))
	return first

def findErrorPure(first, second, a, b):
	second = np.roll(second, a, axis = 1)
	second = np.roll(second, b, axis = 0)
	result = np.absolute(np.subtract(first, second))
	result = [1 - (0.95*x)**0.2 for x in result]
	logResult = [np.log(m) for m in result]
	return np.sum(logResult)

def findError2Pure(first, second, a, b, c, d):
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

def tryOutPure(image1, image2):	
  result = []
  for a in range(-10,10):
	for b in range(-10,10):
		result.append([findErrorPure(image1, image2, a, b), a, b])
  result = sorted(result, key = lambda x: x[0])
  return result

def tryOut2Pure(image1, image2):	
  result = []
  for a in range(-3,3):
	for b in range(-3,3):
		for c in range(-3, 3):
			for d in range(-3, 3):
				result.append([findError2Pure(image1, image2, a, b, c, d), a, b, c, d])
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

def tryOut2MaximumLikelihoodPure(image1, image2):
  result = []
  a = -3
  b = -3
  c = -3
  d = -3
  chainLength = chain_length
  currentParameters = [0,0,0,0]
  i = 0
  previous = -10000000
  while i < chainLength:	  
	  newParameters = copy(currentParameters)		  
	  coinToss1 = random.sample([0,1,2,3],1)[0]
	  coinToss2 = random.sample([-1,1],1)[0]
	  newParameters[coinToss1] = max(min(newParameters[coinToss1] + coinToss2, 3), -3)	  
	  current = findError2Pure(image1, image2, newParameters[0], newParameters[1], newParameters[2], newParameters[3])
	  if current > previous:
		currentParameters = copy(newParameters)
		previous = current
	  i = i + 1
  return [[current, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3]]]

classes = {}
for number in xrange(10):
	classes[number] = [x for x in xrange(len(train_labels)) if train_labels[x] == number]
training_images = {}
for number in xrange(10):
	for m in xrange(training_data_size):
		training_images[str(number)+'_'+str(m)] = returnImage(random.sample(classes[number],1)[0])

def showClass(number):
	return(classes[number])

def tryAgainstRandomExamplePure(image1, x):
	image2 = training_images[str(x)+'_'+str(random.sample(xrange(10), 1)[0])]
	temp = tryOutPure(image1, image2)
	return temp[-1][0]

def tryAgainstRandomExample2Pure(image1, x):
	image2 = training_images[str(x)+'_'+str(random.sample(xrange(10), 1)[0])]
	temp = tryOut2Pure(image1, image2)
	return temp[-1][0]

def tryAgainstRandomExample2MaximumLikelihoodPure(image1, x):
	image2 = training_images[str(x)+'_'+str(random.sample(xrange(10), 1)[0])]
	temp = tryOut2MaximumLikelihoodPure(image1, image2)
	return temp[-1][0]

def tryAgainstRandomExample3(imageNumber1, x):
	imageNumber2 = random.sample(showClass(x),1)[0]
	temp = tryOut3(imageNumber1, imageNumber2)
	return temp[-1][0]

def classify1and2MaximumLikelihoodPure(imageNumber1, trainingExamples = 1):
	results = []
	numericalStability = 200
	for x in range(0,10):
		intermediateResults = []
		total = 0
		image1 = returnImage(imageNumber1)
		for i in xrange(trainingExamples):
			y = tryAgainstRandomExamplePure(image1, x)
			y2 = tryAgainstRandomExample2MaximumLikelihoodPure(image1, x)
			y = max(y, y2)
			intermediateResults.append(y)
		results.append(max(intermediateResults))
	return results.index(max(results)), results

def classify1and2Pure(imageNumber1, trainingExamples = 1):
	results = []
	numericalStability = 200
	for x in range(0,10):
		intermediateResults = []
		total = 0
		image1 = returnImage(imageNumber1)
		for i in xrange(trainingExamples):
			y = tryAgainstRandomExamplePure(image1, x)
			y2 = tryAgainstRandomExample2Pure(image1, x)
			y = max(y, y2)
			intermediateResults.append(y)
		results.append(max(intermediateResults))
	return results.index(max(results)), results	

def classify3(imageNumber1, trainingExamples = 1):
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


def main(unused_argv):
# example image and classification, e.g. image 252  
  image_number = 252
  show(image_number)	
#   print(classify1and2MaximumLikelihood(image_number,training_data_size)[0])
  print(classify1and2MaximumLikelihoodPure(image_number,training_data_size)[0])


# testing 100 images
  
  result = []
  for i in range(300,400):
  	result.append(train_labels[i]==classify1and2Pure(i, training_data_size)[0])
  	print(len(result))
  	print(len([x for x in result if x == True])/len(result))
  print(len([x for x in result if x == True])/len(result))


if __name__ == "__main__":
  tf.app.run()