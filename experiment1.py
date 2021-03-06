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

# set sampling chain length
chain_length = 1

# Load data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
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

def findError(first, second, a, b):
	second = np.roll(second, a, axis = 1)
	second = np.roll(second, b, axis = 0)
	result = np.absolute(np.subtract(first, second))
	result = [1 - (0.95*x)**0.2 for x in result]	
	logResult = [np.log(m) for m in result]
	return np.sum(logResult)

def findError2(first, second, a, b, c, d):
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
	
def tryOut(image1, image2):	
	result = []
	currentParameters = [0,0,0,0]
	for i in xrange(chain_length):
		temp1 = [] 	
		for m in range(-5,5):
			currentParameters[0] = m
			temp1.append([m, findError(image1, image2, currentParameters[0], currentParameters[1])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[0] = temp1[0][0]
		temp1 = []
		for m in range(-5,5):
			currentParameters[1] = m	
			temp1.append([m, findError(image1, image2, currentParameters[0], currentParameters[1])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[1] = temp1[0][0]
	current = findError(image1, image2, currentParameters[0], currentParameters[1])
	return [[current, currentParameters[0], currentParameters[1]]]

def tryOut2(image1, image2):	
	result = []
	currentParameters = [0,0,0,0]
	for i in xrange(chain_length):
		temp1 = [] 	
		for m in range(-4,4):
			currentParameters[0] = m
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[0] = temp1[0][0]
		temp1 = []
		for m in range(-4,4):
			currentParameters[1] = m	
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[1] = temp1[0][0]
		temp1 = []
		for m in range(-4,4):
			currentParameters[2] = m	
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[2] = temp1[0][0]
		temp1 = []
		for m in range(-4,4):
			currentParameters[3] = m	
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[3] = temp1[0][0]		
	current = findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])
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

def tryAgainstRandomExample(image1, x):
	image2 = training_images[str(x)+'_'+str(random.sample(xrange(training_data_size), 1)[0])]
	temp = tryOut(image1, image2)
	return temp[-1][0]

def tryAgainstRandomExample2(image1, x):
	image2 = training_images[str(x)+'_'+str(random.sample(xrange(training_data_size), 1)[0])]
	temp = tryOut2(image1, image2)
	return temp[-1][0]
	
def estimateTranslation(image1, image2):
	result = []
	currentParameters = [0, 0, 0, 0]
	for i in xrange(chain_length):
		temp1 = [] 	
		for m in range(-4,4):
			currentParameters[0] = m
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[0] = temp1[0][0]
		temp1 = []
		for m in range(-4,4):
			currentParameters[1] = m	
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[1] = temp1[0][0]
		temp1 = []
		for m in range(-4,4):
			currentParameters[2] = m	
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[2] = temp1[0][0]
		temp1 = []
		for m in range(-4,4):
			currentParameters[3] = m	
			temp1.append([m, findError2(image1, image2, currentParameters[0], currentParameters[1], currentParameters[2], currentParameters[3])])
		temp1 = sorted(temp1, key = lambda x : x[1], reverse = True)
		currentParameters[3] = temp1[0][0]
	return currentParameters		

def estimateTranslationGeneral(image1, x, number):
	result = []
	for i in xrange(number):
		image2 = randomTrainingImage(x)
		result.append(estimateTranslation(image1, image2))
	return result

def basicTry(image1, x, a, b):
	image2 = training_images[str(x)+'_'+str(random.sample(xrange(training_data_size), 1)[0])]
	return findError(image1, image2, a, b)

def basicTry2(image1, x, a, b, c, d):
	image2 = training_images[str(x)+'_'+str(random.sample(xrange(training_data_size), 1)[0])]
	return findError2(image1, image2, a, b, c, d)

def randomTrainingImage(x):
	return training_images[str(x)+'_'+str(random.sample(xrange(training_data_size), 1)[0])]
	
def classify1and2(imageNumber1, trainingExamples = 1):
	results = []
	for x in range(0,10):
		intermediateResults = []
		total = 0
		image1 = returnImage(imageNumber1)
		for i in xrange(trainingExamples):
			y = tryAgainstRandomExample(image1, x)
			y2 = tryAgainstRandomExample2(image1, x)
			y = max(y, y2)
			intermediateResults.append(y)
		results.append(max(intermediateResults))
	return results.index(max(results)), results	

def basicClassify(imageNumber1, trainingExamples = 1):
	results = []
	for x in range(0,10):	
		image1 = returnImage(imageNumber1)
		image2 = randomTrainingImage(x)
		number = 10
		parametersCandidates = estimateTranslationGeneral(image1, x, number)
		for m in xrange(len(parametersCandidates)):
			parameters = parametersCandidates[m]
			candidates = []
			for i in xrange(trainingExamples):
				image2 = training_images[str(x)+'_'+str(i)]
				y2 = findError2(image1, image2, parameters[0], parameters[1], parameters[2], parameters[3])
				candidates.append([i, m, y2])
		candidates = sorted(candidates, key = lambda x : x[2], reverse = True)
		intermediateResults = []
		indeces = [z[0] for z in candidates[0:5]]
		for index in indeces:
			image2 = training_images[str(x)+'_'+str(index)]		
			y = tryOut2(image1, image2)[0]
			y2 = tryOut(image1, image2)[0]
			y = max(y, y2)
			intermediateResults.append(y)
		results.append(max(intermediateResults))
	return results.index(max(results)), results	 

def main(unused_argv):
# example image and classification, e.g. image 252  
	image_number = 252
	show(image_number)	
	print(basicClassify(image_number,training_data_size)[0])
# testing 100 images
	result = []
	for i in range(400,500):
		result.append(train_labels[i]==basicClassify(i, training_data_size)[0])		
		print(len(result))
		print(len([x for x in result if x == True])/len(result))

if __name__ == "__main__":
	tf.app.run()