# Back-Propagation Feed-Forward Neural Network code 
# (sigmoid with multiple hidden layers and the option to add momentum)
# Written in Python.
# Abhinav Ramakrishnan <gkabhi2@gmail.com>

import numpy as np
import matplotlib.pyplot as plt

# Not strictly needed - just keeps giving the same result every time you start
np.random.seed(0)

# Tanh is a little nicer than the standard 1/(1+e^-x) - logistic. 
# Both are tanh and logistic are coded... one is commented
def sigmoid(x):
	# from -1 to 1
	# return np.tanh(x)
	return 1./(1.+np.exp(-x))

# derivative of our sigmoid function, in terms of the output (i.e. y)
# Both tanh and logistic are coded... one is commented
def dsigmoid(y):
	# d tanh(x)/dx = dy/dx = 1-tanh^2(x)
	# d logistic(x)/dx = dy/dx = y*(1-y)
	# return (1.0 - y**2.)
	return y*(1.-y)

class NN:
	def __init__(self, ni, nh, no):

		# number of input, hidden, and output nodes
		self.ni = ni + 1 # +1 for bias node
		self.nh = nh
		self.no = no

		# Intermediate arrays: only used for hidden-hidden interactions
		self.w = []
		self.c = []
		
		# activations for nodes
		self.ai = np.ones((self.ni, 1)) #input activation
		self.ah = [np.ones(val) for val in self.nh] #hidden activation
		self.ao = np.ones((self.no, 1)) #output activation

		# Initialize internal matrices (for hidden-hidden interactions)
		for i in range(len(self.nh)-1):

			# Intermediate hidden nodes - already initialized
			# hid -> hid standard normal around 0
			self.w.append(np.random.randn(self.nh[i], self.nh[i+1]))
			self.c.append(np.zeros((self.nh[i], self.nh[i+1])))


		# set input and output weights to random values

		# in to hid = (-.2->0.2)
		self.wi = (np.random.rand(self.ni, self.nh[0])-0.5)*2./5.
		
		# hid to out = (-2. -> 2.)
		self.wo = (np.random.rand(self.nh[-1], self.no)-0.5)*4.

		# Matrices to store changes in weights for momentum
		# Initialize them to 0
		self.ci = np.zeros((self.ni, self.nh[0]))  
		self.co = np.zeros((self.nh[-1], self.no))

	def update(self, inputs):

		# basically ignoring the bias which is a constant value of 1: just mediated by the 
		# weight matrix value - CHECK TO SEE IF ENOUGH INPUT VALUES
		if len(inputs) != self.ni-1:
			raise ValueError('Wrong number of inputs')

		# input activations - ASSIGNING THE INPUT VALUES (EXCEPT BIAS)
		inputt = np.array(inputs[:])
		inputt.shape = (len(inputt), 1)
		self.ai[:self.ni-1] = np.array(inputt[:])
		
		# input-hidden interactions and hidden activations - ASSIGNING HIDDEN VALUES
		sum_ = np.dot(self.ai.transpose(), self.wi)
		self.ah[0] = sigmoid(sum_)

		# hidden-hidden interactions "middle interactions" - ASSIGNING HIDDEN-HIDDEN VALUES
		for ind in range(len(self.nh)-1): # Iterate through layers of hidden nodes
			sum_ = np.dot(self.ah[ind], self.w[ind])
			self.ah[ind+1] = sigmoid(sum_)

		# hidden-output interactions - ASSIGNING OUTPUT VALUES
		sum_ = np.dot(self.ah[-1], self.wo)
		self.ao = sigmoid(sum_)
		
		return self.ao[:] # return output activations


	def backPropagate(self, targets, N, M):

		# CHECK TO SEE IF BPROP COULD WORK: see if right number of target values
		if len(targets) != self.no:
			raise ValueError('wrong number of target values')

		# OUTPUT HIDDEN weight updates
		# calculate error terms for output_in
		error = self.ao - np.array(targets)
		
		# dE/dx_output -> if * by y_hidden then get dE/dw_ij
		output_deltas = dsigmoid(self.ao)*error 
		
		# dE/dy_hidden -> need this for dE/dx_hidden
		error = np.dot(self.wo,output_deltas.transpose()) 
		
		# dE/dx_hidden
		hidden_deltas = dsigmoid(self.ah[-1]).transpose() * error 
		
		# update output weights
		# weights := weights + alpha*de(t)/dweights + mom*de(t-1)/dweights
		# print self.ah[-1].shape, output_deltas.shape

		# dE/d_wij matrix
		change = np.dot(self.ah[-1].transpose(), output_deltas) 
		
		# evaluate changes using formula
		self.wo = self.wo - N*change - M*self.co 
		
		# save changes for momentum
		self.co = change 

		# HIDDEN HIDDEN weight updates
		for j in range(len(self.ah[:-1]))[::-1]:
			# calculate error terms for hidden nodes
			# dE/dx_hidden -> calculated earlier
			output_deltas = hidden_deltas 
			
			# dE/dy_hidden-1 layer (for dE/dx_hidden-1 layer)
			error = np.dot(self.w[j], output_deltas) 
			
			# dE/dx_hidden-1 layer
			hidden_deltas = dsigmoid(self.ah[j]).transpose()*error

			# update hidden weights
			# weights := weights + alpha*de(t)/dweights + mom*de(t-1)/dweights
			# dE/dw_ij matrix
			change = np.dot(self.ah[j].transpose(), output_deltas.transpose()) 
			
			# evaluate changes using formula
			self.w[j] = self.w[j] - N * change - M * self.c[j] 
			
			# save changes for momentum
			self.c[j] = change 

		# INPUT HIDDEN weight updates	
		# update input weights
		# dE/dw_ij matrix
		change = np.dot(self.ai, hidden_deltas.transpose())
	
		# weight updates according to formula
		self.wi = self.wi - N*change - M*self.ci 
		
		# save changes for momentum
		self.ci = change  

		# calculate error
		error = 0.0
		error += np.sum(0.5*(targets-self.ao)**2.)
		return error

	def test(self, pattern):

		# TEST NETWORK - modified - assuming one pattern at a time
		corr = int(np.argmax(self.update(pattern[0])) == np.argmax(pattern[1]))
		return np.sum(0.5*(pattern[1]-self.update(pattern[0]))**2.), corr

	def weights(self):

		# PRINT NETWORK WEIGHTS
		print('Input weights:')
		for i in range(self.ni):
			print(self.wi[i])
		print('Hidden weights:')
		for i in self.w:
			print i
		print('Output weights:')
		for j in range(self.nh[-1]):
			print(self.wo[j])

	def train(self, patterns, iterations=1000, N=0.5, M=0.1):

		# N: learning rate
		# M: momentum factor

		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error = error + self.backPropagate(targets, N, M)
			if i % 100 == 0:
				print('error %-.5f' % error)
		return error

	def net(self, inp):
		return self.update(inp)

# Demo tester case
def demo():
	# Teach network custom function
	print "READING DATA..."
	data = np.loadtxt( 'ocr.dat', delimiter = ' ' )
	inn =  data[:, :64] #first 64 columns - bitmap definition
	target = data[:, 64:] #the rest - 10 columns for 10 digits

	# training cases
	train = [[inn[i], target[i]] for i in range(len(inn[:64]))]
	
	# testing cases - simple feed forward nets are usually shit at digit recognition. 
	# expecting results to be horrible (~95%)
	test = [[inn[i], target[i]] for i in range(64, 64 + len(inn[64:]))]
	
	# create a network with 64 input -> 15 hidden -> 15 hidden -> 10 output nodes
	n = NN(64, [15, 15], 10)
	
	print "TRAINING NETWORK..."
	# train it with some patterns
	err = n.train(train, iterations = 4000, M = 0.1, N = .2)
	
	# testing network
	print "TESTING NETWORK"
	l = 0 
	err, corr = 0.0, 0.0
	for x in (test+train):
		e, cor = n.test(x)
		err += e
		corr += cor
		l+=1
		
		# Plot cool images
		val = np.argmax(x[1]+1)
		net_val = n.net(x[0])
		x_val = range(1,10)+[0] #0> 1,2,3,...,9,0
		plt.subplot(2,1,1)
		plt.plot(x_val, net_val.transpose(), 'o')
		plt.title("Actual Value is {0}".format((val+1)%10))
		plt.subplot(2,1,2)
		plt.imshow(x[0].reshape(8,8), interpolation = 'nearest')
		plt.title('Actual Image')
		plt.savefig("image{0}.png".format(l))
		plt.close()
		
	print "AVERAGE ERROR: {0}".format(err/l)
	print "AVERAGE ACCURACY: {0}%".format(100.*corr/l)

demo()