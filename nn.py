import numpy as np
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()
########Training data processing#########
np.seterr(invalid='ignore')    #ignoring NaN type warning
data = np.loadtxt("hw2_data4.txt")
num_examples = int(len(data))   # training set size
np.random.shuffle(data)         #Data shuffling as part of data preprocessing
X=[]                            #Training data feature set value
y = []                          #Training data labels
print num_examples
for i in range(0,num_examples):
    X.append([data[i,0],data[i,1],data[i,2]])

# define training set
X = np.asarray(X)
#standardization of training data for better result
#To standardize the data over each column
#X = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
#####################################
#Normalization of training data
X = (X-np.min(X))/(np.max(X)-np.min(X))
#####################################
#Training data class labels
for i in range(0,num_examples):
    if data[i,3] == 1: #class 1 which is 1
        y.append(0)
    elif(data[i,3] == 2):
        y.append(1) # class 2 which is 2
y = np.array(y).astype(int)
#Preprocessing of training data ends here

# Display plots inline and change default figure size
# Display data plot for data set 1
#%matplotlib inline
#fig = plt.figure()
#matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()
#fig.suptitle('Data plot for Data set 1', fontsize=20)
#plt.xlabel('x coordinate', fontsize=18)
#plt.ylabel('y coordinate', fontsize=16)
#fig.savefig('hw2_data1.png')

loss = []
iteration = []

nn_input_dim = 3 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
dropout = 0.5
fold = 10
# Gradient descent parameters (picked by hand)
epsilon = 0.001 # learning rate for gradient descent
reg_lambda = 0.1 # regularization strength
k = 100 #size of minibatches for stochastic gradient desent
#sigmoid as activation function
def sigmoid(x):
    return np.reciprocal(1 + np.exp(-1*x))

def sigmoid_derivative(x):
    return x*(1-x)

#tanh as activation function
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.power(x, 2)

#ReLU as activation function
def ReLU(x):
    return np.maximum(x,0.)

def ReLU_deriv(x):
    return np.array(x > 0.).astype(int)

class NeuralNetwork:
    def __init__(self, activation='sigmoid'):
        if activation == 'sigmoid':
           self.activation = sigmoid
           self.activation_deriv = sigmoid_derivative
        elif activation == 'tanh':
           self.activation = tanh
           self.activation_deriv = tanh_deriv
        elif activation == 'relu':
           self.activation = ReLU
           self.activation_deriv = ReLU_deriv
    # Helper function to evaluate the total loss on the dataset
    def calculate_loss(self,model):
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
        # Forward propagation to calculate our predictions
        z1 = X.dot(W1) + b1
        a1 = self.activation(z1)
        z2 = a1.dot(W2) + b2
        a2 = self.activation(z2)
        z3 = a2.dot(W3) + b3
        exp_scores = np.exp(z3)
        #print exp_scores
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W2)))
        return 1./num_examples * data_loss

    # Helper function to predict an output (0 or 1)
    def predict(self, model, x):
        W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = self.activation(z1)
        #dropout for testing, multiple array with 0.5 with first hidden layer
        z2 = dropout*a1.dot(W2) + b2
        a2 = self.activation(z2)
        z3 = a2.dot(W3) + b3
        exp_scores =np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def build_model(self, X, y, nn_hdim1, nn_hdim2,  num_passes=1000, print_loss=False):

        # Initialize the parameters to random values. We need to learn these.
        np.random.seed(0)
        W1 = np.random.randn(nn_input_dim, nn_hdim1) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim1))
        W2 = np.random.randn(nn_hdim1, nn_hdim2) / np.sqrt(nn_hdim1)
        b2 = np.zeros((1, nn_hdim2))
        W3 = np.random.randn(nn_hdim2, nn_output_dim) / np.sqrt(nn_hdim2)
        b3 = np.zeros((1, nn_output_dim))
        #Dimensions are correct
        # This is what we return at the end
        model = {}

        # Gradient descent. For each batch...
        for i in xrange(0, num_passes):

            #stochastic gardient desent
            X_stoch = []
            y_stoch = []
            B = np.random.randint(len(X),size=k)
            X_stoch = X[B,:]
            y_stoch = y[B]
            #print X_stoch
            #print y_stoch

            # Forward propagation
            z1 = np.dot(X_stoch,W1) + b1
            #masking for dropout at layer 1
            m2 = np.random.binomial(1, dropout, size=z1.shape)
            a1 = self.activation(z1)
            a1 *= m2
            #####################
            z2 = a1.dot(W2) + b2
            a2 = self.activation(z2)
            z3 = a2.dot(W3) + b3
            exp_scores = np.exp(z3)
            #print exp_scores
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(k), y_stoch] -= 1
            dW3 = (a2.T).dot(delta3)
            db3 = np.sum(delta3, axis=0, keepdims=True)

            delta2 = delta3.dot(W3.T) * self.activation_deriv(a2)
            dW2 = np.dot(a1.T, delta2)
            db2 = np.sum(delta2, axis=0)

            #Dropout in back propagation m2 = binary mask
            delta1 = delta2.dot(W2.T) * self.activation_deriv(a1)*m2
            dW1 = np.dot(X_stoch.T, delta1)
            db1 = np.sum(delta1, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW3 += reg_lambda * W3
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2
            W3 += -epsilon * dW3
            b3 += -epsilon * db3

            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 5 == 0:
                iteration.append(i)
                loss.append(self.calculate_loss(model))
                print "Loss after iteration %i: %f" %(i, self.calculate_loss(model))
        return model
nn = NeuralNetwork('tanh')
#10 fold cross validation
fold = 10
training_fold = (num_examples/fold)*(fold-1)
training_data = X[:-training_fold]
training_label = y[:-training_fold]
testing_data = X[(training_fold+1):]
testing_label = y[(training_fold+1):]
length_test_data = len(testing_data)
##

model = nn.build_model(X, y, 10,5, print_loss=True)
count = 0.0
num_of_test_data = 40
for i in range(0,num_of_test_data):
    result = nn.predict(model,X[i])
    print(i,result[0])
    if result == y[i]:
        count = count + 1
print count
accuracy = (count/num_of_test_data)*100
print "Accuracy is %f" %accuracy

stop = timeit.default_timer()
print "Total running time of Neural Network " + str(stop - start)

#print iteration[1:]
fig = plt.figure()
plt.plot(iteration[1:],loss[1:])
fig.suptitle('Comparison of loss with iteration', fontsize=20)
plt.xlabel('Number of iteration', fontsize=18)
plt.ylabel('loss using cross loss entropy', fontsize=16)
fig.savefig('loss_function.png')
plt.show()