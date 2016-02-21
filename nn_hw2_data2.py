import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv

#data = np.loadtxt("hw2_data1.txt")
myfile = open('NeuralNetworkHw2/hw2_data2.txt',"r")

# Read in the data
X=[]
y = []
csv_reader = csv.reader(myfile)
for row in csv_reader:
    X.append([row[0],row[1]])
    y.append(row[2])
    #num_examples = len(X) # training set size
X = np.asarray(X)
#print X
#fig = plt.figure()
#matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()
#fig.suptitle('Data plot for Data set 1', fontsize=20)
#plt.xlabel('x coordinate', fontsize=18)
#plt.ylabel('y coordinate', fontsize=16)
#fig.savefig('hw2_data2.png')
#print num_examples
    # myfile.txt contains 4 columns of numbers
    #x1, x2, y = data[:,0], data[:,1], data[:,2]
#for i in range(0,num_examples):
    #X.append([data[i,0],data[i,1]])

    # define training set
X = np.array(X).astype(float)
y = np.array(y).astype(float).astype(int)


# Display plots inline and change default figure size
#%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
loss = []
iteration = []

num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
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

#sigmoid as activation function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

#tanh as activation function
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return (1-np.power(x,2))

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    exp_scores =np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def build_model(nn_hdim1, nn_hdim2,  num_passes=5000, print_loss=False):

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

        # Forward propagation
        z1 = np.dot(X,W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        a2 = sigmoid(z2)
        z3 = a2.dot(W3) + b3
        exp_scores = np.exp(z3)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW3 = (a2.T).dot(delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W3.T) * (a2*(a2-1))
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0)
        delta1 = delta2.dot(W2.T) * (a1*(a1-1))
        dW1 = np.dot(X.T, delta1)
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
        if print_loss and i % 1000 == 0:
            iteration.append(i)
            loss.append(calculate_loss(model))
            print "Loss after iteration %i: %f" %(i, calculate_loss(model))
    return model

model = build_model(10,5, print_loss=True)
num_test = 20
count = 0.0
test = np.zeros(shape=(num_test,2))
for i in range(0,num_test):
    test[i] = [X[i,0],X[i,1]]
    result = predict(model,test[i])
    print(i,result)
    if result == y[i]:
        count = count + 1
print count
accuracy = (count/num_test)*100
print "Accuracy is %f" %accuracy
#print iteration[1:]
fig = plt.figure()
plt.plot(iteration[1:],loss[1:])
fig.suptitle('Comparison of loss with iteration', fontsize=20)
plt.xlabel('Number of iteration', fontsize=18)
plt.ylabel('loss using cross loss entropy', fontsize=16)
fig.savefig('NeuralNetworkHw2/loss_function_data2.png')
plt.show()