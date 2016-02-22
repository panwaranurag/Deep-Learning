import numpy as np
import matplotlib.pyplot as plt
import timeit

########Training data processing#########
np.seterr(invalid='ignore')    #ignoring NaN type warning
#file = raw_input("Please enter file name with extention:")
data = np.loadtxt("hw2_data4.txt")
num_examples = int(len(data))   # training set size
num_of_features = int(len(data[0]))
np.random.shuffle(data)         #Data shuffling as part of data preprocessing
X=[]                            #Training data feature set value
y = []                          #Training data labels
no_of_iteartion = []
loss = []
iteration = []

print "Number of rows in file is: " + str(num_examples)
for i in range(0,num_examples):
    X.append([data[i,j] for j in range(num_of_features-1)])

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
    if data[i,num_of_features-1] == 1: #class 1 which is 1
        y.append(0)
    elif(data[i,num_of_features-1] == 2):
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

nn_input_dim = num_of_features-1 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
dropout = 0.5

# Gradient descent parameters (picked by hand)
epsilon = 0.001 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
#mini_batch = 100 #size of minibatches for stochastic gradient desent
neuron_layers = []
#neuron_layers=[3,10,5,2]
neuron_layers.append(nn_input_dim)
hidden_layers = int(raw_input("Enter number of hidden layers:"))
for i in range(1,hidden_layers+1):
    neuron_layers.append(int(raw_input("Number of neurons in hidden Layer " + str(i) + " ")))
neuron_layers.append(nn_output_dim)
no_of_layers = len(neuron_layers)

mini_batch = int(raw_input("Please enter batch size for stochastic gradient descent in range 0-" + str(num_examples) + " "))
fold = int(raw_input("Please enter fold size for testing in range 2-10 "))

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

###Timer Start here###
start = timeit.default_timer()

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
        W   =[0] * (no_of_layers-1)
        b =[0] * (no_of_layers-1)
        z    =[0] * (no_of_layers-1)
        a    =[0] * (no_of_layers-1)

        # Forward propagation to calculate our predictions
        for k in range(0, no_of_layers-1):
            W[k]=model["wt" + `k`]
            b[k]=model["bias" + `k`]
            if k==0:
                z[k]=X.dot(W[k]) + b[k]
                a[k]=self.activation(z[k])
            else:
                z[k]=a[k-1].dot(W[k]) + b[k]
                a[k]=self.activation(z[k])

        exp_scores  = np.exp(z[no_of_layers-2])
        probs = exp_scores  / np.sum(exp_scores , axis=1, keepdims=True)

        # Calculating the cross entropy loss
        cross_entropy = -np.log(probs[range(len(X)), y])
        total_loss = np.sum(cross_entropy)
        # Add regulatization term to loss (optional)
        #total_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W2)))
        return 1./len(X) * total_loss

    # Helper function to predict an output (0 or 1)
    def predict(self, model, x):
        W  = [0] * (no_of_layers-1)
        b =[0] * (no_of_layers-1)
        z    =[0] * (no_of_layers-1)
        a    =[0] * (no_of_layers-1)

        # Forward propagation
        for k in range(0, no_of_layers-1):
            W[k]=model["wt" + `k`]
            b[k]=model["bias" + `k`]
        #dropout for testing, multiple array with 0.5 with first hidden layer
            if k==0:
                z[k]=x.dot(W[k]) + b[k]
                a[k]=self.activation(z[k])
                #a[k] *= 0.5
            else:
                z[k]=a[k-1].dot(W[k]) + b[k]
                a[k]=self.activation(z[k])

        softmax_value = np.exp(z[no_of_layers-2])
        probs = softmax_value / np.sum(softmax_value, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def build_model(self, training_x, training_y, num_passes=20000, print_loss=False):

        ###DECLARING NEURAL NETWORK MODEL#####
        model = {}
        W  = [0] * (no_of_layers-1)
        b =[0] * (no_of_layers-1)
        z    =[0] * (no_of_layers-1)
        a    =[0] * (no_of_layers-1)
        delta=[0] * (no_of_layers-1)
        dw  =[0] * (no_of_layers-1)
        dbias=[0] * (no_of_layers-1)
        ##INITATE RANDOM SEED########
        np.random.seed(0)

        # Initialize the parameters to random values. We need to learn these
        for i in range(0, no_of_layers-1):
            W[i]=np.random.randn(neuron_layers[i], neuron_layers[i+1]) / np.sqrt(neuron_layers[i])
            b[i]=np.zeros((1, neuron_layers[i+1]))

        for i in xrange(0, num_passes):
            #stochastic gardient desent
            training_x_length = len(training_x)
            B = np.random.randint(training_x_length,size=mini_batch)
            X_stoch = training_x[B,:]
            y_stoch = training_y[B]
            #print X_stoch
            #print y_stoch

            # Forward propagation
            for k in range(0, no_of_layers-1):
                if k==0:
                     #masking for dropout at layer 1
                    z[k]=   np.dot(X_stoch,W[k]) + b[k]
                    a[k]=   self.activation(z[k])
                   # m2 = np.random.binomial(1, dropout, size=z[k].shape)
                    #a[k] *= m2
                else:
                    z[k]=   a[k-1].dot(W[k]) + b[k]
                    a[k]=   self.activation(z[k])

            softmax_value = np.exp(z[no_of_layers-2])
            probs = softmax_value / np.sum(softmax_value, axis=1, keepdims=True)
            temp=probs

            # Backpropagation
            for k in range(no_of_layers-2,0,-1):
                if k== no_of_layers-2:
                    temp[range(mini_batch), y_stoch] -= 1
                    delta[k] = temp
                    dw[k] = np.dot( a[k-1].T,delta[k])
                    dbias[k] = np.sum(delta[k], axis=0, keepdims=True)
                else:
                    delta[k] = delta[k+1].dot(W[k+1].T) * self.activation_deriv(a[k])
                    dw[k] = np.dot(a[k-1].T, delta[k])
                    dbias[k] = np.sum(delta[k], axis=0, keepdims=True)

            #Dropout in back propagation m2 = binary mask
            delta[0] = delta[1].dot(W[1].T) * self.activation_deriv(a[0])#*m2
            dw[0] = np.dot(X_stoch.T, delta[0])
            dbias[0] = np.sum(delta[0], axis=0, keepdims=True)


            #Updating weight and bias
            for k in range(0, no_of_layers-1):
                # Add regularization terms (b1 and b2 don't have regularization terms)
                dw[k] += reg_lambda * W[k]
                ###################################################################
                W[k] = W[k] -epsilon * dw[k]
                b[k] = b[k] -epsilon * dbias[k]
                model["wt" + `k`] = W[k]
                model["bias" + `k`] = b[k]

            if print_loss and i % 100 == 0:
                iteration.append(i)
                loss.append(self.calculate_loss(model))
                print "Updated loss function value after %i iteration: %f" %(i, self.calculate_loss( model))

        return model
nn = NeuralNetwork('sigmoid')
#10 fold cross validation
training_fold = (num_examples/fold)*(fold-1)
training_data = X[:training_fold]
training_label = y[:training_fold]
testing_data = X[(training_fold+1):]
testing_label = y[(training_fold+1):]

length_test_data = len(testing_data)
##
model = nn.build_model(training_data, training_label, print_loss=True)
count = 0.0
num_of_test_data = 40
for i in range(0,length_test_data):
    result = nn.predict(model,testing_data[i])
    print "Iteration: " + str(i) + " Actual label: " + str(testing_label[i]) + " Predicted label: " + str(result[0])
    if result == testing_label[i]:
        count = count + 1
accuracy = (count/length_test_data)*100
print "Accuracy = %f" %accuracy

#Timer stopping here#######################################
stop = timeit.default_timer()
print "Total running time of Neural Network is " + str(stop - start) + " seconds"

##############Plot of loss function#########################
fig = plt.figure()
plt.plot(iteration[1:],loss[1:])
fig.suptitle('Comparison of loss with iteration', fontsize=20)
plt.xlabel('Number of iteration', fontsize=18)
plt.ylabel('loss using cross loss entropy', fontsize=16)
fig.savefig('loss_function.png')
plt.show()