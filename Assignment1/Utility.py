# Defining activation functions

def sigmoid(x):
    return 1.0 / (1 + np.exp(-(x)))

def tanh(x):
    return np.tanh(x)

def sin(x):
    return np.sin(x)

def relu(x):
    return (x>0)*(x) + ((x<0)*(x)*0.01)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid_derivative(x):
    return  (1.0 / (1 + np.exp(-(x))))*(1 -  1.0 / (1 + np.exp(-(x))))

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return (x>0)*np.ones(x.shape) + (x<0)*(0.01*np.ones(x.shape) )

# helper functions
def oneHotEncode(matrix_size, Y_data):
    oneHotEncodedMatrix = np.zeros(matrix_size)
    for i in range(Y_data.shape[0]):
        value = Y_data[i]
        oneHotEncodedMatrix[int(value)][i] = 1.0
    return oneHotEncodedMatrix
	
def accuracy(Y_true, Y_pred, data_size):
    Y_true_label = []
    Y_pred_label = []
    ctr = 0
    for i in range(data_size):
        Y_true_label.append(np.argmax(Y_true[:, i]))
        Y_pred_label.append(np.argmax(Y_pred[:, i]))
        if Y_true_label[i] == Y_pred_label[i]:
            ctr += 1
    accuracy = ctr / data_size
    return accuracy, Y_true_label, Y_pred_label	

# Loss functions
def meanSquaredErrorLoss(Y_true, Y_pred):
    MSE = np.mean((Y_true - Y_pred) ** 2)
    return MSE

def crossEntropyLoss(Y_true, Y_pred):
    CE = [-Y_true[i] * np.log(Y_pred[i]) for i in range(len(Y_pred))]
    crossEntropy = np.mean(CE)
    return crossEntropy

def L2RegularisationLoss(weight_decay, weights):
    ALPHA = weight_decay
    return ALPHA * np.sum(
        [
            np.linalg.norm(weights[str(i + 1)]) ** 2
            for i in range(len(weights))
        ]
    )

# initializer functions
def random_initializer(size):
    in_dim = size[1]
    out_dim = size[0]
    return np.random.normal(0, 1, size=(out_dim, in_dim)) 

def Xavier_initializer(size):
    in_dim = size[1]
    out_dim = size[0]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return np.random.normal(0, xavier_stddev, size=(out_dim, in_dim))