import numpy as np
import scipy as sp
import wandb
import time

class FeedForwardNeuralNetwork:
    def __init__(
        self, 
        no_of_hidden_layers, no_of_hidden_neurons_per_layer, 
        X_train_orig, Y_train_orig, N_train, 
        X_val_orig, Y_val_orig, N_val,
        X_test_orig, Y_test_orig, N_test,
        optimizer,
        batch_size,
        weight_decay,
        learning_rate,
        max_epochs,
        activation,
        initializer,
        loss
    ):
      # Initializing FeedForwardNeuralNetwork class with the no_of_hidden_layers, no_of_hidden_neurons_per_layer and original training data
      # Input layer
      self.img_height_px = X_train_orig.shape[1]
      self.img_width_px = X_train_orig.shape[2]
      self.img_size_px = self.img_height_px * self.img_width_px

      # Hidden layer
      self.no_of_hidden_layers = no_of_hidden_layers
      self.no_of_hidden_neurons_per_layer = no_of_hidden_neurons_per_layer
      
      # Output layer
      self.no_of_classes = np.max(Y_train_orig) + 1
      self.output_layer_size = self.no_of_classes
          

      # self.layers = layers
      self.layers = ([self.img_size_px] + no_of_hidden_layers * [no_of_hidden_neurons_per_layer] + [self.output_layer_size])

      self.N_train = N_train
      self.N_val = N_val
      self.N_test = N_test        


      # [img_height_px*img_width_px X N_train]
      self.X_train = np.transpose(
          X_train_orig.reshape(X_train_orig.shape[0], X_train_orig.shape[1] * X_train_orig.shape[2]))

      # [img_height_px*img_width_px X N_test]
      self.X_test = np.transpose(
          X_test_orig.reshape(X_test_orig.shape[0], X_test_orig.shape[1] * X_test_orig.shape[2]))
      
      # [img_height_px*img_width_px X N_val]
      self.X_val = np.transpose(X_val_orig.reshape(X_val_orig.shape[0], X_val_orig.shape[1] * X_val_orig.shape[2]))


      # Normalizing the inputs
      self.X_train = self.X_train / 255
      self.X_test = self.X_test / 255
      self.X_val = self.X_val / 255
      
      # Matrix size is [no_of_classes X number of samples]
      self.Y_train = oneHotEncode((self.no_of_classes, Y_train_orig.shape[0]),Y_train_orig)  
      self.Y_val = oneHotEncode((self.no_of_classes, Y_val_orig.shape[0]),Y_val_orig)
      self.Y_test = oneHotEncode((self.no_of_classes, Y_test_orig.shape[0]), Y_test_orig)
      

      self.activations_dict = {
          "SIGMOID": sigmoid, 
          "TANH": tanh, 
          "RELU": relu
      }

      self.activation_derivative_dict = {
          "SIGMOID": sigmoid_derivative,
          "TANH": tanh_derivative,
          "RELU": relu_derivative,
      }

      self.initializer_dict = {
          "XAVIER": Xavier_initializer,
          "RANDOM": random_initializer
          #"HE": self.He_initializer
      }

      self.optimizer_dict = {
          "SGD": self.sgd,
          "MGD": self.mgd,
          "NAG": self.nag,
          "RMSPROP": self.rmsProp,
          "ADAM": self.adam,
          "NADAM": self.nadam,
      }
      
      self.activation = self.activations_dict[activation]
      
      self.activation_derivative = self.activation_derivative_dict[activation]
      
      self.optimizer = self.optimizer_dict[optimizer]
      self.initializer = self.initializer_dict[initializer]
      self.loss_function = loss
      self.max_epochs = max_epochs
      self.batch_size = batch_size
      self.learning_rate = learning_rate
      
      self.weights, self.biases = self.initializeNeuralNetwork(self.layers)


    def initializeNeuralNetwork(self, layers):
        weights = {}
        biases = {}
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
          W = self.initializer(size=[layers[l + 1], layers[l]])
          b = np.zeros((layers[l + 1], 1))
          weights[str(l + 1)] = W
          biases[str(l + 1)] = b
        return weights, biases   
    
    def forwardPropagation(self, X_train, weights, biases):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                 : X - input matrix
                 : Weights  - Weights matrix
                 : biases - Bias vectors 
        """
        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        X = X_train
		
        # A - Preactivations
        # H - Activations        
        H = {}
        A = {}		
        H["0"] = X
        A["0"] = X
		
	    # Input and hidden layers
        for l in range(0, num_layers - 2):
          W = weights[str(l + 1)]
          b = biases[str(l + 1)]          
            
          # pre activation is different for layer 0 and rest of the layers
          if l == 0:                
              A[str(l + 1)] = np.add(np.matmul(W, X), b)
          else:                
              A[str(l + 1)] = np.add(np.matmul(W, H[str(l)]), b)
          H[str(l + 1)] = self.activation(A[str(l + 1)])         
        
        # Output layer - Softmax is used as the output function
        W = weights[str(num_layers - 1)]
        b = biases[str(num_layers - 1)]
        A[str(num_layers - 1)] = np.add(np.matmul(W, H[str(num_layers - 2)]), b)
        Y = softmax(A[str(num_layers - 1)])
        H[str(num_layers - 1)] = Y
        return Y, H, A
		
    def backPropagation(self, Y, H, A, Y_train_batch):        
        gradients_weights = []
        gradients_biases = []
        num_layers = len(self.layers)

        # Gradient with respect to the output layer
        if self.loss_function == "CROSS":
            globals()["grad_a" + str(num_layers - 1)] = -(Y_train_batch - Y)
        elif self.loss_function == "MSE":
            globals()["grad_a" + str(num_layers - 1)] = np.multiply(2 * (Y - Y_train_batch), np.multiply(Y, (1 - Y)))

        for l in range(num_layers - 2, -1, -1):
            globals()["grad_W" + str(l + 1)] = (np.outer(globals()["grad_a" + str(l + 1)], H[str(l)]))            
            globals()["grad_b" + str(l + 1)] = globals()["grad_a" + str(l + 1)]
			
            gradients_weights.append(globals()["grad_W" + str(l + 1)])
            gradients_biases.append(globals()["grad_b" + str(l + 1)])
            
            globals()["grad_h" + str(l)] = np.matmul(self.weights[str(l + 1)].transpose(),globals()["grad_a" + str(l + 1)],)
            if l != 0:                
                globals()["grad_a" + str(l)] = np.multiply(globals()["grad_h" + str(l)], self.activation_derivative(A[str(l)]))
            elif l == 0:                
                globals()["grad_a" + str(l)] = np.multiply(globals()["grad_h" + str(l)], (A[str(l)]))
				
        return gradients_weights, gradients_biases
