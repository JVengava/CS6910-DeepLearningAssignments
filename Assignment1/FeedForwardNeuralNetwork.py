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
        
        
    def predict(self,X,length_dataset):
        Y_pred = []        
        for i in range(length_dataset):
            Y, H, A = self.forwardPropagation(X[:, i].reshape(self.img_size_px, 1),self.weights,self.biases,)
            Y_pred.append(Y.reshape(self.no_of_classes,))
			
        Y_pred = np.array(Y_pred).transpose()
        return Y_pred

    def sgd(self, epochs, length_dataset, batch_size, learning_rate, weight_decay=0):
        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        num_points_seen = 0

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]

		# begin of epoch for loop
        for epoch in range(epochs):
            start_time = time.time()
            
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_size_px, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.no_of_classes, length_dataset)

            CE = []           
			
			# setting up deltaw & deltab for updating weights
            deltaw = [
                np.zeros((self.layers[l + 1], self.layers[l]))
                for l in range(0, len(self.layers) - 1)
            ]
            deltab = [
                np.zeros((self.layers[l + 1], 1))
                for l in range(0, len(self.layers) - 1)
            ]

			# begin of for loop -- looping over entire data set for a given epoch
            for i in range(length_dataset):

                Y, H, A = self.forwardPropagation(
                    X_train[:, i].reshape(self.img_size_px, 1),
                    self.weights,
                    self.biases,
                )
                grad_weights, grad_biases = self.backPropagation(
                    Y, H, A, Y_train[:, i].reshape(self.no_of_classes, 1)
                )
                deltaw = [
                    grad_weights[num_layers - 2 - i] for i in range(num_layers - 1)
                ]
                deltab = [
                    grad_biases[num_layers - 2 - i] for i in range(num_layers - 1)
                ]
                
                CE.append(
                    crossEntropyLoss(
                        self.Y_train[:, i].reshape(self.no_of_classes, 1), Y
                    )
                    + L2RegularisationLoss(weight_decay, self.weights)
                )

                num_points_seen +=1

                if int(num_points_seen) % batch_size == 0:
                    self.weights = {str(i+1):(self.weights[str(i+1)] - learning_rate*deltaw[i]) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):(self.biases[str(i+1)] - learning_rate*deltab[i]) for i in range(len(self.biases))}
                    
                    #resetting gradient updates
                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
                
            # end of for loop -- looping over data set
				
            elapsed = time.time() - start_time            
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])
            
            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch, })        
		    # end of epoch for loop
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred
        
    def mgd(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        GAMMA = 0.9

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        

        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_size_px, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.no_of_classes, length_dataset)

            CE = []
            #Y_pred = []
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            

            for i in range(length_dataset):
                Y,H,A = self.forwardPropagation(self.X_train[:,i].reshape(self.img_size_px,1), self.weights, self.biases) 
                grad_weights, grad_biases = self.backPropagation(Y,H,A,self.Y_train[:,i].reshape(self.no_of_classes,1))
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(self.no_of_classes,))
                CE.append(crossEntropyLoss(self.Y_train[:,i].reshape(self.no_of_classes,1), Y) + L2RegularisationLoss(weight_decay, self.weights))
                
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
                    
                    self.weights = {str(i+1) : (self.weights[str(i+1)] - v_w[i]) for i in range(len(self.weights))}
                    self.biases = {str(i+1): (self.biases[str(i+1)] - v_b[i]) for i in range(len(self.biases))}

                    prev_v_w = v_w
                    prev_v_b = v_b

                    #resetting gradient updates
                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred

    def nag(self,epochs,length_dataset, batch_size,learning_rate, weight_decay = 0):
        GAMMA = 0.9

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        


        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_size_px, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.no_of_classes, length_dataset)

            CE = []
            #Y_pred = []  
            
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(self.layers)-1)]

            for i in range(length_dataset):
                winter = {str(i+1) : self.weights[str(i+1)] - v_w[i] for i in range(0, len(self.layers)-1)}
                binter = {str(i+1) : self.biases[str(i+1)] - v_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagation(self.X_train[:,i].reshape(self.img_size_px,1), winter, binter) 
                grad_weights, grad_biases = self.backPropagation(Y,H,A,self.Y_train[:,i].reshape(self.no_of_classes,1))
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(self.no_of_classes,))
                CE.append(crossEntropyLoss(self.Y_train[:,i].reshape(self.no_of_classes,1), Y) + L2RegularisationLoss(weight_decay, self.weights))

                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:                            

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
        
                    self.weights ={str(i+1):self.weights[str(i+1)]  - v_w[i] for i in range(len(self.weights))}
                    self.biases = {str(i+1):self.biases[str(i+1)]  - v_b[i] for i in range(len(self.biases))}
                
                    prev_v_w = v_w
                    prev_v_b = v_b

                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

    
            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred       
