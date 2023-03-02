#Kavadi,Suresh
# 1002_040_703 (100x_xxx_xxx)
# 2023_02_26 (yyyy_mm_dd)
# Assignment_01_01

import numpy as np

def sigmoid(n):
     a = 1/(1+np.exp(-n))
     return a

def  multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):

     np.random.seed(seed)
     num_layers = len(layers)
     weights = []
     for i in range(num_layers):
         if i == 0:
             input_size =  X_train.shape[0]
         else:
             input_size = layers[i-1]
         output_size = layers[i]
         weight_matrix = np.random.randn(output_size, input_size+1)
         weights.append(weight_matrix)

     mse_errors = []
     test_outputs = []
     for epoch in range(epochs):
         for n in range(len(X_train[0])):  # length of input sample
             a = [X_train[:, n]]
             for i in range(num_layers):
                 c = weights[i] @ np.append(1, a[i])
                 a_i = sigmoid(c)
                 a.append(a_i)

          # Calculate MSE with random weights
             diff_in_output = a[-1] - Y_train[:, n]
             mse = np.sum(diff_in_output ** 2)/len(Y_train)

         # Calculate partial derivatives with respect to each weight using centered difference approximation
         for n in range(len(X_train[0])):
             a1 = [X_train[:, n]]
             temp_weights = []
             for i in range(num_layers):
                 temp_weight_matrix = np.zeros_like(weights[i])
                 weights[i] += h
                 temp_weights.append(weights[i])
                 z1 = temp_weights[i] @ np.append(1, a1[i])
                 #print(f"z1  is {z1}")
                 a1_i = sigmoid(z1)
                 a1.append([a1_i])
             grad_output_layer1 = a1[-1] - Y_train[:, n]
             mse_new = np.sum(grad_output_layer1 ** 2)/len(Y_train)

        # Calculate partial derivative using centered difference approximation
             partial_derivative = (mse_new - mse) / (2 * h)
         new_weights=[]
         for i in range (len(temp_weights)):

             temp_weight_matrix = temp_weights[i]- alpha * partial_derivative
             new_weights.append(temp_weight_matrix)

              # Update all weights using temporary matrices
         for i in range(num_layers):
             new_weights[i] = temp_weights[i]
         mse_errors.append(mse)

     # performing test with samples and weights have been updated
         a_test = [X_test[:, n]]
         for i in range(num_layers):
             z_test = np.dot(new_weights[i] , np.append(1, a_test[i]))
             a_test_i = sigmoid(z_test)
             a_test.append(a_test_i)
             test_output = a_test[-1]
         test_outputs.append(test_output)

           # calculating mse_mean for given epochs
     mse_mean_errors = []
     for epoch in range (epochs):
         mse_test_errors = []
         for n in range(len(X_test[0])):
             diff_in_test_output =  a_test[-1] - Y_test[:, n]
             mse_test = np.sum(diff_in_test_output ** 2)/len(Y_test)
             mse_test_errors.append(mse_test)
         mean = sum(mse_test_errors)/len(mse_test_errors)
         mse_mean_errors.append(mean)
     num_test=len(Y_test[0])

     return [weights, mse_mean_errors, (np.array(test_outputs).T)]






# I took below sample as input while writing the code

#import numpy as np

# Define the training and testing data
#X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1],[2,1]]).T
#Y_train = np.array([[0, 1, 1, 0,2]])
#X_test = X_train
#Y_test = Y_train

# Define the neural network architecture
#layers = [2, 3, 2]

# Define the learning rate, number of epochs, step size, and random seed
#alpha = 0.1
#epochs = 3
#h = 0.00001
#seed = 2

# Call the multi_layer_nn function
#weights, errors, predictions = multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h, seed)

# Print the results
#print("Final weights:", weights)
#print("Errors:", errors)
#print("Predictions:", predictions)




