import numpy as np
import time
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * np.sqrt(1 / input_size) #Xavier initialisation
        self.bias = 0
        self.value = 0
        self.activated_value = 0
        
    def activate(self, activation_function):
        if activation_function == 'sigmoid':
            self.activated_value = 1 / (1 + np.exp(-self.value))
        elif activation_function == 'relu':
            self.activated_value = max(0, self.value)
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")
    
    def activation_derivative(self, activation_function):
        if activation_function == 'sigmoid' :
            return self.activated_value * (1 - self.activated_value)
        elif activation_function == 'relu':
            if self.activated_value > 0 :
                return 1
            else :
                return 0
            
            
            
    # def activate(self):
    #     self.activated_value = self.sigmoid()

    def forward_pass(self, input_vec, activation_function):
        self.value = np.dot(self.weights, input_vec) + self.bias
        self.activate(activation_function)

    def cost_derivative_by_layer(self, output_ref) : # (1/2) * 2(y - y_ref)
        return (self.activated_value - output_ref)

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate, activation_function):
        self.total_layers = []
        self.activation_function = activation_function
        for i in range(len(layer_sizes) - 1):
            self.total_layers.append([Neuron(layer_sizes[i]) for _ in range(layer_sizes[i + 1])])
        self.learning_rate = learning_rate

    def forward_pass(self, input_vec):
        current_input = input_vec
        for layer in self.total_layers:
            for neuron in layer:
                neuron.forward_pass(current_input, self.activation_function)
            current_input = [neuron.activated_value for neuron in layer]
    
    def update_weights_and_biases(self, weight_gradients, bias_gradients):
        i = 0
        for l in range(len(self.total_layers) - 1, -1, -1):
            for neuron in self.total_layers[l]:
                neuron.weights -= self.learning_rate * weight_gradients[i]
                neuron.bias -= self.learning_rate * bias_gradients[i]
                i += 1

    def backpropagation(self, output_vec, input_vec):
        
        layer_outputs = [input_vec]
        for layer in self.total_layers: # we store all the a(L=n) values
            layer_outputs.append([neuron.activated_value for neuron in layer])

        errors = [] # store the error of each layer
        weight_gradients = [] # store all the weights gradient of each layers (a list of np arrays)
        bias_gradients = [] #store all the bias_gradient of each layers (a list of values)

        
        output_layer = self.total_layers[-1]
        output_errors = [] # store the propagate errors from each output neuron to the previous layer 
        for i, neuron in enumerate(output_layer): # Output backprop
            bias_gradient = neuron.cost_derivative_by_layer(output_vec[i]) * neuron.activation_derivative(self.activation_function) # da(L)/dz(L) * dC0/da(L) ***the last term is a(L-1)***
            output_errors.append(bias_gradient) # we use this list for propagating the error
            bias_gradients.append(bias_gradient) # we use this list for the bias update
            weight_gradients.append(np.array(layer_outputs[-2]) * bias_gradient)

        errors.append(output_errors)

        # hidden layers backprop
        for l in range(len(self.total_layers) - 2, -1, -1): #we start from the last hidden layer, to the first hidden layer
            layer = self.total_layers[l]
            next_layer = self.total_layers[l + 1]
            hidden_errors = [] # store the propagate errors from each a(L) neuron to the a(L-1) layer
            for j, neuron in enumerate(layer): # we calculate the propagate sum error from layer L to layer L-1 : dC/dz(L-1) = (Σ(dC0/dz(L)) * w(L)) * (da(L-1)/dz(L-1))
                s = sum(next_layer[k].weights[j] * errors[-1][k] for k in range(len(next_layer))) # error[-1][k] represent the errors of each neuron of the last layer (-1 is the last layer append to the errors list)
                bias_gradient = s * neuron.activation_derivative(self.activation_function)
                hidden_errors.append(bias_gradient)
                bias_gradients.append(bias_gradient)
                weight_gradients.append(np.array(layer_outputs[l]) * bias_gradient) # bias_gradient * dC0/da(L-1)
            errors.append(hidden_errors) # we will use the error of the a(l-1) for the next iteration with error[-1]

        return (weight_gradients, bias_gradients)

    def train(self, input_data, output_data, epochs, batch_size, progress_callback=None):
        errors = []
        for epoch in range(epochs):
            if progress_callback: # for the progress bar
                progress_callback(epoch)
            # shuffle the data while keeping pairs of input/output
            permutation = np.random.permutation(len(input_data))
            input_data = input_data[permutation]
            output_data = output_data[permutation]
            
            total_error = 0
            for i in range(0, len(input_data), batch_size):
                batch_input = input_data[i:i+batch_size]
                batch_output = output_data[i:i+batch_size]
                batch_weight_gradients = []
                batch_bias_gradients = []
                for j, sample in enumerate(batch_input):
                    self.forward_pass(sample)
                    gradients = self.backpropagation(batch_output[j], sample)
                    if batch_weight_gradients == []:
                        batch_weight_gradients = gradients[0]
                        batch_bias_gradients = gradients[1]
                    else: # we aggregate all previous gradients with the one of the current sample
                        batch_weight_gradients = [wg + g for wg, g in zip(batch_weight_gradients, gradients[0])]
                        batch_bias_gradients = [bg + g for bg, g in zip(batch_bias_gradients, gradients[1])]
                    # calculate total error
                    total_error += sum((neuron.activated_value - batch_output[j][k])**2 for k, neuron in enumerate(self.total_layers[-1]))
                
                # Average gradients over the batch
                avg_weight_gradients = [wg / batch_size for wg in batch_weight_gradients]
                avg_bias_gradients = [bg / batch_size for bg in batch_bias_gradients]
                self.update_weights_and_biases(avg_weight_gradients, avg_bias_gradients)
            errors.append(total_error / len(input_data))
            if epoch % 1000 == 0:
                # show the MSE every x epoch
                print(f"Epoch {epoch}, MSE: {errors[-1]}")
                
        #create the error plot
        plt.clf()
        plt.plot(range(epochs), errors, label='Total error')
        plt.xlabel('Epoch')
        plt.ylabel('Mean error')
        plt.title('Error evolution through epoch')
        plt.legend()
        plt.savefig("error_plot.png")
        print("Plot saved as 'error_plot.png'")

    def test(self, input_set, output_set):
        for i, sample in enumerate(input_set):
            self.forward_pass(sample)
            prediction = [neuron.activated_value for neuron in self.total_layers[-1]] # retrieve the values on the output layer
            predicted_class = np.argmax(prediction)
            reference_class = np.argmax(output_set[i])
            
            print(f"Sample {i + 1}:")
            print(f"  Predicted class: {predicted_class} (values: {prediction})")
            print(f"  Reference class: {reference_class} (output: {output_set[i]})")
            print()
    
    def inference(self, sample) :
        self.forward_pass(sample)
        prediction = [neuron.activated_value for neuron in self.total_layers[-1]] # retrieve the values on the output layer
        predicted_class = np.argmax(prediction)
        print(f"  Predicted class: {predicted_class} (values: {prediction})")
        return prediction, predicted_class
        
