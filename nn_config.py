from neural_network import NeuralNetwork
import numpy as np
import time
import os

# Function to read the files and get input-output sets
def load_samples_and_outputs(samples_dir):
    """Load activation states from files and generate input-output sets."""
    input_set = []
    output_set = []
    file_names = []
    
    # Get all files in the directory
    sample_files = sorted(os.listdir(samples_dir))  # Sorted for consistent ordering
    num_classes = len(sample_files) # Number of classes = number of files (each file represents a class)
    # Create a one-hot encoding matrix for outputs
    output_matrix = np.eye(num_classes)
    
    for idx, filename in enumerate(sample_files):
        file_path = os.path.join(samples_dir, filename)
        file_name_without_extension = os.path.splitext(filename)[0] # it's for retrieving the name for canva2.py
        file_names.append(file_name_without_extension)
        with open(file_path, "r") as file:
            for line in file:
                activation_states = line.strip().split(",")
                input_set.append([int(x) for x in activation_states])
                # Assign the corresponding one-hot encoded output for the entire file
                output_set.append(output_matrix[idx])
    return np.array(input_set), np.array(output_set), file_names



def init_and_train_nn(epochs, lr, activation_function, hidden_size, progress_callback=None) :
    input_set, output_set, file_names = load_samples_and_outputs("samples") # We load the input and output
    # Print sample results for debugging
    print("Input samples shape:", input_set.shape)
    print("Output samples shape:", output_set.shape)
    print("First input sample:", input_set[0])
    print("First output sample:", output_set[0])
    
    nn = NeuralNetwork(layer_sizes=[input_set.shape[1], hidden_size, output_set.shape[1]], learning_rate=lr, activation_function=activation_function) # Create the neural network
    print("Starting training...")
    start_time = time.time()
    end_time = time.time()
    print("Training over")

    training_time = end_time - start_time
    nn.train(input_set, output_set, epochs, batch_size=1, progress_callback=progress_callback)
    print(f"Training time: {training_time:.2f} seconds.")

    # Test the network
    print("\nTest result:")
    nn.test(input_set, output_set)
    
    return nn, file_names
    
def test_nn(nn, input_set) :
    return nn.inference(input_set)


"""
Pour la suite :
    - Ca marche !
    - Il faut maintenant faire en sorte d'avoir plusieurs sample qui ont le même nom et sont lié à un seul output.
"""