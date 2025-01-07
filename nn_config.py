from neural_network import NeuralNetwork
import numpy as np
import time
import os

# Function to read the files and get input-output sets
def load_samples_and_outputs(samples_dir):
    """Load activation states from files and generate input-output sets."""
    input_set = []
    output_set = []
    
    # Get all files in the directory
    sample_files = sorted(os.listdir(samples_dir))  # Sorted for consistent ordering
    
    # Number of samples determines the number of output neurons
    num_samples = len(sample_files)
    
    # Create a one-hot encoding matrix for outputs
    output_matrix = np.eye(num_samples)
    
    for idx, filename in enumerate(sample_files):
        file_path = os.path.join(samples_dir, filename)
        
        # Read activation states from the file
        with open(file_path, "r") as file:
            activation_states = file.read().strip().split(",")
            input_set.append([int(x) for x in activation_states])
        
        # Add the corresponding output vector (one-hot encoded)
        output_set.append(output_matrix[idx])
    
    return np.array(input_set), np.array(output_set)

# Load the samples and outputs
samples_dir = "samples"
input_set, output_set = load_samples_and_outputs(samples_dir)

# Check the shapes of input and output sets
print(f"Input set shape: {input_set.shape}")
print(f"Output set shape: {output_set.shape}")

# Create the neural network
# nn = NeuralNetwork(layer_sizes=[input_set.shape[1], 100, output_set.shape[1]], learning_rate=0.2, activation_function='sigmoid')

nn = NeuralNetwork(layer_sizes=[input_set.shape[1], 100, output_set.shape[1]], learning_rate=0.2, activation_function='sigmoid')

# Measure the training time
print("Starting training...")
start_time = time.time()
end_time = time.time()
print("Training over")

training_time = end_time - start_time
nn.train(input_set, output_set, epochs=4000, batch_size=1)
print(f"Training time: {training_time:.2f} seconds.")

# Test the network
print("\nTest result:")
nn.test(input_set, output_set)

"""
Pour la suite :
    - Tester si le réseau marche lorsque il y a un output avec + de 1 valeurs. Donc tester avec des petit truc custom (un peu en mode Xor)
"""