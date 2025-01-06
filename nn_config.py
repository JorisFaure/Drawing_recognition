from neural_network import NeuralNetwork
import numpy as np
import time

# Function to read the file and get the list of activated vectors
def load_activation_states(file_path):
    """Load the activation states list from the file."""
    with open(file_path, "r") as file:
        activation_states = file.read().strip().split(",")
        # Convert the values to integers (0 or 1)
        return np.array([int(x) for x in activation_states])

# Load the activation states from the file
activation_states = load_activation_states("activation_states.txt")

# Check that you have 200 entries
print(f"Number of entries: {len(activation_states)}")
if len(activation_states) != 200:
    raise ValueError("The activation file must contain 200 values.")

# -----------------------Test of the entire NN-------------------------------
# Using the activation states as input for the network
input_set = activation_states.reshape(1, -1)  # Reshape to match the expected input (1, 200)
output_set = np.array([[1]])  # Example output, you can adjust this according to your problem

# Create the neural network with 200 input neurons
nn = NeuralNetwork(layer_sizes=[200, 100, 1], learning_rate=0.2, activation_function='sigmoid')

# Measure the training time
print("Starting training...")
start_time = time.time()
nn.train(input_set, output_set, epochs=4000, batch_size=1)
end_time = time.time()
print("Training over")

training_time = end_time - start_time
print(f"Training time: {training_time:.2f} seconds.")

# Test the network
print("\nTest result:")
nn.test(input_set, output_set)

"""
Pour la suite :
    - Faire en sorte que le réseau ai autant d'output que de différents label de samples
    - Mettre en place l'enregistrement de plusieurs samples
        - Mettre une case qui permette d'entrer un label.
        - L'orsque l'on clique sur "save", ca enregistre la liste des activations en un fichier qui a le nom du label.
            - C'est la représentation du dessin actuelle en valeur neuronal.
        - Ensuite ca clear automatiquement le canva pour refaire un dessin.
        - A la fin on devrait avoir un fichier pour chaque enregistrement. Ca représente un sample.
        - La valeur x de chaque sample est le contenue du fichier, et la valeur y le nom du fichier.