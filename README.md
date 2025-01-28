# Drawing_recognition
### Description
This project is an interactive application that allows users to draw sketches, save them, and train a neural network to recognize those sketches. It demonstrates the basics of supervised learning with a neural network implemented entirely from scratch, without using any machine learning libraries.

---

### Project Structure

**Main Files:**
- `neural_network.py`: Contains the `NeuralNetwork` class with training and prediction methods.
- `nn_config.py`: Handles dataset preparation and initializes the neural network.
- `canva2.py`: Provides the application interface for drawing, saving, and training.

---

### Usage Guide

#### Step 1: Drawing and Saving
1. Run the application using `python canva2.py`.
2. Draw a sketch on the left canvas.
3. Assign a name to the sketch and click **Save** to store it.

#### Step 2: Training
1. Save multiple sketches across different categories (e.g., "house", "tree", "car").
2. Click **Train** to train the neural network on the saved sketches.
   - By default, the training uses the following parameters: 
     - Learning rate (`lr`): 0.2.
     - Activation function: Sigmoid.
     - Hidden layer size: 100 neurons.
     - Epochs: 1000.
   - These parameters work well for recognizing 3 different sketch categories with 2 samples per category.
3. Training is very fast (less than 10 seconds) and works with very few samples.

#### Step 3: Prediction
1. Draw a new sketch on the left canvas.
2. Observe the predicted category on the right canvas and as text below it.

---

### Customizable Parameters
You can adjust the neural network parameters directly in the `training()` function of the `canva2.py` file at **line 203**:
- At **line 205**, change the number of epochs:
  ```python
  epochs = 1000
  ```
At line 215, modify the neural network configuration:
  ```python
  nn, file_names = init_and_train_nn(
    epochs=epochs, 
    lr=0.2, 
    activation_function='sigmoid', 
    hidden_size=100, 
    progress_callback=update_progress_bar
)
```
### Key features
1. **Custom Implementation**: The neural network and data preparation are fully implemented from scratch using only `numpy` for numerical computations and `Tkinter` for the interface. No machine learning libraries are used.

2. **Efficient Recognition**: Activation vectors are used to reduce the dimensionality of input data and improve generalization. This method allows the model to work effectively without relying on convolutional networks.

3. **Minimal Training Data**: The neural network can recognize drawings with as little as 2 samples per category, making it highly data-efficient.

4. **Fast Training**: Training the neural network takes less than 10 seconds, even on a modest setup.
