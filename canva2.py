import tkinter as tk
from tkinter import ttk
import time
import threading
from nn_config import init_and_train_nn, test_nn
import numpy as np
import os
import shutil



def start_drawing(canvas, event):
    global last_position, last_time # we use it to specify that we use the variable declared at the beginning and not a new variable
    last_position = (event.x, event.y)
    last_time = time.time()
    canvas.bind("<Motion>", lambda e: draw_motion(canvas, e))

def stop_drawing(canvas, event):
    canvas.unbind("<Motion>")
    
def normalize_drawing():
    global drawing_vectors, norm_drawing_vectors, canvas1
    
    min_x = min([x[0] for vector in drawing_vectors for x in vector])
    max_x = max([x[0] for vector in drawing_vectors for x in vector])
    min_y = min([x[1] for vector in drawing_vectors for x in vector])
    max_y = max([x[1] for vector in drawing_vectors for x in vector])

    drawing_width = max_x - min_x
    drawing_height = max_y - min_y
    canvas_width = canvas1.winfo_width()
    canvas_height = canvas1.winfo_height()

    # Ratio calculus for keeping proportions
    scale_x = canvas_width / drawing_width
    scale_y = canvas_height / drawing_height
    scale_factor = min(scale_x, scale_y)
    # Centering
    offset_x = (canvas_width - drawing_width * scale_factor) / 2 - min_x * scale_factor
    offset_y = (canvas_height - drawing_height * scale_factor) / 2 - min_y * scale_factor

    normalized_vectors = []
    for (x1, y1), (x2, y2) in drawing_vectors:
        normalized_vectors.append(
            ((x1 * scale_factor + offset_x, y1 * scale_factor + offset_y), 
             (x2 * scale_factor + offset_x, y2 * scale_factor + offset_y))
        )

    # We replace the old version of normalized vectors by the new one
    norm_drawing_vectors = normalized_vectors
    
def draw_motion(canvas, event): # draws a vector between two positions following the mouse movement.
    global last_position, last_time, activation_states, prediction_frame, norm_drawing_vectors, canvas2
    current_time = time.time()
    if current_time - last_time >= 0.05:  # draw every 50ms
        x1, y1 = last_position
        x2, y2 = event.x, event.y
        canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
        drawing_vectors.append(((x1, y1), (x2, y2)))
        
        # We normalize before checking intersections
        normalize_drawing()
        
        # Resets canvas2
        canvas2.delete("all")
        activation_vectors.clear()
        activation_states.clear()

        # Resets the activation vectors
        init_canvas_2(canvas2)
        
        check_intersection_all(norm_drawing_vectors)
        
        last_position = (x2, y2)
        last_time = current_time
        if is_trained == 1 : # inference
            prediction, predicted_class = test_nn(nn, activation_states)
            # Clear previous prediction results (if any)
            for widget in prediction_frame.winfo_children():
                widget.destroy()
 
            # Display the prediction results
            for idx, file_name in enumerate(file_names):
                confidence = prediction[idx] * 100
                file_label = tk.Label(prediction_frame, text=f"{file_name}: {confidence:.1f}%")
                file_label.pack(anchor="w")
 
            # Display the highest confidence prediction
            max_confidence_idx = np.argmax(prediction)
            highest_prediction_label = tk.Label(prediction_frame, text=f"I think it's: {file_names[max_confidence_idx]}", font=("Arial", 14, "bold"))
            highest_prediction_label.pack(anchor="w")
            
            
def check_intersection_all(vector_list) :
    for vector in vector_list :
        (x1, y1), (x2, y2) = vector
        check_intersection(x1, y1, x2, y2)

def check_intersection(x1, y1, x2, y2): # Checks if the drawn vector intersects an activation vector and updates it if so.
    global activation_states
    for i, ((ax1, ay1), (ax2, ay2)) in enumerate(activation_vectors):
        if activation_states[i] == 0 and is_intersecting((x1, y1), (x2, y2), (ax1, ay1), (ax2, ay2)):
            activation_states[i] = 1  # Activates the vector and redraw it in red
            canvas2.create_line(ax1, ay1, ax2, ay2, fill="red", width=2)

def is_intersecting(p1, p2, q1, q2): # Determines if two segments (p1-p2 and q1-q2) intersect.
    def orientation(a, b, c):
        # calculates the orientation
        return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    
    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)
    
    # Checks the orientations
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    
    return False

def clear_canvas(): # Clears canvas1 and resets canvas2.
    global activation_vectors, activation_states

    # Clears canvas1
    canvas1.delete("all")
    drawing_vectors.clear()

    # Resets canvas2
    canvas2.delete("all")
    activation_vectors.clear()
    activation_states.clear()

    # Resets the activation vectors
    init_canvas_2(canvas2)

def save_vectors(): # Saves the drawn vectors in two files.
    if not os.path.exists("samples"):
        os.makedirs("samples")
    with open("canvas1_vectors.txt", "w") as f1:
        for vector in drawing_vectors:
            f1.write(f"{vector[0]},{vector[1]}\n")
    with open("canvas2_vectors.txt", "w") as f2:
        for vector in activation_vectors:
            f2.write(f"{vector[0]},{vector[1]}\n")
    
    sample_name = label_input.get()  # Get the sample label from the Entry field
    if not sample_name.strip():
        sample_name = "default"  # Use "default" if no label is provided

    # Save activation states to a file named after the sample label
    activation_file = f"samples/{sample_name}.txt"
    with open(activation_file, "a") as f:
        f.write(",".join(map(str, activation_states)) + "\n")
    
    clear_canvas()

def init_canvas_2(canvas2): # Initializes canvas_2 and draws a sawtooth pattern, vector by vector.
    global activation_vectors, activation_states
    canvas2.delete("all")  # Clears the current content

    # Canvas dimensions
    width = 400
    height = 400
    step = 40  # Spacing between points (length of a vector)
    demi_step = 20
    # Draws diagonals from top-left to bottom-right
    for y in range(0, height - step + 1, step):
        for x in range(0, width - step + 1, step):
            canvas2.create_line(x, y, x + step/2, y + step/2, fill="black")
            activation_vectors.append(((x, y), (x + step/2, y + step/2)))
    for y in range(demi_step, height - demi_step + 1, step):
        for x in range(demi_step, width - demi_step + 1, step):
            canvas2.create_line(x, y, x + step/2, y + step/2, fill="black")
            activation_vectors.append(((x, y), (x + step/2, y + step/2))) 
    

    # Draws diagonals from bottom-left to top-right
    for y in range(demi_step, height - demi_step + 1, step):
        for x in range(demi_step, width - demi_step + 1, step):
            canvas2.create_line(x, y, x + step/2, y - step/2, fill="black")
            activation_vectors.append(((x, y), (x + step/2, y - step/2)))
    for y in range(step, height + 1, step):
        for x in range(0, width - step + 1, step):
            canvas2.create_line(x, y, x + step/2, y - step/2, fill="black")
            activation_vectors.append(((x, y), (x + step/2, y - step/2)))
    
    # Draws diagonals from bottom-left to top-right
    for y in range(0, height, demi_step):
        for x in range(demi_step, width, demi_step):
            canvas2.create_line(x, y, x, y + step/2, fill="black")
            activation_vectors.append(((x, y), (x, y + step/2)))
    for y in range(demi_step, height - demi_step + 1, demi_step):
        for x in range(0, width, demi_step):
            canvas2.create_line(x, y, x + step/2, y, fill="black")
            activation_vectors.append(((x, y), (x + step/2, y)))
         
    
    # Initializes activation states
    activation_states.extend([0] * len(activation_vectors))
    return canvas2

def training():
    global is_trained, nn, progress_bar
    epochs = 1000

    # Update progress bar callback function
    def update_progress_bar(epoch):
        progress_bar["value"] = (epoch / epochs) * 100
        progress_bar.update_idletasks()  # update graphic interface

    # Run the training in a separate thread to keep the UI responsive
    def thread_training():
        global nn, is_trained, file_names
        nn, file_names = init_and_train_nn(epochs=epochs, lr=0.2, activation_function='sigmoid', hidden_size=100, progress_callback=update_progress_bar)
        is_trained = 1
        print("Entraînement terminé.")
    
    threading.Thread(target=thread_training).start()

def on_close():
    # delete 'samples' folder
    if os.path.exists("samples"):
        shutil.rmtree("samples")
    root.destroy()

# Create the main window
root = tk.Tk()
root.title("Drawing Canvas")

# Create the main frame
frame = tk.Frame(root, bg="white")
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Initialize vectors for each canvas
drawing_vectors = []  # Vectors drawn on canvas1
norm_drawing_vectors = [] # Vectors drawn on canvas1 but normalized for canva2 intersections calculus
activation_vectors = []  # Activation vectors on canvas2
activation_states = []  # Activation states for each vector on canvas2

# Create the two canvases
canvas1 = tk.Canvas(frame, width=400, height=400, bg="white", highlightbackground="black", highlightthickness=1)
canvas1.grid(row=0, column=0, padx=5, pady=5)

canvas2 = tk.Canvas(frame, width=400, height=400, bg="white", highlightbackground="black", highlightthickness=1)
canvas2.grid(row=0, column=1, padx=5, pady=5)

# Initialize canvas2 by calling init_canvas_2
canvas2 = init_canvas_2(canvas2)

# Bind mouse events for the first canvas
canvas1.bind("<ButtonPress-1>", lambda e: start_drawing(canvas1, e))
canvas1.bind("<ButtonRelease-1>", lambda e: stop_drawing(canvas1, e))

# Add Clear and Save buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

input_text = tk.Label(button_frame, text="Enter your sample label :")
input_text.pack(side=tk.LEFT, padx=5)
label_input = tk.Entry(button_frame)
label_input.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=5)

save_button = tk.Button(button_frame, text="Save", command=save_vectors)
save_button.pack(side=tk.LEFT, padx=5)

train_button = tk.Button(button_frame, text="Train", command=training)
train_button.pack(side=tk.LEFT, padx=5)

progress_frame = tk.Frame(root)
progress_frame.pack(pady=10)
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
progress_bar["maximum"] = 100
progress_bar.pack(pady=10)

# Create a frame for the prediction results to display under 'Prediction results:'
prediction_results_frame = tk.Frame(root)
prediction_results_frame.pack(pady=10)

results_text = tk.Label(prediction_results_frame, text="Prediction results :")
results_text.pack(side=tk.LEFT, padx=5)

# Frame to contain the dynamic prediction results
prediction_frame = tk.Frame(prediction_results_frame)
prediction_frame.pack(pady=5)



# Initialize global variables
last_position = None
last_time = None
is_trained = 0 # Know if training is finish, to starting inference
nn = None
file_names = None

root.protocol("WM_DELETE_WINDOW", on_close)
# Start the main loop
tk.mainloop()