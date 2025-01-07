import tkinter as tk
import time

def start_drawing(canvas, event): # Initialises the drawing process on the canvas.
    global last_position, last_time
    last_position = (event.x, event.y)
    last_time = time.time()
    canvas.bind("<Motion>", lambda e: draw_motion(canvas, e))

def stop_drawing(canvas, event): # Stops the drawing process on the canvas.
    canvas.unbind("<Motion>")

def draw_motion(canvas, event): # Draws a vector between two positions following the mouse movement.
    global last_position, last_time, activation_states
    current_time = time.time()
    if current_time - last_time >= 0.05:  # Draw every 50ms
        x1, y1 = last_position
        x2, y2 = event.x, event.y
        canvas.create_line(x1, y1, x2, y2, fill="black", width=1)
        drawing_vectors.append(((x1, y1), (x2, y2)))
        check_intersection(x1, y1, x2, y2)
        last_position = (x2, y2)
        last_time = current_time

def check_intersection(x1, y1, x2, y2): # Checks if the drawn vector intersects an activation vector and updates it if so.
    global activation_states
    for i, ((ax1, ay1), (ax2, ay2)) in enumerate(activation_vectors):
        if activation_states[i] == 0 and is_intersecting((x1, y1), (x2, y2), (ax1, ay1), (ax2, ay2)):
            activation_states[i] = 1  # Activates the vector
            # Redraws the vector in red
            canvas2.create_line(ax1, ay1, ax2, ay2, fill="red", width=2)

def is_intersecting(p1, p2, q1, q2): # Determines if two segments (p1-p2 and q1-q2) intersect.
    def orientation(a, b, c):
        # Calculates the orientation
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
    with open(activation_file, "w") as f:
        f.write(",".join(map(str, activation_states)))

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
         
    
    # Initializes activation states
    activation_states.extend([0] * len(activation_vectors))
    print(f"Activation Vectors: {activation_vectors}")  # Check the vectors
    print(f"Activation States: {activation_states}")  # Check the states
    return canvas2

# Create the main window
root = tk.Tk()
root.title("Drawing Canvas")

# Create the main frame
frame = tk.Frame(root, bg="white")
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Initialize vectors for each canvas
drawing_vectors = []  # Vectors drawn on canvas1
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



# Initialize global variables
last_position = None
last_time = None

# Start the main loop
tk.mainloop()