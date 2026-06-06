import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# --- 1. The Millipede's Brain (Data Generation) ---
def generate_data(n_points=2000, radius=2.0):
    # Create a grid of points to visualize deformations clearly
    r = np.sqrt(np.random.rand(n_points)) * radius
    theta = np.random.rand(n_points) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # We also add a specific "rainbow" coloring based on the x-axis
    # This helps us identify "folding" (when red touches blue)
    colors = x 
    return np.column_stack((x, y)), colors

# Initial Data
X_orig, C_orig = generate_data()

# --- 2. The Setup ---
fig = plt.figure(figsize=(16, 6))
plt.subplots_adjust(bottom=0.35) # Make room for controls

ax_input = fig.add_subplot(131)
ax_hidden = fig.add_subplot(132)
ax_output = fig.add_subplot(133)

# Initial Parameters
init_w1_angle = 0.0
init_w2_angle = np.pi / 2 # Orthogonal initially
init_b1 = 0.0
init_b2 = 0.0
init_mag = 1.0 # Scale of weights
act_type = 'ReLU' # Default

# --- 3. The Math (The Engine) ---
def forward_pass(w1_a, w2_a, b1, b2, mag, activation):
    # Create Weight Matrix from angles
    w1 = np.array([np.cos(w1_a), np.sin(w1_a)]) * mag
    w2 = np.array([np.cos(w2_a), np.sin(w2_a)]) * mag
    
    W = np.vstack((w1, w2))
    b = np.array([b1, b2])
    
    # 1. Linear Step
    z = X_orig.dot(W.T) + b
    
    # 2. Non-linear Step (Layer 1 Output)
    if activation == 'ReLU':
        h = np.maximum(0, z)
    else: # Tanh
        h = np.tanh(z)
        
    # 3. Aggregation (Layer 2 - Simple Sum to show folding)
    # We sum the two neurons to see if they create a "ridge"
    y_final = h[:, 0] + h[:, 1]
    
    return z, h, y_final, w1, w2

# --- 4. The Visualization ---
def update(val):
    # Get values
    ang1 = s_ang1.val
    ang2 = s_ang2.val
    b1 = s_b1.val
    b2 = s_b2.val
    mag = s_mag.val
    
    # Compute
    z, h, y_final, w1, w2 = forward_pass(ang1, ang2, b1, b2, mag, act_type)
    
    # -- Plot 1: Input Space --
    ax_input.clear()
    ax_input.scatter(X_orig[:,0], X_orig[:,1], c=C_orig, cmap='jet', s=5, alpha=0.5)
    # Draw Weight Vectors
    ax_input.arrow(0, 0, w1[0], w1[1], head_width=0.2, color='red', linewidth=2, label='N1 Direction')
    ax_input.arrow(0, 0, w2[0], w2[1], head_width=0.2, color='blue', linewidth=2, label='N2 Direction')
    ax_input.set_title("1. Input Disk (Color = X position)")
    ax_input.set_xlim(-3, 3); ax_input.set_ylim(-3, 3)
    ax_input.grid(True, linestyle=':')
    ax_input.legend()

    # -- Plot 2: Hidden Layer (The Manifold) --
    ax_hidden.clear()
    ax_hidden.scatter(h[:,0], h[:,1], c=C_orig, cmap='jet', s=5, alpha=0.5)
    ax_hidden.set_title(f"2. Hidden Layer (2 Neurons)\nActivation: {act_type}")
    
    # Set limits based on activation to see the "crust" or "cut"
    if act_type == 'ReLU':
        ax_hidden.set_xlim(-0.5, 4); ax_hidden.set_ylim(-0.5, 4)
        ax_hidden.text(0.1, 0.1, "Dead Zone (0,0)", color='red')
    else:
        ax_hidden.set_xlim(-1.1, 1.1); ax_hidden.set_ylim(-1.1, 1.1)
    ax_hidden.grid(True)
    ax_hidden.set_xlabel("Neuron 1 Activation")
    ax_hidden.set_ylabel("Neuron 2 Activation")

    # -- Plot 3: Final Output (The Fold) --
    ax_output.clear()
    # We plot Input X vs Final Output Y to see the "landscape"
    ax_output.scatter(X_orig[:,0], y_final, c=C_orig, cmap='jet', s=5, alpha=0.5)
    ax_output.set_title("3. Aggregation (Sum of Neurons)\nLook for V-shapes (Folds)")
    ax_output.set_xlabel("Input X position")
    ax_output.set_ylabel("Total Output Signal")
    ax_output.grid(True)
    
    fig.canvas.draw_idle()

# --- 5. Controls ---
ax_color = 'lightgoldenrodyellow'
# Sliders
ax_ang1 = plt.axes([0.1, 0.25, 0.3, 0.03], facecolor=ax_color)
ax_ang2 = plt.axes([0.1, 0.20, 0.3, 0.03], facecolor=ax_color)
ax_b1 = plt.axes([0.5, 0.25, 0.3, 0.03], facecolor=ax_color)
ax_b2 = plt.axes([0.5, 0.20, 0.3, 0.03], facecolor=ax_color)
ax_mag = plt.axes([0.1, 0.15, 0.7, 0.03], facecolor=ax_color)

s_ang1 = Slider(ax_ang1, 'Angle N1', 0, 2*np.pi, valinit=init_w1_angle)
s_ang2 = Slider(ax_ang2, 'Angle N2', 0, 2*np.pi, valinit=init_w2_angle)
s_b1 = Slider(ax_b1, 'Bias N1', -2.0, 2.0, valinit=init_b1)
s_b2 = Slider(ax_b2, 'Bias N2', -2.0, 2.0, valinit=init_b2)
s_mag = Slider(ax_mag, 'W Magnitude', 0.1, 5.0, valinit=init_mag)

# Radio Button for Activation
rax = plt.axes([0.85, 0.15, 0.12, 0.15], facecolor=ax_color)
radio = RadioButtons(rax, ('ReLU', 'Tanh'))

def change_act(label):
    global act_type
    act_type = label
    update(None)
radio.on_clicked(change_act)

# Connect sliders
s_ang1.on_changed(update)
s_ang2.on_changed(update)
s_b1.on_changed(update)
s_b2.on_changed(update)
s_mag.on_changed(update)

# Initial draw
update(None)
plt.show()