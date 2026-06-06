import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

# --- 1. The Millipede's Brain (Data Generation) ---
def generate_data(n_points=2000, radius=2.0):
    # Create a grid-like structure to see "Flow" better
    # Concentric circles
    r_space = np.linspace(0.1, radius, 10)
    t_space = np.linspace(0, 2*np.pi, 40)
    
    # Grid points
    x_list, y_list, c_list = [], [], []
    for r in r_space:
        for t in t_space:
            x_list.append(r * np.cos(t))
            y_list.append(r * np.sin(t))
            # Color by angle (Hue) to track rotation/twisting
            c_list.append(t)
            
    # Add random filler points for density
    r_rand = np.sqrt(np.random.rand(n_points)) * radius
    t_rand = np.random.rand(n_points) * 2 * np.pi
    x_rand = r_rand * np.cos(t_rand)
    y_rand = r_rand * np.sin(t_rand)
    
    X_grid = np.column_stack((x_list, y_list))
    C_grid = np.array(c_list)
    
    X_rand = np.column_stack((x_rand, y_rand))
    C_rand = t_rand # Color by angle
    
    return X_grid, C_grid, X_rand, C_rand

# Initial Data
X_g, C_g, X_r, C_r = generate_data()

# --- 2. The Setup ---
fig = plt.figure(figsize=(16, 6))
plt.subplots_adjust(bottom=0.35) 

ax_input = fig.add_subplot(131)
ax_delta = fig.add_subplot(132)
ax_output = fig.add_subplot(133)

# Initial Parameters
init_w1_angle = 0.0
init_w2_angle = np.pi / 2 
init_b1 = 0.0
init_b2 = 0.0
init_mag = 1.0 
act_type = 'ReLU'
is_resnet = False # Start in Standard Mode

# --- 3. The Math (The Engine) ---
def forward_pass(w1_a, w2_a, b1, b2, mag, activation, resnet_mode):
    # Weight Matrix (2 neurons, 2 inputs)
    # This maps R^2 -> R^2 directly for visualization simplicity
    w1 = np.array([np.cos(w1_a), np.sin(w1_a)]) * mag
    w2 = np.array([np.cos(w2_a), np.sin(w2_a)]) * mag
    W = np.vstack((w1, w2))
    b = np.array([b1, b2])
    
    # Combine grid and random points
    X_all = np.vstack((X_g, X_r))
    
    # 1. The Transformation F(x)
    # Projection
    z = X_all.dot(W.T) + b
    
    # Activation (The "Delta" or "Texture")
    if activation == 'ReLU':
        delta = np.maximum(0, z)
    else: # Tanh
        delta = np.tanh(z)
        
    # 2. The Final Output
    if resnet_mode:
        # y = x + F(x)
        # We treat the two neurons as defining a vector field (dx, dy)
        # Note: In real nets, there's usually a projection back to dimensions
        # Here we assume Neuron 1 drives X-shift, Neuron 2 drives Y-shift
        y_final = X_all + delta 
    else:
        # y = F(x)
        y_final = delta
    
    return X_all, delta, y_final, w1, w2

# --- 4. The Visualization ---
def update(val):
    # Get values
    ang1 = s_ang1.val
    ang2 = s_ang2.val
    b1 = s_b1.val
    b2 = s_b2.val
    mag = s_mag.val
    
    # Compute
    X_in, delta, y_final, w1, w2 = forward_pass(ang1, ang2, b1, b2, mag, act_type, is_resnet)
    
    # Split back to grid/random for plotting order
    n_grid = len(X_g)
    
    # -- Plot 1: Input Space --
    ax_input.clear()
    # Plot faint random points
    ax_input.scatter(X_in[n_grid:,0], X_in[n_grid:,1], c='gray', s=1, alpha=0.1)
    # Plot grid points
    ax_input.scatter(X_in[:n_grid,0], X_in[:n_grid,1], c=C_g, cmap='hsv', s=15, alpha=0.8)
    
    ax_input.arrow(0, 0, w1[0], w1[1], head_width=0.1, color='black', linewidth=1.5)
    ax_input.arrow(0, 0, w2[0], w2[1], head_width=0.1, color='black', linewidth=1.5)
    ax_input.set_title("1. Input Manifold\n(Color = Angle)")
    ax_input.set_xlim(-4, 4); ax_input.set_ylim(-4, 4)
    ax_input.grid(True, linestyle=':')

    # -- Plot 2: The Transform F(x) --
    ax_delta.clear()
    ax_delta.scatter(delta[n_grid:,0], delta[n_grid:,1], c='gray', s=1, alpha=0.1)
    ax_delta.scatter(delta[:n_grid,0], delta[:n_grid,1], c=C_g, cmap='hsv', s=15, alpha=0.8)
    
    title_mode = "The Perturbation F(x)" if is_resnet else "Standard Output F(x)"
    ax_delta.set_title(f"2. {title_mode}\n(Hidden Layer Activity)")
    ax_delta.grid(True)
    
    # Dynamic limits
    if act_type == 'ReLU':
        limit = max(4, mag*3 + abs(b1))
        ax_delta.set_xlim(-0.5, limit); ax_delta.set_ylim(-0.5, limit)
    else:
        ax_delta.set_xlim(-1.1, 1.1); ax_delta.set_ylim(-1.1, 1.1)

    # -- Plot 3: The Result --
    ax_output.clear()
    ax_output.scatter(y_final[n_grid:,0], y_final[n_grid:,1], c='gray', s=1, alpha=0.1)
    ax_output.scatter(y_final[:n_grid,0], y_final[:n_grid,1], c=C_g, cmap='hsv', s=15, alpha=0.8)
    
    final_title = "y = x + F(x)\n(Flow / Distortion)" if is_resnet else "y = F(x)\n(Projection / Collapse)"
    ax_output.set_title(f"3. Final Manifold\n{final_title}")
    ax_output.set_xlim(-4, 4); ax_output.set_ylim(-4, 4)
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
s_mag = Slider(ax_mag, 'W Magnitude', 0.1, 3.0, valinit=init_mag)

# Radio Button for Activation
rax = plt.axes([0.85, 0.20, 0.12, 0.10], facecolor=ax_color)
radio = RadioButtons(rax, ('ReLU', 'Tanh'))

# Checkbox for ResNet
cax = plt.axes([0.85, 0.10, 0.12, 0.08], facecolor=ax_color)
check = CheckButtons(cax, ['ResNet Mode'], [False])

def change_act(label):
    global act_type
    act_type = label
    update(None)

def toggle_resnet(label):
    global is_resnet
    is_resnet = not is_resnet
    update(None)

radio.on_clicked(change_act)
check.on_clicked(toggle_resnet)

# Connect sliders
s_ang1.on_changed(update)
s_ang2.on_changed(update)
s_b1.on_changed(update)
s_b2.on_changed(update)
s_mag.on_changed(update)

# Initial draw
update(None)
plt.show()