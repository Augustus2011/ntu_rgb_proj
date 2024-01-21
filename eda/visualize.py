import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import polars as pl

df = pl.read_parquet("/Users/kunkerdthaisong/ipu/ntu_rgb_proj/ntu_rgb/30actions_10class.parquet")
action = df.filter(pl.col("file_path") == df["file_path"].unique(maintain_order=True)[4])
max_f = action[len(action) - 1]["frame"].item()

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Define the joint connections (assuming each row has columns "x" and "y")
skeleton_connections = [
    (4, 3),(3, 21),(21, 2),(2, 1),
    (21, 9),(9,10),(10,11),(11,12),(12,25),(12,24),
    (21, 5),(5,6),(6,7),(7,8),(8,22),(8,23),
    (1, 13),(13,14),(14,15),(15,16),
    (1, 17),(17,18),(18,19),(19,20),
]

# Function to update the plot in each frame
def update(frame):
    ax.clear()  # Clear the previous frame
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    for i in range(1,max_f+1):
        positions = action.filter(pl.col("frame") == i)[["x", "y"]]
        ax.scatter(positions['x'], positions['y'], c='r', marker='o')
    
    # Plot skeleton connections
        for connection in skeleton_connections:
            joint1 = connection[0]
            joint2 = connection[1]
            x_values = [positions['x'][joint1], positions['x'][joint2]]
            y_values = [positions['y'][joint1], positions['y'][joint2]]
            ax.plot(x_values, y_values, c='b')

# Function to generate 2D joint positions for each frame
def get_joint_positions(frame):
    for i in range(1,max_f+1):
        positions = action.filter(pl.col("frame") == max_f)[["x", "y"]]
    return positions

# Create the 2D animation
animation = FuncAnimation(fig, update, frames=max_f, interval=1000)

plt.show()
