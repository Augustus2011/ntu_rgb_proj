import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import polars as pl

df = pl.read_parquet("/Users/kunkerdthaisong/ipu/ntu_rgb_proj/ntu_rgb/30actions_10class.parquet")
action = df.filter(pl.col("file_path") == df["file_path"].unique(maintain_order=True)[6])
max_f = action[len(action) - 1]["frame"].item()


fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)


skeleton_connections = [
    (4, 3), (3, 21), (21, 2), (2, 1),
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (12, 24),
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 22), (8, 23),
    (1, 13), (13, 14), (14, 15), (15, 16),
    (1, 17), (17, 18), (18, 19), (19, 20),
]

def update(frame):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    positions = action.filter(pl.col("frame") == frame)[["x", "y"]]
    ax.scatter(positions['x'], positions['y'], c='r', marker='o')


#for connection in skeleton_connections:
    #joint1 = connection[0]
   # joint2 = connection[1]
  #  x_values = []
 #   y_values = []
#
    #for i in range(1, max_f + 1):
   #     positions = action.filter(pl.col("frame") == i)[["x", "y"]]
  #      x_values.extend([positions['x'][joint1 - 1], positions['x'][joint2 - 1], None])
 #       y_values.extend([positions['y'][joint1 - 1], positions['y'][joint2 - 1], None])
#
   # ax.plot(x_values, y_values, c='b')


animation = FuncAnimation(fig, update, frames=max_f, interval=100)
video_filename ="a7_throw.gif"
animation.save(video_filename, dpi=300)
plt.show()
