
#visualize 2d 25joint and save .gif
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import polars as pl
import os

df = pl.read_parquet("/Users/kunkerdthaisong/ipu/ntu_rgb_proj/ntu_rgb/two_person_30actions_10class.parquet")
action = df.filter(pl.col("file_path") == df["file_path"].unique(maintain_order=True)[2])

if "skel_body" in action.columns:
    action = action.filter(pl.col("skel_body") ==1)
max_f = action[len(action) - 1]["frame"].item()

output_dir="/Users/kunkerdthaisong/ipu/ntu_rgb_proj/ex_animation"
#file_path=df["file_path"].unique(maintain_order=True)[10]

title_action="2p_a53_pushing"

fig, ax = plt.subplots()
fig.suptitle(title_action)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)


skeleton_connections = [
    (4, 3), (3, 21), (21, 2), (2, 1),
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (12, 24),
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 22), (8, 23),
    (1, 13), (13, 14), (14, 15), (15, 16),
    (1, 17), (17, 18), (18, 19), (19, 20),
]

def update(frame):
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1)

    positions = action.filter(pl.col("frame") == frame)[["x", "y"]]
    ax.scatter(positions['x'], positions['y'], c='r', marker='o')


title_action=title_action.replace("/",".")
animation = FuncAnimation(fig, update, frames=max_f, interval=100)
video_filename =os.path.join(output_dir,f"{title_action}.gif")
#animation.save(filename=video_filename, dpi=300) #save to .gif
plt.show()
