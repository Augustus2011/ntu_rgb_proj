from gen_spec import GenSpec
import os
import glob

path="/Users/kunkerdthaisong/ipu/ntu_rgb_proj/SampleSkeleton/"
save_to="/Users/kunkerdthaisong/ipu/ntu_rgb_proj/spec/"

program = GenSpec(path=path, save_to=save_to, drop_col=None) #example drop_col=["x","y","z"]
program.run_all(gen_spec=True)
