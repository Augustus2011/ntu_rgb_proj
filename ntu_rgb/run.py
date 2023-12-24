from gen_spec import GenSpec


path="/Users/kunkerdthaisong/ipu/SampleSkeleton/"
save_to="/Users/kunkerdthaisong/ipu/spec/"

program = GenSpec(path=path, save_to=save_to, drop_col=None) #example drop_col=["x","y","z"]
program.run_all(gen_spec=True)