# ntu_rgb_proj



```bash
#to gen spectogram you need to run
ntu_rgb/run.py
before run you need to change confix first


from gen_spec import GenSpec


path="/Users/kunkerdthaisong/ipu/SampleSkeleton/" <<<<<<<<< change this
save_to="/Users/kunkerdthaisong/ipu/spec/" <<<<<<<<< change this

program = GenSpec(path=path, save_to=save_to, drop_col=None) #example drop_col=["x","y","z"]
program.run_all(gen_spec=True)

```

## members
1.Worawit Tepsan
<a href='https://github.com/PitiwatL'> 2.Pitiwat Lueangwitchajaroen <a/>
3.Kun kerdthausing

## advisors
1.Sitapa Watcharapinchai
2.Sorn Sooksatra
