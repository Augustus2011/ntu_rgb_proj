# ntu_rgb_proj
## my dataframe.parquet https://drive.google.com/file/d/1C0z6zbYpT_daiI1kLGHYe_VN3GyJKPZk/view?usp=share_link

## my specto_gram.png https://drive.google.com/file/d/1lh-A5VL0d0re_DweEHCcwO5ENhcV0jXP/view?usp=share_link

#to gen spectogram you need to run

pip install -r requirements.txt

```bash

change configs in ntu_rgb/genconfig.yaml

#initial settings

path_npy_skeleton: /Users/kunkerdthaisong/ipu/ntu_rgb_proj/SampleSkeleton/ #ex:/Users/kunkerdthaisong/ipu/ntu_rgb_proj/SampleSkeleton/    
save_to: /Users/kunkerdthaisong/ipu/ntu_rgb_proj/spec/
drop_col: None #example drop_col=["x","y","z"]  #all feature x,y,z,frame,joint,zone,dis_from_00,dis_from_hop1,angle_from_hop1



gen_type: 0 #0:genboth, 1:gentable, 2:genspec, 3:gen_spec_from_exist_table

```

```bash
after that run python file run.py
```


## members
1.Worawit Tepsan
<a href='https://github.com/PitiwatL'> 2.Pitiwat Lueangwitchajaroen <a/>
3.Kun kerdthausing

## advisors
1.Sitapa Watcharapinchai
2.Sorn Sooksatra
