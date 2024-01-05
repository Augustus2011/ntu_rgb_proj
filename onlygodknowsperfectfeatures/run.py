import os
import yaml
from ntu_rgb.gen_spec import GenSpec
from onlygodknowsperfectfeatures.randomgenoperators import randomgen_feature
import polars as pl


CONFIG_PATH = "/Users/kunkerdthaisong/ipu/ntu_rgb_proj/onlygodknowsperfectfeatures/config.yaml"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

path=config["path_npy_skeleton"] #"/Users/kunkerdthaisong/ipu/ntu_rgb_proj/SampleSkeleton/"
drop_col=config["drop_col"]  #defualt=None
gen_type=config["gen_type"] #0
feature_imp=config["feature_imp"] #True or False
save_parquet_to=config["save_parquet_to"]
save_img_to=config["save_img_to"]


randomgen_feature(df=,list_operator=["+","-","*","/"],save_parquet_to=save_parquet_to, save_img_to=save_img_to)
program = GenSpec(path=path,drop_col=drop_col, gen_type=gen_type, feature_imp=feature_imp)
program.run_all()
#shutil.make_archive(path_zip,format="zip")