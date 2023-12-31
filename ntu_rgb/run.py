from gen_spec import GenSpec
import os
import yaml
#import shutil

CONFIG_PATH = "/Users/kunkerdthaisong/ipu/ntu_rgb_proj/ntu_rgb/"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config("genconfig.yaml")


path=config["path_npy_skeleton"] #"/Users/kunkerdthaisong/ipu/ntu_rgb_proj/SampleSkeleton/"
save_to=config["save_to"] #"/Users/kunkerdthaisong/ipu/ntu_rgb_proj/spec/"
drop_col=config["drop_col"]  #defualt=None
gen_type=config["gen_type"] #0
path_parquet=config["path_parquet"] #path of .parquet
feature_imp=config["feature_imp"] #True or False

program = GenSpec(path=path, save_to=save_to, drop_col=drop_col, gen_type=gen_type, path_parquet=path_parquet, feature_imp=feature_imp)
program.run_all()
#shutil.make_archive(path_zip,format="zip")