import os
import yaml
from ntu_rgb.gen_spec import GenSpec

CONFIG_PATH = "/Users/kunkerdthaisong/ipu/ntu_rgb_proj/onlygodknowsperfectfeatures/config.yaml"


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config("config.yaml")
drop_col=config["drop_col"]  #defualt=None
gen_type=config["gen_type"] #3