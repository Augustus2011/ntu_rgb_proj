import yaml
import os

def load_config(CONFIG_PATH:str,config_name:str):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config
