import yaml
import os

def load_config(path="boya/config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    # İstersen path’leri absolute yap
    for key in config.get("paths", {}):
        config["paths"][key] = os.path.expanduser(config["paths"][key])
    return config
