import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out = cfg["output_dir"]
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)
    cfg["output_dir"] = out
    for sub in ["data", "models", "sim", "figures"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    return cfg


def out_path(cfg, *parts):
    return os.path.join(cfg["output_dir"], *parts)


def root_path(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
