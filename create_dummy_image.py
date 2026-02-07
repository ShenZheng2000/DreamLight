import numpy as np
from PIL import Image
import os

def make_dummy_bg(path, size=1024):
    bg = np.full((size, size, 3), 127, dtype=np.uint8)
    Image.fromarray(bg).save(path)

def make_dummy_env(path, h=256, w=512):
    # lat-long env map
    env = np.full((h, w, 3), 127, dtype=np.uint8)
    Image.fromarray(env).save(path)

if __name__ == "__main__":
    os.makedirs("dummy_inputs", exist_ok=True)

    make_dummy_bg("dummy_inputs/dummy_bg.png")
    make_dummy_env("dummy_inputs/dummy_env.png")

    print("✔ Dummy BG and ENV images created")