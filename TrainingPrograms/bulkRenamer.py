import os
from PIL import Image

storage_dir = 'traffic_cone2'
prefix = 'cone_'  # set this to whatever string you want prepended

for filename in os.listdir(storage_dir):
    filepath = os.path.join(storage_dir, filename)
    new_filepath = os.path.join(storage_dir, f"{prefix}{filename}")
    try:
        with Image.open(filepath) as img:
            img.save(new_filepath)
    except Exception as e:
        print(f"Skipped {filename}: {e}")
