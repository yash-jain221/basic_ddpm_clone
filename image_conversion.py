import os
from PIL import Image

# ----------- CONFIG ------------
input_dir = "C:\\Users\\Yash\\Desktop\\textures_raw"             # Folder with raw generated images
output_dir = "C:\\Users\\Yash\\Desktop\\textures_unity_ready"     # Folder to save Unity-compatible textures
target_size = (128, 128)                 # Resize to this for Unity

os.makedirs(output_dir, exist_ok=True)

# ----------- PROCESSING ----------
image_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
print(image_files)

for i, file_name in enumerate(image_files):
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, f"texture_{i}.png")

    try:
        img = Image.open(input_path).convert("RGB")  # ensure it's RGB
        img = img.resize(target_size, resample=Image.BICUBIC)
        img.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")
