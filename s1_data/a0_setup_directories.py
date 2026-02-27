import os

base_folder = "data"
os.makedirs(base_folder, exist_ok=True)

model_data_dir = "model_data"
os.makedirs(os.path.join(base_folder, model_data_dir), exist_ok=True)