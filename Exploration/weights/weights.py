import os
import torch

def load_weights(folder_path):
    pt_files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]

    models = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for pt in pt_files:
        path = os.path.join(folder_path, pt)

        obj = torch.load(path, map_location=device)

        models[pt] = obj
        print(f"Loaded raw object from: {pt}")

    return models


if __name__ == "__main__":
    folder = os.path.join(os.getcwd(), "Exploration/weights")
    print("Loading from:", folder)
    all_models = load_weights(folder)