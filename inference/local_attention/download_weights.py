import os
import gdown

weights_folder = os.path.join(os.getcwd(), "Exploration/weights")
os.makedirs(weights_folder, exist_ok=True)


file_urls = [
    "https://drive.google.com/file/d/1YopVInqfhNyRph2fWYt63XCwQGOrhC6s/view",
    "https://drive.google.com/file/d/1oYtnFqY5aetkEZnU3VBKV0TrPxpSvtay/view?usp=sharing",
    "https://drive.google.com/file/d/1U5CHXKyK8_rmvmWPGrdTgzOO-64aeh3e/view?usp=sharing",
    "https://drive.google.com/file/d/1VDG22YKJJEfcqCg0bi9u3uVYqC2DAxGB/view?usp=sharing"
]
file_names = [
    "pca.resnet.pt",
    "cnn.pt",
    "random_resnet.pt",
    "resnet_18.pt"
]

def download_weights():
    for url, name in zip(file_urls, file_names):
        file_id = url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = os.path.join(weights_folder, name)

        # Download
        print(f"Downloading {name}...")
        gdown.download(download_url, output_path, quiet=False)
    print("done downloaded to 'Exploration/weights")
if __name__ == "__main__":
    download_weights()