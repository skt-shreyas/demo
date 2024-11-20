import os
import requests

# Set the model path and the direct download link
MODEL_PATH = "/var/model_storage/model.pth"
MODEL_URL = "https://drive.google.com/uc?id=1BI5t5NK1UB2im4AayJg_yGCMpG2JucIB&export=download"

def download_model():
    if not os.path.exists(MODEL_PATH):  # Check if the model already exists
        print("Model not found. Downloading...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded and saved to {MODEL_PATH}")
    else:
        print(f"Model already exists at {MODEL_PATH}. Skipping download.")

if __name__ == "__main__":
    download_model()
