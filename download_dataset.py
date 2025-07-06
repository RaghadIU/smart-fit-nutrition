import os
import zipfile
import urllib.request

def download_and_extract(url, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    zip_path = os.path.join(dest_path, "food_dataset.zip")

    print("🔽 Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_path)

    os.remove(zip_path)
    print("✅ Done! Dataset extracted to:", dest_path)

if __name__ == "__main__":
    url = "https://github.com/mohamedgamaleldin/food-vision-mini/releases/download/v1.0/10_food_classes_10_percent.zip"
    dest = "./data"
    download_and_extract(url, dest)
