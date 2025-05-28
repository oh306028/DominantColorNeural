from PIL import Image
import numpy as np
from pathlib import Path

class imageSet:
    def __init__(self, path_to_dataset):
        self.images = []
        self.image_names = []
        try:
            for img_path in Path(path_to_dataset).iterdir():
                if img_path.is_file():
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((50, 50))
                    self.images.append(img)
                    self.image_names.append(img_path.name)
                else:
                    print(f"Didnt load {img_path}")
        except IOError as e:
            print(f"Error loading images: {e}")

    def asArray(self, flatten=True):
        result = []
        for img in self.images:
            arr = np.array(img) / 255.0
            if flatten:
                arr = arr.flatten()
            result.append(arr)
        return np.array(result)
