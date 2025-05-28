import numpy as np

class DominantColorLabelManager:
    def __init__(self):
        pass

    def generate_labels(self, images_array):
        # images_array shape = (num_images, 7500) flattened RGB 50x50x3
        # z powrotem reshaping do (num_images, 50, 50, 3)
        num_images = images_array.shape[0]
        reshaped = images_array.reshape(num_images, 50, 50, 3)

        labels = []
        for img in reshaped:
            avg_colors = img.mean(axis=(0, 1))  # średnia po pikselach dla R, G, B
            dominant_color_index = np.argmax(avg_colors)
            label = np.zeros(3)
            label[dominant_color_index] = 1
            labels.append(label)
        return np.array(labels)
