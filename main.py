from pathlib import Path
from NeuralNetwork import NeuralNetwork
from DominantColorLabelManager import DominantColorLabelManager
from ImageSet import imageSet
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = Path('input/')  # folder z obrazami
    
    # Wczytaj obrazy
    dataset = imageSet(p)
    images_array = dataset.asArray()  # (num_images, 7500)
    print(f"Liczba obrazów: {len(dataset.images)}")
    print(f"Kształt danych wejściowych: {images_array.shape}")

    # Generuj etykiety heurystycznie
    label_manager = DominantColorLabelManager()
    labels = label_manager.generate_labels(images_array)
    print(f"Kształt etykiet: {labels.shape}")

    # Stwórz i wytrenuj sieć
    nn = NeuralNetwork(learning_rate=0.01)
    predictions, losses = nn.train(images_array, labels, epochs=100, print_every=10)

    # Wykres strat
    plt.plot(losses)
    plt.title("Wykres strat (loss) podczas treningu")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Wypisz predykcje dla wszystkich obrazów
    for pred, label, name in zip(predictions, labels, dataset.image_names):
        pred_rounded = [f"{p:.3f}" if p > 0.001 else "0.000" for p in pred]
        label_int = [int(x) for x in label]
        print(f"Plik: {name} | Predykcja: {pred_rounded} | Etykieta: {label_int}")

if __name__ == "__main__":
    main()
