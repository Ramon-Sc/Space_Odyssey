import os
import pandas as pd
import matplotlib.pyplot as plt

def load_history(path, label):
    df = pd.read_csv(path)
    # Expect columns: epoch, train_loss, train_acc, val_loss, val_acc, epoch_time_sec
    return df["epoch"], df["val_acc"], label

def main():
    output_dir = "/home/ramon/Space_Odyssey/src/Space_Odyssey/experiment_outputs/lung"
    dataset_name = output_dir.split("/")[-1]
    runs = [
        ("full.csv", "Full"),
        ("craig.csv", "CRAIG"),
        ("herding.csv", "Herding"),
    ]

    plt.figure(figsize=(8, 5))
    for fname, label in runs:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        epoch, val_acc, label = load_history(path, label)
        plt.plot(epoch, val_acc, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy by Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_accuracy.png"))

if __name__ == "__main__":
    main()