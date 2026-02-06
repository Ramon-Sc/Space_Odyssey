import os
import sys
import time
import csv

import numpy as np
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision.datasets.folder import default_loader
from cs_shapley import TorchLogisticRegression

from data_preprocesssor import IndexedImageFolder, get_config
from craig import craig
from herding import herding
from cs_shapley import cs_shapley

from tqdm import tqdm



#build a linear classifier on top of the embeddings for craig - this is the model that is used for the coreset selection
def build_embedding_classifier(embedding_dim, num_classes):
    #craig implementation takes the last module of the model for the alignment with the residual gradient --> hence wrap in nn.sequential
    return nn.Sequential(nn.Linear(embedding_dim, num_classes))




def make_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)


def make_loader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    criterion = nn.CrossEntropyLoss()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                _, inputs, labels = batch
            else:
                inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            with amp_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += inputs.size(0)

    mean_loss = total_loss / max(1, total_count)
    mean_acc = total_correct / max(1, total_count)
    return mean_loss, mean_acc


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs,
    batch_size,
    device,
    sample_weights=None,
    lr=1e-3,
):
    """
    Args:
        sample_weights: Optional tensor of shape (len(full_train_dataset),).
                        Requires train dataset to return (index, input, label).
    """
    model.to(device)
    model.train()
    optimizer = make_optimizer(model, lr=lr)
    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction="none")

    if sample_weights is not None:
        sample_weights = sample_weights.to(device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    history = []
    with tqdm(total=epochs, desc="Training") as pbar:
        for epoch in range(epochs):
            start = time.perf_counter()
            model.train()

            running_loss = 0.0
            running_correct = 0
            running_count = 0

            for batch in train_loader:
                if len(batch) == 3:
                    indices, inputs, labels = batch
                else:
                    indices = None
                    inputs, labels = batch

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with amp_ctx:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if sample_weights is not None:
                        if indices is None:
                            raise ValueError(
                                "Sample weights require dataset to return indices."
                            )
                        weights = sample_weights[indices.to(device)]
                        loss = (loss * weights).mean()
                    else:
                        loss = loss.mean()

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_count += inputs.size(0)

            train_loss = running_loss / max(1, running_count)
            train_acc = running_correct / max(1, running_count)
            val_loss, val_acc = evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - start

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_sec": elapsed,
                }
            )

        return history


def save_history(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"]
        )
        for row in history:
            writer.writerow(
                [
                    row["epoch"],
                    row["train_loss"],
                    row["train_acc"],
                    row["val_loss"],
                    row["val_acc"],
                    row["epoch_time_sec"],
                ]
            )


def run_herding_subset(train_dataset, n_samples, feature_model, batch_size):
    herding_selector = herding(
        model=feature_model,
        train_dataset=train_dataset,
        n_samples=n_samples,
        batch_size=batch_size,
    )
    selected_indices = herding_selector.run_herding()
    return selected_indices


def run_cs_shapley_subset(
    train_dataset,
    val_dataset,
    n_samples,
    feature_model,
    batch_size,
    num_classes,
    num_permutations,
    resample,
    clf_max_iter,
    clf_batch_size=None,
):
    clf = TorchLogisticRegression(
        max_iter=clf_max_iter, lr=1e-1, batch_size=clf_batch_size
    )
    selector = cs_shapley(
        model=feature_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        clf=clf,
        n_samples=n_samples,
        num_classes=num_classes,
        num_permutations=num_permutations,
        resample=resample,
        batch_size=batch_size,
    )
    selected_indices, shapley_values = selector.run_cs_shapley()
    return selected_indices, shapley_values


def run_craig_subset(
    train_dataset,
    val_dataset,
    n_samples,
    num_epochs_warmup,
    batch_size,
    num_classes,
    embedding_dim,
):
    model = build_embedding_classifier(embedding_dim, num_classes)
    optimizer = make_optimizer(model)
    craig_selector = craig(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        num_epochs_warmup=num_epochs_warmup,
        num_epochs=1,
        n_samples=n_samples,
        n_reselections=1,
        batch_size=batch_size,
    )

    if num_epochs_warmup > 0:
        craig_selector.train_model_warmup()

    selected_indices, optimized_weights = craig_selector.select_samples_craig()

    full_sample_weights = torch.zeros(
        len(train_dataset), dtype=optimized_weights.dtype
    )
    full_sample_weights[
        torch.tensor([int(i) for i in selected_indices], dtype=torch.long)
    ] = optimized_weights.cpu()

    return selected_indices, full_sample_weights


class GroupImageFolder(torch.utils.data.Dataset):
    """
    ImageFolder-style dataset that assigns class by top-level group folder,
    ignoring any deeper subdirectories.
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.targets = []

        group_names = [
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        ]
        group_names = sorted(group_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(group_names)}
        self.classes = group_names

        for group in group_names:
            group_root = os.path.join(root, group)
            for dirpath, _, filenames in os.walk(group_root):
                for fname in filenames:
                    if fname.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".svs")
                    ):
                        path = os.path.join(dirpath, fname)
                        self.samples.append(path)
                        self.targets.append(self.class_to_idx[group])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.targets[index]
        img = default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return index, img, target


def build_datasets(dataset_name, batch_size, seed=42):
    """
    Supported datasets:
      - "lung": lung_image_sets
      - "colon": colon_image_sets
      - "bracs": BRACS WSI with Group_* top-level classes
    """
    if dataset_name == "lung":
        train_dir = "/mnt/raid0/data/ramon/LUNG_COLON_DATA/lung_image_sets/train"
        val_dir = "/mnt/raid0/data/ramon/LUNG_COLON_DATA/lung_image_sets/test"
        train_dataset = IndexedImageFolder(root=train_dir, transform=None)
        val_dataset = IndexedImageFolder(root=val_dir, transform=None)
        return train_dataset, val_dataset

    if dataset_name == "colon":
        train_dir = "/mnt/raid0/data/ramon/LUNG_COLON_DATA/colon_image_sets/train"
        # testset is called 'val here ..
        val_dir = "/mnt/raid0/data/ramon/LUNG_COLON_DATA/colon_image_sets/val"
        train_dataset = IndexedImageFolder(root=train_dir, transform=None)
        val_dataset = IndexedImageFolder(root=val_dir, transform=None)
        return train_dataset, val_dataset

    if dataset_name == "bracs":
        train_dir = "/mnt/raid0/data/ramon/BRACS/histoimage.na.icar.cnr.it/BRACS_WSI/train"
        val_dir = "/mnt/raid0/data/ramon/BRACS/histoimage.na.icar.cnr.it/BRACS_WSI/val"
        train_dataset = GroupImageFolder(root=train_dir, transform=None)
        val_dataset = GroupImageFolder(root=val_dir, transform=None)
        return train_dataset, val_dataset

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def get_num_classes(dataset):
    if hasattr(dataset, "classes"):
        return len(dataset.classes)
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "classes"):
        return len(dataset.dataset.classes)
    raise ValueError("Unable to infer num_classes from dataset.")


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, index):
        return index, self.embeddings[index], self.labels[index]


def gigapath_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def ensure_gigapath_repo_on_path():
    repo_path = "/home/ramon/Space_Odyssey/prov-gigapath/prov-gigapath"
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def load_gigapath_tile_encoder_only(device):
    """
    Load only the tile encoder using local weights.
    This avoids importing the slide encoder and its dependencies.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required to load the Prov-GigaPath tile encoder."
        ) from exc

    tile_path = "/home/ramon/Space_Odyssey/prov-gigapath/pytorch_model.bin"
    model = timm.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        pretrained=False,
        checkpoint_path=tile_path,
    )
    model.to(device).eval()
    return model


def load_gigapath_encoders(device):
    """
    Load tile + slide encoders from the local prov-gigapath repo using local weights.
    """
    ensure_gigapath_repo_on_path()
    try:
        from gigapath.pipeline import load_tile_slide_encoder
    except ImportError as exc:
        raise ImportError(
            "Could not import gigapath from the local repo. "
            "Check /home/ramon/Space_Odyssey/prov-gigapath/prov-gigapath."
        ) from exc

    tile_path = "/home/ramon/Space_Odyssey/prov-gigapath/pytorch_model.bin"
    slide_path = "/home/ramon/Space_Odyssey/prov-gigapath/slide_encoder.pth"

    tile_encoder, slide_encoder = load_tile_slide_encoder(
        local_tile_encoder_path=tile_path,
        local_slide_encoder_path=slide_path,
        global_pool=True,
    )

    tile_encoder.to(device).eval()
    slide_encoder.to(device).eval()
    return tile_encoder, slide_encoder


def precompute_image_embeddings(dataset, encoder, device, batch_size, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["labels"]

    transform = gigapath_transform()
    def collate_pil(batch):
        indices, images, labels = zip(*batch)
        return list(indices), list(images), torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_pil,
    )

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        from tqdm import tqdm
        for _, inputs, labels in tqdm(loader, desc="Computing embeddings"):
            inputs = torch.stack([transform(img) for img in inputs])
            inputs = inputs.to(device)
            feats = encoder(inputs).detach().cpu().numpy()
            embeddings_list.append(feats)
            labels_list.append(labels.numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, labels=labels)

    return embeddings, labels


def iter_bracs_slides(dataset):
    for idx in range(len(dataset)):
        path = dataset.samples[idx]
        label = dataset.targets[idx]
        yield path, label


def extract_wsi_tiles(slide, tile_size=256, stride=256, max_tiles=256):
    width, height = slide.dimensions
    tiles = []
    coords = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            if len(tiles) >= max_tiles:
                return tiles, coords
            tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
            tiles.append(tile)
            coords.append([x, y])

    return tiles, coords


def precompute_bracs_slide_embeddings(
    dataset,
    tile_encoder,
    slide_encoder,
    device,
    batch_size,
    cache_path=None,
):  
    # cache the embeddings 
    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["labels"]

    try:
        import openslide
    except ImportError as exc:
        raise ImportError(
            "openslide is required to read .svs files for BRACS."
        ) from exc

    transform = gigapath_transform()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
   
        for slide_path, label in tqdm(iter_bracs_slides(dataset), desc="Computing slide embeddings"):
            slide = openslide.OpenSlide(slide_path)
            tiles, coords = extract_wsi_tiles(slide)
            if len(tiles) == 0:
                continue
            tile_tensors = torch.stack([transform(t) for t in tiles]).to(device)
            coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)

            tile_embeddings = []
            for i in range(0, tile_tensors.size(0), batch_size):
                batch = tile_tensors[i : i + batch_size]
                feats = tile_encoder(batch)
                tile_embeddings.append(feats)
            tile_embeddings = torch.cat(tile_embeddings, dim=0)

            slide_embedding = slide_encoder(tile_embeddings, coords_tensor)
            slide_embedding = slide_embedding.squeeze().detach().cpu().numpy()

            embeddings_list.append(slide_embedding)
            labels_list.append(label)

    embeddings = np.stack(embeddings_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, labels=labels)

    return embeddings, labels





#################################################################################################################################
#MAIN FUNCTION
#################################################################################################################################


def main():
    config = get_config(config_path="./config.yaml")


    ############################################################################################################################
    #build the datasets
    ############################################################################################################################
    # 1. Choose dataset: "lung", "colon", or "bracs"
    dataset_name = config["dataset_name"]
    train_dataset, val_dataset = build_datasets(
        dataset_name=dataset_name,
        batch_size=config.get("batch_size", config.get("batch_size_craig", 128)),
    )

    num_classes = get_num_classes(train_dataset)
    n_samples = int(config["n_samples"])
    num_epochs = int(config["num_epochs"])
    num_epochs_warmup = int(config["num_epochs_warmup"])
    batch_size_full = int(
        config.get("batch_size_full", config.get("batch_size", config["batch_size_craig"]))
    )
    batch_size_provgiga = int(config["batch_size_provgiga"])
    batch_size_craig = int(config["batch_size_craig"])
    batch_size_cs_shapley = int(config["batch_size_cs_shapley"])
    batch_size_cs_shapley_lr = config.get("batch_size_cs_shapley_lr", None)
    if batch_size_cs_shapley_lr is not None:
        batch_size_cs_shapley_lr = int(batch_size_cs_shapley_lr)
    num_permutations = int(config.get("num_permutations", 50))
    resample = int(config.get("resample", 1))
    shapley_lr_max_iter = int(config.get("shapley_lr_max_iter", 200))
    batch_size_herding = int(config["batch_size_herding"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join("./experiment_outputs", dataset_name)
    cache_dir = os.path.join(output_dir, "gigapath_cache")

    ############################################################################################################################
    # Precompute embeddings using Prov-GigaPath
    ############################################################################################################################
    if dataset_name in ("lung", "colon"):
        tile_encoder = load_gigapath_tile_encoder_only(device)
        slide_encoder = None
    else:
        tile_encoder, slide_encoder = load_gigapath_encoders(device)

    #MHIST DATASET 
    if dataset_name in ("lung", "colon"):
        train_embeddings, train_labels = precompute_image_embeddings(
            train_dataset,
            tile_encoder,
            device,
            batch_size=batch_size_provgiga,
            cache_path=os.path.join(cache_dir, "train_tile_embeddings.npz"),
        )
        val_embeddings, val_labels = precompute_image_embeddings(
            val_dataset,
            tile_encoder,
            device,
            batch_size=batch_size_provgiga,
            cache_path=os.path.join(cache_dir, "val_tile_embeddings.npz"),
        )

    #BRACS DATASET WSI Images
    else:
        train_embeddings, train_labels = precompute_bracs_slide_embeddings(
            train_dataset,
            tile_encoder,
            slide_encoder,
            device,
            batch_size=batch_size_provgiga,
            cache_path=os.path.join(cache_dir, "train_slide_embeddings.npz"),
        )
        val_embeddings, val_labels = precompute_bracs_slide_embeddings(
            val_dataset,
            tile_encoder,
            slide_encoder,
            device,
            batch_size=batch_size_provgiga,
            cache_path=os.path.join(cache_dir, "val_slide_embeddings.npz"),
        )

    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset = EmbeddingDataset(val_embeddings, val_labels)

    ###########################################################################################################################
    # FULL DATA PART
    ###########################################################################################################################
    if not os.path.exists(os.path.join(output_dir, "fullmodel.pt")):
        model_full = build_embedding_classifier(train_embeddings.shape[1], num_classes)
        history_full = train_model(
            model=model_full,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=num_epochs,
            batch_size=batch_size_full,
            device=device,
        )
        save_history(os.path.join(output_dir, "full.csv"), history_full)
        torch.save(model_full.state_dict(), os.path.join(output_dir, "fullmodel.pt"))
    else:
        print("Loading full model from checkpoint")
        model_full = build_embedding_classifier(train_embeddings.shape[1], num_classes)
        model_full.load_state_dict(torch.load(os.path.join(output_dir, "fullmodel.pt")))
    # Full data baseline


    ###########################################################################################################################
    # CRAIG PART
    ###########################################################################################################################
    
    craig_csv_path = os.path.join(output_dir, "craig.csv")
    if os.path.exists(craig_csv_path):
        print(f"Skipping CRAIG: found existing {craig_csv_path}")
    else:
        # CRAIG subset (coreset)
        craig_indices, craig_weights = run_craig_subset(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_samples=n_samples,
            num_epochs_warmup=num_epochs_warmup,
            batch_size=batch_size_craig,
            num_classes=num_classes,
            embedding_dim=train_embeddings.shape[1],
        )
        craig_subset = Subset(train_dataset, craig_indices)

        #train the model on the craig subset and evaluate on the validation set
        model_craig = build_embedding_classifier(train_embeddings.shape[1], num_classes)
        print(f"Training classifier model on craig subset for {num_epochs} epochs with batch size {batch_size_craig}")
        history_craig = train_model(
            model=model_craig,
            train_dataset=craig_subset,
            val_dataset=val_dataset,
            epochs=num_epochs,
            batch_size=batch_size_craig,
            device=device,
            sample_weights=craig_weights,
        )
        save_history(craig_csv_path, history_craig)

    ###########################################################################################################################
    # HERDING PART
    ###########################################################################################################################

    herding_csv_path = os.path.join(output_dir, "herding.csv")
    if os.path.exists(herding_csv_path):
        print(f"Skipping herding: found existing {herding_csv_path}")
    else:
        # HERDING SUBSET
        feature_model_herding = nn.Identity()
        herding_indices = run_herding_subset(
            train_dataset=train_dataset,
            n_samples=n_samples,
            feature_model=feature_model_herding,
            batch_size=batch_size_herding,
        )
        herding_subset = Subset(train_dataset, herding_indices)

        #train the model on the herding subset and evaluate on the validation set
        print(f"Training classifier model on herding subset for {num_epochs} epochs with batch size {batch_size_herding}")
        model_herding = build_embedding_classifier(train_embeddings.shape[1], num_classes)
        history_herding = train_model(
            model=model_herding,
            train_dataset=herding_subset,
            val_dataset=val_dataset,
            epochs=num_epochs,
            batch_size=batch_size_herding,
            device=device,
        )
        save_history(herding_csv_path, history_herding)

    ###########################################################################################################################
    # CS-SHAPLEY PART
    ###########################################################################################################################

    # CS-SHAPLEY SUBSET 
    feature_model_shapley = None
    shapley_fraction = float(config.get("shapley_train_fraction", 0.25))
    shapley_fraction = min(max(shapley_fraction, 0.0), 1.0)
    shapley_train_size = max(1, int(len(train_dataset) * shapley_fraction))
    rng = np.random.default_rng(42)
    shapley_indices_subset = rng.choice(
        len(train_dataset), size=shapley_train_size, replace=False
    )
    shapley_train_subset = Subset(train_dataset, shapley_indices_subset.tolist())
    shapley_n_samples = min(n_samples, len(shapley_train_subset))
    print(
        f"CS-Shapley using {len(shapley_train_subset)}/{len(train_dataset)} "
        f"samples ({shapley_fraction:.2f} of train set)"
    )
    shapley_indices, _ = run_cs_shapley_subset(
        train_dataset=shapley_train_subset,
        val_dataset=val_dataset,
        n_samples=shapley_n_samples,
        feature_model=feature_model_shapley,
        batch_size=batch_size_cs_shapley,
        num_classes=num_classes,
        num_permutations=num_permutations,
        resample=resample,
        clf_max_iter=shapley_lr_max_iter,
        clf_batch_size=batch_size_cs_shapley_lr,
    )
    shapley_subset = Subset(train_dataset, shapley_indices)

    #train the model on the cs-shapley subset and evaluate on the validation set
    print(f"Training classifier model on cs-shapley subset for {num_epochs} epochs with batch size {batch_size_cs_shapley}")
    model_shapley = build_embedding_classifier(train_embeddings.shape[1], num_classes)
    history_shapley = train_model(
        model=model_shapley,
        train_dataset=shapley_subset,
        val_dataset=val_dataset,
        epochs=num_epochs,
        batch_size=batch_size_cs_shapley,
        device=device,
    )
    save_history(os.path.join(output_dir, "cs_shapley.csv"), history_shapley)



if __name__ == "__main__":
    main()
