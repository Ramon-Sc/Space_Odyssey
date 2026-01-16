

import torch
from torch.utils.data import DataLoader


class herding():
    """
    Herding coreset selection ("Herding Dynamical weights to learn" Max Welling ,2009).

    This implementation uses a pretrained network as the feature map phi(x).
    It greedily selects `n_samples` points from `train_dataset` whose feature
    mean approximates the full dataset feature mean.

   
    """

    def __init__(self, model, train_dataset, n_samples, model_path=None, batch_size=64):
        """
        Args:
            model: Pretrained network used as feature extractor φ(x).
            train_dataset: Training dataset (e.g. IndexedImageFolder).
            n_samples: Number of herding points to select.
            model_path: Optional path to a checkpoint for `model`.
            batch_size: Batch size for feature extraction.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.n_samples = n_samples
        self.model_path = model_path
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_features(self):
        """
        Compute phi(x) for all samples in `train_dataset`.

        Returns:
            all_indices: Tensor of dataset indices (N,)
            features: Tensor of features (N, D)
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.model.to(self.device)
        self.model.eval()

        all_indices = []
        all_features = []

        with torch.no_grad():
            for indices, inputs, _ in loader:
                inputs = inputs.to(self.device)

                feats = self.model(inputs)
                # Flatten everything except batch dim
                feats = feats.view(feats.size(0), -1)

                all_indices.append(indices)
                all_features.append(feats.cpu())

        all_indices = torch.cat(all_indices, dim=0).long()
        features = torch.cat(all_features, dim=0).float()  # (N, D)

        return all_indices, features

    def run_herding(self):
        """
        Run Max Welling's (2009) herding algorithm to select a representative subset.

        Returns:
            selected_dataset_indices (List[int]): Indices into `train_dataset`
                                                 of the selected points.
        """
        

        # Compute features phi(x) for all training samples 
        dataset_indices, features = self.compute_features()  # (N,), (N, D)

        N, D = features.shape
        if self.n_samples > N:
            raise ValueError(
                f"Requested n_samples={self.n_samples} but dataset size is only N={N}"
            )

        # Compute feature mean μ over the whole dataset
        mean_feature = features.mean(dim=0)  # (D,)

        #if muliple moments are required:
        #examples:
        #feature1 = (features**2).mean(dim=0)  # (N,)
        #feature2 = (features**3).mean(dim=0)  # (N,)
        #feature3 = (features**4).mean(dim=0)  # (N,)
        #...
        #feature_n = (features**n).mean(dim=0)  # (N,)
 
        #mean_feature = concatenate(feature1, feature2, feature3, ..., feature_n)  # (N,)

        # Herding loop
        w = mean_feature.clone()

        selected_mask = torch.zeros(N, dtype=torch.bool)
        selected_dataset_indices = []

        for _ in range(self.n_samples):
            # Scores for all candidates: dot product weights * features <w, phi(x_i)>

            # to gpu
            w.to(self.device)
            features.to(self.device)
            
            scores = features @ w  # (N,)

            # Mask out already selected points
            scores[selected_mask] = -float("inf")

            # Greedily pick best-aligned point
            best_idx = torch.argmax(scores).item()  # index into features array

            selected_mask[best_idx] = True
            selected_dataset_indices.append(int(dataset_indices[best_idx].item()))

            # Update w: w_t = w_{t-1} + mean_feature - phi(x_{i_t})
            w = w + mean_feature - features[best_idx]

        return selected_dataset_indices