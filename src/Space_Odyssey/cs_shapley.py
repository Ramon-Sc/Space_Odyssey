import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from random import shuffle, randint, sample
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class TorchLogisticRegression:
    def __init__(
        self,
        max_iter=50,
        lr=1e-1,
        weight_decay=0.0,
        batch_size=None,
        device=None,
    ):
        self.max_iter = max_iter
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self._class_labels = None
        self._class_to_index = None
        self._constant_class = None

    def _prepare_xy(self, X, y):
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X).float()
        else:
            X_t = X.float()
        if isinstance(y, np.ndarray):
            y_t = torch.from_numpy(y).long()
        else:
            y_t = y.long()
        return X_t.to(self.device), y_t.to(self.device)

    def fit(self, X, y):
        X_t, y_t = self._prepare_xy(X, y)
        unique = torch.unique(y_t)
        if unique.numel() == 1:
            self._constant_class = int(unique.item())
            self.model = None
            return self

        self._constant_class = None
        self._class_labels = unique.sort().values
        self._class_to_index = {
            int(label.item()): idx
            for idx, label in enumerate(self._class_labels)
        }
        y_idx = torch.tensor(
            [self._class_to_index[int(v.item())] for v in y_t],
            device=self.device,
            dtype=torch.long,
        )

        num_features = X_t.shape[1]
        num_classes = len(self._class_labels)
        self.model = nn.Linear(num_features, num_classes).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        if self.batch_size is None or self.batch_size >= X_t.shape[0]:
            for _ in range(self.max_iter):
                optimizer.zero_grad()
                logits = self.model(X_t)
                loss = criterion(logits, y_idx)
                loss.backward()
                optimizer.step()
        else:
            dataset = torch.utils.data.TensorDataset(X_t, y_idx)
            loader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(dataset)),
                shuffle=True,
            )
            for _ in range(self.max_iter):
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
        return self

    def predict(self, X):
        if self._constant_class is not None:
            return np.full(len(X), self._constant_class, dtype=np.int64)
        if self.model is None:
            raise ValueError("Model is not fitted.")
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X).float().to(self.device)
        else:
            X_t = X.float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        pred_labels = self._class_labels.cpu().numpy()[pred_idx]
        return pred_labels.astype(np.int64)


class cs_shapley():
    """
    Class-wise Shapley coreset selection using a classifier utility.

    This adapts the reference implementation to datasets and optional feature
    extractors. Shapley values are computed per class by estimating marginal
    contributions to validation accuracy under random permutations.
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        clf,
        n_samples,
        num_classes=None,
        num_permutations=200,
        epsilon=1e-4,
        normalized_score=True,
        resample=1,
        batch_size=64,
        model_path=None,
        use_tqdm=True,
    ):
        """
        Args:
            model: Optional feature extractor. If None, raw inputs are flattened.
            train_dataset: Training dataset (e.g. IndexedImageFolder).
            val_dataset: Validation dataset.
            clf: sklearn-style classifier with fit/predict.
            n_samples: Total number of samples to select across all classes.
            num_classes: Number of classes. If None, inferred from dataset.
            num_permutations: Number of permutations per class.
            epsilon: Early stopping threshold for class utility computation.
            normalized_score: Normalize Shapley values within each class.
            resample: Multiplier for permutations (num_permutations * resample).
            batch_size: Batch size for feature extraction.
            model_path: Optional path to a checkpoint for `model`.
            use_tqdm: Whether to show progress bars.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.clf = clf
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.num_permutations = num_permutations
        self.epsilon = epsilon
        self.normalized_score = normalized_score
        self.resample = resample
        self.batch_size = batch_size
        self.model_path = model_path
        self.use_tqdm = use_tqdm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer_num_classes(self, labels):
        if self.num_classes is not None:
            return self.num_classes
        if hasattr(self.train_dataset, "classes"):
            return len(self.train_dataset.classes)
        return int(labels.max()) + 1

    def _dataset_to_numpy(self, dataset):
        """
        Convert a torch dataset into numpy arrays (X, Y, indices).
        Applies model feature extraction if provided.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        all_indices = []
        all_features = []
        all_labels = []

        start_idx = 0
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    indices, inputs, labels = batch
                else:
                    inputs, labels = batch
                    indices = torch.arange(
                        start_idx, start_idx + inputs.size(0), dtype=torch.long
                    )
                    start_idx += inputs.size(0)

                inputs = inputs.to(self.device)
                if self.model is not None:
                    feats = self.model(inputs)
                else:
                    feats = inputs

                feats = feats.view(feats.size(0), -1)

                all_indices.append(indices.cpu())
                all_features.append(feats.cpu())
                all_labels.append(labels.cpu())

        all_indices = torch.cat(all_indices, dim=0).long()
        features = torch.cat(all_features, dim=0).float()
        labels = torch.cat(all_labels, dim=0).long()

        # ensure consistent ordering by dataset index
        order = torch.argsort(all_indices)
        all_indices = all_indices[order]
        features = features[order]
        labels = labels[order]

        return features.numpy(), labels.numpy(), all_indices.numpy()

    def _allocate_samples_per_class(self, class_counts):
        total = int(self.n_samples)
        if total <= 0:
            raise ValueError("n_samples must be positive.")

        total_count = int(sum(class_counts))
        if total > total_count:
            raise ValueError(
                f"Requested n_samples={total} but dataset size is {total_count}."
            )

        # proportional allocation with remainder distribution
        raw = [total * (c / total_count) for c in class_counts]
        alloc = [int(np.floor(x)) for x in raw]
        remainder = total - sum(alloc)

        frac = [r - np.floor(r) for r in raw]
        order = np.argsort(frac)[::-1]

        for idx in order:
            if remainder == 0:
                break
            if alloc[idx] < class_counts[idx]:
                alloc[idx] += 1
                remainder -= 1

        # if any remainder still left due to caps, fill from any class with room
        if remainder > 0:
            for idx in range(len(class_counts)):
                if remainder == 0:
                    break
                if alloc[idx] < class_counts[idx]:
                    alloc[idx] += 1
                    remainder -= 1

        return alloc

    def _class_conditional_sampling(self, Y, label_set):
        idx_nonlabel = []
        for label in label_set:
            label_indices = list(np.where(Y == label)[0])
            s = randint(1, len(label_indices))
            idx_nonlabel += sample(label_indices, s)
        shuffle(idx_nonlabel)
        return idx_nonlabel

    def _cs_shapley_class(self, trnX, trnY, devX, devY, label):
        """
        Reference class-wise Shapley algorithm for a single class.
        Returns:
            val (np.ndarray): Shapley values for the class subset
            orig_indices (np.ndarray): Indices of class samples in the full train set
        """
        # Select data based on class label
        orig_indices = np.array(list(range(trnX.shape[0])))[trnY == label]
        trnX_label = trnX[trnY == label]
        trnY_label = trnY[trnY == label]
        trnX_nonlabel = trnX[trnY != label]
        trnY_nonlabel = trnY[trnY != label]
        devX_label = devX[devY == label]
        devY_label = devY[devY == label]
        devX_nonlabel = devX[devY != label]
        devY_nonlabel = devY[devY != label]
        N_nonlabel = trnX_nonlabel.shape[0]
        
        nonlabel_set = list(set(trnY_nonlabel)) # list of unique classes in the non-label set



        ##################################################################################################
        # Permutation and resampling loop
        ##################################################################################################

        # Create indices and initialize the shapley value array
        N = trnX_label.shape[0]
        idx = list(range(N))
        val, k = np.zeros((N)), 0


        total_iters = self.num_permutations * int(self.resample)
        pbar = (
            tqdm(
                total=total_iters,
                desc=f"CS-Shapley class {label}",
                leave=False,
            )
            if self.use_tqdm
            else None
        )

        for permutation in range(self.num_permutations):

            ##############################################################
            # shuffle the indices
            ##############################################################
            shuffle(idx)
            # For each permutation, resample r times from the other classes
            for r in range(int(self.resample)):
                k += 1
                val_i = np.zeros((N + 1))
                val_i_non = np.zeros((N + 1))

                    # val_i: [acc(non_label_only), acc(non_label+1label), acc(non_label+2label), ..., acc(non_label+Nlabel)]
                    # val_i_non: [acc(non_label_only), acc(non_label+1label), acc(non_label+2label), ..., acc(non_label+Nlabel)]

                # Sample a subset of training data from other labels
                if len(nonlabel_set) == 1:
                    s = randint(1, N_nonlabel)
                    idx_nonlabel = sample(list(range(N_nonlabel)), s)
                else:
                    idx_nonlabel = self._class_conditional_sampling(
                        trnY_nonlabel, nonlabel_set
                    )
                trnX_nonlabel_i = trnX_nonlabel[idx_nonlabel, :]
                trnY_nonlabel_i = trnY_nonlabel[idx_nonlabel]


                ##############################################################
                #1. With no data from the target class
                val_i[0] = 0.0
                try:
                    self.clf.fit(trnX_nonlabel_i, trnY_nonlabel_i)
                    val_i_non[0] = (
                        accuracy_score(
                            devY_nonlabel,
                            self.clf.predict(devX_nonlabel),
                            normalize=False,
                        )
                        / len(devY)
                    )
                except ValueError:
                    val_i_non[0] = (
                        accuracy_score(
                            devY_nonlabel,
                            [trnY_nonlabel_i[0]] * len(devY_nonlabel),
                            normalize=False,
                        )
                        / len(devY)
                    )
                ##############################################################
                #2. With all data from the target class
                tempX = np.concatenate((trnX_nonlabel_i, trnX_label))
                tempY = np.concatenate((trnY_nonlabel_i, trnY_label))
                self.clf.fit(tempX, tempY)
                val_i[N] = (
                    accuracy_score(devY_label, self.clf.predict(devX_label), normalize=False)
                    / len(devY)
                )
                val_i_non[N] = (
                    accuracy_score(
                        devY_nonlabel,
                        self.clf.predict(devX_nonlabel),
                        normalize=False,
                    )
                    / len(devY)
                )
                ##############################################################
                ##############################################################
                #MARGINAL CONTRIBUTION PER SAMPLE CALCULATION
                ##############################################################
                for j in range(1, N + 1):
                    if abs(val_i[N] - val_i[j - 1]) < self.epsilon:
                        val_i[j] = val_i[j - 1]
                    else:
                        trnX_j = trnX_label[idx[:j], :]
                        trnY_j = trnY_label[idx[:j]]
                        ##############################################################
                        # concatenate the non-label set and the label set up to the j-th sample
                        tempX = np.concatenate((trnX_nonlabel_i, trnX_j))
                        tempY = np.concatenate((trnY_nonlabel_i, trnY_j))
                        ##############################################################
                        # fit the classifier on the concatenated data
                        self.clf.fit(tempX, tempY)
                        ##############################################################
                        # calculate the accuracy of the classifier on the label set
                        val_i[j] = (
                            accuracy_score(
                                devY_label,
                                self.clf.predict(devX_label),
                                normalize=False,
                            )
                            / len(devY)
                        )
                        ##############################################################
                        # calculate the marginal contribution of the j-th sample
                        val_i_non[j] = (
                            accuracy_score(
                                devY_nonlabel,
                                self.clf.predict(devX_nonlabel),
                                normalize=False,
                            )
                            / len(devY)
                        )

                # weighted values: v_yi(S_yi|S_-yi)= accuracy_S(D_yi)* e^(accuracy_S(D_-yi))
                
                wvalues = np.exp(val_i_non) * val_i
                # [acc(non_label_only), acc(non_label+1label), acc(non_label+2label), ..., acc(non_label+Nlabel)] - [acc(non_label_only), acc(non_label+1label), acc(non_label+2label), ..., acc(non_label+Nlabel)]
                diff = wvalues[1:] - wvalues[:N]

                # update the shapley values for the current permutation 
                val[idx] = ((1.0 * (k - 1) / k)) * val[idx] + (1.0 / k) * diff
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Normalize within class if requested
        if self.normalized_score:
            if val.sum() != 0:
                val = val / val.sum()
            self.clf.fit(trnX, trnY)
            score = (
                accuracy_score(devY_label, self.clf.predict(devX_label), normalize=False)
                / len(devY)
            )
            val = val * score

        return val, orig_indices

    def run_cs_shapley(self):
        """
        Compute class-wise Shapley values and select coreset samples.

        Returns:
            selected_indices (List[int]): Selected dataset indices.
            shapley_values (np.ndarray): Shapley values for all dataset points.
        """
        trnX, trnY, trn_idx = self._dataset_to_numpy(self.train_dataset)
        devX, devY, _ = self._dataset_to_numpy(self.val_dataset)
        num_classes = self._infer_num_classes(trnY)

        if isinstance(self.train_dataset, torch.utils.data.Subset):
            total_len = len(self.train_dataset.dataset)
        else:
            total_len = len(self.train_dataset)
        shapley_values = np.zeros(total_len, dtype=np.float32)
        class_counts = []
        class_indices_list = []

        class_iter = (
            tqdm(range(num_classes), desc="CS-Shapley classes")
            if self.use_tqdm
            else range(num_classes)
        )
        for label in class_iter:
            val, orig_indices = self._cs_shapley_class(trnX, trnY, devX, devY, label)
            shapley_values[trn_idx[orig_indices]] = val.astype(np.float32)
            class_counts.append(int(len(orig_indices)))
            class_indices_list.append(trn_idx[orig_indices])

        # allocate how many to take per class, then select top points
        per_class_alloc = self._allocate_samples_per_class(class_counts)
        selected_indices = []

        for c in range(num_classes):
            k_c = per_class_alloc[c]
            if k_c <= 0:
                continue
            class_indices = class_indices_list[c]
            if class_indices.size == 0:
                continue
            class_vals = shapley_values[class_indices]
            topk = np.argsort(class_vals)[::-1][:k_c]
            selected_indices.extend(class_indices[topk].tolist())

        return selected_indices, shapley_values
