import os
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from stage2_sghf.Kvasir_V2.custom_dataset import MedicalImageDataset

def read_all_data(root: str):
    assert os.path.exists(root), f"dataset root: {root} does not exist."

    class_names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    class_names.sort()

    class_indices = {k: v for v, k in enumerate(class_names)}

    json_str = json.dumps({v: k for k, v in class_indices.items()}, indent=4)
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    supported = {".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"}
    all_paths, all_labels = [], []
    every_class_num = []

    for cla in class_names:
        cla_path = os.path.join(root, cla)
        images = [
            os.path.join(cla_path, i) for i in os.listdir(cla_path)
            if os.path.splitext(i)[-1] in supported
        ]
        images.sort()
        every_class_num.append(len(images))

        label = class_indices[cla]
        for p in images:
            all_paths.append(p)
            all_labels.append(label)

    print(f"{sum(every_class_num)} images were found in the dataset.")
    for cla, n in zip(class_names, every_class_num):
        print(f"  - {cla}: {n}")

    assert len(all_paths) > 0, "not find any images."
    return all_paths, all_labels, class_indices


def build_kfold_loaders(
    data_root,
    transform_train,
    transform_valid,
    k=5,
    batch_size=32,
    num_workers=8,
    seed=10,
    shuffle_train=True,
    pin_memory=True
):
    all_paths, all_labels, class_indices = read_all_data(data_root)

    all_paths = np.array(all_paths)
    all_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels), start=1):
        train_paths = all_paths[train_idx].tolist()
        train_labels = all_labels[train_idx].tolist()
        val_paths = all_paths[val_idx].tolist()
        val_labels = all_labels[val_idx].tolist()

        train_dataset = MedicalImageDataset(train_paths, train_labels, transform=transform_train)
        val_dataset   = MedicalImageDataset(val_paths,   val_labels,   transform=transform_valid)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        tr_counts = np.bincount(np.array(train_labels), minlength=len(class_indices))
        va_counts = np.bincount(np.array(val_labels),   minlength=len(class_indices))
        print(f"\n[Fold {fold_id}/{k}] train={len(train_dataset)} val={len(val_dataset)}")
        print("  train class counts:", tr_counts.tolist())
        print("  val   class counts:", va_counts.tolist())

        folds.append((fold_id, train_loader, val_loader))

    return folds, class_indices