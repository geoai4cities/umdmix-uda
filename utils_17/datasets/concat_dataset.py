import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from sampling.voxelizer import Voxelizer

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


class ConcatDataset(Dataset):
    def __init__(
        self,
        source1_dataset,
        source2_dataset,
        target_dataset,
        augment_mask_data,
        augment_data,
        remove_overlap,
    ) -> None:
        r"""
        Desc: Wrapper for 2 BaseDataset instances. Used for non-source free UDA methods;
        :param source_dataset: the source dataset (labelled)
        :param target_dataset: the target dataset (labels used ONLY for evaluation)
        """
        super().__init__()

        self.source1_dataset = source1_dataset
        self.source2_dataset = source2_dataset
        self.target_dataset = target_dataset

        self.voxel_size = self.target_dataset.voxel_size

        self.target_len = int(len(source1_dataset))

        self.class2names = self.target_dataset.class2names
        self.colormap = self.source2_dataset.color_map

        self.ignore_label = self.target_dataset.ignore_label

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        self.augment_mask_data = augment_mask_data
        self.augment_data = augment_data
        self.remove_overlap = remove_overlap

        self.clip_bounds = None
        self.scale_augmentation_bound = (0.95, 1.05)
        self.rotation_augmentation_bound = (
            (-np.pi / 20, np.pi / 20),
            (-np.pi / 20, np.pi / 20),
            (-np.pi / 20, np.pi / 20),
        )
        self.translation_augmentation_ratio_bound = None

        self.scale_augmentation_bound_mask = (0.95, 1.05)
        self.rotation_augmentation_bound_mask = (None, None, (-np.pi / 20, np.pi / 20))
        self.translation_augmentation_ratio_bound_mask = None

        self.mask_voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=self.clip_bounds,
            use_augmentation=self.augment_mask_data,
            scale_augmentation_bound=self.scale_augmentation_bound_mask,
            rotation_augmentation_bound=self.rotation_augmentation_bound_mask,
            translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound_mask,
            ignore_label=vox_ign_label,
        )

        self.voxelizer = Voxelizer(
            voxel_size=self.voxel_size,
            clip_bound=self.clip_bounds,
            use_augmentation=self.augment_data,
            scale_augmentation_bound=self.scale_augmentation_bound,
            rotation_augmentation_bound=self.rotation_augmentation_bound,
            translation_augmentation_ratio_bound=self.translation_augmentation_ratio_bound,
            ignore_label=vox_ign_label,
        )

        self.weights = self.source1_dataset.weights

    def voxelize(self, data, do_divide=True):
        data_pts = data["coordinates"]
        data_labels = data["labels"]
        data_features = data["features"]

        _, _, _, voxel_idx = ME.utils.sparse_quantize(
            coordinates=data_pts,
            features=data_features,
            labels=data_labels,
            quantization_size=self.voxel_size,
            return_index=True,
        )

        if do_divide:
            data_pts = data_pts[voxel_idx] / self.voxel_size
        else:
            data_pts = data_pts[voxel_idx]
        data_labels = data_labels[voxel_idx]
        data_features = data_features[voxel_idx]

        if not isinstance(voxel_idx, torch.Tensor):
            voxel_idx = torch.from_numpy(voxel_idx)

        return {
            "coordinates": torch.from_numpy(data_pts).floor(),
            "labels": torch.from_numpy(data_labels),
            "features": torch.from_numpy(data_features),
            "idx": voxel_idx,
        }

    def merge(self, source1_data, source2_data, target_data):
        # "coordinates"
        # "features"
        # "labels"
        # "sampled_idx"
        # "idx"
        # data come in a dict with keys [coordinates, features and labels]
        # the output will contain also a mixed points, mixed labels and the idx to separate point clouds
        source1_data = self.voxelize(source1_data)
        source2_data = self.voxelize(source2_data)
        target_data = self.voxelize(target_data)

        source1_data = {f"source1_{k}": v for k, v in source1_data.items()}
        source2_data = {f"source2_{k}": v for k, v in source2_data.items()}
        target_data = {f"target_{k}": v for k, v in target_data.items()}

        data = {**source1_data, **source2_data, **target_data}

        return data

    def __getitem__(self, idx):
        if idx < len(self.target_dataset):
            source1_data = self.source1_dataset.get_data(idx)
            target_data = self.target_dataset.get_data(idx)
            source2_data = self.source2_dataset.get_data(
                np.random.choice(len(self.source2_dataset), 1)[0]
            )
            # print(np.unique(source2_data["labels"]))
        else:
            idx_mod = idx % len(self.target_dataset)
            target_data = self.target_dataset.get_data(idx_mod)
            source1_data = self.source1_dataset.get_data(idx)
            source2_data = self.source2_dataset.get_data(
                np.random.choice(len(self.source2_dataset), 1)[0]
            )

        return self.merge(source1_data, source2_data, target_data)

    def __len__(self):
        return self.target_len // 100
