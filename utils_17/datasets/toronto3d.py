import os
import torch
import yaml
import numpy as np
import tqdm

import MinkowskiEngine as ME
from datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class Toronto3D(BaseDataset):
    def __init__(
        self,
        version: str = "full",
        phase: str = "train",
        dataset_path: str = "/data3/vaibhav/Toronto_3D",
        mapping_path: str = "_resources/toronto3d.yaml",
        weights_path: str = None,
        voxel_size: float = 0.05,
        use_intensity: bool = False,
        augment_data: bool = False,
        sub_num: int = 150000,
        device: str = None,
        num_classes: int = 7,
        ignore_label: int = -1,
    ):

        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        super().__init__(
            version=version,
            phase=phase,
            dataset_path=dataset_path,
            voxel_size=voxel_size,
            sub_num=sub_num,
            use_intensity=use_intensity,
            augment_data=augment_data,
            device=device,
            num_classes=num_classes,
            ignore_label=ignore_label,
            weights_path=weights_path,
        )

        if self.version == "full":
            self.split = {
                "train": [
                    "L001_2",
                    "L001_3",
                    "L001_4",
                    "L001_5",
                    "L001_6",
                    "L001_7_1",
                    "L001_7_2",
                    "L001_8",
                    "L001_9",
                    "L002_2",
                    "L002_3",
                    "L002_4",
                    "L002_5",
                    "L002_6",
                    "L002_7",
                    "L002_8",
                    "L002_9",
                    "L003_10",
                    "L003_3",
                    "L003_4",
                    "L003_5",
                    "L003_6_2",
                    "L003_6_3",
                    "L003_6_4",
                    "L003_7",
                    "L003_8",
                    "L003_9",
                ],
                "validation": [
                    "L004_10",
                    "L004_2",
                    "L004_3",
                    "L004_4",
                    "L004_5",
                    "L004_6",
                    "L004_7",
                    "L004_8",
                    "L004_9",
                ],
            }
        elif self.version == "mini":
            self.split = {
                "train": [
                    "L001_2",
                    "L001_3",
                    "L001_4",
                    "L001_5",
                    "L001_6",
                    "L001_7_1",
                    "L001_7_2",
                    "L001_8",
                    "L001_9",
                    "L002_2",
                    "L002_3",
                    "L002_4",
                    "L002_5",
                    "L002_6",
                    "L002_7",
                    "L002_8",
                    "L002_9",
                    "L003_10",
                    "L003_3",
                    "L003_4",
                    "L003_5",
                    "L003_6_2",
                    "L003_6_3",
                    "L003_6_4",
                    "L003_7",
                    "L003_8",
                    "L003_9",
                ],
                "validation": [
                    "L004_10",
                    "L004_2",
                    "L004_3",
                    "L004_4",
                    "L004_5",
                    "L004_6",
                    "L004_7",
                    "L004_8",
                    "L004_9",
                ],
            }
        elif self.version == "sequential":
            self.split = {
                "train": [
                    "L001_2",
                    "L001_3",
                    "L001_4",
                    "L001_5",
                    "L001_6",
                    "L001_7_1",
                    "L001_7_2",
                    "L001_8",
                    "L001_9",
                    "L002_2",
                    "L002_3",
                    "L002_4",
                    "L002_5",
                    "L002_6",
                    "L002_7",
                    "L002_8",
                    "L002_9",
                    "L003_10",
                    "L003_3",
                    "L003_4",
                    "L003_5",
                    "L003_6_2",
                    "L003_6_3",
                    "L003_6_4",
                    "L003_7",
                    "L003_8",
                    "L003_9",
                ],
                "validation": [
                    "L004_10",
                    "L004_2",
                    "L004_3",
                    "L004_4",
                    "L004_5",
                    "L004_6",
                    "L004_7",
                    "L004_8",
                    "L004_9",
                ],
            }
        else:
            raise NotImplementedError

        self.name = "Toronto3D"
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), "r"))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val
        # print("Toronto remap", self.remap_lut_val)

        # self.rotation_augmentation_bound = (
        #     (0.2242, 0.2242),
        #     (0.0037, 0.0037),
        #     (-np.pi, -np.pi/2),
        # )

        for files in self.split[self.phase]:
            pcd_path = os.path.join(self.dataset_path, f"{files}.csv")
            self.pcd_path.append(pcd_path)

        self.color_map = (
            np.array(
                [
                    (255, 255, 255),  # unlabeled: -1
                    (13, 13, 247),  # cars: 0
                    (2, 111, 107),  # cycle: 1
                    (44, 223, 223),  # motorbike: 2
                    (75, 112, 184),  # other-vehicle: 3
                    (255, 228, 196),  # person: 4
                    (255, 40, 200),  # bicyclist: 5
                    (255, 128, 64),  # motorcyclist: 6
                    (199, 0, 199),  # road: 7
                    (166, 0, 166),  # sidewalk: 8
                    (172, 115, 112),  # other-ground: 9
                    (235, 0, 1),  # buildings: 10
                    (245, 145, 69),  # fences: 11
                    (0, 161, 0),  # vegetations: 12
                    (86, 101, 89),  # traffic-sign: 13
                    (223, 190, 0),  # pole: 14
                    (128, 0, 255),  # parking: 15
                    (255, 255, 0),  # wires: 16
                ]
            )
            / 255.0
        )  # traffic-sign

        self.classes_to_sample = [0, 1, 2, 3, 4]

    def __len__(self):
        return len(self.pcd_path)

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i]

        data = np.genfromtxt(pcd_tmp, delimiter=",", skip_header=1)

        # Initialize an empty array to hold the subsampled rows
        subsampled_rows = np.empty((0, data.shape[1]))

        # Loop through each label to subsample
        for label_to_subsample in self.classes_to_sample:
            # Identify rows with the specified label
            label_column = data[:, -1]
            rows_to_subsample = data[label_column == label_to_subsample]

            if label_to_subsample in [3, 4]:
                num_to_subsample = int(len(rows_to_subsample) * 0.50)
            else:
                num_to_subsample = int(len(rows_to_subsample) * 1e-3)

            if num_to_subsample > 0 and len(rows_to_subsample) > 0:
                # Randomly subsample the specified label
                subsampled_indices = np.random.choice(
                    len(rows_to_subsample), num_to_subsample, replace=False
                )
                subsampled_rows = np.vstack(
                    (subsampled_rows, rows_to_subsample[subsampled_indices])
                )
            elif len(rows_to_subsample) == 0:
                print(f"No rows found with label {label_to_subsample}")

        # Identify rows that do not match any of the labels to be subsampled
        rows_to_keep = data[~np.isin(label_column, self.classes_to_sample)]
        data = np.vstack((rows_to_keep, subsampled_rows))

        points = data[:, :3].reshape(-1, 3).astype(np.float32)
        label = data[:, -1].reshape(-1).astype(np.int32)
        # label = self.remap_lut_val[label]
        mapping = {0: -1, 1: -1, 2: -1, 3: 12, 4: 10, 5: 16, 6: 14, 7: 0}

        # Vectorize the mapping function with default value for missing keys
        map_func = np.vectorize(lambda x: mapping.get(x, -1))

        # Apply the mapping
        label = map_func(label)
        label = label.astype(np.int32)

        if self.use_intensity:
            colors = data[:, 6][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)

        data = {"points": points, "colors": colors, "labels": label}
        data_mean = [
            np.mean(points[:, 0]),
            np.mean(points[:, 1]),
            np.mean(points[:, 2]),
        ]
        points = data["points"] - data_mean
        colors = data["colors"]
        labels = data["labels"]

        # print("toronto3d before transformation - mean", points)

        sampled_idx = np.arange(points.shape[0])

        if self.phase == "train" and self.augment_data:
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack(
                (points, np.ones((points.shape[0], 1), dtype=points.dtype))
            )
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        # print(labels.shape, points.shape, colors.shape)

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
            points,
            colors,
            labels=labels,
            ignore_label=vox_ign_label,
            quantization_size=self.voxel_size,
            return_index=True,
        )

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None

        # print("toronto quantized_coords", quantized_coords)

        return {
            "coordinates": quantized_coords,
            "features": feats,
            "labels": labels,
            "sampled_idx": sampled_idx,
            "idx": torch.tensor(i),
        }

    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max() + 1)
        for l in tqdm.tqdm(
            range(len(self.pcd_path)), desc="Loading weights", leave=True
        ):
            pcd_tmp = self.pcd_path[l]
            data = np.genfromtxt(pcd_tmp, delimiter=",", skip_header=1)
            label = data[:, -1].reshape(-1, 1).astype(np.int32)
            label = self.remap_lut_val[label]
            label = label.astype(np.int32)
            lbl, count = np.unique(label, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights

    def get_data(self, i: int):
        pcd_tmp = self.pcd_path[i]
        data = np.genfromtxt(pcd_tmp, delimiter=",", skip_header=1)

        # Initialize an empty array to hold the subsampled rows
        subsampled_rows = np.empty((0, data.shape[1]))

        # Loop through each label to subsample
        for label_to_subsample in self.classes_to_sample:
            # Identify rows with the specified label
            label_column = data[:, -1]
            rows_to_subsample = data[label_column == label_to_subsample]

            if label_to_subsample in [3, 4]:
                num_to_subsample = int(len(rows_to_subsample) * 0.4)
            elif label_to_subsample in [0, 1, 2]:
                num_to_subsample = int(len(rows_to_subsample) * 1e-3)

            if num_to_subsample > 0 and len(rows_to_subsample) > 0:
                # Randomly subsample the specified label
                subsampled_indices = np.random.choice(
                    len(rows_to_subsample), num_to_subsample, replace=False
                )
                subsampled_rows = np.vstack(
                    (subsampled_rows, rows_to_subsample[subsampled_indices])
                )
            # elif len(rows_to_subsample) == 0:
            # print(f"No rows found with label {label_to_subsample}")

        # Identify rows that do not match any of the labels to be subsampled
        rows_to_keep = data[~np.isin(label_column, self.classes_to_sample)]
        data = np.vstack((rows_to_keep, subsampled_rows))

        points = data[:, :3].reshape(-1, 3).astype(np.float32)
        label = data[:, -1].reshape(-1).astype(np.int32)
        label = self.remap_lut_val[label]
        label = label.astype(np.int32)
        # print("Unqiue labels in toronto3d: ", np.unique(label))
        points = points[:, :3]
        if self.use_intensity:
            colors = data[:, 6][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {"points": points, "colors": colors, "labels": label}
        data_mean = [
            np.mean(points[:, 0]),
            np.mean(points[:, 1]),
            np.mean(points[:, 2] * 1.0005),
        ]
        points = data["points"] - data_mean
        colors = data["colors"]
        labels = data["labels"]

        return {"coordinates": points, "features": colors, "labels": labels, "idx": i}
