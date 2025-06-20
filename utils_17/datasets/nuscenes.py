import os

import numpy as np
import torch
from torch.nn import functional as F
import yaml
import tqdm
from nuscenes import NuScenes as NuScenes_
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.splits import create_splits_scenes

import MinkowskiEngine as ME
from datasets.dataset import BaseDataset

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class NuScenesDataset(BaseDataset):
    def __init__(
        self,
        version: str = "full",
        phase: str = "train",
        dataset_path: str = "/data4/vaibhav/lidardata/nuscenes/",
        mapping_path: str = "_resources/nuscenes.yaml",
        weights_path: str = None,
        voxel_size: float = 0.05,
        use_intensity: bool = False,
        augment_data: bool = False,
        sub_num: int = 50000,
        device: str = None,
        num_classes: int = 7,
        ignore_label: int = -1,
    ):
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
        self.name = "NuScenesDataset"
        self.nusc = NuScenes_(
            version="v1.0-trainval", dataroot=self.dataset_path, verbose=True
        )
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), "r"))

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
        self.remap_lut_val = remap_lut_val

        self.color_map = (
            np.array(
                [
                    (255, 255, 255),  # unlabelled
                    (25, 25, 255),  # car
                    (187, 0, 255),  # bicycle
                    (187, 50, 255),  # motorcycle
                    (0, 247, 255),  # truck
                    (50, 162, 168),  # other-vehicle
                    (250, 178, 50),  # person
                    (255, 196, 0),  # bicyclist
                    (255, 196, 0),  # motorcyclist
                    (0, 0, 0),  # road
                    (148, 148, 148),  # parking
                    (255, 20, 60),  # sidewalk
                    (164, 173, 104),  # other-ground
                    (233, 166, 250),  # building
                    (255, 214, 251),  # fence
                    (157, 234, 50),  # vegetation
                    (107, 98, 56),  # trunk
                    (78, 72, 44),  # terrain
                    (83, 93, 130),  # pole
                    (173, 23, 121),  # traffic-sign
                ]
            )
            / 255.0
        )  # traffic-sign
        # get the scenes
        assert self.phase in ["train", "val", "test"]
        phase_scenes = create_splits_scenes()[self.phase]

        # create a list of camera & lidar scans
        self.list_keyframes = []
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                current_sample_token = scene["first_sample_token"]

                # Loop to get all successive keyframes
                list_data = []
                while current_sample_token != "":
                    current_sample = self.nusc.get("sample", current_sample_token)
                    list_data.append(current_sample["data"])
                    current_sample_token = current_sample["next"]

                # Add new scans in the list
                self.list_keyframes.extend(list_data)

    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, i: int):
        data = self.list_keyframes[i]

        # get the lidar
        lidar_token = data["LIDAR_TOP"]
        lidar_rec = self.nusc.get("sample_data", lidar_token)
        pc = LidarPointCloud.from_file(
            os.path.join(self.nusc.dataroot, lidar_rec["filename"])
        )
        pc = pc.points.T

        points = pc[:, :3].reshape(-1, 3).astype(np.float32)
        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)

        lidarseg_label_filename = os.path.join(
            self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"]
        )
        label = load_bin_file(lidarseg_label_filename).reshape(-1).astype(np.int32)
        label = self.remap_lut_val[label]
        data = {"points": points, "colors": colors, "labels": label}

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

    def get_data(self, i: int):
        data = self.list_keyframes[i]

        # get the lidar
        lidar_token = data["LIDAR_TOP"]
        lidar_rec = self.nusc.get("sample_data", lidar_token)
        pc = LidarPointCloud.from_file(
            os.path.join(self.nusc.dataroot, lidar_rec["filename"])
        )
        pc = pc.points.T

        points = pc[:, :3].reshape(-1, 3).astype(np.float32)
        if self.use_intensity:
            colors = points[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)

        lidarseg_label_filename = os.path.join(
            self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"]
        )
        label = load_bin_file(lidarseg_label_filename).reshape(-1).astype(np.int32)
        # print(label)
        label = self.remap_lut_val[label]
        data = {"points": points, "colors": colors, "labels": label}
        return {
            "coordinates": data["points"],
            "features": data["colors"],
            "labels": data["labels"],
            "idx": i,
        }
