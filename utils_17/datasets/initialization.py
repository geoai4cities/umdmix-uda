import os
import numpy as np
from datasets.dataset import BaseDataset
from datasets.concat_dataset import ConcatDataset
from datasets.toronto3d import Toronto3D
from datasets.semantickitti import SemanticKITTIDataset
from datasets.synlidar import SynLiDARDataset
from datasets.semanticposs import SemanticPOSSDataset
from datasets.nuscenes import NuScenesDataset

classnamearray = np.array(
    [
        "car",
        "bicycle",
        "motorcycle",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "traffic-sign",
        "pole",
        "parking",
        "wires",
    ]
)

classcolorarray = (
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
            (255, 255, 255),  # wires: 16
        ]
    )
    / 255.0
)


def get_dataset(
    dataset_name: str,
    dataset_path: str,
    dataset2_name: str = None,
    dataset2_path: str = None,
    voxel_size: float = 0.02,
    sub_num: int = 50000,
    augment_data: bool = False,
    version: str = "mini",
    num_classes: int = 7,
    ignore_label: int = -1,
    mapping_path: str = None,
    target_name: str = None,
    weights_path: str = None,
):
    target_dataset = None
    if dataset_name == "SemanticKITTI":
        training1_dataset = SemanticKITTIDataset(
            dataset_path=dataset_path,
            mapping_path=mapping_path,
            version=version,
            phase="train",
            voxel_size=voxel_size,
            augment_data=augment_data,
            sub_num=sub_num,
            num_classes=num_classes,
            ignore_label=ignore_label,
            weights_path=weights_path,
        )
        validation1_dataset = SemanticKITTIDataset(
            dataset_path=dataset_path,
            mapping_path=mapping_path,
            version=version,
            phase="validation",
            voxel_size=voxel_size,
            augment_data=False,
            num_classes=num_classes,
            ignore_label=ignore_label,
            weights_path=weights_path,
        )
        target_dataset = None
        training2_dataset = None
        validation2_dataset = None

        training1_dataset.class2names = classnamearray
        validation1_dataset.class2names = classnamearray
        training1_dataset.color_map = classcolorarray
        validation1_dataset.color_map = classcolorarray

    elif dataset_name == "SynLiDAR" and dataset2_name == "Toronto3D":

        training1_dataset = SynLiDARDataset(
            dataset_path=dataset_path,
            version=version,
            phase="train",
            voxel_size=voxel_size,
            augment_data=augment_data,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path=mapping_path,
            weights_path=weights_path,
        )

        validation1_dataset = SynLiDARDataset(
            dataset_path=dataset_path,
            version=version,
            phase="validation",
            voxel_size=voxel_size,
            augment_data=False,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path=mapping_path,
            weights_path=weights_path,
        )

        training2_dataset = Toronto3D(
            dataset_path=dataset2_path,
            version=version,
            phase="train",
            voxel_size=voxel_size,
            augment_data=augment_data,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path="_resources/toronto3d.yaml",
            weights_path=weights_path,
        )

        validation2_dataset = Toronto3D(
            dataset_path=dataset2_path,
            version=version,
            phase="validation",
            voxel_size=voxel_size,
            augment_data=False,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path="_resources/toronto3d.yaml",
            weights_path=weights_path,
        )

        if target_name == "SemanticKITTI":
            main_path, _ = os.path.split(dataset_path)
            target_dataset_path = "/data4/vaibhav/lidardata/semantic_kitti/sequences/"

            target_mapping_path = "_resources/semantic-kitti.yaml"
            target_dataset = SemanticKITTIDataset(
                dataset_path=target_dataset_path,
                mapping_path=target_mapping_path,
                version=version,
                phase="validation",
                voxel_size=voxel_size,
                augment_data=False,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )

            training1_dataset.class2names = classnamearray
            validation1_dataset.class2names = classnamearray

            training2_dataset.class2names = classnamearray
            validation2_dataset.class2names = classnamearray

            target_dataset.class2names = classnamearray

            training1_dataset.color_map = classcolorarray
            validation1_dataset.color_map = classcolorarray
            training2_dataset.color_map = classcolorarray
            validation2_dataset.color_map = classcolorarray
            target_dataset.color_map = classcolorarray

        if target_name == "NuScenes":
            main_path, _ = os.path.split(dataset_path)
            target_dataset_path = "/data4/vaibhav/lidardata/nuscenes/"

            target_mapping_path = "_resources/nuscenes.yaml"
            target_dataset = NuScenesDataset(
                dataset_path=target_dataset_path,
                mapping_path=target_mapping_path,
                version=version,
                phase="train",
                voxel_size=voxel_size,
                augment_data=False,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )

            training1_dataset.class2names = classnamearray
            validation1_dataset.class2names = classnamearray

            training2_dataset.class2names = classnamearray
            validation2_dataset.class2names = classnamearray

            target_dataset.class2names = classnamearray

            training1_dataset.color_map = classcolorarray
            validation1_dataset.color_map = classcolorarray
            training2_dataset.color_map = classcolorarray
            validation2_dataset.color_map = classcolorarray
            target_dataset.color_map = classcolorarray

    elif dataset_name == "SynLiDAR" and dataset2_name == "SemanticPOSS":
        training1_dataset = SynLiDARDataset(
            dataset_path=dataset_path,
            version=version,
            phase="train",
            voxel_size=voxel_size,
            augment_data=augment_data,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path=mapping_path,
            weights_path=weights_path,
        )
        validation1_dataset = SynLiDARDataset(
            dataset_path=dataset_path,
            version=version,
            phase="validation",
            voxel_size=voxel_size,
            augment_data=False,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path=mapping_path,
            weights_path=weights_path,
        )
        training2_dataset = SemanticPOSSDataset(
            dataset_path=dataset2_path,
            version=version,
            phase="train",
            voxel_size=voxel_size,
            augment_data=augment_data,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path="_resources/semanticposs.yaml",
            weights_path=weights_path,
        )

        validation2_dataset = SemanticPOSSDataset(
            dataset_path=dataset2_path,
            version=version,
            phase="validation",
            voxel_size=voxel_size,
            augment_data=False,
            num_classes=num_classes,
            ignore_label=ignore_label,
            mapping_path="_resources/semanticposs.yaml",
            weights_path=weights_path,
        )

        if target_name == "SemanticKITTI":
            main_path, _ = os.path.split(dataset_path)
            target_dataset_path = "/data4/vaibhav/lidardata/semantic_kitti/sequences/"

            target_mapping_path = "_resources/semantic-kitti.yaml"
            target_dataset = SemanticKITTIDataset(
                dataset_path=target_dataset_path,
                mapping_path=target_mapping_path,
                version=version,
                phase="validation",
                voxel_size=voxel_size,
                augment_data=False,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )
            training1_dataset.class2names = classnamearray
            validation1_dataset.class2names = classnamearray

            training2_dataset.class2names = classnamearray
            validation2_dataset.class2names = classnamearray

            target_dataset.class2names = classnamearray

            training1_dataset.color_map = classcolorarray
            validation1_dataset.color_map = classcolorarray
            training2_dataset.color_map = classcolorarray
            validation2_dataset.color_map = classcolorarray
            target_dataset.color_map = classcolorarray

        if target_name == "NuScenes":
            main_path, _ = os.path.split(dataset_path)
            target_dataset_path = "/data4/vaibhav/lidardata/nuscenes/"

            target_mapping_path = "_resources/nuscenes.yaml"
            target_dataset = NuScenesDataset(
                dataset_path=target_dataset_path,
                mapping_path=target_mapping_path,
                version=version,
                phase="train",
                voxel_size=voxel_size,
                augment_data=False,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )

            training1_dataset.class2names = classnamearray
            validation1_dataset.class2names = classnamearray

            training2_dataset.class2names = classnamearray
            validation2_dataset.class2names = classnamearray

            target_dataset.class2names = classnamearray

            training1_dataset.color_map = classcolorarray
            validation1_dataset.color_map = classcolorarray
            training2_dataset.color_map = classcolorarray
            validation2_dataset.color_map = classcolorarray
            target_dataset.color_map = classcolorarray

    else:
        raise NotImplementedError
    return (
        training1_dataset,
        validation1_dataset,
        training2_dataset,
        validation2_dataset,
        target_dataset,
    )


def get_concat_dataset(
    source1_dataset,
    source2_dataset,
    target_dataset,
    augment_data=False,
    augment_mask_data=False,
    remove_overlap=False,
):
    return ConcatDataset(
        source1_dataset=source1_dataset,
        source2_dataset=source2_dataset,
        target_dataset=target_dataset,
        augment_mask_data=augment_mask_data,
        augment_data=augment_data,
        remove_overlap=remove_overlap,
    )
