import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn import functional as F
from torch.optim import Adam, SGD
from sklearn.metrics import jaccard_score
import open3d as o3d
import MinkowskiEngine as ME
from datetime import datetime
from scipy.spatial import cKDTree
import csv
from tqdm import tqdm
import sys
import pandas as pd

torch.manual_seed(42)  # Set seed for CPU
torch.cuda.manual_seed(42)  # Set seed for current GPU
np.random.seed(42)

sys.path.append("/data4/vaibhav/musmix/")
sys.path.append("/data4/vaibhav/musmix/utils_17/")

from models.minkunet import MinkUNet34
from datasets.initialization import get_concat_dataset, get_dataset
from collation.collation import CollateMerged, CollateFN
from losses.losses import SoftDICELoss

try:
    synlidar_mapping_path = "_resources/synlidar_semantickitti.yaml"
except AttributeError("No mapping found!"):
    synlidar_mapping_path = None

selection_perc = 0.6
print(f"Selection ratio: {selection_perc}", flush=True)
project_name = "23_sept_mod_mix"

torch.cuda.set_device(0)
print("CURRENT CUDA DEVICE:\t cuda:", torch.cuda.current_device())
device = "cuda:0"

(
    training1_dataset,
    validation1_dataset,
    training2_dataset,
    validation2_dataset,
    target_dataset,
) = get_dataset(
    dataset_name="SynLiDAR",
    dataset_path="/data3/vaibhav/SynLiDAR/sequences",
    dataset2_name="SemanticPOSS",
    dataset2_path="/data4/vaibhav/lidardata/SemanticPOSS_dataset/dataset/sequences",
    voxel_size=0.05,
    sub_num=50000,
    augment_data=True,
    version="full",
    num_classes=17,
    ignore_label=-1,
    mapping_path=synlidar_mapping_path,
    target_name="SemanticKITTI",
    weights_path=None,
)

train_collation = CollateMerged()
validation_collation = CollateFN()

training_dataset = get_concat_dataset(
    source1_dataset=training1_dataset,
    source2_dataset=training2_dataset,
    target_dataset=target_dataset,
    augment_data=True,
    augment_mask_data=True,
)

training_loader = DataLoader(
    training_dataset,
    batch_size=3,
    drop_last=True,
    collate_fn=train_collation,
    pin_memory=True,
    shuffle=True,
    num_workers=4,
)

# validation1_loader = DataLoader(
#     validation1_dataset,
#     batch_size=8,
#     collate_fn=validation_collation,
#     pin_memory=True,
#     shuffle=True,
#     num_workers=16,
# )

# validation2_loader = DataLoader(
#     validation2_dataset,
#     batch_size=4,
#     collate_fn=validation_collation,
#     pin_memory=True,
#     shuffle=False,
#     num_workers=16,
# )

target_validation_loader = DataLoader(
    target_dataset,
    batch_size=8,
    collate_fn=validation_collation,
    pin_memory=True,
    shuffle=False,
    num_workers=4,
)

teacher_model = MinkUNet34(1, 17)
# teacher_model_state_dict = torch.load(
#     "/data4/vaibhav/musmix/output/model_checkpoints/adaptation/teacher_23_sept_mod_mix_0_374_05_11_2024_16.pth",
#     map_location=device,
# )
# teacher_model.load_state_dict(teacher_model_state_dict)
teacher_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(teacher_model)
# print("Teacher Model loaded:\n", teacher_model)

student_model = MinkUNet34(1, 17)
# student_model_state_dict = torch.load(
#     "/data4/vaibhav/musmix/output/model_checkpoints/adaptation/student_23_sept_mod_mix_0_374_05_11_2024_16.pth",
#     map_location=device,
# )
# student_model.load_state_dict(student_model_state_dict)
student_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(student_model)
# print("Student Model loaded:\n", student_model)

# loss functions
sd_loss = SoftDICELoss(ignore_label=-1, is_kitti=False)
entropy = CrossEntropyLoss(
    ignore_index=-1,
)
kl_div = KLDivLoss(reduction="mean")
target_confidence_th = 0.85


def sample_classes(origin_classes, num_classes, origin_weights=None, is_pseudo=False):
    if not is_pseudo:
        if origin_weights is not None:
            sampling_weights = origin_weights[origin_classes] * (
                1 / origin_weights[origin_classes].sum()
            )
            sampling_weights = 1 - sampling_weights + 1e-2

            selected_classes = np.random.choice(
                origin_classes, num_classes, replace=False, p=sampling_weights
            )
        else:
            selected_classes = np.random.choice(
                origin_classes, num_classes, replace=False
            )
    else:
        selected_classes = origin_classes
    return selected_classes


def remove_occluded_pts(origin_pts, dest_pts, radius=1):
    """
    :makes kdtree of dest_pts
    :remove the origin_pts neighborhood from dest_pts
    """
    dest_tree = cKDTree(dest_pts)
    indices_to_remove = dest_tree.query_ball_point(origin_pts, radius)
    indices_to_remove = np.unique(np.hstack(indices_to_remove)).astype(int)

    return indices_to_remove


def random_sample(points, sub_num):
    """
    :param points: input points of shape [N, 3]
    :return: np.ndarray of N' points sampled from input points
    """

    num_points = points.shape[0]

    if sub_num is not None:
        if sub_num <= num_points:
            sampled_idx = np.random.choice(
                np.arange(num_points), sub_num, replace=False
            )
        else:
            over_idx = np.random.choice(
                np.arange(num_points), sub_num - num_points, replace=False
            )
            sampled_idx = np.concatenate([np.arange(num_points), over_idx])
    else:
        sampled_idx = np.arange(num_points)

    return sampled_idx


def mask(
    origin_pts,
    origin_labels,
    origin_features,
    dest_pts,
    dest_labels,
    dest_features,
    is_pseudo=False,
):
    mask = np.ones(dest_pts.shape[0], dtype=np.bool_)
    if (origin_labels == -1).sum() < origin_labels.shape[0]:
        origin_present_classes = np.unique(origin_labels)
        origin_present_classes = origin_present_classes[origin_present_classes != -1]
        # print("origin_present_classes", origin_present_classes)

        dest_present_classes = np.unique(dest_labels)
        dest_present_classes = dest_present_classes[dest_present_classes != -1]
        # print("dest_present_classes", dest_present_classes)

        # print("Condition on class selection", len(origin_present_classes), len(dest_present_classes))

        if len(origin_present_classes) < len(dest_present_classes):
            num_classes = origin_present_classes.shape[0]
        else:
            num_classes = int(selection_perc * origin_present_classes.shape[0])

        selected_classes = sample_classes(
            origin_present_classes, num_classes, is_pseudo=is_pseudo
        )
        # print("Selected classes:", selected_classes)

        selected_idx = []
        selected_pts = []
        selected_labels = []
        selected_features = []

        if not training_dataset.augment_mask_data:
            for sc in selected_classes:
                class_idx = np.where(origin_labels == sc)[0]
                selected_idx.append(class_idx)
                selected_pts.append(origin_pts[class_idx])
                selected_labels.append(origin_labels[class_idx])
                selected_features.append(origin_features[class_idx])
            if len(selected_pts) > 0:
                # selected_idx = np.concatenate(selected_idx, axis=0)
                selected_pts = np.concatenate(selected_pts, axis=0)
                selected_labels = np.concatenate(selected_labels, axis=0)
                selected_features = np.concatenate(selected_features, axis=0)
        else:
            for sc in selected_classes:
                class_idx = np.where(origin_labels == sc)[0]
                class_pts = origin_pts[class_idx]

                # num_pts = class_pts.shape[0]
                # sub_num = int(0.9 * num_pts)

                # # random subsample
                # random_idx = random_sample(class_pts, sub_num=sub_num)
                # class_idx = class_idx[random_idx]
                # class_pts = class_pts[random_idx]

                voxel_mtx, affine_mtx = (
                    training_dataset.mask_voxelizer.get_transformation_matrix()
                )
                rigid_transformation = affine_mtx @ voxel_mtx
                # apply transformations
                homo_coords = np.hstack(
                    (
                        class_pts,
                        np.ones((class_pts.shape[0], 1), dtype=class_pts.dtype),
                    )
                )
                class_pts = homo_coords @ rigid_transformation.T[:, :3]
                class_labels = (
                    np.ones(
                        origin_labels[class_idx].shape[0], dtype=origin_labels.dtype
                    )
                    * sc
                )
                # class_labels = np.ones_like(origin_labels[class_idx])*sc
                # print("origin_class_labels", class_labels[:10], class_labels.shape, sc)
                class_features = origin_features[class_idx]
                # print(class_pts.shape, class_features.shape, class_labels.shape)

                selected_idx.append(class_idx)
                selected_pts.append(class_pts)
                selected_labels.append(class_labels)
                selected_features.append(class_features)

            if len(selected_pts) > 0:
                selected_idx = np.concatenate(selected_idx, axis=0)
                selected_pts = np.concatenate(selected_pts, axis=0)
                selected_labels = np.concatenate(selected_labels, axis=0)
                selected_features = np.concatenate(selected_features, axis=0)

        if len(selected_pts) > 0:
            dest_idx = dest_pts.shape[0]
            # print("height minimas in dest, selected pts before diff:", np.sort(dest_pts[:, -1])[90:100], np.sort(selected_pts[:, -1])[90:100])
            # diff_height = (
            #     np.sort(selected_pts[:, -1])[100] - np.sort(dest_pts[:, -1])[100]
            # )
            # selected_pts[:, -1] = selected_pts[:, -1] - diff_height
            remove_indices = remove_occluded_pts(selected_pts, dest_pts)
            dest_pts = np.delete(dest_pts, remove_indices, axis=0)
            dest_features = np.delete(dest_features, remove_indices, axis=0)
            dest_labels = np.delete(dest_labels, remove_indices, axis=0)
            # print("height minimas in dest, selected pts after diff:", np.min(dest_pts[:, -1]), np.min(selected_pts[:, -1]))
            dest_pts = np.concatenate([dest_pts, selected_pts], axis=0)
            # print("Masking pts shape", dest_pts, "\n", selected_pts)
            dest_labels = np.concatenate([dest_labels, selected_labels], axis=0)
            dest_features = np.concatenate([dest_features, selected_features], axis=0)

            mask = np.ones(dest_pts.shape[0])
            mask[:dest_idx] = 0

        if training_dataset.augment_data:
            # get transformation
            voxel_mtx, affine_mtx = (
                training_dataset.voxelizer.get_transformation_matrix()
            )
            rigid_transformation = affine_mtx @ voxel_mtx
            # apply transformations
            homo_coords = np.hstack(
                (dest_pts, np.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype))
            )
            dest_pts = homo_coords @ rigid_transformation.T[:, :3]

    return dest_pts, dest_labels, dest_features, mask.astype(np.bool_)


def mask_data(batch, device, batch_idx, save_mixing=True):

    print(f"__MAKING MASKED DATA__ for BATCH:{batch_idx}")

    batch_source1_pts = batch["source1_coordinates"].cpu().numpy()
    # print("BATCH_S1_PTS:", batch_source1_pts.shape)
    batch_source1_labels = batch["source1_labels"].cpu().numpy()
    # print(f"batch_source1_labels:", np.unique(batch_source1_labels))
    batch_source1_features = batch["source1_features"].cpu().numpy()
    # print("Batch_source2_idx", batch["source2_coordinates"][:, 0].cpu().numpy())
    # print("Batch_source1_idx", batch["source1_coordinates"][:, 0].cpu().numpy())
    batch_source2_idx = batch["source2_coordinates"][:, 0].cpu().numpy()
    batch_source2_pts = batch["source2_coordinates"].cpu().numpy()
    # print("BATCH_S2_PTS:", batch_source2_pts.shape)
    batch_source2_labels = batch["source2_labels"].cpu().numpy()
    # print(f"batch_source2_labels:", np.unique(batch_source2_labels))
    batch_source2_features = batch["source2_features"].cpu().numpy()

    batch_target_idx = batch["target_coordinates"][:, 0].cpu().numpy()
    batch_target_pts = batch["target_coordinates"].cpu().numpy()
    batch_target_features = batch["target_features"].cpu().numpy()
    batch_target_labels = batch["pseudo_labels"].cpu().numpy()

    # print("Labels shape:", batch_source1_labels.shape, batch_source2_labels.shape)
    batch_size = int(np.max(batch_source2_idx).item() + 1)
    # print("batch size in mask_data a/c to source 2:", batch_size)

    new_batch = {
        # "masked_source2_pts": [],
        # "masked_source2_labels": [],
        # "masked_source2_features": [],
        # "masked_source1_pts": [],
        # "masked_source1_labels": [],
        # "masked_source1_features": [],
        "masked_target1_pts": [],
        "masked_target1_labels": [],
        "masked_target1_features": [],
        "masked_target2_pts": [],
        "masked_target2_labels": [],
        "masked_target2_features": [],
    }

    target_order = np.arange(batch_size)

    for batch in range(batch_size):
        source1_batch_idx = batch_source1_pts[:, 0] == batch
        source2_batch_idx = batch_source2_pts[:, 0] == batch
        target_batch = target_order[batch]
        target_batch_idx = batch_target_idx == target_batch

        source1_pts = (
            batch_source1_pts[source1_batch_idx, 1:] * training_dataset.voxel_size
        )
        source1_labels = batch_source1_labels[source1_batch_idx]
        source1_features = batch_source1_features[source1_batch_idx]

        source2_pts = (
            batch_source2_pts[source2_batch_idx, 1:] * training_dataset.voxel_size
        )
        source2_labels = batch_source2_labels[source2_batch_idx]
        source2_features = batch_source2_features[source2_batch_idx]

        target_pts = (
            batch_target_pts[target_batch_idx, 1:] * training_dataset.voxel_size
        )
        target_labels = batch_target_labels[target_batch_idx]
        target_features = batch_target_features[target_batch_idx]

        (
            masked_target1_pts,
            masked_target1_labels,
            masked_target1_features,
            masked_target1_mask,
        ) = mask(
            origin_pts=source1_pts,
            origin_labels=source1_labels.reshape(-1),
            origin_features=source1_features,
            dest_pts=target_pts,
            dest_labels=target_labels.reshape(-1),
            dest_features=target_features,
        )

        (
            masked_target2_pts,
            masked_target2_labels,
            masked_target2_features,
            masked_target2_mask,
        ) = mask(
            origin_pts=source2_pts,
            origin_labels=source2_labels.reshape(-1),
            origin_features=source2_features,
            dest_pts=target_pts,
            dest_labels=target_labels.reshape(-1),
            dest_features=target_features,
        )

        # print("_"*20)
        # print(f"Shapes of masked_target2_pts: {masked_target2_pts.shape}, masked_target2_features: {masked_target2_features.shape}")
        # print(f"Shapes of masked_target1_pts: {masked_target1_pts.shape}, masked_target1_features: {masked_target1_features.shape}")

        # (
        #     masked_source1_pts,
        #     masked_source1_labels,
        #     masked_source1_features,
        #     masked_source1_mask,
        # ) = mask(
        #     origin_pts=target_pts,
        #     origin_labels=target_labels.reshape(-1),
        #     origin_features=target_features,
        #     dest_pts=source1_pts,
        #     dest_labels=source1_labels.reshape(-1),
        #     dest_features=source1_features,
        #     is_pseudo=True,
        # )

        # (
        #     masked_source2_pts,
        #     masked_source2_labels,
        #     masked_source2_features,
        #     masked_source2_mask,
        # ) = mask(
        #     origin_pts=target_pts,
        #     origin_labels=target_labels.reshape(-1),
        #     origin_features=target_features,
        #     dest_pts=source2_pts,
        #     dest_labels=source2_labels.reshape(-1),
        #     dest_features=source2_features,
        #     is_pseudo=True,
        # )

        # print("Unique labels after mixing: ", np.unique(masked_source1_labels), np.unique(masked_source2_labels))

        if save_mixing and batch_idx % 25 == 0:
            # print(batch)
            os.makedirs("/data4/vaibhav/musmix/output/vis_mix", exist_ok=True)
            os.makedirs(
                f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/ttos2",
                exist_ok=True,
            )
            os.makedirs(
                f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/ttos1",
                exist_ok=True,
            )
            os.makedirs(
                f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/s1tot",
                exist_ok=True,
            )
            os.makedirs(
                f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/s2tot",
                exist_ok=True,
            )
            os.makedirs(
                f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/source2",
                exist_ok=True,
            )
            os.makedirs(
                f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/source1",
                exist_ok=True,
            )

            # source1_pcd = o3d.geometry.PointCloud()
            # valid_source1 = source1_labels != -1
            # valid_source2 = source2_labels != -1
            # valid_masked_source1 = masked_source1_labels != -1
            # valid_masked_source2 = masked_source2_labels != -1
            # # print("Valid source",valid_source.shape, source_labels.shape)
            # # print(self.source_validation_dataset.color_map.shape)
            # source1_pcd.points = o3d.utility.Vector3dVector(source1_pts[valid_source1])
            # # print("source1 colormap:", source1_labels[valid_source1], training_dataset.colormap[source1_labels[valid_source1] + 1], source1_labels[source1_labels==5], training_dataset.colormap[source1_labels[source1_labels==5]+1])
            # source1_pcd.colors = o3d.utility.Vector3dVector(
            #     training_dataset.colormap[source1_labels[valid_source1] + 1]
            # )  # remove +1
            # # source1_pcd.labels = source1_labels[valid_source1]

            # source2_pcd = o3d.geometry.PointCloud()
            # source2_pcd.points = o3d.utility.Vector3dVector(source2_pts[valid_source2])
            # # print("source2_pcd Colormap:", source2_labels[valid_source2], training_dataset.colormap[source2_labels[valid_source2]+1])
            # source2_pcd.colors = o3d.utility.Vector3dVector(
            #     training_dataset.colormap[source2_labels[valid_source2] + 1]
            # )
            # # source2_pcd.labels = source2_labels[valid_source2]

            # target_pcd = o3d.geometry.PointCloud()
            # target_pcd.points = o3d.utility.Vector3dVector(target_pts)
            # target_pcd.colors = o3d.utility.Vector3dVector(
            #     training_dataset.colormap[target_labels + 1]
            # )

            # s1tot_pcd = o3d.geometry.PointCloud()
            # s1tot_pcd.points = o3d.utility.Vector3dVector(masked_target1_pts)
            # # print("s1tos2 colormap:", masked_source2_labels[valid_masked_source2], training_dataset.colormap[masked_source2_labels[valid_masked_source2] + 1])
            # s1tot_pcd.colors = o3d.utility.Vector3dVector(
            #     training_dataset.colormap[masked_target1_labels + 1]
            # )
            # # s1tos2_pcd.labels = masked_source2_labels[valid_masked_source2]

            # ttos1_pcd = o3d.geometry.PointCloud()
            # # valid_source1 = masked_source1_labels != -1
            # ttos1_pcd.points = o3d.utility.Vector3dVector(
            #     masked_source1_pts[valid_masked_source1]
            # )
            # ttos1_pcd.colors = o3d.utility.Vector3dVector(
            #     training_dataset.colormap[
            #         masked_source1_labels[valid_masked_source1] + 1
            #     ]
            # )

            # s2tot_pcd = o3d.geometry.PointCloud()
            # s2tot_pcd.points = o3d.utility.Vector3dVector(masked_target2_pts)
            # # print("s1tos2 colormap:", masked_source2_labels[valid_masked_source2], training_dataset.colormap[masked_source2_labels[valid_masked_source2] + 1])
            # s2tot_pcd.colors = o3d.utility.Vector3dVector(
            #     training_dataset.colormap[masked_target2_labels + 1]
            # )
            # # s1tos2_pcd.labels = masked_source2_labels[valid_masked_source2]

            # ttos2_pcd = o3d.geometry.PointCloud()
            # # valid_source1 = masked_source1_labels != -1
            # ttos2_pcd.points = o3d.utility.Vector3dVector(
            #     masked_source2_pts[valid_masked_source2]
            # )
            # ttos2_pcd.colors = o3d.utility.Vector3dVector(
            #     training_dataset.colormap[
            #         masked_source2_labels[valid_masked_source2] + 1
            #     ]
            # )

            # # s2tos1_pcd.labels = masked_source1_labels[valid_masked_source1]

            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/source1/source1_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     source1_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/source2/source2_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     source2_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/ttos2/ttos2_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     ttos2_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/ttos1/ttos1_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     ttos1_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/s1tot/s1tot_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     s1tot_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/s2tot/s2tot_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     s2tot_pcd,
            # )

            # os.makedirs(
            #     f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/s1tot_mask",
            #     exist_ok=True,
            # )
            # os.makedirs(
            #     f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/s2tot_mask",
            #     exist_ok=True,
            # )
            # os.makedirs(
            #     f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/ttos1_mask",
            #     exist_ok=True,
            # )
            # os.makedirs(
            #     f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/ttos2_mask",
            #     exist_ok=True,
            # )
            # os.makedirs(
            #     f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/source1_mask",
            #     exist_ok=True,
            # )
            # os.makedirs(
            #     f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/adaptation/source2_mask",
            #     exist_ok=True,
            # )

            # source1_pcd.paint_uniform_color([1, 0.706, 0])
            # source2_pcd.paint_uniform_color([0, 0.651, 0.929])
            # target_pcd.paint_uniform_color([0.7, 0.354, 0.555])

            # ttos2_pcd = o3d.geometry.PointCloud()
            # ttos2_pcd.points = o3d.utility.Vector3dVector(
            #     masked_source2_pts[valid_masked_source2]
            # )
            # masked_source2_mask = masked_source2_mask[valid_masked_source2]
            # ttos2_colors = np.zeros_like(masked_source2_pts[valid_masked_source2])
            # ttos2_colors[masked_source2_mask] = [0.7, 0.354, 0.555]
            # ttos2_colors[np.logical_not(masked_source2_mask)] = [0, 0.651, 0.929]
            # ttos2_pcd.colors = o3d.utility.Vector3dVector(ttos2_colors)

            # ttos1_pcd = o3d.geometry.PointCloud()
            # # valid_source1 = masked_source1_labels != -1
            # ttos1_pcd.points = o3d.utility.Vector3dVector(
            #     masked_source1_pts[valid_masked_source1]
            # )
            # ttos1_colors = np.zeros_like(masked_source1_pts[valid_masked_source1])
            # masked_source1_mask = masked_source1_mask[valid_masked_source1]
            # ttos1_colors[masked_source1_mask] = [0, 0.651, 0.929]
            # ttos1_colors[np.logical_not(masked_source1_mask)] = [1, 0.706, 0]
            # ttos1_pcd.colors = o3d.utility.Vector3dVector(ttos1_colors)

            # s1tot_pcd = o3d.geometry.PointCloud()
            # s1tot_pcd.points = o3d.utility.Vector3dVector(
            #     masked_target1_pts
            # )
            # s1tot_colors = np.zeros_like(masked_target1_pts)
            # s1tot_colors[masked_target1_mask] = [1, 0.706, 0]
            # s1tot_colors[np.logical_not(masked_target1_mask)] = [0.7, 0.354, 0.555]
            # s1tot_pcd.colors = o3d.utility.Vector3dVector(s1tot_colors)

            # s2tot_pcd = o3d.geometry.PointCloud()
            # s2tot_pcd.points = o3d.utility.Vector3dVector(
            #     masked_target2_pts
            # )
            # s2tot_colors = np.zeros_like(masked_target2_pts)
            # s2tot_colors[masked_target2_mask] = [1, 0.706, 0]
            # s2tot_colors[np.logical_not(masked_target2_mask)] = [0, 0.651, 0.929]
            # s2tot_pcd.colors = o3d.utility.Vector3dVector(s2tot_colors)

            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/source1_mask/source1_mask_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     source1_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/source2_mask/source2_mask_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     source2_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/s1tos2_mask/s1tos2_mask_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     s1tos2_pcd,
            # )
            # o3d.io.write_point_cloud(
            #     f'/data4/vaibhav/musmix/output/vis_mix/{project_name}/s2tos1_mask/s2tos1_mask_{project_name}_{batch_idx}_{datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S_%MS")}_{batch}.ply',
            #     s2tos1_pcd,
            # )

        # _, _, _, masked_source2_voxel_idx = ME.utils.sparse_quantize(
        #     coordinates=masked_source2_pts,
        #     features=masked_source2_features,
        #     labels=masked_source2_labels,
        #     quantization_size=training_dataset.voxel_size,
        #     return_index=True,
        # )

        # _, _, _, masked_source1_voxel_idx = ME.utils.sparse_quantize(
        #     coordinates=masked_source1_pts,
        #     features=masked_source1_features,
        #     labels=masked_source1_labels,
        #     quantization_size=training_dataset.voxel_size,
        #     return_index=True,
        # )

        _, _, _, masked_target1_voxel_idx = ME.utils.sparse_quantize(
            coordinates=masked_target1_pts,
            features=masked_target1_features,
            labels=masked_target1_labels,
            quantization_size=training_dataset.voxel_size,
            return_index=True,
        )

        _, _, _, masked_target2_voxel_idx = ME.utils.sparse_quantize(
            coordinates=masked_target2_pts,
            features=masked_target2_features,
            labels=masked_target2_labels,
            quantization_size=training_dataset.voxel_size,
            return_index=True,
        )

        # masked_source2_pts = masked_source2_pts[masked_source2_voxel_idx]
        # masked_source2_labels = masked_source2_labels[masked_source2_voxel_idx]
        # masked_source2_features = masked_source2_features[masked_source2_voxel_idx]

        # masked_source1_pts = masked_source1_pts[masked_source1_voxel_idx]
        # masked_source1_labels = masked_source1_labels[masked_source1_voxel_idx]
        # masked_source1_features = masked_source1_features[masked_source1_voxel_idx]

        masked_target1_pts = masked_target1_pts[masked_target1_voxel_idx]
        masked_target1_labels = masked_target1_labels[masked_target1_voxel_idx]
        masked_target1_features = masked_target1_features[masked_target1_voxel_idx]

        masked_target2_pts = masked_target2_pts[masked_target2_voxel_idx]
        masked_target2_labels = masked_target2_labels[masked_target2_voxel_idx]
        masked_target2_features = masked_target2_features[masked_target2_voxel_idx]

        # masked_source2_pts = np.floor(masked_source2_pts / training_dataset.voxel_size)
        # masked_source1_pts = np.floor(masked_source1_pts / training_dataset.voxel_size)
        masked_target1_pts = np.floor(masked_target1_pts / training_dataset.voxel_size)
        masked_target2_pts = np.floor(masked_target2_pts / training_dataset.voxel_size)

        # batch_index = np.ones([masked_source2_pts.shape[0], 1]) * batch
        # masked_source2_pts = np.concatenate([batch_index, masked_source2_pts], axis=-1)

        # batch_index = np.ones([masked_source1_pts.shape[0], 1]) * batch
        # masked_source1_pts = np.concatenate([batch_index, masked_source1_pts], axis=-1)

        batch_index = np.ones([masked_target1_pts.shape[0], 1]) * batch
        masked_target1_pts = np.concatenate([batch_index, masked_target1_pts], axis=-1)

        batch_index = np.ones([masked_target2_pts.shape[0], 1]) * batch
        masked_target2_pts = np.concatenate([batch_index, masked_target2_pts], axis=-1)

        # new_batch["masked_source2_pts"].append(masked_source2_pts)
        # new_batch["masked_source2_labels"].append(masked_source2_labels)
        # new_batch["masked_source2_features"].append(masked_source2_features)
        # new_batch["masked_source1_pts"].append(masked_source1_pts)
        # new_batch["masked_source1_labels"].append(masked_source1_labels)
        # new_batch["masked_source1_features"].append(masked_source1_features)
        new_batch["masked_target1_pts"].append(masked_target1_pts)
        new_batch["masked_target1_labels"].append(masked_target1_labels)
        new_batch["masked_target1_features"].append(masked_target1_features)
        new_batch["masked_target2_pts"].append(masked_target2_pts)
        new_batch["masked_target2_labels"].append(masked_target2_labels)
        new_batch["masked_target2_features"].append(masked_target2_features)

    for k, i in new_batch.items():
        if k in [
            # "masked_source2_pts",
            # "masked_source2_features",
            # "masked_source1_pts",
            # "masked_source1_features",
            "masked_target1_pts",
            "masked_target1_features",
            "masked_target2_pts",
            "masked_target2_features",
        ]:
            new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0)).to(device)
        else:
            print(f"{k}", flush=True)
            new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0))

    return new_batch


def training_step(teacher_model, student_model, batch, batch_id, global_epoch):
    torch.cuda.empty_cache()
    student_model.train()
    target_stensor = ME.SparseTensor(
        coordinates=batch["target_coordinates"].int(),
        features=batch["target_features"],
        device=device,
    )

    print("__CALCULATING PSEUDO_LABELS__", flush=True)

    with torch.no_grad():
        teacher_model.eval()
        target_pseudo = teacher_model(target_stensor).F.cpu()
        target_pseudo = F.softmax(target_pseudo, dim=-1)
        target_confidence, target_pseudo = target_pseudo.max(dim=-1)
        filtered_target_pseudo = -torch.ones_like(target_pseudo)
        valid_idx = target_confidence > target_confidence_th
        filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
        target_pseudo = filtered_target_pseudo.long()

    print("__CALCULATED PSEUDO_LABELS__", flush=True)

    batch["pseudo_labels"] = target_pseudo
    masked_batch = mask_data(batch, device, batch_id)

    print("__GOT THE MASKED BATCH__", flush=True)

    s1tot_stensor = ME.SparseTensor(
        coordinates=masked_batch["masked_target1_pts"].int(),
        features=masked_batch["masked_target1_features"],
    )
    s1tot_labels = masked_batch["masked_target1_labels"].long()
    s1tot_out = student_model(s1tot_stensor).F.cpu()

    s2tot_stensor = ME.SparseTensor(
        coordinates=masked_batch["masked_target2_pts"].int(),
        features=masked_batch["masked_target2_features"],
    )
    s2tot_labels = masked_batch["masked_target2_labels"].long()
    s2tot_out = student_model(s2tot_stensor).F.cpu()

    s1tot_loss = sd_loss(s1tot_out, s1tot_labels)
    s2tot_loss = sd_loss(s2tot_out, s2tot_labels)
    student_loss = s1tot_loss + s2tot_loss

    teacher_model.train()
    teacher_target_out = teacher_model(target_stensor).F.cpu()
    teacher_target_probs = F.softmax(teacher_target_out, dim=1)
    teacher_target_probs_clone = teacher_target_probs.clone().detach()
    student_target_out = student_model(target_stensor).F.cpu()
    student_output_prob_clone = (
        torch.softmax(student_target_out, dim=1).clone().detach() + 1e-8
    )
    teacher_output_log_prob = torch.log_softmax(teacher_target_out, dim=1)

    target_entropy = -torch.sum(
        teacher_target_probs * torch.log(teacher_target_probs_clone + 1e-10), dim=1
    )
    target_entropy = target_entropy.mean()
    teacher_student_div = kl_div(teacher_output_log_prob, student_output_prob_clone)
    teacher_loss = teacher_student_div + target_entropy
    print(
        f"TEACHER LOSS for batch: {batch_id}: {target_entropy}, {teacher_student_div}"
    )
    return teacher_model, student_model, teacher_loss, student_loss


def update_student_params(optim, student_loss):
    # Update student model
    optim.zero_grad()
    student_loss.backward()
    optim.step()


def update_teacher_params(optim, accumulated_losses):
    # Calculate average loss
    avg_loss = torch.mean(torch.stack(accumulated_losses))
    # Update teacher model with averaged loss
    optim.zero_grad()
    avg_loss.backward()
    optim.step()
    return avg_loss


def ema_update(teacher_model, student_model, alpha=0.999):
    print("__Updated EMA PARAMS__", flush=True)
    for teacher_param, student_param in zip(
        teacher_model.parameters(), student_model.parameters()
    ):
        teacher_param.data = (
            alpha * teacher_param.data + (1 - alpha) * student_param.data
        )


def validation_step(model, batch, batch_idx, phase, output_dir, epoch=0):
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        # Input batch
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"].int(),
            features=batch["features"],
            device=device,
        )

        out = model(stensor).F.cpu()
        labels = batch["labels"].long().cpu()

        loss = sd_loss(out, labels)

        soft_pseudo = F.softmax(out[:, :-1], dim=-1)
        conf, preds = soft_pseudo.max(1)

        # Calculate IoU
        iou_tmp = jaccard_score(
            preds.detach().numpy(),
            labels.numpy(),
            average=None,
            labels=np.arange(0, 17),
            zero_division=0.0,
        )

    # Get present labels and their occurrence
    present_labels, class_occurs = np.unique(labels.numpy(), return_counts=True)
    present_labels = present_labels[present_labels != -1]
    present_names = training_dataset.class2names[present_labels].tolist()

    # Create a results dictionary with NaN for missing classes
    results_dict = {
        f"{phase}/{name}_iou": iou
        for name, iou in zip(present_names, iou_tmp[present_labels])
    }

    # Handle missing classes by adding NaN for those not present
    for class_idx in range(17):
        if class_idx not in present_labels:
            results_dict[f"{phase}/{training_dataset.class2names[class_idx]}_iou"] = (
                np.nan
            )

    results_dict[f"{phase}/loss"] = loss.item()  # Ensure loss is a float
    results_dict[f"{phase}/mean_iou"] = np.nanmean(iou_tmp)  # Average IoU

    results_dict[f"{phase}/val_idx"] = batch_idx
    results_dict[f"{phase}/val_idx"] = epoch

    # Save results to CSV
    results_df = pd.DataFrame([results_dict])
    output_csv_path = os.path.join(output_dir, f"{phase}_metrics.csv")

    os.makedirs(output_dir, exist_ok=True)

    # Append to CSV if it exists, otherwise create a new one
    if os.path.exists(output_csv_path):
        results_df.to_csv(output_csv_path, mode="a", header=False, index=False)
    else:
        results_df.to_csv(output_csv_path, index=False)


def train_function(teacher_model, student_model, training_dataloader, epochs=5):
    torch.cuda.empty_cache()
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    teacher_model.eval()
    student_model.train()
    teacher_optim = Adam(teacher_model.parameters(), lr=0.001, betas=(0.9, 0.999))
    student_optim = Adam(student_model.parameters(), lr=0.001, betas=(0.9, 0.999))

    accumulated_teacher_losses = []

    for epoch in range(epochs):

        print("_" * 50, flush=True)
        print(f"__TRAINING__:{epoch} out of {epochs}", flush=True)

        for train_idx, train_batch in tqdm(
            enumerate(training_dataloader), desc=f"Training:{epoch}/{epochs}"
        ):
            print("_" * 50, flush=True)

            teacher_model.eval()
            teacher_model, student_model, teacher_loss, student_loss = training_step(
                teacher_model, student_model, train_batch, train_idx, epoch
            )

            # Always update student model
            update_student_params(student_optim, student_loss)

            # Store teacher loss for accumulation
            # We detach the loss to prevent gradient accumulation in computational graph
            accumulated_teacher_losses.append(teacher_loss.detach().requires_grad_())

            # Update teacher model every 25 iterations
            if (train_idx + 1) % 1 == 0:
                avg_teacher_loss = update_teacher_params(
                    teacher_optim, accumulated_teacher_losses
                )
                print(
                    f"Updated teacher model at iteration {train_idx + 1} with average loss: {avg_teacher_loss.item()}"
                )
                # Clear accumulated losses
                accumulated_teacher_losses = []

            print(
                f"TRAIN EPOCH:{epoch} ITER: {train_idx} TEACHER_LOSS: {teacher_loss.detach().item()} STUDENT_LOSS: {student_loss.detach().item()}",
                flush=True,
            )

            if (train_idx + 1) % 5 == 0:
                ema_update(teacher_model, student_model, alpha=0.995)
                os.makedirs(
                    "/data4/vaibhav/musmix/output/model_checkpoints/adaptation",
                    exist_ok=True,
                )
                torch.save(
                    student_model.state_dict(),
                    f"/data4/vaibhav/musmix/output/model_checkpoints/adaptation/teacher_{project_name}_{epoch}_{train_idx}_{datetime.now().strftime(f'%d_%m_%Y_%H')}.pth",
                )
                torch.save(
                    student_model.state_dict(),
                    f"/data4/vaibhav/musmix/output/model_checkpoints/adaptation/student_{project_name}_{epoch}_{train_idx}_{datetime.now().strftime(f'%d_%m_%Y_%H')}.pth",
                )
                # for val_idx, val_batch in tqdm(
                #     enumerate(target_validation_loader), desc="Validation:"
                # ):
                #     validation_step(
                #         student_model,
                #         val_batch,
                #         val_idx,
                #         "target_validation",
                #         f"/data4/vaibhav/musmix/output/results/{project_name}/adaptation/",
                #         epoch,
                #     )

        # Handle any remaining accumulated losses at the end of epoch
        if accumulated_teacher_losses:
            avg_teacher_loss = update_teacher_params(
                teacher_optim, accumulated_teacher_losses
            )
            print(
                f"Updated teacher model at end of epoch with average loss: {avg_teacher_loss.item()}"
            )
            accumulated_teacher_losses = []

        # for val_idx, val_batch in tqdm(
        #     enumerate(target_validation_loader), desc="Validation"
        # ):
        #     validation_step(
        #         student_model,
        #         val_batch,
        #         val_idx,
        #         "target_validation",
        #         f"/data4/vaibhav/musmix/output/vis_mix/{project_name}/results/adaptation/",
        #         epoch,
        #     )

        print(f"__TRAINING__:{epoch} out of {epochs} Finished", flush=True)
        print("_" * 40, flush=True)
    print(f"__TRAINING__:{epoch} out of {epochs} Finished!", flush=True)


if __name__ == "__main__":
    train_function(teacher_model, student_model, training_loader)
