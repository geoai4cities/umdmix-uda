import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam, SGD
from sklearn.metrics import jaccard_score
import open3d as o3d
import MinkowskiEngine as ME
from datetime import datetime
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse

import sys

sys.path.append("/data4/vaibhav/musmix/")
sys.path.append("/data4/vaibhav/musmix/utils_17/")

from models.minkunet import MinkUNet34
from datasets.initialization import get_concat_dataset, get_dataset
from collation.collation import CollateMerged, CollateFN
from losses.losses import SoftDICELoss, get_neigbors_idx, PosAwareLoss
from common.config import get_config

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg_file", default="config/train_cfg.yaml", type=str, help="Path to config file"
)

args = parser.parse_args()
cfg = get_config(args.cfg_file)

torch.cuda.set_device(cfg.hwconfig.num_cuda)


def get_model(model, device, pretrained=False):
    def clean_state_dict(state):
        for k in list(ckpt.keys()):
            if "model" in k:
                ckpt[k.replace("model.", "")] = ckpt[k]
            del ckpt[k]
        return state

    if pretrained:
        ckpt_path = cfg.model.pretrained
        ckpt = torch.load(ckpt_path, map_location=device)["state_dict"]
        ckpt = clean_state_dict(ckpt)
        model.load_state_dict(ckpt)
    return model


class TrainPipeline:
    def __init__(self, pretrained=False):
        self.pa_loss = PosAwareLoss(ign_label=cfg.datasets.ignore_label)
        self.dice_loss = SoftDICELoss(ignore_label=cfg.datasets.ignore_label)
        self.pretrained = pretrained
        self.device = f"cuda:{cfg.hwconfig.num_cuda}"
        self.model = MinkUNet34(cfg.model.in_feat_size, cfg.model.out_classes)
        self.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model)
        self.epochs = cfg.pipelines.epochs
        self.optim_lr = cfg.pipelines.optim_lr

        if self.pretrained:
            ckpt_path = cfg.model.pretrained
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt)

    def get_data(self):
        (
            train_src1_data,
            valid_src1_data,
            train_src2_data,
            valid_src2_data,
            target_dataset,
        ) = get_dataset(
            dataset_name=cfg.datasets.source1_name,
            dataset_path=cfg.datasets.source1_path,
            dataset2_name=cfg.datasets.source2_name,
            dataset2_path=cfg.datasets.source2_path,
            voxel_size=cfg.datasets.voxel_size,
            sub_num=cfg.datasets.sub_num,
            augment_data=cfg.datasets.augment_data,
            version=cfg.datasets.version,
            num_classes=cfg.datasets.num_classes,
            ignore_label=cfg.datasets.ignore_label,
            mapping_path=cfg.datasets.source1_mapping_path,
            target_name=cfg.datasets.target_name,
            weights_path=None,
        )

        train_collation = CollateMerged()
        valid_collation = CollateFN()

        self.training_dataset = get_concat_dataset(
            train_src1_data,
            train_src2_data,
            target_dataset,
            augment_data=cfg.datasets.augment_data,
            augment_mask_data=cfg.datasets.augment_mask_data,
        )

        self.training_loader = DataLoader(
            self.training_dataset,
            batch_size=cfg.dataloaders.batch_size,
            collate_fn=train_collation,
            pin_memory=cfg.dataloaders.pin_memory,
            drop_last=True,
            shuffle=cfg.dataloaders.shuffle,
            num_workers=cfg.hwconfig.num_workers,
        )

        self.valid_src1_loader = DataLoader(
            valid_src1_data,
            batch_size=cfg.dataloaders.batch_size,
            collate_fn=valid_collation,
            pin_memory=cfg.dataloaders.pin_memory,
            drop_last=True,
            shuffle=cfg.dataloaders.shuffle,
            num_workers=cfg.hwconfig.num_workers,
        )

        self.valid_src2_loader = DataLoader(
            valid_src2_data,
            batch_size=cfg.dataloaders.batch_size,
            collate_fn=valid_collation,
            pin_memory=cfg.dataloaders.pin_memory,
            drop_last=True,
            shuffle=cfg.dataloaders.shuffle,
            num_workers=cfg.hwconfig.num_workers,
        )

    def remove_occluded_pts(self, origin_pts, dest_pts, radius=1):
        """
        :makes kdtree of dest_pts
        :remove the origin_pts neighborhood from dest_pts
        """
        dest_tree = cKDTree(dest_pts)
        indices_to_remove = dest_tree.query_ball_point(origin_pts, radius)
        indices_to_remove = np.unique(np.hstack(indices_to_remove)).astype(int)

        return indices_to_remove

    def sample_classes(self, origin_classes, num_classes, origin_weights=None):
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
        return selected_classes

    def random_sample(self, points, sub_num):
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
        self,
        origin_pts,
        origin_labels,
        origin_features,
        dest_pts,
        dest_labels,
        dest_features,
    ):
        mask = np.ones(dest_pts.shape[0], dtype=np.bool_)
        if (origin_labels == -1).sum() < origin_labels.shape[0]:
            origin_present_classes = np.unique(origin_labels)
            origin_present_classes = origin_present_classes[
                origin_present_classes != -1
            ]

            dest_present_classes = np.unique(dest_labels)
            dest_present_classes = dest_present_classes[dest_present_classes != -1]

            if len(origin_present_classes) < len(dest_present_classes):
                num_classes = origin_present_classes.shape[0]
            else:
                num_classes = int(
                    cfg.mask_fn.select_ratio * origin_present_classes.shape[0]
                )

            selected_classes = self.sample_classes(origin_present_classes, num_classes)

            selected_idx = []
            selected_pts = []
            selected_labels = []
            selected_features = []

            if not self.training_dataset.augment_mask_data:
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

                    voxel_mtx, affine_mtx = (
                        self.training_dataset.mask_voxelizer.get_transformation_matrix()
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
                    class_features = origin_features[class_idx]

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
                remove_indices = self.remove_occluded_pts(selected_pts, dest_pts)
                dest_pts = np.delete(dest_pts, remove_indices, axis=0)
                dest_features = np.delete(dest_features, remove_indices, axis=0)
                dest_labels = np.delete(dest_labels, remove_indices, axis=0)
                # diff_height = (
                #     np.sort(selected_pts[:, -1])[100] - np.sort(dest_pts[:, -1])[100]
                # )
                # selected_pts[:, -1] = selected_pts[:, -1] - diff_height
                dest_pts = np.concatenate([dest_pts, selected_pts], axis=0)
                dest_labels = np.concatenate([dest_labels, selected_labels], axis=0)
                dest_features = np.concatenate(
                    [dest_features, selected_features], axis=0
                )

                mask = np.ones(dest_pts.shape[0])
                mask[:dest_idx] = 0

            if self.training_dataset.augment_data:
                # get transformation
                voxel_mtx, affine_mtx = (
                    self.training_dataset.voxelizer.get_transformation_matrix()
                )
                rigid_transformation = affine_mtx @ voxel_mtx
                # apply transformations
                homo_coords = np.hstack(
                    (dest_pts, np.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype))
                )
                dest_pts = homo_coords @ rigid_transformation.T[:, :3]

        return dest_pts, dest_labels, dest_features, mask.astype(np.bool_)

    def mask_data(self, batch, batch_idx, device, save_mixing=False):
        batch_source1_pts = batch["source1_coordinates"].cpu().numpy()
        batch_source1_labels = batch["source1_labels"].cpu().numpy()
        batch_source1_features = batch["source1_features"].cpu().numpy()
        batch_source2_idx = batch["source2_coordinates"][:, 0].cpu().numpy()
        batch_source2_pts = batch["source2_coordinates"].cpu().numpy()
        batch_source2_labels = batch["source2_labels"].cpu().numpy()
        batch_source2_features = batch["source2_features"].cpu().numpy()

        batch_size = int(np.max(batch_source2_idx).item() + 1)

        new_batch = {
            "masked_source2_pts": [],
            "masked_source2_labels": [],
            "masked_source2_features": [],
            "masked_source1_pts": [],
            "masked_source1_labels": [],
            "masked_source1_features": [],
        }

        for batch in range(batch_size):
            source1_batch_idx = batch_source1_pts[:, 0] == batch
            source2_batch_idx = batch_source2_pts[:, 0] == batch

            source1_pts = (
                batch_source1_pts[source1_batch_idx, 1:] * cfg.datasets.voxel_size
            )
            source1_labels = batch_source1_labels[source1_batch_idx]
            source1_features = batch_source1_features[source1_batch_idx]

            source2_pts = (
                batch_source2_pts[source2_batch_idx, 1:] * cfg.datasets.voxel_size
            )
            source2_labels = batch_source2_labels[source2_batch_idx]
            source2_features = batch_source2_features[source2_batch_idx]

            (
                masked_source2_pts,
                masked_source2_labels,
                masked_source2_features,
                masked_source2_mask,
            ) = self.mask(
                origin_pts=source1_pts,
                origin_labels=source1_labels.reshape(-1),
                origin_features=source1_features,
                dest_pts=source2_pts,
                dest_labels=source2_labels.reshape(-1),
                dest_features=source2_features,
            )

            (
                masked_source1_pts,
                masked_source1_labels,
                masked_source1_features,
                masked_source1_mask,
            ) = self.mask(
                origin_pts=source2_pts,
                origin_labels=source2_labels.reshape(-1),
                origin_features=source2_features,
                dest_pts=source1_pts,
                dest_labels=source1_labels.reshape(-1),
                dest_features=source1_features,
            )

            _, _, _, masked_source2_voxel_idx = ME.utils.sparse_quantize(
                coordinates=masked_source2_pts,
                features=masked_source2_features,
                labels=masked_source2_labels,
                quantization_size=cfg.datasets.voxel_size,
                return_index=True,
            )

            _, _, _, masked_source1_voxel_idx = ME.utils.sparse_quantize(
                coordinates=masked_source1_pts,
                features=masked_source1_features,
                labels=masked_source1_labels,
                quantization_size=cfg.datasets.voxel_size,
                return_index=True,
            )

            masked_source2_pts = masked_source2_pts[masked_source2_voxel_idx]
            masked_source2_labels = masked_source2_labels[masked_source2_voxel_idx]
            masked_source2_features = masked_source2_features[masked_source2_voxel_idx]

            masked_source1_pts = masked_source1_pts[masked_source1_voxel_idx]
            masked_source1_labels = masked_source1_labels[masked_source1_voxel_idx]
            masked_source1_features = masked_source1_features[masked_source1_voxel_idx]

            masked_source2_pts = np.floor(masked_source2_pts / cfg.datasets.voxel_size)
            masked_source1_pts = np.floor(masked_source1_pts / cfg.datasets.voxel_size)

            batch_index = np.ones([masked_source2_pts.shape[0], 1]) * batch
            masked_source2_pts = np.concatenate(
                [batch_index, masked_source2_pts], axis=-1
            )

            batch_index = np.ones([masked_source1_pts.shape[0], 1]) * batch
            masked_source1_pts = np.concatenate(
                [batch_index, masked_source1_pts], axis=-1
            )

            new_batch["masked_source2_pts"].append(masked_source2_pts)
            new_batch["masked_source2_labels"].append(masked_source2_labels)
            new_batch["masked_source2_features"].append(masked_source2_features)
            new_batch["masked_source1_pts"].append(masked_source1_pts)
            new_batch["masked_source1_labels"].append(masked_source1_labels)
            new_batch["masked_source1_features"].append(masked_source1_features)

        for k, i in new_batch.items():
            if k in [
                "masked_source2_pts",
                "masked_source2_features",
                "masked_source1_pts",
                "masked_source1_features",
            ]:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0)).to(device)
            else:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0))
        return new_batch

    def train_step(self, batch, batch_idx, epoch):
        torch.cuda.empty_cache()
        self.model.train()

        masked_batch = self.mask_data(batch, batch_idx, self.device)

        s1tos2_tensor = ME.SparseTensor(
            coordinates=masked_batch["masked_source2_pts"].int(),
            features=masked_batch["masked_source2_features"],
            device=self.device,
        )
        s2tos1_tensor = ME.SparseTensor(
            coordinates=masked_batch["masked_source1_pts"].int(),
            features=masked_batch["masked_source1_features"],
            device=self.device,
        )

        s1tos2_labels = masked_batch["masked_source2_labels"]
        s1tos2_predictions = self.model(s1tos2_tensor).F.cpu()
        sd_s1s2 = self.dice_loss(s1tos2_predictions, s1tos2_labels.long())
        idx_s1s2 = get_neigbors_idx(s1tos2_tensor)
        posaware_s1s2 = self.pa_loss(s1tos2_predictions, s1tos2_labels, idx_s1s2)
        s1tos2_loss = sd_s1s2 + posaware_s1s2

        s2tos1_labels = masked_batch["masked_source1_labels"]
        s2tos1_predictions = self.model(s2tos1_tensor).F.cpu()
        sd_s2s1 = self.dice_loss(s2tos1_predictions, s2tos1_labels.long())
        idx_s2s1 = get_neigbors_idx(s2tos1_tensor)
        posaware_s1s2 = self.pa_loss(s2tos1_predictions, s2tos1_labels, idx_s2s1)
        s2tos1_loss = sd_s2s1 + posaware_s1s2

        final_loss = (
            cfg.datasets.s2s1_wt * s2tos1_loss + cfg.datasets.s1s2_wt * s1tos2_loss
        )

        with torch.no_grad():
            self.model.eval()

            s1tensor = ME.SparseTensor(
                coordinates=batch["source1_coordinates"].int(),
                features=batch["source1_features"],
                device=self.device,
            )
            s1labels = batch["source1_labels"].long().cpu()

            s2tensor = ME.SparseTensor(
                coordinates=batch["source2_coordinates"].int(),
                features=batch["source2_features"],
                device=self.device,
            )
            s2labels = batch["source2_labels"].long().cpu()

            out1 = self.model(s1tensor).F.cpu()
            soft_pseudo_1 = F.softmax(out1, dim=-1)
            _, preds_1 = soft_pseudo_1.max(1)

            out2 = self.model(s2tensor).F.cpu()
            soft_pseudo_2 = F.softmax(out2, dim=-1)
            _, preds_2 = soft_pseudo_2.max(1)

            iou_tmp_1 = jaccard_score(
                preds_1.detach().numpy(),
                s1labels.numpy(),
                average=None,
                labels=np.arange(0, 19),
                zero_division=0.0,
            )

            iou_tmp_2 = jaccard_score(
                preds_2.detach().numpy(),
                s2labels.numpy(),
                average=None,
                labels=np.arange(0, 19),
                zero_division=0.0,
            )

            all_labels = np.arange(-1, 19)
            eval_labels = all_labels[all_labels != -1]
            eval_names = self.training_dataset.class2names[eval_labels].tolist()
            eval_names = [f"{p}_iou" for p in eval_names]
            iou_dict_1 = {f"src1/{name}": np.nan for name in eval_names}
            iou_dict_2 = {f"src2/{name}": np.nan for name in eval_names}

            # Assign IoU values to corresponding class names
            for label, iou in zip(np.arange(0, len(iou_tmp_1)), iou_tmp_1):
                class_name = f"src1/{self.training_dataset.class2names[label]}_iou"
                if (
                    class_name in iou_dict_1
                ):  # Check if the class name exists in eval_names
                    iou_dict_1[class_name] = iou

            for label, iou in zip(np.arange(0, len(iou_tmp_2)), iou_tmp_2):
                class_name = f"src2/{self.training_dataset.class2names[label]}_iou"
                if (
                    class_name in iou_dict_2
                ):  # Check if the class name exists in eval_names
                    iou_dict_2[class_name] = iou

            iou_dict_1["meanIoU"] = np.nanmean(iou_tmp_1)
            iou_dict_1["train_loss_s1s2"] = s1tos2_loss.detach().item()
            iou_dict_1["train_loss_s2s1"] = s2tos1_loss.detach().item()
            iou_dict_1["total_train_loss"] = final_loss.detach().item()
            iou_dict_1["idx"] = batch_idx
            iou_dict_1["epoch"] = epoch

            iou_dict_2["meanIoU"] = np.nanmean(iou_tmp_2)
            iou_dict_2["train_loss_s1s2"] = s1tos2_loss.detach().item()
            iou_dict_2["train_loss_s2s1"] = s2tos1_loss.detach().item()
            iou_dict_2["total_train_loss"] = final_loss.detach().item()
            iou_dict_2["idx"] = batch_idx
            iou_dict_2["epoch"] = epoch

        self.model.train()
        return final_loss, iou_dict_1, iou_dict_2

    def validation(self, dataloader, phase, epoch, train_idx):
        torch.cuda.empty_cache()
        loop = tqdm(dataloader, leave=True, desc=f"Validation for {phase}")
        self.model.eval()
        iou_results = []
        with torch.no_grad():
            for val_idx, val_batch in enumerate(loop):
                stensor = ME.SparseTensor(
                    coordinates=val_batch["coordinates"].int(),
                    features=val_batch["features"],
                    device=self.device,
                )
                # print(model, stensor.device)
                out = self.model(stensor).F.cpu()
                labels = val_batch["labels"].long().cpu()
                loss = self.dice_loss(out, labels)

                soft_pseudo = F.softmax(out, dim=-1)
                conf, preds = soft_pseudo.max(1)

                iou_tmp = jaccard_score(
                    preds.detach().numpy(),
                    labels.numpy(),
                    average=None,
                    labels=np.arange(0, 19),
                    zero_division=0.0,
                )

                if val_idx % 10 == 0:
                    loop.set_postfix(
                        val_idx=val_idx,
                        meanIoU=np.nanmean(iou_tmp).item(),
                        val_loss=loss.item(),
                    )

                all_labels = np.arange(-1, 19)
                eval_labels = all_labels[all_labels != -1]
                eval_names = self.training_dataset.class2names[eval_labels].tolist()
                eval_names = [f"{p}_iou" for p in eval_names]
                iou_dict = {f"{phase}/{name}": np.nan for name in eval_names}

                # Assign IoU values to corresponding class names
                for label, iou in zip(np.arange(0, len(iou_tmp)), iou_tmp):
                    class_name = (
                        f"{phase}/{self.training_dataset.class2names[label]}_iou"
                    )
                    if (
                        class_name in iou_dict
                    ):  # Check if the class name exists in eval_names
                        iou_dict[class_name] = iou

                iou_dict["meanIoU"] = np.nanmean(iou_tmp)
                iou_dict["val_loss"] = loss.item()
                iou_dict["val_idx"] = val_idx
                iou_dict["train_idx"] = train_idx
                iou_dict["epoch"] = epoch

                iou_results.append(iou_dict)

        df = pd.DataFrame(iou_results)
        df.to_csv(f"{phase}_iou_results_ep_{epoch}_tridx_{train_idx}.csv", index=False)

    def train_function(self):
        torch.cuda.empty_cache()
        self.model = self.model.to(self.device)
        optim = Adam(self.model.parameters(), lr=self.optim_lr, betas=(0.99, 0.999))
        train_iou_results = []
        for epoch in tqdm(
            range(self.epochs), desc=f"Training for epochs {self.epochs}", leave=True
        ):
            train_loop = tqdm(self.training_loader, leave=True)
            for train_idx, train_batch in enumerate(train_loop):
                train_loss, iou_src1, iou_src2 = self.train_step(
                    train_batch, train_idx, epoch
                )
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                train_iou_results.append(iou_src1)
                train_iou_results.append(iou_src2)

                train_loop.set_postfix(
                    iou_src1=iou_src1["meanIoU"],
                    iou_src2=iou_src2["meanIoU"],
                    train_loss=iou_src1["total_train_loss"],
                )

                if (train_idx + 1) % 500 == 0:
                    save_update_path = f"{cfg.results.chk_save_dir}/{cfg.project.name}/{cfg.project.version}/{cfg.project.state}"
                    os.makedirs(save_update_path, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(
                            save_update_path,
                            f"tr_src_ep_{epoch}_tridx_{train_idx}.ckpt",
                        ),
                    )

                    self.validation(
                        self.valid_src1_loader,
                        phase="src1",
                        epoch=epoch,
                        train_idx=train_idx,
                    )
                    self.validation(
                        self.valid_src2_loader,
                        phase="src2",
                        epoch=epoch,
                        train_idx=train_idx,
                    )
            df = pd.DataFrame(train_iou_results)
            df.to_csv(
                f"train_iou_results_ep_{epoch}_tridx_{train_idx}.csv", index=False
            )


if __name__ == "__main__":
    os.environ["PYTHONHASHSEED"] = str(cfg.pipelines.seed)
    np.random.seed(cfg.pipelines.seed)
    torch.manual_seed(cfg.pipelines.seed)
    torch.cuda.manual_seed(cfg.pipelines.seed)
    torch.backends.cudnn.benchmark = True
    tp = TrainPipeline()
    tp.get_data()
    tp.train_function()
