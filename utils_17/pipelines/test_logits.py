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
    dataset2_name="Toronto3D",
    dataset2_path="/data3/vaibhav/Toronto_3D",
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

target_validation_loader = DataLoader(
    target_dataset,
    batch_size=8,
    collate_fn=validation_collation,
    pin_memory=True,
    shuffle=False,
    num_workers=4,
)

teacher_model = MinkUNet34(1, 17)
teacher_model_state_dict = torch.load(
    "/data4/vaibhav/musmix/output/model_checkpoints/adaptation/teacher_23_sept_mod_mix_4_3999_10_11_2024_21.pth",
    map_location=device,
)
teacher_model.load_state_dict(teacher_model_state_dict)
teacher_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(teacher_model)
# print("Teacher Model loaded:\n", teacher_model)

student_model = MinkUNet34(1, 17)
student_model_state_dict = torch.load(
    "/data4/vaibhav/musmix/output/model_checkpoints/adaptation/student_23_sept_mod_mix_4_3999_10_11_2024_21.pth",
    map_location=device,
)
student_model.load_state_dict(student_model_state_dict)
student_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(student_model)

# loss functions
sd_loss = SoftDICELoss(ignore_label=-1, is_kitti=False)
entropy = CrossEntropyLoss(
    ignore_index=-1,
)
kl_div = KLDivLoss(reduction="mean")
target_confidence_th = 0.85


def entropy(prob_logits):
    entropy = -torch.sum(prob_logits * torch.log(prob_logits + 1e-10), dim=1)
    return entropy


def validation_step(
    student_model, teacher_model, batch, batch_idx, phase, output_dir, epoch=0
):
    torch.cuda.empty_cache()
    with torch.no_grad():
        student_model.eval()
        teacher_model.eval()
        # Input batch
        stensor = ME.SparseTensor(
            coordinates=batch["coordinates"].int(),
            features=batch["features"],
            device=device,
        )

        student_out = student_model(stensor).F.cpu()
        teacher_out = teacher_model(stensor).F.cpu()
        labels = batch["labels"].long().cpu()

        # loss = sd_loss(student_out, labels)

        soft_student_out = F.softmax(student_out[:, :-1], dim=1)
        soft_teacher_out = F.softmax(teacher_out[:, :-1], dim=1)

        teacher_entropy = entropy(soft_teacher_out)
        student_entropy = entropy(soft_student_out)

        p_ratio = student_entropy / (teacher_entropy + student_entropy)
        print(soft_student_out.shape, soft_teacher_out.shape, p_ratio.shape, flush=True)

        final_logits = (
            p_ratio.view(-1, 1) * soft_student_out
            + (1 - p_ratio.view(-1, 1)) * soft_teacher_out
        )

        soft_final = F.softmax(final_logits, dim=-1)
        final_conf, final_preds = soft_final.max(1)

        # Calculate IoU
        iou_tmp = jaccard_score(
            final_preds.detach().numpy(),
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

    # print(val_idx)
    print(soft_student_out.shape, soft_teacher_out.shape, results_dict)


for val_idx, val_batch in tqdm(enumerate(target_validation_loader), desc="Validation:"):
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    validation_step(
        student_model,
        teacher_model,
        val_batch,
        val_idx,
        "target_validation",
        f"/data4/vaibhav/musmix/output/results/{project_name}/adaptation/",
        0,
    )
