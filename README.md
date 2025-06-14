# UMDMix: Urban Multi-Domain Mixing (UMDMix) Based Unsupervised Domain Adaptation for LiDAR Semantic Segmentation
### Abstract
3D semantic maps generated from Light Detection and Ranging (LiDAR) point clouds enable scene understanding in diverse applications such as autonomous driving and urban planning. However, existing deep learning models struggle when tested on different domains, worsened by limited labeled data. Unsupervised domain adaptation (UDA) can bridge this gap, but existing UDA methods often face adaptation challenges due to domain shifts occurring from variations due to physical environment, data sparsity, and sensor variations. To address these limitations, we propose *UMDMix*, a novel UDA architecture that operates on the mixing of multiple labeled source domains with unlabeled target domains to make the predictive model robust to cross-domain variations. *UMDMix* integrates a teacher-student learning scheme to produce a robust teacher model and an adaptable student model. The performance of the teacher model in the source domain is further strengthened by a position-aware loss that assigns more significance to semantically rich neighborhoods. A combination of entropy regularizer and KL-divergence loss in the target domain updates the knowledge of the teacher model to the student model during adaptation. Our extensive experiments across diverse environments show that *UMDMix* achieves an average improvement of 13\% on minor classes such as bicycle, traffic sign, and person in target domain datasets, outperforming previous state-of-the-art (SOTA) UDA methods.

## Data preparation

### SynLiDAR
Download SynLiDAR dataset from [here](https://github.com/xiaoaoran/SynLiDAR), then prepare data folders as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    └──sequences/
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        └── 12/
```

### SemanticKITTI
To download SemanticKITTI follow the instructions [here](http://www.semantic-kitti.org). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
```

### SemanticPOSS
To download SemanticPOSS follow the instructions [here](http://www.poss.pku.edu.cn/semanticposs.html). Then, prepare the paths as follows:
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── tag
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── instances.txt
            └── 06/
```

### Pip/Venv/Conda
In your virtual environment follow [MinkowskiEnginge](https://github.com/NVIDIA/MinkowskiEngine).
This will install all the base packages.
This code implementation has taken various code snippets from [CoSMix: Compositional Semantic Mix for Domain Adaptation in 3D LiDAR Segmentation](https://github.com/saltoricristiano/cosmix-uda) 
Additionally, you need to install:
- [open3d 0.13.0](http://www.open3d.org)
- [pytorch-lighting 1.4.1](https://www.pytorchlightning.ai)
- [wandb](https://docs.wandb.ai/quickstart)
- tqdm
- pickle
### Run repo
- For training the source model (that will act as teacher model), run the ```train_source.py``` in ```pipleines``` folder within ```utils_17``` folder.
- For adaptation task, run the ```train_adaptation.py``` in ```pipleines``` folder within ```utils_17``` folder.
