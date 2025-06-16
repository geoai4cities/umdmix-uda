### This is official implementation of :--> UMDMix: Urban Multi-Domain Mixing (UMDMix) Based Unsupervised Domain Adaptation for LiDAR Semantic Segmentation


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
