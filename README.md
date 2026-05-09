# CAM-GCN

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML==5.4.1, tqdm==4.53.0, tensorboardX
- Torchvision==0.2.1
- Numpy==1.19.4
- Scipy==1.5.4
- thop==0.0.31.post2005241907 , fvcore==0.1.3.post20210311
- iopath==0.1.4
- yacs==0.1.8
- loguru==0.5.3

## Architecture of CAM-GCN ( Motion-guided Graph)
![Model Architecture](src/MA.png)
# Data Preparation

#### This project uses the NTU RGB+D skeleton datasets. Before running training or evaluation, please prepare the data as follows.

#### Required datasets

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton

#### How to obtain the data

#### NTU RGB+D 60 and 120

1. Visit the official NTU RGB+D dataset page and submit a request:
    https://rose1.ntu.edu.sg/dataset/actionRecognition
2. After approval, download the skeleton-only archives:
   - `nturgbd_skeletons_s001_to_s017.zip` (for NTU RGB+D 60)
   - `nturgbd_skeletons_s018_to_s032.zip` (for NTU RGB+D 120)
   - unzip all downloaded files into the following directory:   ./data/nturgbd_raw

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```



# Training 

`python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/temgcn --device 0`

to train model on NTU RGB+D 60/120 with motion modalities. 

```
python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --device 0
```



 
