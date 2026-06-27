# MC-HGCN: Motion-Guided Channel-Wise Hypergraph Convolutional Network


Official PyTorch implementation of **MC-HGCN** for skeleton-based action recognition.

> **MC-HGCN: Motion-Guided Channel-Wise Hypergraph Convolutional Network for Skeleton-Based Action Recognition**

## Overview

MC-HGCN models skeleton sequences through a unified hypergraph structure that captures higher-order joint group dependencies. Unlike conventional graph convolutional networks limited to pairwise edges, hyperedges can connect arbitrary numbers of joints, explicitly encoding coordinated joint groups such as entire limbs or cross-limb pairs. The framework introduces three complementary mechanisms:

- **Hypergraph Construction**: body-part and cross-limb hyperedges encode anatomical and symmetric priors.
- **Motion-Guided Modulation**: per-joint motion intensity adaptively reweights the incidence matrix at each frame.
- **Channel-Wise Hyperedge Learning**: each feature channel learns distinct hyperedge correlation patterns.

## Architecture

![MC-HGCN Architecture](assets/architecture.png)

## Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX
- Numpy, Scipy
- thop, fvcore

Install dependencies:

```bash
pip install torch>=1.1.0 pyyaml tqdm tensorboardX numpy scipy thop fvcore
```

## Data Preparation

This project uses the NTU RGB+D skeleton datasets.

### Required Datasets

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton

### Obtaining the Data

1. Visit the [NTU RGB+D dataset page](https://rose1.ntu.edu.sg/dataset/actionRecognition) and submit a request.
2. After approval, download the skeleton archives:
   - `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   - `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
3. Unzip into the following directory structure:

```
data/
  nturgbd_raw/
    nturgb+d_skeletons/       # from nturgbd_skeletons_s001_to_s017.zip
    nturgb+d_skeletons120/    # from nturgbd_skeletons_s018_to_s032.zip
```

### Processing

```bash
# For NTU RGB+D 60
cd ./data/ntu
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py

# For NTU RGB+D 120
cd ./data/ntu120
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

## Training

Train on NTU RGB+D 120 (cross-subject):

```bash
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/mc-hgcn --device 0
```

Train on NTU RGB+D 60 (cross-subject):

```bash
python main.py --config config/nturgbd60-cross-subject/default.yaml --work-dir work_dir/ntu60/csub/mc-hgcn --device 0
```

All experiments use only 3D joint coordinates as input (no bone or motion modalities).

## Results

| Benchmark     | X-Sub | X-View | X-Set |
| ------------- | ----- | ------ | ----- |
| NTU RGB+D 60  | 93.1  | 96.7   | --    |
| NTU RGB+D 120 | 90.3  | --     | 91.2  |

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2025mchgcn,
  title={MC-HGCN: Motion-Guided Channel-Wise Hypergraph Convolutional Network for Skeleton-Based Action Recognition},
  author={Li, Yongqi and others},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License.

