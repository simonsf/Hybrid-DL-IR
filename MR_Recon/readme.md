## Data Preparation
### Modify your data format referring to the test case in ./data/test_case before your training.
- Subsample Data: MRI subsampling data in k-space(our code) or image damain (Modified data loader)
- Mask: Sampling mask
- Fullsample Data: Groud truth.

## Run
### Modified json config file in  ./config/{share_config}.json before running the following training code.
`CUDA_VISIBLE_DEVICES=1 python train.py`

## Requirements
pytorch>=1.8.1 tqdm SimpleITK sigpy matplotlib
