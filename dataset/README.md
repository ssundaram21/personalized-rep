This page contains information on the dataset structure expected by our pipeline, details on the PODS dataset (+ the reformulated DOGS and DF2 datasets), and links to download all real and synthetic datasets. 

**Table of Contents:**
* [Dataset structure](#dataset-structure)
* [Benchmarks](#benchmarks)
  * [PODS](#pods)
  * [DF2 & DOGS](#df2--dogs)
* [Synthetic datasets](#synthetic-datasets)

# Dataset Structure
Our pipeline expects real datasets in the following form:
```
real_dataset_name/
├── train/
│   ├── class_0/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └──  class_1/ class_2/ ...
├── test/
│   ├── class_0/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └──  class_1/ class_2/ ...
├── train_masks/
│   ├── class_0/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   └── ...
│   └──  class_1/ class_2/ ...
└── test_masks/
    ├── class_0/
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ...
    └──  class_1/ class_2/ ...
```
Each class is a different instance/object. The `train_masks` and `test_masks` folders are required for running dense evaluation (detection, segmentation), but optional otherwise.

Synthetic datasets are expected in the following form:
```
synthetic_dataset_name/
├── class_0/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└──  class_1/ class_2/ ...
```

Negatives are expected in the following form:
```
negatives/
├── 0.jpg
├── 1.jpg
└── ...
```

# Benchmarks 
## PODS
The PODS dataset is new a benchmark for personalized vision tasks. It includes:
* 100 common household objects from 5 semantic categories
* 4 tasks (classification, retrieval, segmentation, detection)
* 4 test splits with different distribution shifts.
* 71-201 test images per instance with classification label annotations (`test` split).
* 12 test images per instance (3 per split) with segmentation annotations (`test_dense` split).

PODS is [available on HuggingFace](https://huggingface.co/datasets/chaenayo/PODS), or can be directly downloaded [here](https://data.csail.mit.edu/personal_rep/pods.zip).

PODS is split *class-wise* into a validation set (6 classes per semantic category) and a test set (14 classes per semantic category). All test performance reported in our paper is from the test set of classes.

*Within each class*, images are divided into a train/retrieval set (3 images) and a test/query set. The test/query set is then further divided into 4 test splits reflecting different distribution shifts.

Metadata is stored in two files:
* `pods_info.json`:
  * `classes`: A list of class names
  * `class_to_idx`: Mapping of each class to an integer id
  * `class_to_sc`: Mapping of each class to a broad, single-word semantic category
  * `class_to_split`: Mapping of each class to the `val` or `test` split.
* `pods_image_annos.json`: Maps every image ID to a dictionary:
  * `class`: The class name that the image belongs to
  * `split`: One of `[train, test]` indicating if the image is in the train or test set for that class.
  * `test_split`: For images in the `test` split, denotes which distribution-shift test split the image is in: One of `[in_distribution, pose, distractors, pose_and_distractors]`

## DF2 & Dogs
DF2 and Dogs are reformulated subsets of the [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) and [DogFaceNet](https://github.com/GuillaumeMougeot/DogFaceNet) datasets to enable evaluation across the same 4 tasks as PODS.

DF2 and DOGS each contain a metadata file `{dataset}_info.json` with the same information as `pods_info.json`.

## Download
| **Dataset**     | **Download**    |
|-----------------|-----------------|
| PODS            | [HuggingFace](https://huggingface.co/datasets/chaenayo/PODS) or [Direct](https://data.csail.mit.edu/personal_rep/pods.zip) |
| DF2             | [Mapping Notebook]   |
| DOGS            | [Direct](https://data.csail.mit.edu/personal_rep/dogs.zip)    |

## Training on your own dataset
To train on your own data, set your training data up according to the schema above. The minimal requirements are a `real_dataset` containing a `train` folder, and a `negatives` dataset.

# Synthetic Datasets
All synthetic datasets from our paper are available for download below. All were generated using Stable Diffusion 1.5.

**PODS**
| Synthetic Dataset | Download |
|----------------|----------------|
| DreamBooth + standard caption | [CFG 4](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_cfg_4.zip), [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_cfg_5.zip), [CFG 7.5](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_cfg_7.5.zip) |
| DreamBooth + LLM captions | [CFG 4](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_llm_cfg_4.zip), [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_llm_cfg_5.zip), [CFG 7.5](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_llm_cfg_7.5.zip) |
| Masked DreamBooth + LLM captions + Filtering | [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_llm_masked_filtered_cfg_5.zip) |
| Cut/Paste | [Generated backgrounds](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_cut_and_paste_sd_background.zip), [Real backgrounds](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_cut_and_paste_real_background.zip) |
| Negatives | [Generated](https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_negatives.zip)|

**DF2**
| Synthetic Dataset | Download |
|----------------|----------------|
| DreamBooth + standard caption | [CFG 4](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_cfg_4.zip), [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_cfg_5.zip), [CFG 7.5](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_cfg_7.5.zip) |
| DreamBooth + LLM captions | [CFG 4](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_llm_cfg_4.zip), [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_llm_cfg_5.zip), [CFG 7.5](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_llm_cfg_7.5.zip) |
| Masked DreamBooth + LLM captions + Filtering | [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_llm_masked_filtered_cfg_5.zip) |
| Cut/Paste | [Generated backgrounds](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_cut_and_paste_sd_background.zip), [Real backgrounds](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_cut_and_paste_real_background.zip) |
| Negatives | [Generated](https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_negatives.zip)|

**DOGS**
| Synthetic Dataset | Download |
|----------------|----------------|
| DreamBooth + standard caption | [CFG 4](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_cfg_4.zip), [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_cfg_5.zip), [CFG 7.5](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_cfg_7.5.zip) |
| DreamBooth + LLM captions | [CFG 4](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_llm_cfg_4.zip), [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_llm_cfg_5.zip), [CFG 7.5](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_llm_cfg_7.5.zip) |
| Masked DreamBooth + LLM captions + Filtering | [CFG 5](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_llm_masked_filtered_cfg_5.zip) |
| Cut/Paste | [Generated backgrounds](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_cut_and_paste_sd_background.zip), [Real backgrounds](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_cut_and_paste_real_background.zip) |
| Negatives | [Generated](https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_negatives.zip)|
