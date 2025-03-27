# Efficient Deep Ladle-Net

## Associated Publications
Wang et al. (Under submission) Fast universal 3D lesion segmentation on chest-abdomen-pelvis computed tomography using deep learning

## Setup

#### Requirerements
- Ubuntu 18.04
- GPU Memory => 16 GB
- GPU driver version >= 530.30.02
- GPU CUDA >= 12.1
- Python (3.9.18), PyTorch (2.2.1), matplotlib (3.8.3), numpy (1.26.4), nibabel (5.2.1), SimpleITK (2.3.1).

#### Download
Execution file, configuration file, and model weights are download from the [zip](https://drive.google.com/) file.  (For reviewers, please use the password provided in the Code Availability section of the associated manuscript to decompress the file.)

## Steps
#### 0.Installation

Please refer to the following instructions in the terminal.
```
# create and activate the conda environment
conda create --name ladlenet python=3.9
conda activate ladlenet

# install related package
pip install -e .
pip install acvl_utils==0.2
```

#### 1. Set Export Folder Paths

In the terminal run:
```
export nnUNet_raw="... .../LadleNet/LadleNet_raw"
export nnUNet_preprocessed="... .../LadleNet/LadleNet_preprocessed"
export nnUNet_results="... .../LadleNet/LadleNet_results"
```

#### 2. Analyze Statistical Voxel Intensity Properties

A sample version of 'dataset.json' is provided in the "./files" folder. Copy this file into "LadleNet_raw/Dataset001_ULSexd".

Place the CT images and labels in "./LadleNet_raw"
```
./LadleNet_raw/Dataset001_ULSexd/
├── imagesTr/
│   ├── ct_0001_01_0000.nii.gz
│   ├── ct_0002_01_0000.nii.gz
│   ├── ...
│        ⋮
│   └── ct_xxxx_xx_0000.nii.gz
│
├── labelsTr/
│   ├── ct_0001_01.nii.gz
│   ├── ct_0002_01.nii.gz
│        ⋮
│   └── ct_xxxx_xx.nii.gz
│
└── dataset.json
```

Then in a terminal run:
```
nnUNetv2_extract_fingerprint -d 001

```

After running in a terminal, the result file 'LadleNet_preprocessed/Dataset001_ULSexd/dataset_fingerprint.json' will be generated, containing foreground intensity properties for each data channel.

A preprocessed version of 'dataset.json' and 'nnUNetPlans.json' is provided in the "./files" folder. Copy both files into "LadleNet_preprocessed/Dataset001_ULSexd".

```
./LadleNet_preprocessed/Dataset001_ULSexd/
├── dataset_fingerprint.json
├── dataset.json
└── nnUNetPlans.json
```

'dataset.json' includes the metadata for the training data (number of channels,label value,number of cases, etc.).
'nnUNetPlans.json' includes the network structure setting, batch_size, patch_size, custom resampling setting, etc.


#### 3. Data Preprocessing Based on Intensity Properties

In the terminal run:
```
nnUNetv2_preprocess -d 001 -c 3d_fullres -np 4 
```

After preprocessing, the output directory structure will be:
```
./LadleNet_preprocessed/Dataset001_ULSexd/
├── nnUNetPlans_3d_fullres/
│   ├── ct_0001_01_0000.npz
│   ├── ct_0001_01_0000.pkl
│   ├── ct_0002_01_0000.npz
│   ├── ct_0002_01_0000.pkl
│   │       ⋮
│   ├── ct_xxxx_xx_0000.npz
│   └── ct_xxxx_xx_0000.pkl
│
├── gt_segmentations/
│   ├── ct_0001_01.nii.gz
│   ├── ct_0002_01.nii.gz
│        ⋮
│   └── ct_xxxx_xx.nii.gz
│
├── dataset_fingerprint.json
├── dataset.json
└── nnUNetPlans.json
```


#### 4. Testing data format
Prepare the testing data using the same naming format like training data, and place them into "./LadleNet_inference/testingset" folder.

Place the CT images and lables in testing set folder:
./LadleNet_inference/testingset/
├── imagesTs/
│   ├── ct_xxx1_01_0000.nii.gz
│   ├── ct_xxx2_01_0000.nii.gz
│   ├── ...
│        ⋮
│   └── ct_xxxn_xx_0000.nii.gz
│
└── labelsTs/
    ├── ct_xxx1_01.nii.gz
    ├── ct_xxx2_01.nii.gz
         ⋮
    └── ct_xxxn_xx.nii.gz


#### 5. Testing model place
Place the trained model weights under "./LadleNet_results" folder. The download link for the model weights is provided in the file 'README.md' inside the zip archive.

After downloading and unzipping the weights folder from Google Drive, place it as following:
./LadleNet_results
└── Dataset023_ULSabl


#### 6. Inference 

To generate the prediction outcome of the Ladle-Net model with Test-Time data Augmentation:
```
nnUNetv2_predict -i "... .../LadleNet/LadleNet_inference/testingset/imagesTs/" -o "... .../LadleNet/LadleNet_inference/prediction/Proposed" -d Dataset023_ULSabl -c 3d_fullres_resenc -tr nnUNetTrainer_ULS_500_000003 -f all -chk "checkpoint_best.pth"
```
On the other hand, to generate the prediction outcome of the Ladle-Net model without Test-Time data Augmentation:
```
nnUNetv2_predict -i "... .../LadleNet/LadleNet_inference/testingset/imagesTs/" -o "... .../LadleNet/LadleNet_inference/prediction/Proposed_woTTA" -d Dataset023_ULSabl -c 3d_fullres_resenc -tr nnUNetTrainer_ULS_500_000003 -f all -chk "checkpoint_best.pth" --disable_tta
```

After inference, the output directory structure will be:
```
./LadleNet_inference/
├── prediction/
│   ├── Proposed/
│   │   ├── ct_xxx1_01.nii.gz
│   │   ├── ct_xxx2_01.nii.gz
│   │        ⋮
│   │   ├── ct_xxxn_xx.nii.gz
│   │   ├── plans.json
│   │   ├── dataset.json
│   │   └── predict_from_raw_data_args.json
│   
└── testingset/
    ├── imagesTs/
    └── labelsTs/
```
The predicted masks in "Proposed_TTA" are able to evaluted with manually annotated labels in "labelsTs".

If you are using our provided model weights for the ULS23 Challenge, you can run the following command as an example:
```
nnUNetv2_predict -i "... .../LadleNet/LadleNet_inference/testingset/images/" -o "... .../LadleNet/LadleNet_inference/result/Challenge" -d Dataset400_FSUP_ULS -c 3d_fullres_resenc -tr nnUNetTrainer_ULS_500_QuarterLR -p nnUNetPlansNoRs -f all -chk "checkpoint_best.pth" --disable_tta
```


## Training
#### Training from scratch
Run this code in the terminal to train:
```
nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train Dataset001_ULSexd 3d_fullres_resenc all -tr nnUNetTrainer_ULS_500_000003
```

After training, the output directory structure will be:
```
./LadleNet_results/
├── Dataset001_ULSexd/nnUNetTrainer_ULS_500_000003__nnUNetPlans__3d_fullres_resenc/
├── fold_all/
│   ├── checkpoint_best.pth
│   ├── checkpoint_final.pth
│   ├── progress.png
│   └── training_log_xxxx_x_xx_xx_xx_xx.txt
│
├── dataset.json
├── dataset_fingerprint.json
└── plans.json
```


#### Training from pretrained weights
To fine-tune the model from pretrained weights, follow the steps below: 
The pretrained weights download link for the pretrained weights is provided in the file 'README.md' inside the zip archive. Alternatively, search The ULS23 Baseline Model on the Zenodo platform.

After downloading the weights from Zenodo, locate the weights 'checkpoint_best.pth', which under "Dataset901_Filtered_FSUP" folder. Move this weights file to "fold_all" folder. A preprocessed version of 'dataset.json', 'dataset_fingerprint.json' and 'plans.json' is provided in the "./files" folder. Copy these three files into folder as following:
```
./LadleNet_results/Dataset001_ULSexd/nnUNetTrainer_ULS_500_000003__nnUNetPlans__3d_fullres_resenc/
├── fold_all/
│   └── checkpoint_best.pth
│
├── dataset.json
├── dataset_fingerprint.json
└── plans.json
```

Run this code in the terminal to train with pretrained weights:
```
nnUNet_compile=T nnUNet_n_proc_DA=8 nnUNetv2_train Dataset001_ULSexd 3d_fullres_resenc all -tr nnUNetTrainer_ULS_500_000003 --c
```


## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology

