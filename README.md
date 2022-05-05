# Tactile Prediction
This is a git repo for tactile prediction of household items for pick and move operations.

## Dataset
### 1. General Info
The dataset is a set of kinesthetic motions, via human operation, of pick and move operations of houshold objects. These objects are grasped with a tactile enabled end effector, using the Xela USkin sensor (16x3).

The dataset is stored in three formats depending on use case:
  - as a zip file containing bare data
  - Linear dataset:
    - Tactile sequence, "d" = 1x48 float16, [d0, d1, ... , dN]
    - Robot sequence, "r" = 1x6 float16, [[x0,y0,z0,r0,p0,y0], [x1,y1,z1,r1,p1,y1], ... , [xN,yN,zN,rN,pN,yN]]
    - Slip Classification, "s" bool, [s0, s1, ... , sN]
  - Image dataset:
    - Tactile sequence, "d" = 1x48 float16, [d0, d1, ... , dN]
    - Tactile image sequence, "d" = 1x48 -> 64x64x3 int8, [folder_name/image_name_0, folder_name/image_name_1, ..., folder_name/image_name_N]
    - Robot sequence, "r" = 1x6 float16, [[x0,y0,z0,r0,p0,y0], [x1,y1,z1,r1,p1,y1], ..., [xN,yN,zN,rN,pN,yN]]
    - Slip Classification, "s" bool, [s0, s1, ..., sN]

### 2. Formatting the datasets for training & testing
There are two different folders, one training, one testing.

Use: data_formatting/generate_image_dataset.py to create an image dataset. Adjust the global variable "data_dir" to point to the correct dataset "test_dataset", "train_dataset" or other subset datasets you may be interested in.


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