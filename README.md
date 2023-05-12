# Action Conditioned Tactile Prediction During Robot Manipulation
This repository contains the code and resources for the RSS 2022 paper: **Action Conditioned Tactile Prediction**.

This repository will provide everything you need to get started with tacitle prediction. We will explain how to install the relevant requirements, to format the data for training and how to train and test the models we explore in our work. 

## Paper:
Action Conditioned Tactile Prediction: case study on slip prediction

- **First Author**: Willow Mandil
- **Second Author**: Kiyanoush Nazari
- **Third Author**: Amir Ghalamzan E


Bibtex:
```bash
@article{mandil2022action,
  title={Action conditioned tactile prediction: a case study on slip prediction},
  author={Mandil, Willow and Nazari, Kiyanoush and others},
  journal={arXiv preprint arXiv:2205.09430},
  year={2022}
}
```
### TLDR
- We show that during manipulation of objects, we can predict tactile sensation over a future time horizon. 
- We show that these tactile predictions can be used to predict object slip. 
- We use an LSTM based RNN. 

## Paper Abstract:
Tactile predictive models can be useful across several robotic manipulation tasks, e.g. robotic pushing, robotic grasping, slip avoidance, and in-hand manipulation. However, available tactile prediction models are mostly studied for image-based tactile sensors and there is no comparison study indicating the best performing models. In this paper, we presented two novel data-driven action-conditioned models for predicting tactile signals during real-world physical robot interaction tasks (1) action condition tactile prediction and (2) action conditioned tactile-video prediction models. We use a magnetic-based tactile sensor that is challenging to analyse and test state-of-the-art predictive models and the only existing bespoke tactile prediction model. We compare the performance of these models with those of our proposed models. We perform the comparison study using our novel tactile enabled dataset containing 51,000 tactile frames of a real-world robotic manipulation task with 11 flat-surfaced household objects. Our experimental results demonstrate the superiority of our proposed tactile prediction models in terms of qualitative, quantitative and slip prediction scores.

## Requirements

- GPU access is recommended for faster training (we used two Nvidia RTX A6000 GPUs)
- Python 3.8
- tensorflow (for CDNA baseline test)
- chainer (for CDNA baseline test)
- PyTorch, torchvision
- SciPy
- NumPy
- Matplotlib
- OpenCV
- tqdm

To install dependencies, use:
```bash
pip install -r requirements.txt
```

## Dataset
To run training and testing of the models and pipeline presented in this work, please download the dataset from:  https://github.com/imanlab/tactile_prediction_flat_dataset

The dataset consists of four key features:
- **Tactile sensation** from the Xela uSkin magnetic based tactile sensor (4x4 taxels). Each taxel  reads sheer X and Y forces as well as a normal force (uncallibrated).
- **Robot state** in both joint space and task space (euler and quaternion available). For our research we use euler task space robot pose.
- **Object pose** in euler angles. This pose is collected from an aruco marker placed on the object and observed with the intel realsense D435 camera.
- **Slip signal** which is generated from the object pose itself. This signal is used for training the random forrests algorithm to learn the correlation between tactile sensation and object slip.

An example of this dataset for one grasp and move sequence in shown below:

<p align="center">
<img src="https://github.com/imanlab/action_conditioned_tactile_prediction/blob/main/assets/dataset_structure_example.png" alt= “” width="500">
<p/>

The data set was collected use teleoperation of two robots. the human controller attempted to create motions that generate high volumes of object slip (10% of the data is slip).

A gif of dataset collection is shown below:
<p align="center">
<img src="https://github.com/imanlab/action_conditioned_tactile_prediction/blob/main/assets/datacollection_example.gif" alt= “” width="500">
<p/>

### Dataset formatting:
The dataset requires formatting for use in the training and testing scripts.

To format the data, apply [gen_image_dataset_2.py](https://github.com/imanlab/action_conditioned_tactile_prediction/blob/main/code/data_formatting/gen_image_dataset_2.py). The function requires several user inputs, for example, the length of the context window, the length of the prediction horizon, where to save the data, whether to convert the tactile data to tactile images and finally, desired image height and width. The default is 10 context frames and 10 prediction frames.

```bash
python3 gen_image_dataset_2.py
```

The output of this method outputs two key items:
- The formated frames for tactile data, tactile image data, robot data, slip signals.
- A map file, that contains the names of each individual frames data. This stops us from saving the full sequence. If we did not do this the dataset size would become too large as individual frames would be repeated throught the dataset.



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







# Simultaneous Prediction of Optical and Tactile Sensation

This repository contains the code and resources for the T-RO 2023 paper: **Combining Vision and Touch for Physical Robot Interaction**.

**First Author**: Willow Mandil
**Second Author**: Amir Ghalamzan E

We examine the benefits of incorporating tactile sensation into video prediction models for physical robot interactions. By proposing three multi-modal integration approaches and comparing the performance of these tactile-enhanced video prediction models, we demonstrate the potential of using both visual and tactile feedback for improved scene prediction accuracy and a better understanding of cause-effect relationships during robot interactions. We also introduce two new datasets of robot pushing using a magnetic-based tactile sensor for unsupervised learning.

<p align="center">
<img src="https://github.com/imanlab/SPOTS_IML/blob/main/assets/SPOTS_abstract_5%20(1).jpg" alt= “” width="500">
<p/>

## Datasets

Two datasets and their descriptions can be found at:

  - [Marked Friction Dataset](https://github.com/imanlab/object_pushing_MarkedFrictionDataset)
  - [Household Objects Dataset](https://github.com/imanlab/)

<p align="center">
<img src="https://github.com/imanlab/SPOTS_IML/blob/main/assets/data_collection_household%20(1).jpg" alt= “” width="500">
<p/>

<p align="center">
<img src="https://github.com/imanlab/SPOTS_IML/blob/main/assets/DatasetExampleLarge%20(1).jpeg" alt= “” width="500">
<p/>

## Requirements

- GPU access is recommended for faster training (we used two Nvidia RTX A6000 GPUs)
- Python 3.8
- PyTorch, torchvision
- SciPy
- NumPy
- Matplotlib
- OpenCV
- tqdm

To install dependencies, use:
```bash
pip install -r requirements.txt
```

### Dataset formatting:
Download the dataset you wish from the section above using:
```bash
git clone https://github.com/imanlab/object_pushing_MarkedFrictionDataset.git
```

The dataset requires formatting for use in the training and testing scripts.

To format the data, apply [format_data.py](https://github.com/imanlab/SPOTS_IML/data_formatting/format_data.py). The function requires several user inputs, for example, the length of the context window, the length of the prediction horizon, where to save the data, whether to convert the tactile data to tactile images and finally, desired image height and width.

```bash
python3 format_data.py
```

### Training and Testing:
We have simplified the training and testing procedure of the models presented in this paper. 

To train the model run:  
```bash
python3 model_trainer.py
```

To test the model run:  
```bash
python3 model_trainer.py
```

There are a variety of input arguments that can be adjusted to suit your needs. The extensive list below explains each argument for the two programs. An example of the use of these input arguments is shown below: 

```bash
python3 model_trainer.py --model_name="SPOTS_SVG_ACTP" --batch_size=32 --epochs=100 --device="cuda:0" --model_save_path= "/home/.../SPOTS_SVG_ACTP/" --train_data_dir="/home/.../test_dataset/" --scaler_dir="/home/.../scalar_dataset/" 
```