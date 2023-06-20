# ADoTO-AI
Automatic Detection of Topographic Objects (ADoTO)

## This repository is part of ADoTO project:

| Responsibility | Link                                               |
|----------------|----------------------------------------------------|
| Architecture   | https://github.com/RadekBucki/ADoTO-Architecture   |
| Backend        | https://github.com/RadekBucki/ADoTO-Backend        |
| AI             | https://github.com/RadekBucki/ADoTO-AI             |
| Frontend       | https://github.com/RadekBucki/ADoTO-Frontend       |

## Prerequisites

The program includes three modules:
- Train
- Predict
- Flask API

If you want to run program, configure the `config.file`. Below is a table with summaries about env variables:

| Name                | Description                                               |
|-------------------------------|----------------------------------------------------|
| TRAIN_IMAGE_SIZE              | Size of the photos on which we want to train, e.g: have `1000x1000` images and we want that model will train them at `256x256` size, so we enter `256` there |
| HOUSE_TRAIN_IMAGE_SIZE        | Size of the trained photos (photos size in trained model)             |
| ROADS_TRAIN_IMAGE_SIZE        | Size of the trained photos (photos size in trained model)             |
| WATER_TRAIN_IMAGE_SIZE        | Size of the trained photos (photos size in trained model)       |
| FOREST_TRAIN_IMAGE_SIZE       | Size of the trained photos (photos size in trained model)        |
| EPOCH_SIZE                    | The amount of epoch training             |
| BATCH_SIZE                    | Photos are trained in groups. Batch size determines how many photos belong to that group. The number can cause differences in training results. **IMPORTANT:** *Higher number = more RAM consumption*       |
| MODEL_NAME                    | Final name of the trained model        |
| IMAGE_PATH                    | Path to folder where folder with photos exists            |
| MASK_PATH                     | Path to folder where folder with masks exists      |
| IMAGE_FOLDER_NAME                    | Name of folder with photos             |
| MASK_FOLDER_NAME                     | Name of folder with masks      |
| IMAGE_FOLDERS_COUNT                     | Number of subfolders with photos/masks. *Examples of project configuration with multiple pictures subfolders will be described below.*    |
| POINT_ACC                     | How much predicted points of the object will be returned. *Higher number = less points*       |
| CUDA_TOOLKIT_BIN              | Path to CUDA folder        |
| CUDA_TOOLKIT_LIB              | Path to CUDA folder        |
| CUDA_TOOLKIT_ZLIBWAPI         | Path to ZLIBWAPI library folder        |

The `config.file` is initially (for demonstration) completed.

### Software requirements

- **Operating system:** `Ubuntu/Debian/Arch` - reason here: [Ended support of tensorflow tool for Windows](https://discuss.tensorflow.org/t/2-10-last-version-to-support-native-windows-gpu/12404)

- **Python:** recommended `3.10.X`

- **Packages:** included in the `requirements.txt` file


#### Below are the steps to configure project for Ubuntu 22.04:

- Verify your Python version:

```bash
python3 --version
```
- *(If you don't have)* install venv:
```bash
sudo apt update
sudo apt install python3.10-venv1
```
- Create and active venv:
```bash
python3 -m venv [your_venv_name] 
source [your_venv_name]/bin/activate
```
- Clone Github project into created venv.

- Install packages from `requirements.txt`:
```bash
pip3 install requirements.txt
```
- Venv with project is ready to use.


### Hardware requirements

#### To predict:
- **CPU:** not verified
- **GPU:** not verified
- **RAM:** not verified

#### To learn:
- **CPU:** If use processor for training - minimum Intel Core i7 10Gen or AMD Ryzen 7, preferred Intel Core i9 10Gen or AMD Ryzen 9
- **GPU:** If use graphic card for training - minimum Nvidia RTX 2070, preferred RTX 30XX
- **RAM:** minimum 32GB 

If you don't have required graphic card, it's possible to split the folder with photos/masks into subfolders. This will reduce the consumption of the graphics card. 

**Example:**
- We have a folder with photos as in the example below:
```
/data/pictures/photos
/data/pictures/masks
```
- We divide photos into smaller groups and place in subfolders
```
/data/pictures/photos_1
/data/pictures/photos_2
/data/pictures/photos_3

/data/pictures/masks_1
/data/pictures/masks_2
/data/pictures/masks_3
```
**IMPORTANT**: Folder index must start on 1.


- In `config.file` we set up:
```
IMAGE_PATH=/data/pictures
MASK_PATH=/data/pictures
IMAGE_FOLDER_NAME=images
MASK_FOLDER_NAME=masks
IMAGE_FOLDERS_COUNT=3
```

### Model datasets
Dataset files have their specific names and are located in the `Datasets` folder by default. The file must start with the prefix `dataset_` and then the model name e.g. `forest`. Datasets are saved by default with the `.h5` extension. 

**Examples:**

- `Datasets/dataset_forest.h5`
- `Datasets/dataset_water.h5`
- `Datasets/dataset_roads.h5`
- `Datasets/dataset_house.h5`


### U-Net 
The neural network architecture was based on:
- [Github repository](https://github.com/msoczi/unet_water_bodies_segmentation/tree/main)
- [Documentation 1](https://arxiv.org/pdf/1505.04597.pdf)
- [Documentation 2](https://arxiv.org/pdf/2207.11222v1.pdf)

### Images datasets

Dataset was mostly prepared by us using images downloaded from `geoportal.gov.pl`. Each object has prepared around 1000 photos.
[Google Drive with Datasets](https://drive.google.com/drive/folders/1Z-9dsvDqO_iDyi6dHUKkNgShgGFi-sTD?usp=sharing)

## Project launch

Project can be run locally or on a Docker container. The container runs the Flask API by default and allows using the prediction module

### How to run project on Docker container

- Go to the folder where the project is located.
- Configure/Fill `config.file`.
- Build docker image:
```bash
docker build -t [container_name]:[version] .
```
- Run docker image:
```bash
docker run -d -p 5000:5000 [container_name]:[version]
```
Docker will be started on port 5000. You can check that docker is working properly using the `docker ps` command.


### How to run project locally

- Go to the folder where the project is located.
- Configure/Fill `config.file`.
- Build docker image:
```bash
docker build -t [container_name]:[version] .
```
- Run docker image:
```bash
docker run -d -p 5000:5000 [container_name]:[version]
```
Docker will be started on port 5000. You can check that docker is working properly using the `docker ps` command.

- Configure venv as in the example shown above.
- Go to the folder where the project is located.
- Configure/Fill `config.file`.
- a) To run the train model use:
```bash
python3 Train.py
```
- b) To run the predict model use:
```bash
python3 Predict.py
```

