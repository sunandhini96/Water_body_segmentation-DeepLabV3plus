# Task : Water Body Segmentation using DeepLabV3plus
This project implements water body segmentation using the DeepLabV3+ model. Mainly focus on the importance of ASPP in Deeplab V3+ for extracting the water bodies. It includes code for training the model, evaluating its performance, and metrics calculations.

# Files Required:

- `deeplabv3Ex.py`: DeepLabV3+ model
- `train2.py`: Training the model code
- `eval.py`: Model evaluation code
- `metrics.py`: Evaluation metrics code

# Installation:

git clone https://github.com/sunandhini96/Water_body_segmentation-DeepLabV3plus.git

cd Water_body_segmentation-DeepLabV3plus


# Usage:

### Run the training script to train the model:
   
python train2.py

### To evaluate the trained model:

python eval.py

# Dataset:

The project uses RGB satellite images and corresponding masks from Sentinel-2 A/B satellite. You can obtain the dataset https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies

# Methodology: Deeplab V3+ Architecture

<img width="501" alt="image" src="https://github.com/sunandhini96/Water_body_segmentation-DeepLabV3plus/assets/63030539/226c62c7-3d74-482e-a1b7-62cb21e1ee4b">

# Output:
### RGB image, True mask image, predicted mask image without ASPP, Predicted mask image with ASPP

<img width="562" alt="image" src="https://github.com/sunandhini96/Water_body_segmentation-DeepLabV3plus/assets/63030539/df690262-19da-4c70-beed-d4cb8bf46062">


## Citation:

If you use this code in your research, please cite our paper for more details.

## More Details:

For a detailed explanation of the project and results, refer to our paper.

### Conference Paper Link : https://ieeexplore.ieee.org/document/10116882



