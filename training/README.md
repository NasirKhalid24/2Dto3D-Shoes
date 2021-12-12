# 2D to 3D - Sneakers

The following folder contains files and scripts used to train the model for this project. The train.py is the main file executed to begin training and log training details to WANDB online

```
pip install -r requirements.txt
python train.py
```

The script autosaves weights during training to a folder called "weights" and images of mesh during training to "output"

In order to train you need the dataset of 360 shoe images and it must be preprocessed (resized/scaled) and saved as .npy to a folder called data. A subset of the data is available here so train.py can be run and it will fit on the 6 shoes data and output images during training + save the weights during training

The dataset.py file in modules has parameters for camera postions, width and height which can be adjusted if needed

## Model

in modules/model.py you can inspect the architecture of the whole network from the **NModel** class