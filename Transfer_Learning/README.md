# Transfer Learning 

## Introduction

* In this project, we demonstrate the use of transfer learning using keras.
* Our task is to distinguish between dogs and cats using pre-trained image-net weights.
* Our base model is InceptionV3.

## Dependencies

* Python 3.6
* Keras
* h5py

## Getting started

* Download the dataset(train and val) from [here](https://drive.google.com/file/d/1hwztyVYGhRv08TIzmBrp9TtynioHStY4/view?usp=sharing) and place it in the folder containing the code.
* Download the model from [here]() and place it in the same directory.

## Usage

* In case you want to train, execute `python fine_tune.py --train_dir train_dir --val_dir val_dir`. This trains the model and saves the weights.
* Execute `python predict.py`. 

