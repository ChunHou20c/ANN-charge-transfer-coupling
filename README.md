# ANN-charge-transfer-coupling
A framework for deployment of artificial neural network with tensorflow for predicting charge transfer coupling.

This is the codes for my final year project.

## How to run the code
Add training input data file and output data file (in .npy format) into this directory
Modify hyperparameters in config.py and model architechture in model.py 
Optionally, change the optimizer in main.py 

## Use other schedular (for training rate control)
Define the schedular in schedular.py, then change the line in main.py import <schedular function name> as schedular
  
## Customize the output graph
Change the content in postprocessing.py
  
## Dependancies
python, tensorflow, numpy, matplotlib, sklearn
