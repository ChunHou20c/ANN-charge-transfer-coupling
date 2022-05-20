# ANN-charge-transfer-coupling
A framework for deployment of artificial neural network with tensorflow for predicting charge transfer coupling.

This is the codes for my final year project.

## How to run the code
Add training input data file and output data file (in .npy format) into this directory (also change the FEATURE and RESULT name to your input and output data respectively)
Modify hyperparameters in config.py and model architechture in model.py
Create a directory to store the data or set the directory in config (relative path)
For windows user, the save path in postprocessing.py might need to be modified
Optionally, change the optimizer in main.py 

## Use other schedular (for training rate control)
Define the schedular in schedular.py, then change the line in main.py import <schedular function name> as schedular
  
## Customize the output graph
Change the content in postprocessing.py
  
## Dependancies
python, tensorflow, numpy, matplotlib, sklearn
