# Visual analytics assignment 2
This repository is assignment 2 out of 4, to be sumbitted for the exam of the university course [Visual Analytics](https://kursuskatalog.au.dk/en/course/115695/Visual-Analytics) at Aarhus Univeristy.

The first section describes the assignment task as defined by the course instructor. The section __Student edit__ is the student's description of how the repository solves the task and how to use it.

## Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

## Student edit
### Solution
The code written for this assignment can be found within the ```src``` directory. The directory contains three scripts, all with parameter values that can be set from the command line. _The scripts assume that ```src``` is the working driectory_. Here follows a description of the funcionality of each script:

- __preprocess.py__: Loads the ```Cifar10``` dataset - which is inherently train-test splitted - from the ```Tensorflow``` package and preprocesses the data so that it is conformable with machine learning algorithms in the ```Scikit-learn``` package. This involves converting images to grey scale, scaling values to the interval [0,1], and reshaping the data so that each image is represented as a 1d array. The data is saved to ```data``` as ```preprocessed_data.pkl``` (which is passed to ```.gitignore``` since it exceeds the Github size limit). This script need only be run once.

- __logistic.py__: Takes a data ``pkl``-file of choice and performs logistic regression on the data. See ```python3 logistic.py -h``` for an overview of manipulatable parameters. The model has the ```solver```-parameter set to ```saga```. Other non user-defined parameters are set to their default (see documentation at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The script outputs a fitted logistic model (gitignored), a text file indicating the set parameter values of the model in ```models```, and a classification report text file in ```out```.

- __neuralnet.py__: Takes a data ``pkl``-file of choice and builds a neural network classification model based on the data. See ```python3 neuralnet.py -h``` for an overview of manipulatable parameters. The model has the ```early_stopping```-parameter set to ```True```. Other non user-defined parameters are set to their default (see documentation at https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). The script outputs a fitted model (gitignored), a text file indicating the parameter values of the model in ```models```, and a classification report text file in ```out```.

### Results
The two different models were run: A logistic regression (log1) and a neural network (multi-layer perceptron classifier, nn1). See corresponding ``txt``-files in ``models`` for model parameters.

|Model|Overall accuracy|
|---|---|
|log1|0.29|
|nn1|0.30|

Both models perform quite badly. The issue is likely due to the nature of the data; Predicting image contents is very difficult from a flattened array of pixel values since the appearance and position of an object varies greatly on an image. 

### Setup
The scripts require the following to be run from the terminal:

```shell
bash setup.sh
```

This will create a virtual environment, ```assignment2_env``` (git ignored), to which the packages listed in ```requirements.txt``` will be downloaded. This might take some time. __Note__, ```setup.sh``` works only on computers running POSIX. Remember to activate the environment running the following line in a terminal before running the ```.py```-scripts.

```shell 
source ./assignment2_env/bin/activate
```