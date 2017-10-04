# Kaggle Titanic challenge
Here is my implementation of the Kaggle Titanic challenge. My purpose was to use this challenge
to do my first Python script, my first Machine Learning work, and a practical use of Pandas and
Scikit-Learn.

Most of the models currently implemented allowed to me to hit 78.9% accuracy on test data provided
by Kaggle, so if you have something else using this, well there is a problem somewhere! I mostly 
used the Supprot Vector Classifier for this task, despite most of people on Kaggle reporting better 
results using a Decision Tree. Well for me it is not the case. You can test yourself, there is an 
option to train a Decision Tree.

I will detail the content in subsequent sections, but would like to make a summary of the files here:
- all the text files are monitoring outputs from the code. You can skip it, there is no code inside.
- the main part of the code (the executable script) is **titanic.py**.
- the directory **titalib** contains custom-made librairies. Those weren't *necessary* but greatly
helps improving the readability of the script and its versatility.
- the **data** directory contains data downloaded from the Kaggle challenge page, as well as the 
prediction file produced by the code.

I will not discuss feature engineering here as I wrote a Jupyter notebook for this purpose 
(*analysis.ipynb*).

## Requirements
I created this script using:
- Python 3.6.2
- Scikit-Learn 0.19.0
- Pandas 0.20.3

## Usage
Simply execute the script to train the default SVC with the parameters C=2, gamma=0.0173 and 
kernel='rbf'. It will produce a prediction file formatted to be submitted on Kaggle under the 'data/'
directory. You can go into *titanic.py* script to change those settings.

Alternatively, you might want to play around with other models, or explore different parameters settings.
There are numerous flags for this. The full option list is detailed below. They are all defined into 
*titanic.py* script.  

Paths:
- **--train_data** will specify the full path to the training file.
- **--test_data** will specify the full path to the test file.
- **--pred_output** will specify the full path to the prediction file.

Mode (accepts 0 or 1):
- **--grid**: wether or not you want to run a grid search on the hyperparameters. The dictionaries
still have to be written inside *titanic.py* by yourself if you want to change the default grid search settings
I used.
- **--test**: I just wanted to be able to perform quick testings on models an manually split the data into
a cross-validation set and a training set after shuffling the data. Its purpose is more about testing code 
rather than models so you can totally skip this section.
- **--multi**: the multi-mode is a simple ensemble method in this code. It is used to produce a prediciton file
made from the equal-weight-averaged predictions from a k-Nearest Neighbors, a Support Vector Classifier, and a 
Logistic Regression.
- **--model**: this option specify the model you want to use for your grid search, your testing or if you run
in single mode (the default option, that's to say *grid*, *test*, *multi* and *mixed* are all set to 0).
The available models are for instance a Support Vector Classifier ('SVC'), a Decision Tree ('Tree'),
 a Logistic Regression ('LR') or k-Nearest Neighbors ('KNN'). By default, this option is set to 'SVC'. Also 
 note that this field is append to the output text files.

Data settings:
- **--child**: the limit in years below which a passenger will be treated as a child.
- **--sibsp**: the maximum number of vanilla categories preserved for *SibSp* feature after label-encoding. 
All numbers above in the data will be put into the same additional category. I don't use this feature anymore 
so changing this option will have no effect unless you remove it from the list of features to be drop.
- **--parch**: same as above but for the *Parch* features. Both those features have been combined into a new one 
called *Relatives*. As above, it needs to be removed from the list of features to be drop if you want this option 
to be of any use.
- **--fare1**: the upper limit of the "poor" category of passengers.
- **--fare2**: the lower limit of the "rich" category of passengers. A third category lies between those two.
- **--title**: the minimum number of occurences of a given title to be placed into its own category. Anything below 
that will be put into the same category.

## The files
