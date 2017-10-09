# Kaggle Titanic challenge
Here is my implementation of the Kaggle Titanic challenge. My purpose was to use this challenge
to do my first Python script, my first Machine Learning work, and a practical use of Pandas and
Scikit-Learn.

Most of the models currently implemented allowed to me to hit 78.9% accuracy on test data provided
by Kaggle, so if you have something else using this, well there is a problem somewhere! I mostly 
used the Support Vector Classifier for this task, despite most of peoples on Kaggle reporting better 
results using a Decision Tree. Well for me it was not the case. You can test yourself, there is an 
option to train a Decision Tree.

I will detail the content in subsequent sections, but would like to make a summary of the files here:
- all the text files are monitoring outputs from the code. You can skip them, there is no code inside.
- the main part of the code (the executable script) is **titanic.py**. This is where all options and
models are defined.
- the directory **titalib** contains custom-made librairies. Those weren't *necessary* but greatly
helps improving the readability of the script and its versatility. The feature engineering is performed
into one of the scripts from this directory, we will come back later on it.
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
Simply execute the script to train the default SVC with the parameter C=2 and use a Radial Basis Function
as a kernel of parameter gamma=0.0173. It will produce a prediction file formatted to be submitted on 
Kaggle under the `data/` directory. You can go into `titanic.py` script to change those settings.

All the features are transformed into categorical features, one-hot-encoded (for non-binary ones) and
are scaled to mean=0 and std=1. Currently, there are 5 categorical features left (*Age*, *Fare*, *Sex*, 
*Relatives* and *Location*), 3 being binary (*Age*, *Sex* and *Relatives*), one with 3 categories (*Fare*)
and the last with 4 categories (*Location*). This give a total of 96 combinations, which is about one order
of magnitude below the data count (which is about 800). We then can reasonably hope that each of our 
combination will be populated, and that our models will be able to train on each possible case. The former is not
necessarly true, but I still think it is preferable to train on as many configurations as possible.

Alternatively, you might want to play around with other models, or explore different parameters settings.
There are numerous flags for this. The full option list is detailed below. They are all defined into 
`titanic.py` script. You can also go for a different feature engineering route. For this, you will have 
to go into the `dataformat()` function in the `data/preproc.py` file.

Paths:
- `--train_data` will specify the full path to the training file.
- `--test_data` will specify the full path to the test file.
- `--pred_output` will specify the full path to the prediction file.

Mode (accepts 0 or 1):
- `--grid`: wether or not you want to run a grid search on the hyperparameters. The dictionaries
still have to be written inside `titanic.py` by yourself if you want to change the default grid search settings
I used.
- `--test`: I just wanted to be able to perform quick testings on models an manually split the data into
a cross-validation set and a training set after shuffling the data. Its purpose is more about testing code 
rather than models so you can totally skip this section.
- `--multi`: the multi-mode is a simple ensemble method in this code. It is used to produce a prediciton file
made from the equal-weight-averaged predictions from a k-Nearest Neighbors, a Support Vector Classifier, and a 
Logistic Regression. Scikit-learn provides functions to do that, but I didn`t know at that time.
- `--model`: this option specify the model you want to use for your grid search, your testing or if you run
in single mode (the default option, that's to say `grid`, `test` and `multi` are all set to 0).
The available models are for instance a Support Vector Classifier ( `SVC` ), a Decision Tree ( `Tree` ),
 a Logistic Regression ( `LR` ) or k-Nearest Neighbors ( `KNN` ). By default, this option is set to `SVC`. Also 
 note that this field is append to the output text files.

Note that except for the model option, those modes are all exclusive to each other.

Data settings:
- `--child`: the limit in years below which a passenger will be treated as a child.
- `--sibsp`: the maximum number of vanilla categories preserved for *SibSp* feature after label-encoding. 
All numbers above in the data will be put into the same additional category. I don`t use this feature anymore 
so changing this option will have no effect unless you remove it from the list of features to be drop.
- `--parch`: same as above but for the *Parch* features. Both those features have been combined into a new one 
called *Relatives*. As above, it needs to be removed from the list of features to be drop if you want this option 
to be of any use.
- `--fare1`: the upper limit of the "poor" category of passengers.
- `--fare2`: the lower limit of the "rich" category of passengers. A third category lies between those two.
- `--title`: the minimum number of occurences of a given title to be placed into its own category. Anything below 
that will be put into the same category.

## The files
### titanic.py
This is the main script. You will find inside all the options to run the code (described above) and the settings
of data slicing. The most important things you need to know here are about the following lines, near the beginning 
of the code:

- `droplist = ['Name','Ticket','Survived','Cabin','Embarked','Pclass','SibSp','Parch']`  
The droplist is the list of pandas columns to be kicked after the data have been pre-processed. You see here
that the features *SibSp* and *Parch* are on the droplist, that's why their cut option defined above is currently
useless. If you want them back, remove them from this list and add instead *Relatives*, which is a new feature built
on those two.

- `mask = [False,True,False,False,True]`  
The mask indicates whether or not a column has to be one-hot encoded or not. Binary features shouldn't foo obvious
reasons, as well as numerical features. To know precisely which fields corresponds to which feature, simply print
the first element of the pandas dataframe output by the `prep.dataformat()` function.

### titalib/preproc.py
This files contains functions related to the pre-processing of data, namely the feature engineering, the one-hot
encoding and the feature scaling. The `dataformat()` function is in charge to perform the feature engineering
and is probably the most important of all the files if you had to choose one to read.

### titalib/models.py
This files contains functions to run the different operating modes described above. Those are just shortcuts
to train a model, print infos to terminal and write an output file in order to make the main script lighter.

### titalib/printer.py
This files contains the routines to write all the output files, and basic infos about the chosen settings 
in the terminal.

### grid_output_*.txt
Those files give more detailed results of the grid search provided by scikit-learn for a given model. It will 
give you all the parameters combination tested, their rank (the first being the best parameters combination), 
as well as the mean train/test score and the associated standard deviation for each data split used.

### cv_log_*.txt
This files simply contains the full output of scikit-learn's grid search ( the `cv_results_` attribute of
the `GridSearchCV` object), in case you need more informations.

### fails_*.txt
The funnier of output files. This one contains a list of all the data entries the given model fails to 
reproduce. Can be usefull before deciding or not to use an ensemble method and which weight to use for their
vote, or simply to understand why fails the models (if it is because of outliers, if there is a particular
pattern etc...).
