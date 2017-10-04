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
help improve readability of the script and its versatility.
- the **data** directory contains data downloaded from the Kaggle challenge page, as well as the 
prediction file produced by the code.

## Requirements
I created this script using:
- Python 3.6.2
- Scikit-Learn 0.19.0
- Pandas 0.20.3

## The model
