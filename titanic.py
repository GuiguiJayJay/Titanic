"""
================================
Titanic challenge draft
================================

v.021017
Supports:
  - multi-model mode:
      predictions of different models are averaged to give the final predictions.
  - mixed-model mode:
      use the predictions of an SVC as a new feature for a Logistic Regression.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import titalib.preproc as prep
import titalib.printer as prt
import titalib.models as mdl

from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn import tree

########################
# PARSE INPUTS
########################
parser = argparse.ArgumentParser(description='Define important inputs.')

# Path section
parser.add_argument("--train_data",
                    type=str,
                    default='data/train.csv',
                    help="Path to the training data")
parser.add_argument("--test_data",
                    type=str,
                    default='data/test.csv',
                    help="Path to the test data")
parser.add_argument("--pred_output",
                    type=str,
                    default='data/predictions.csv',
                    help="Path to output predictions file")

# Options section
parser.add_argument("--grid",
                    type=int,
                    default=0,
                    help="Switch to activate(=1)/deactivate(=0) grid search")
parser.add_argument("--test",
                    type=int,
                    default=0,
                    help="activate(=1)/deactivate(=0) test mode for manual data splitting")
parser.add_argument("--multi",
                    type=int,
                    default=0,
                    help="activate(=1)/deactivate(=0) multi model mode")
parser.add_argument("--mixed",
                    type=int,
                    default=0,
                    help="activate(=1)/deactivate(=0) mixed model mode")
parser.add_argument("--model",
                    type=str,
                    default='SVC',
                    help="model to be used for single-model mode ('SVC', 'LR', 'KNN', 'Tree')")

# Settings section
parser.add_argument("--child",
                    type=int,
                    default=15,
                    help="Limit of age for childrens.")
parser.add_argument("--sibsp",
                    type=int,
                    default=2,
                    help="Number of categories for SibSp to be kept (see 'preproc.dataformat')")
parser.add_argument("--parch",
                    type=int,
                    default=3,
                    help="Number of categories for Parch to be kept (see 'preproc.dataformat')")
parser.add_argument("--fare1",
                    type=float,
                    default=25,
                    help="Upper limit of lowest fare category.")
parser.add_argument("--fare2",
                    type=float,
                    default=50,
                    help="Lower limit of highest fare category.")
parser.add_argument("--title",
                    type=int,
                    default=20,
                    help="Min of data required for Title field (see 'preproc.dataformat')")
                    
FLAGS, unparsed = parser.parse_known_args()
limits = {"child": FLAGS.child,
          "sibsplimit": FLAGS.sibsp,
          "parchlimit": FLAGS.parch,
          "farebound1": FLAGS.fare1,
          "farebound2": FLAGS.fare2,
          "titlelimit": FLAGS.title}


########################
# DATA PRE-PROCESSING
########################
# load the data
data_train_raw = prep.dataload(FLAGS.train_data)
data_test_raw = prep.dataload(FLAGS.test_data)

# save index of test set as a list for reconstruction of ouput file
index = data_test_raw.index.tolist()

# format and clean up the data
droplist = ['Name','Ticket','Survived','Cabin','Embarked','Pclass','SibSp','Parch']
fullset, trainsize, labels = prep.dataformat(data_train_raw,
                                             data_test_raw,
                                             droplist=droplist,
                                             limits=limits)

# one-hot encoding and split back data 
mask = [False,True,False,False,True]
data_train, data_test, labels = prep.dataonehot(dataset=fullset,
                                                mask=mask,
                                                labels=labels,
                                                trainsize=trainsize)

if FLAGS.model != 'Tree':
  # mean normalization to mean=0 std=1
  data_train = prep.datanorm(data_train)
  data_test = prep.datanorm(data_test)

# model and parameters definitions
classif = svm.SVC(cache_size=2000)
params_SVC = {'C': 2, 'gamma': 0.0173, 'kernel': 'rbf'}

knear = neighbors.KNeighborsClassifier(algorithm='kd_tree')
params_KNN = {'n_neighbors': 5, 'weights': 'distance' }

logistic = LogisticRegression()
params_LR = {'C': 0.27, 'penalty': 'l1' }

dtree = tree.DecisionTreeClassifier()
params_Tree = {}
  
###############################
# grid search on the parameters
###############################
if FLAGS.grid == 1:
  # prepare the 'shuffle and split' tool
  n_splits = 5
  n_repeats = 10
  test_size = 1/n_splits
  rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
  
  # parameters to be tested
  param_C = np.linspace(1,10,10)
  param_Gam = np.linspace(0.001,0.05,10)
  params_SVC = [{'C': param_C, 'gamma': param_Gam, 'kernel': ['rbf']}]
  
  param_neighbors = range(2,20)
  params_KNN = [{'n_neighbors': param_neighbors, 'weights': ['uniform', 'distance']}]
  
  param_C2 = np.linspace(0.0001,1,20)
  params_LR = [{'C': param_C2, 'penalty': ['l2', 'l1']}]
  
  params_Tree = {}

  # grid search
  if FLAGS.mixed == 1:
    # train the SVC on the full set, and create a new feature made of its predictions 
    # train each model separately
    feeds = mdl.gridsearch(model = classif,
                           switch = FLAGS.model,
                           params = params_SVC,
                           X_train = data_train,
                           y_train = labels,
                           splits = rkf,
                           test_size = test_size,
                           n_repeats = n_repeats)
                   
    # train the SVC on the full set, and create a new feature made of its predictions                      
    classif.set_params(C = feeds['C'], 
                       gamma = feeds['gamma'])
    classif.fit(data_train, labels)
    data_train_custom = mdl.newfeat(classif, data_train)
    
    # train the final model
    dummy = mdl.gridsearch(model = logistic,
                           switch = FLAGS.model,
                           params = params_LR,
                           X_train = data_train_custom,
                           y_train = labels,
                           splits = rkf,
                           test_size = test_size,
                           n_repeats = n_repeats)
  
  else:
    # train each model separately
    if FLAGS.model == 'SVC':
      dummy = mdl.gridsearch(model = classif,
                             switch = FLAGS.model,
                             params = params_SVC,
                             X_train = data_train,
                             y_train = labels,
                             splits = rkf,
                             test_size = test_size,
                             n_repeats = n_repeats)
    elif FLAGS.model == 'KNN':
      dummy = mdl.gridsearch(model = knear,
                             switch = FLAGS.model,
                             params = params_KNN,
                             X_train = data_train,
                             y_train = labels,
                             splits = rkf,
                             test_size = test_size,
                             n_repeats = n_repeats)
    elif FLAGS.model == 'LR':
      dummy = mdl.gridsearch(model = logistic,
                             switch = FLAGS.model,
                             params = params_LR,
                             X_train = data_train,
                             y_train = labels,
                             splits = rkf,
                             test_size = test_size,
                             n_repeats = n_repeats)
    elif FLAGS.model == 'Tree':
      dummy = mdl.gridsearch(model = dtree,
                             switch = FLAGS.model,
                             params = params_Tree,
                             X_train = data_train,
                             y_train = labels,
                             splits = rkf,
                             test_size = test_size,
                             n_repeats = n_repeats)
    else:
      print("No model selected. Please choose one with the --model option.")
            
########################################################
# Fitting and predictions (no-grid, failed pred outputs)
########################################################
elif FLAGS.test == 1:
  # manually split into test/train sets
  n_samples = len(data_train_raw)
  testsize = int(0.3*n_samples)

  np.random.seed(int(np.random.uniform(0,100,1)[0]))
  order = np.random.permutation(n_samples)
  data_train = data_train[order]
  labels = labels[order]

  X_train = data_train[testsize:]
  y_train = labels[testsize:]
  X_test = data_train[:testsize]
  y_test = labels[:testsize]
  
  # train, predict and write output files for failed predictions:
  if FLAGS.mixed == 1:                 
    # train the SVC                   
    classif.set_params(C = params_SVC['C'], 
                       gamma = params_SVC['gamma'])
    classif.fit(X_train, y_train)
    
    # create a new feature made of its predictions for training and test sets
    X_train_custom = mdl.newfeat(classif, X_train)
    X_test_custom = mdl.newfeat(classif, X_test)
    
    # train the final model
    mdl.traintest(model = logistic,
                  switch = FLAGS.model,
                  params = params_LR,
                  X_train = X_train_custom,
                  y_train = y_train,
                  X_test = X_test_custom,
                  y_test = y_test,
                  testsize = testsize,
                  order = order)
                  
  else:
    # train each model separately
    if FLAGS.model == 'SVC':
      mdl.traintest(model = classif,
                    switch = FLAGS.model,
                    params = params_SVC,
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_test,
                    y_test = y_test,
                    testsize = testsize,
                    order = order)
    elif FLAGS.model == 'KNN':                    
      mdl.traintest(model = knear,
                    switch = FLAGS.model,
                    params = params_KNN,
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_test,
                    y_test = y_test,
                    testsize = testsize,
                    order = order)
    elif FLAGS.model == 'LR':                  
      mdl.traintest(model = logistic,
                    switch = FLAGS.model,
                    params = params_LR,
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_test,
                    y_test = y_test,
                    testsize = testsize,
                    order = order)
    elif FLAGS.model == 'Tree':                  
      mdl.traintest(model = dtree,
                    switch = FLAGS.model,
                    params = params_Tree,
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_test,
                    y_test = y_test,
                    testsize = testsize,
                    order = order)
    else:
      print("No model selected. Please choose one with the --model option.")
      
##############################################
# Fitting and predictions outputs (multi mode)
##############################################
elif FLAGS.multi == 1:
  # train a KNN, SVC, LR on the data
  knn_pred = mdl.trainpred(model = knear,
                           switch = FLAGS.model,
                           params = params_KNN,
                           X_train = data_train,
                           y_train = labels,
                           X_test = data_test)
  svm_pred = mdl.trainpred(model = classif,
                           switch = FLAGS.model,
                           params = params_SVC,
                           X_train = data_train,
                           y_train = labels,
                           X_test = data_test)
  lr_pred = mdl.trainpred(model = logistic,
                          switch = FLAGS.model,
                          params = params_LR,
                          X_train = data_train,
                          y_train = labels,
                          X_test = data_test)
                           
  # average the predictions of the different models
  predictions = ((knn_pred + svm_pred + lr_pred)/3 + 0.5).astype(int)
  
  # score on training test
  pred = ((knear.predict(data_train) + classif.predict(data_train) + logistic.predict(data_train))/3 + 0.5).astype(int)
  score = 1 - ((abs(labels-pred)).sum() / len(data_train))
  print("Total training score: %0.3f" % score)
  
  # Write the prediction file
  prt.outwriter(filename = FLAGS.pred_output,
                testindex = index,
                predictions = predictions)
              
##############################################
# Fitting and predictions outputs (mixed mode)
##############################################
elif FLAGS.mixed == 1:
  # train the SVC                   
  classif.set_params(C = params_SVC['C'], 
                     gamma = params_SVC['gamma'])
  classif.fit(data_train, labels)
  
  # create a new feature made of its predictions for training and test sets
  data_train_custom = mdl.newfeat(classif, data_train)
  data_test_custom = mdl.newfeat(classif, data_test)
  
  # train the final model
  predictions = mdl.trainpred(model = logistic,
                              switch = FLAGS.model,
                              params = params_LR,
                              X_train = data_train_custom,
                              y_train = labels,
                              X_test = data_test_custom)
  
  # Write the prediction file
  prt.outwriter(filename = FLAGS.pred_output,
                testindex = index,
                predictions = predictions)


###############################################
# Fitting and predictions outputs (single mode)
###############################################
else:
  # train a single model on the data
  if FLAGS.model == 'SVC':
    predictions = mdl.trainpred(model = classif,
                                switch = FLAGS.model,
                                params = params_SVC,
                                X_train = data_train,
                                y_train = labels,
                                X_test = data_test)
  elif FLAGS.model == 'KNN':                             
    predictions = mdl.trainpred(model = knear,
                                switch = FLAGS.model,
                                params = params_KNN,
                                X_train = data_train,
                                y_train = labels,
                                X_test = data_test)
  elif FLAGS.model == 'LR':                               
    predictions = mdl.trainpred(model = logistic,
                                switch = FLAGS.model,
                                params = params_LR,
                                X_train = data_train,
                                y_train = labels,
                                X_test = data_test)
  elif FLAGS.model == 'Tree':                               
    predictions = mdl.trainpred(model = dtree,
                                switch = FLAGS.model,
                                params = params_Tree,
                                X_train = data_train,
                                y_train = labels,
                                X_test = data_test)
  else:
    print("No model selected. Please choose one with the --model option.")  
    
  # Write the prediction file
  prt.outwriter(filename = FLAGS.pred_output,
                testindex = index,
                predictions = predictions)
  

