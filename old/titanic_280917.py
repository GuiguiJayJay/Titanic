"""
================================
Titanic challenge draft
================================

v.280917
Supports mixed-model mode, where a KNN feeds an SVC via a new feature
made of its predictions.
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
from sklearn import neighbors

########################
# PARSE INPUTS
########################
parser = argparse.ArgumentParser(description='Define important inputs.')
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
parser.add_argument("--years",
                    type=int,
                    default=15,
                    help="Limit of age for childrens.")
parser.add_argument("--sibsp",
                    type=int,
                    default=2,
                    help="Number of categories for SibSp to be kept (see 'preproc.dataformat')")
parser.add_argument("--parch",
                    type=int,
                    default=1,
                    help="Number of categories for Parch to be kept (see 'preproc.dataformat')")
parser.add_argument("--fare1",
                    type=float,
                    default=10,
                    help="Upper limit of lowest fare category.")
parser.add_argument("--fare2",
                    type=float,
                    default=50,
                    help="Lower limit of highest fare category.")
parser.add_argument("--title",
                    type=int,
                    default=10,
                    help="Min of data required for Title field (see 'preproc.dataformat')")
parser.add_argument("--cabin",
                    type=int,
                    default=50,
                    help="Min of data required for Cabin field (see 'preproc.dataformat')")
                    
FLAGS, unparsed = parser.parse_known_args()
limits = {"years": FLAGS.years,
          "sibsplimit": FLAGS.sibsp,
          "parchlimit": FLAGS.parch,
          "farebound1": FLAGS.fare1,
          "farebound2": FLAGS.fare2,
          "titlelimit": FLAGS.title,
          "cabinlimit": FLAGS.cabin}


########################
# DATA PRE-PROCESSING
########################
# load the data
data_train_raw = prep.dataload(FLAGS.train_data)
data_test_raw = prep.dataload(FLAGS.test_data)

# format and clean up the data
droplist = ['Ticket', 'Survived']
fullset, trainsize, labels = prep.dataformat(data_train_raw,
                                             data_test_raw,
                                             droplist=droplist,
                                             limits=limits)

# one-hot encoding and split back data
mask=[True,True,True,True,True,True,True,True,True]  
data_train, data_test, labels = prep.dataonehot(dataset=fullset,
                                                labels=labels,
                                                mask=mask,
                                                trainsize=trainsize)

# scaling numerical features
data_train = prep.datanorm(data_train)
data_test = prep.datanorm(data_test)

# model and parameters definitions
classif = svm.SVC(class_weight='balanced', cache_size=2000)
params_SVC = {'C': 15.51, 'gamma': 0.0019, 'kernel': 'rbf'}

knear = neighbors.KNeighborsClassifier(algorithm='kd_tree')
params_KNN = {'n_neighbors': 6, 'weights': 'distance' }
  
###############################
# grid search on the parameters
###############################
if FLAGS.grid == 1:
  # prepare the 'shuffle and split' tool
  n_splits = 5
  n_repeats = 20
  test_size = 0.2
  shuffle = ShuffleSplit(n_splits=n_splits, test_size=test_size,random_state=None)
  rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)
  
  # parameters to be tested
  param_C = np.linspace(10,100,10)
  param_Gam = np.linspace(0.0001,0.01,10)
  param_neighbors = range(2,20)
  params_SVC = [{'C': param_C, 'gamma': param_Gam, 'kernel': ['rbf']}]
  params_KNN = [{'n_neighbors': param_neighbors, 'weights': ['uniform', 'distance']}]

  data_train_custom = data_train
  if FLAGS.multi == 1:
    # grid search
    feeds = mdl.gridsearch(model = knear,
                           switch = 'KNN',
                           params = params_KNN,
                           X_train = data_train,
                           y_train = labels,
                           splits = rkf,
                           test_size = test_size,
                           n_splits = n_splits)
                           
    # train the knn on the full set, and create a new feature made of its predictions                      
    knear.set_params(n_neighbors = feeds['n_neighbors'], 
                     weights = feeds['weights'])
    knear.fit(data_train, labels)
    pred_knn_train = knear.predict(data_train)
    newfeat_train = pred_knn_train.reshape([len(pred_knn_train), 1])
    data_train_custom = np.concatenate((data_train, newfeat_train), axis=1)
  
  # grid search
  nothing = mdl.gridsearch(model = classif,
                           switch = 'SVC',
                           params = params_SVC,
                           X_train = data_train_custom,
                           y_train = labels,
                           splits = rkf,
                           test_size = test_size,
                           n_splits = n_splits)
            
###################################
# Fitting and predictions (no grid)
###################################
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
  mdl.trainpred(model = classif,
                switch = 'SVC',
                params = params_SVC,
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                testsize = testsize,
                order = order)
  mdl.trainpred(model = knear,
                switch = 'KNN',
                params = params_KNN,
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                testsize = testsize,
                order = order)

elif FLAGS.multi == 1:
  # train a knn on the vanilla data
  knear.set_params(n_neighbors = params_KNN['n_neighbors'], 
                   weights = params_KNN['weights'])
  knear.fit(data_train, labels)
  
  # add knn predictions to train features
  pred_knn_train = knear.predict(data_train)
  newfeat_train = pred_knn_train.reshape([len(pred_knn_train), 1])
  data_train_custom = np.concatenate((data_train, newfeat_train), axis=1)
  
  # add knn predictions to test features
  pred_knn_test = knear.predict(data_test)
  newfeat_test = pred_knn_test.reshape([len(pred_knn_test), 1])
  data_test_custom = np.concatenate((data_test, newfeat_test), axis=1)
  
  # train a SVC on the new data
  classif.set_params(C = params_SVC['C'],
                     gamma = params_SVC['gamma'],
                     kernel = params_SVC['kernel'])
  classif.fit(data_train_custom, labels)
  predictions = classif.predict(data_test_custom)

  # display accuracy
  print("Train accuracy: %0.3f" % classif.score(data_train_custom, labels))
  print("\tC: %0.2f" % params_SVC['C'])
  print("\tgamma: %0.4f" % params_SVC['gamma'])
  print("\tneighbors: %0.4f" % params_KNN['n_neighbors'])
  print("\tweights: %r" % params_KNN['weights'])

  # Write the prediction file
  index = data_test_raw.index.tolist()
  prt.outwriter(filename = FLAGS.pred_output,
                testindex = index,
                predictions = predictions)

else:
  # train a simple SVC on the data
  classif.set_params(C = params_SVC['C'],
                     gamma = params_SVC['gamma'],
                     kernel = params_SVC['kernel'])
  classif.fit(data_train, labels)
  predictions = classif.predict(data_test)

  # display accuracy
  print("Train accuracy: %0.3f" % classif.score(data_train, labels))
  print("\tC: %0.2f" % params_SVC['C'])
  print("\tgamma: %0.4f" % params_SVC['gamma'])
  
  # Write the prediction file
  index = data_test_raw.index.tolist()
  prt.outwriter(filename = FLAGS.pred_output,
                testindex = index,
                predictions = predictions)
  

