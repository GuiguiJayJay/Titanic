import numpy as np

import titalib.printer as prt

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

############################################
def trainpred(model, switch, params, X_train, y_train, X_test):
  """ training and predictions for an input model.
  
  Parameters
  ==========
  model: scikit learn model
    the input scikit learn model.
    
  switch: string
    a simple (but not supple) way of determining the kind
    of model is input. This helps to set model's parameters.
    Possible values are 'SVC','KNN','LR'
    
  params: python dict
    parameters for the model.
    
  X_train, y_train, X_test: numpy array (2D for X, 1D for y)
    arrays containing the training/test data, or the training
    labels.
    
  Returns
  =======
  predictions: numpy array (1D)
    array containing the predictions for the given test file.
    
  """

  # set model parameters
  if switch == 'SVC':
    model.set_params(C = params['C'],
                     gamma = params['gamma'],
                     kernel = params['kernel'])
  elif switch == 'KNN':
    model.set_params(n_neighbors = params['n_neighbors'],
                     weights = params['weights'])
  elif switch == 'LR':
    model.set_params(C = params['C'],
                     penalty = params['penalty'])
  else:
    print('No model type has been specified, parameters cannot be set and trainig will use default parameters.')
  
  # train and test the models
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)

  # display accuracy and parameters
  print("--------------------------")
  print("Model: %r" % switch)
  print("Train accuracy: %0.3f" % model.score(X_train, y_train))
  for name in params.keys():
    if type(params[name]) == str:
      print("\t%r: \t%r" % (name,params[name]))
    else:
      print("\t%r: \t%0.4f" % (name,params[name]))
  print("--------------------------")

  return predictions
  

############################################
def traintest(model, switch, params, X_train, y_train, X_test, y_test, testsize, order):
  """ training and predictions for an input model given manually shuffled and split
  data from the training set (model testing purposes).
  
  Parameters
  ==========
  model: scikit learn model
    the input scikit learn model.
    
  switch: string
    a simple (but not supple) way of determining the kind
    of model is input. This helps to set model's parameters.
    Possible values are 'SVC','KNN','LR'
    
  params: python dict
    parameters for the model.
    
  X_train, y_train, X_test, y_test: numpy array (2D for X, 1D for y)
    arrays containing the training/test data, or the training/test
    labels.
    
  testsize: integer
    the size of the test set.
    
  order: unumpy array (1D)
    the order of permutations applied to data sets. Necessary if one
    wants to look at those precise data in pandas dataframe. This
    array is written in a text file.
    
  Returns
  =======
  """

  # set model parameters
  if switch == 'SVC':
    model.set_params(C = params['C'],
                     gamma = params['gamma'],
                     kernel = params['kernel'])
  elif switch == 'KNN':
    model.set_params(n_neighbors = params['n_neighbors'],
                     weights = params['weights'])
  elif switch == 'LR':
    model.set_params(C = params['C'],
                     penalty = params['penalty'])
  else:
    print('No model type has been specified, parameters cannot be set and trainig \
            will use default parameters.')
  
  # train and test the models
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  
  true_posneg = 0
  fails = np.ndarray(shape=(testsize,1),dtype=bool)
  for i in range(0,testsize):
    if predictions[i] == y_test[i]:
      true_posneg = true_posneg + 1
      fails[i] = True
    else:
      fails[i] = False
  test_acc = true_posneg/testsize

  # display accuracy and parameters
  print("--------------------------")
  print("Model: %r" % switch)
  print("\tTrain accuracy: %0.3f" % model.score(X_train, y_train)) 
  print("\tTest accuracy: %0.3f" % test_acc)
  print("--------------------------")

  # Write the fails output file
  prt.fails(fails,order,switch)

  return None


############################################
def gridsearch(model, switch, params, X_train, y_train, splits, test_size, n_repeats):
  """ training and predictions for an input model.
  
  Parameters
  ==========
  model: scikit learn model
    the input scikit learn model.
    
  switch: string
    a simple (but not supple) way of determining the kind
    of model is input. This helps to set model's parameters.
    Possible values are 'SVC','KNN','LR'
    
  params: python dict
    parameters for the model.
    
  X_train, y_train: numpy array (2D for X, 1D for y)
    arrays containing the training data or the labels.
    
  splits: sklearn ShuffleSplit class object
    class describing a shuffle of the data, and a number of splits,
    each containing a training and a cross-validation set.
    
  test_size: float
    the size of the test set.
    
  n_repeats: integer
    the number of repetitions of splitting, fitting, and testing
    the model.
    
  Returns
  =======
  best_parameters: python dict
    the dictionnary of the best parameters set from the grid search.
  """

  # grid search
  grid_search = GridSearchCV( model,
                              params,
                              cv=splits,
                              n_jobs=-1,
                              verbose=1)
  grid_search.fit(X_train, y_train)

  # display basic setting infos
  bestind = grid_search.best_index_
  print("--------------------------")
  print("Model: %r" % switch)
  print("\ttest_size = %0.2f" % test_size)
  print("\tn_repeats = ", n_repeats)
  print("--------------------------")
  
  # display top models results
  duplicates1 = prt.topmod(results=grid_search.cv_results_, 
                           params=params,
                           top=1, 
                           skip=0)
  #duplicates2 = prt.topmod(results=grid_search.cv_results_,
                           #params=params,
                           #top=2,
                           #skip=duplicates1-1)

  # Write grid search results for eventual analysis
  includelist = [ 'params',
                  'mean_test_score',
                  'std_test_score',
                  'rank_test_score',
                  'mean_train_score',
                  'std_train_score']
  prt.monitor(results = grid_search.cv_results_,
              includelist = includelist,
              full_log = True,
              model = switch)
              
  best_parameters = grid_search.best_estimator_.get_params()

  return best_parameters
  
  
############################################
def newfeat(model, data):
  """ predicts outputs given a model, and add a new (binary) feature 
      to the data from it.
  
  Parameters
  ==========
  model: scikit learn model
    the input scikit learn model.
    
  data: numpy array (2D)
    arrays containing the training data.
    
  Returns
  =======
  data_custom: numpy array (2D)
    array containing the data with the new feature appended as a last column
    for all the data entries.
  """

  predictions = model.predict(data)
  newfeature = predictions.reshape([len(predictions), 1])
  data_custom = np.concatenate((data, newfeature), axis=1)

  return data_custom
