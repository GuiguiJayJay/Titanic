import numpy as np
import pandas as pd
import csv


############################################
def topmod(results,params,top=1,skip=0):
  """ print informations about the top models
  
  Parameters
  ==========
  results: python dict
    dictionary of grid search results from scikit-learn
    (the 'cv_results_' attribute of gridsearch CV object).
    
  params: python list of dict
    list of dictionaries of params used by gridsearch.
    
  top: int (optional)
    rank of top score to be printed (for example 1 means 
    print only infos from the model which has the best
    test score, or all the model yeilding the best test
    score in case of duplicates).
    
  skip: int (optional)
    if you wnat to print the rank 2 top scores, you have 
    to skip all the rank 1 scores. It is thus important 
    to know what is the number of duplicates for rank 1
    top scores to know how many elements have to be skip.
    
  Returns
  =======
  duplicates: integer
    the number of models yielding the same top best score.
  """

  TopResults = np.where(results['rank_test_score']==top+skip) # index of all top score
  Top_train_score = results['mean_train_score'][TopResults[0]]
  Top_train_std = results['std_train_score'][TopResults[0]]
  Top_test_score = results['mean_test_score'][TopResults[0]]
  Top_test_std = results['std_test_score'][TopResults[0]]
  
  duplicates = len(TopResults[0])
  print("Top results %s (multiplicity %s):" % (top,duplicates))
  for i in range(0,duplicates):
    print("  mean test score: %0.3f +- %0.3f" % (Top_test_score[i],
                                                Top_test_std[i]))
    print("  mean train score: %0.3f +- %0.3f" % (Top_train_score[i],
                                                Top_train_std[i]))
    for j in range(0,len(params)):                          
      for name in params[j].keys():
        paramval = results['param_'+name][TopResults[0]][i]
        if type(paramval) == str:
          print("\t%r: \t%r" % (name,paramval))
        else:
          print("\t%r: \t%0.4f" % (name,paramval))

  return duplicates


############################################
def monitor(results, includelist, full_log=False, model=''):
  """ write informations about the fitting of splits in a file
  
  Parameters
  ==========
  results: python dict
    dictionary of grid search results from scikit-learn
    (the 'cv_results_' attribute of gridsearch CV object).
    
  includelist: python list
    list of all items from grid search to be written in
    output file.
    
  full_log: bool (optional)
    setting it to True will also write a full log output file
    which include the full content of cv_results_.
    
  model: string (optional)
    the name of the model which infos will be written. used for
    the output filename.
    
  Returns
  =======
  """
  
  print()
  filename = 'grid_output_' + model + '.txt'
  outfile = open(filename, 'w')
                  
  for item in includelist:
    if item == 'params':
      outfile.write("%r \n" % item)
      for index in range(0,len(results['params'])):
        outfile.write("\t index %s: %r \n" % (index,results['params'][index]))
    else:
      outfile.write("%r \t %s \n" % (item,results[item]))  
  outfile.close()
  print("Output file written: %r" % filename)

  if full_log == True:
    full_log_name = 'cv_log_' + model + '.txt'
    full = open(full_log_name, 'w+')
    for item in results.keys():
      full.write("%r \t %s \n" % (item,results[item]))
    full.close
    print("Output file written: %r" % full_log_name)
    
  print("============================================")
  print()
      
  return None


############################################
def featinfo(dataset,limits={},settings={}):
  """ print info about pre-processing of features
  
  Parameters
  ==========
  dataset: pandas.DataFrame
    the input data to be processed.
    
  limits: python dict (optional)
    'child':  the limit age to define a passenger as a child or not.
    'sibsplimit': the sibsplimit, the number of categories
              for the 'SibSp' feature to be kept. Remaining elements
              are put into an additional category.
    'parchlimit': the parchlimit, the number of categories
              for the 'Parch' feature to be kept. Remaining elements
              are put into an additional category.
    'farebound1': the fare bound 1, the upper limit below which
              we define the lowest fares category.
    'farebound2': the fare bound 2, the lower limit above which
              we define the highest fares category. Anything between
              those 2 boundaries is considered as the 3rd fare category.
    'titlelimit': the limit amount of passengers registered in with a given
              title category required to keep it. Title categories below will 
              all be put in a 'garbage' category.
    'merged': the number of cabin categories that have been merged into
              the 'garbage' category.
              
  settings: python dict (optional)
    dictionnary of {feature: bool} telling whether or not a 'NaN' is
    in the data for the given feature.
    
  Returns
  =======
  """
  
  # Which features contains NaN (union of test and train sets)
  print("NaN in the features:")
  NanList = []
  for name in settings.keys():
    if settings[name] == True:
      #print(" %s " % name)
      NanList.append(name) 
  print(" %r" % NanList)
  print()
  
  # Print other categories
  for name in dataset.keys():
    if name == 'Age':
      print("%r categories: %s (childs under %s)"
              % (name, len(dataset[name].unique()), limits['child']))
    elif name == 'Cabin':
      print("%r categories: %s (%s merged)" 
              % (name, len(dataset['Cabin'].unique()), limits['merged']))
    elif name == 'Fare':
      print("%r categories: %s (delimiters: %s and %s)" 
              % (name, len(dataset[name].unique()), limits['farebound1'], limits['farebound2']))
    else:
      print("%r categories: %s" % (name,len(dataset[name].unique())) )

  return None


############################################
def outwriter(filename, testindex, predictions):
  """ print info about pre-processing of features
  
  Parameters
  ==========
  filename: string
    name of the output csv file to be written.
    
  testindex: python list
    pandas dataframe's index of test data converted as a list.
    in other words, this is a list of integers, the PassengerId
    of the test data.
    
  predictions: numpy ndarray (1D)
    our predictions stored as an array.
    
  Returns
  =======
  """

  print()
  csvfile = open(filename, 'w', newline='')
  filewriter = csv.writer(csvfile)
  
  # write the header header as required by challenge's rules
  filewriter.writerow(['PassengerId', 'Survived'])
  
  # write the Passenger Id and its corresponding status after sink
  for i in range(0,len(testindex)):
    filewriter.writerow([testindex[i],predictions[i]])
    
  print("Output file written: %r" % filename)
  print("============================================")
  print()
  
  return None


############################################
def fails(datafails, order, model):
  """ print data index associated to failed predictions
  
  Parameters
  ==========
  datafails: numpy array (1D)
    bool array of failed (false) or successful (true) predictions
    
  order: numpy array (1D)
    array of permutations applied on the data. Basically, this array
    is the list of index in the new order.
    
  model: string
    the model used for the predictions used. This string will be
    added to the output filename.
    
  Returns
  =======
  """
  
  print()
  filename = 'fails_' + model + '.txt'
  failedfile = open(filename, 'w+')
  
  for i in range(0,len(datafails)):
    if datafails[i] == False:
      failedfile.write("%s," % order[i])
  failedfile.close()
  
  print("Output file written: %r" % filename)
  print("============================================")
  print()

  return None
