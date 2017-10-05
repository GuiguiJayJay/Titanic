import pandas as pd
import numpy as np
import matplotlib as plt
import math

from sklearn import preprocessing


############################################
def dataload(filename='data/train.csv'):
  """ load the data in a pandas dataframe.
  
  Parameters
  ==========
  filename: string
    location and name of the file to be read.
    
  Returns
  =======
  data: pandas.DataFrame
    the data in a pandas dataframe.
  """

  data = pd.read_csv(filepath_or_buffer=filename, index_col='PassengerId')

  return data


############################################
def dataformat(dataset, years=8):
  """ pre-process the data (removing features, reformat some etc...).
  
  Parameters
  ==========
  dataset: pandas.DataFrame
    the input data to be processed.
    
  years: integer (optional)
    the size in years of slices for the 'Age' feature after conversion
    to a class feature. 8 by default.
  
  Returns
  =======
  dataset: pandas.DataFrame
    the modified ("pre-processed") dataset.
    
  labels: pandas.Series
    the labels associated with dataset.
    
  Gender: python list
    list of class-names for the 'Gender' feature, ordered from lowest
    integer representation to highest.
    
  Embarkport: python list
    list of class-names for the 'Embarkment Port' feature, ordered from lowest
    integer representation to highest.
    
  Cabin: python list
    list of class-names for the 'Cabin' feature, ordered from lowest
    integer representation to highest.    
  """
  
  le = preprocessing.LabelEncoder()
  
  # extract labels as a separate dataframe
  labels = dataset['Survived']
  
  # get rid of irrelevant fields
  dataset.drop('Name', axis=1, inplace=True)
  dataset.drop('Ticket', axis=1, inplace=True)
  dataset.drop('Survived', axis=1, inplace=True)
    
  # For various features:
  # - convert classes into numerical variables
  # - save a list of original classes (order by increasing integer representation)
  # - replace 'Nan' by dummy values for subsequent work
  
  # Gender
  dataset['Sex'] = le.fit_transform(dataset['Sex'])
  Gender = list(le.classes_)
  
  # Embarkment port
  dataset['Embarked'].fillna('Z', inplace=True)
  dataset['Embarked'] = le.fit_transform(dataset['Embarked'])
  Embarkport = list(le.classes_)
  
  # Fare: normalize by number of Cabin booked
  dataset['Cabin'].fillna('Z', inplace=True)
  dataset['Cabin'] = dataset['Cabin'].str.split()
  normfactor = dataset['Cabin'].apply(lambda x: len(x))
  dataset['Fare'] = dataset['Fare']/normfactor
  
  # Cabin
  dataset['Cabin'] = dataset['Cabin'].astype(str).str[2]
  dataset['Cabin'] = le.fit_transform(dataset['Cabin'])
  Cabin = list(le.classes_)
  
  # Age: convert to class feature
  slices = math.ceil(dataset['Age'].max()/years)
  dataset['Age'].fillna(100, inplace=True)
  for i in range(0,slices):
    dataset.loc[(dataset['Age']<=years*(i+1)) & (dataset['Age']>years*i), 'Age'] = i
  dataset.loc[dataset['Age']==100, 'Age'] = slices # manually set the 'unknown' class (the last)
  dataset['Age'] = dataset['Age'].astype(int) # converts to int

  return dataset, labels, Gender, Embarkport, Cabin


############################################  
def dataonehot(dataset,labels,mask='all'):
  """ perform one-hot encoding of categorical features according to a mask.
  
  Parameters
  ==========
  dataset: pandas.DataFrame
    the input data to be processed.
    
  labels: pandas.Series
    the input labels to be processed.
    
  mask: boolean array (optional)
    mask of the input features to be 'one-hot encoded'. By default, all
    features will be treated as categorical.
  
  Returns
  =======
  datamat: sklearn sparse matrix
    the pre-processed data with one-hot encoding converted as a sparse matrix.
    
  labelmat: nd.array
    the labels with one-hot encoding converted as an array.
    
  """

  # converts dataframe into a np array (2D matrix form, entry VS features)
  datamat = dataset.as_matrix()
  labelmat = labels.as_matrix()
  
  # perform one-hot encoding
  enc = preprocessing.OneHotEncoder(sparse=True,categorical_features=mask)
  datamat = enc.fit_transform(datamat)
  
  return datamat, labelmat


############################################  
def datanorm(dataset,mask=0):
  """ perform scaling on input features to mean=0 and variance=1.
  
  Parameters
  ==========
  dataset: sklearn sparse matrix
    the input data to be processed.
    
  mask: integer (optional)
    number of numerical features, assuming all of them have to be
    scaled. The preprocessing.OneHotEncoder() function put all the
    non-categorical features to the last columns. Mask specify here 
    the number of column to be scaled satrting from the end.
  
  Returns
  =======
  dataset_scaled: nd.array (2D)
    the pre-processed data with one-hot encoding and numerical features 
    scaled to mean=0 and variance=1 as a 2D matrix.

  """
  
  dataset_scaled = preprocessing.scale(dataset.toarray())
  #dataset_temp = preprocessing.scale(dataset.toarray()[:,-mask:])
  #dataset_scaled = np.concatenate((dataset.toarray()[:,:-mask], dataset_temp), axis=1)
  
  return dataset_scaled
