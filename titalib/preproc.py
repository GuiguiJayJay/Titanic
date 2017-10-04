import pandas as pd
import numpy as np
import matplotlib as plt
import math

import titalib.printer as prt

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
def dataformat(trainset, testset, limits, droplist=[]):
  """ pre-process the train data (removing features, reformat some etc...).
  
  Parameters
  ==========
  trainset: pandas.DataFrame
    the input training data to be processed.
    
  testset: pandas.DataFrame
    the input training data to be processed.
    
  limits: python dict
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
    
  droplist: python list (optional)
    list of features to throw from data
  
  Returns
  =======
  trainset: pandas.DataFrame
    the modified ("pre-processed") training dataset.
    
  testset: pandas.DataFrame
    the modified ("pre-processed") test dataset.
    
  labels: pandas.Series
    the labels associated with the training set.
  """

  le = preprocessing.LabelEncoder()
  
  trainsize = len(trainset)
  testsize = len(testset)
  
  # add a 'Survived' column to test set and merge the sets
  testset = testset.assign(Survived=pd.Series(np.zeros(testsize,dtype=int)).values)
  dataset = trainset.append(testset,verify_integrity=True)
  
  # add columns to data: 'Relatives', 'Location'
  dataset = dataset.assign(Relatives=pd.Series(np.zeros(testsize+trainsize,dtype=int)).values)
  dataset = dataset.assign(Location=pd.Series(np.zeros(testsize+trainsize,dtype=int)).values)
  
  # create settings dict: NaN in specified feature? (bool)
  settings = dict(dataset.isnull().any())
   
  # For various features:
  # - convert classes into numerical variables
  # - save a list of original classes (order by increasing integer representation)
  # - replace 'Nan' by dummy values for subsequent work
      
  # Gender
  dataset['Sex'] = le.fit_transform(dataset['Sex'])
  
  # Embarkment port: merge the 2 lowest populated categories
  dataset['Embarked'].fillna('S', inplace=True) # the most frequent class (concerns only 2 entries)
  dataset.loc[dataset['Embarked']!='S', 'Embarked'] = 'A'
  dataset['Embarked'] = le.fit_transform(dataset['Embarked'])

  # Ticket: use the number of identical ticket delivered to normalize fare if Cabin is NaN
  dataset['Cabin'].fillna('Z', inplace=True)
  ticket = dataset['Ticket'].value_counts().to_dict()
  ticketreduced = dict((k, v) for k, v in ticket.items() if v >= 2) # what needs to be normalized
  for item in ticketreduced.keys():
    dataset.loc[(dataset['Ticket']==item) & (dataset['Cabin']=='Z'),'Fare'] \
          = dataset.loc[(dataset['Ticket']==item) & (dataset['Cabin']=='Z'),'Fare'] / ticketreduced[item]
  
  # Fare: normalize by number of cabin booked
  dataset['Cabin'] = dataset['Cabin'].str.split()
  normfactor = dataset['Cabin'].apply(lambda x: len(x))
  dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True) # only 1 data entry concerned
  dataset['Fare'] = dataset['Fare']/normfactor
  # Fare = 0 means special ticket. Re-allocate to the 3 fare categories depending on Pclass
  dataset.loc[dataset['Fare']==0, 'Fare'] = (dataset.loc[dataset['Fare']==0, 'Pclass'] - 1)*(limits['farebound1']+limits['farebound2'])/2
  # Fare: convert to cat.-feat. and reduce it to 3 classes
  dataset.loc[dataset['Fare']<limits['farebound1'], 'Fare'] = 0
  dataset.loc[(dataset['Fare']<limits['farebound2']) & (dataset['Fare']>=limits['farebound1']),'Fare'] = 1
  dataset.loc[dataset['Fare']>=limits['farebound2'], 'Fare'] = 2
  dataset['Fare'] = dataset['Fare'].astype(int) # converts to int
  
  # Cabin: convert to cat.-feat.
  dataset['Cabin'] = dataset['Cabin'].astype(str).str[2]
  dataset['Cabin'] = le.fit_transform(dataset['Cabin'])
  
  # Location:  manually fill it depending on Cabin and Pclass
  dataset.loc[dataset['Pclass']==3,'Location'] = 0
  dataset.loc[dataset['Pclass']==2,'Location'] = 1
  dataset.loc[dataset['Pclass']==1,'Location'] = 2
  dataset.loc[(dataset['Pclass']==2) & (dataset['Cabin']<8),'Location'] = 3
  dataset.loc[(dataset['Pclass']==1) & (dataset['Cabin']>=2) & (dataset['Cabin']<=4),'Location'] = 3
  
  # Title: extract it from the 'Name', there are correlations with age that can be used
  dataset['Name'] = dataset['Name'].str.split(',').str.get(1).str.split().str.get(0) # title comes right after comma
  titles = dataset['Name'].value_counts().to_dict() # dict of {titles: counts}
  garbage =  dict((k, v) for k, v in titles.items() if v <= limits['titlelimit']) # titles to be garbaged 
  # Fancy titles are not for kids, but for elder peoples:
  for title in garbage.keys():
    dataset.loc[(dataset['Name']==title) & (dataset['Age'].isnull()), 'Age'] = limits['child']+1
  # Correlation with 'Master.' title is clear
  dataset.loc[(dataset['Name']=='Master.') & (dataset['Age'].isnull()), 'Age'] \
          = dataset.loc[dataset['Name']=='Master.', 'Age'].mean()
  # Kids mostly don't travel alone (Parch = 0 means no kid or no parents, so an adult)
  dataset.loc[(dataset['Name']=='Miss.') & (dataset['Parch']==0) & (dataset['Age'].isnull()), 'Age'] \
          = dataset.loc[(dataset['Name']=='Miss.') & (dataset['Parch']==0) & (dataset['Age']>0), 'Age'].mean()
  dataset.loc[(dataset['Name']=='Mr.') & (dataset['Parch']==0) & (dataset['Age'].isnull()), 'Age'] \
          = dataset.loc[(dataset['Name']=='Mr.') & (dataset['Parch']==0) & (dataset['Age']>0), 'Age'].mean()
  # Married ladies aren't child anymore (I hope so at least)
  dataset.loc[(dataset['Name']=='Mrs.') & (dataset['Age'].isnull()), 'Age'] \
          = dataset.loc[(dataset['Name']=='Mrs.') & (dataset['Age']>0), 'Age'].mean()
  # For the others, let the current statistic decide
  dataset.loc[(dataset['Name']=='Miss.') & (dataset['Parch']>0) & (dataset['Age'].isnull()), 'Age'] \
          = dataset.loc[(dataset['Name']=='Miss.') & (dataset['Parch']>0) & (dataset['Age']>0), 'Age'].mean()
  dataset.loc[(dataset['Name']=='Mr.') & (dataset['Parch']>0) & (dataset['Age'].isnull()), 'Age'] \
          = dataset.loc[(dataset['Name']=='Mr.') & (dataset['Parch']>0) & (dataset['Age']>0), 'Age'].mean()
  
  # slice age in 2 categories
  dataset.loc[dataset['Age']<limits['child'], 'Age'] = 0
  dataset.loc[dataset['Age']>=limits['child'], 'Age'] = 1
  dataset['Age'] = dataset['Age'].astype(int) # converts to int
  
  # Encode title as a feature, who knows?
  keep =  dict((k, v) for k, v in titles.items() if v > limits['titlelimit']) # titles to be kept
  encoding = 0
  for title in keep.keys(): # assign one value per title kept
    dataset.loc[dataset['Name'] == title, 'Name'] = encoding
    encoding = encoding + 1
  for title in garbage.keys(): # assign a single value for all garbaged titles
    dataset.loc[dataset['Name'] == title, 'Name'] = encoding
  dataset['Name'] = le.fit_transform(dataset['Name'])
  
  # Relatives:  fill it depending on SibSp and Parch (family size estimation)
  dataset.loc[(dataset['Parch']==0) & (dataset['SibSp']==1), 'Relatives'] = 1
  dataset.loc[(dataset['Parch']==1) & (dataset['SibSp']<=2), 'Relatives'] = 1
  dataset.loc[(dataset['Parch']==2) & (dataset['SibSp']<=1), 'Relatives'] = 1
  
  # SibSp:  convert to cat.-feat. and reduce it to (sibsplimit+1) classes
  dataset.loc[dataset['SibSp']>=limits['sibsplimit'], 'SibSp'] = limits['sibsplimit']
  
  # Parch:  convert to cat.-feat. and reduce it to (parchlimit+1) classes
  dataset.loc[dataset['Parch']>=limits['parchlimit'], 'Parch'] = limits['parchlimit']
  
  # extract labels as a separate dataframe
  labels = dataset['Survived'].head(trainsize)
  
  # get rid of the features given in droplist if they are in the current dataset
  removelist = [item for item in droplist if item in list(dataset)]
  for item in removelist:
    dataset.drop(item, axis=1, inplace=True)
    
  # print infos regarding the data and features pre-proc
  prt.featinfo(dataset,limits,settings)
  
  return dataset, trainsize, labels
  

############################################  
def dataonehot(dataset, labels, trainsize, mask='all'):
  """ perform one-hot encoding of categorical features according to a mask.
  
  Parameters
  ==========
  dataset: pandas.DataFrame
    the input data to be processed.
    
  labels: pandas.Series
    the labels of training set.
    
  trainsize: integer
    size of training set.
    
  mask: boolean array (optional)
    mask of the input features to be 'one-hot encoded'. By default, all
    features will be treated as categorical.
  
  Returns
  =======
  trainset: numpy ndarray (2D)
    the one-hot encoded training dataset.
    
  testset: numpy ndarray (2D)
    the one-hot encoded test dataset.
    
  labels: numpy ndarray (1D)
    the labels associated with the training set.
  """
  
  # convert dataframes to 2D ndarrays
  dataset = dataset.as_matrix()
  labels = labels.as_matrix()

  # perform one-hot encoding
  enc = preprocessing.OneHotEncoder(sparse=True,categorical_features=mask)
  dataset = enc.fit_transform(dataset)
  
  numfeat = len(dataset.toarray()[0])
  print("----> Total number of Features = %s" % numfeat)
  print("============================================")
  print()
  
  # split back the data to a train and test set
  trainset = dataset.toarray()[:trainsize,:]
  testset = dataset.toarray()[trainsize:,:]
  
  return trainset, testset, labels


############################################  
def datanorm(dataset):
  """ perform scaling on input features to mean=0 and variance=1.
  
  Parameters
  ==========
  dataset: numpy ndarray (2D)
    the input data to be processed.
  
  Returns
  =======
  dataset_scaled: numpy ndarray (2D)
    the pre-processed data with one-hot encoding and numerical features 
    scaled to mean=0 and variance=1 as a 2D matrix.
  """
  
  dataset_scaled = preprocessing.scale(dataset)
  
  return dataset_scaled
