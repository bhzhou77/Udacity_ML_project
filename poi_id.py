#!/usr/bin/python

import sys
import numpy as np
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
# I delete 'email_address' because it is irrelevant. I also delete 'loan_advances', 'total_paments', and 'restricted_stock_deferred', because they contains outliers. Check documented answers for question 1.
features_list = ['poi',
                 'salary',                  # feature 1
                 'deferral_payments',       # feature 2
                 'bonus',                   # feature 3
                 'deferred_income',         # feature 4
                 'total_stock_value',       # feature 5
                 'expenses',                # feature 6
                 'exercised_stock_options', # feature 7
                 'other',                   # feature 8
                 'long_term_incentive',     # feature 9
                 'restricted_stock',        # feature 10
                 'director_fees',           # feature 11
                 'shared_receipt_with_poi', # feature 12
                 'to_messages',             # feature 13
                 'from_messages',           # feature 14
                 'from_poi_to_this_person', # feature 15
                 'from_this_person_to_poi'] # feature 16

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
    
### Task 2: Remove outliers
# Featues containing outliers are not included, which are 'loan_advances', 'total_paments', and 'restricted_stock_deferred'.
# delete the key 'TOTAL'
del data_dict['TOTAL']


### Task 3: Create new feature(s)
# New features will be created after feature extraction below

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# covert labels and features to numpy arrays
labels = np.array(labels)
features = np.array(features)


#################################################################################################################################
# This block is only an experimental section for testing the new features, and the new features are not included in the final model.

# I want to use two new features: 
# 1. the ratio of 'to_messages' and 'from_messages'
# 2. the ratio of 'from_poi_to_this_person' and 'from_this_person_to_poi'
ratio1 = np.zeros_like(labels)
ratio2 = np.zeros_like(labels)
for ii in range(len(ratio1)):
    if features[ii,-3] != 0:
        ratio1[ii] = features[ii,-4]/features[ii,-3]
    if features[ii,-1] != 0:
        ratio2[ii] = features[ii,-2]/features[ii,-1]
# append ratio1 and ratio2 to features
features_extended = np.append(features,ratio1.reshape((len(ratio1),1)),axis=1)
features_extended = np.append(features,ratio2.reshape((len(ratio2),1)),axis=1)

# Now let's rescale the features_extended
features_rescaled = np.zeros_like(features_extended)
for jj in range(features_extended.shape[1]):
    max_value = features_extended[:,jj].max()
    min_value = features_extended[:,jj].min()
    if max_value != min_value:
        features_rescaled[:,jj] = (features_extended[:,jj]-min_value)/(max_value-min_value)
        
# Use SelectKBest to do feature selection
select = SelectKBest(k=12)
select.fit_transform(features_rescaled,labels)
#################################################################################################################################


#################################################################################################################################
# test code only, please ignore
# Now let's rescale the features
def feature_scaler(features=np.array([1]),labels=np.array([1])):
   
    features_rescaled = np.zeros_like(features)
    for jj in range(features.shape[1]):
        max_value = features[:,jj].max()
        min_value = features[:,jj].min()
        if max_value != min_value:
            features_rescaled[:,jj] = (features[:,jj]-min_value)/(max_value-min_value)
    
    return features,labels

features_rescaled,labels = feature_scaler(features,labels)
#################################################################################################################################


# Use SelectKBest to do feature selection
select = SelectKBest(k=10)
select.fit_transform(features_rescaled,labels)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# I will use pipeline and gridsearchcv to do the training
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics
from sklearn.cross_validation import StratifiedShuffleSplit

# use scaler in the gridsearchcv
Min_Max_scaler = MinMaxScaler()

# splitting the data into training and testing
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features_rescaled, labels, test_size=0.3, random_state=42)

# define the SelectKBest without assigning the k value
select = sklearn.feature_selection.SelectKBest()


########################################################
################ 1. Naive bayes ########################
########################################################

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

steps = [('scaler',Min_Max_scaler),
        ('feature_selection',select),
        ('naive_bayes',gnb)]

pipeline = Pipeline(steps)
parameters = dict(feature_selection__k=[5,10])
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)


########################################################
################ 2. Random forest ######################
########################################################
'''
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

steps = [('scaler',Min_Max_scaler),
        ('feature_selection',select),
        ('random_forest',rfc)]

pipeline = Pipeline(steps)
parameters = dict(feature_selection__k=[5,10], 
              random_forest__n_estimators=[50,100,200],
              random_forest__min_samples_split=[2,3,4,5,10])
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
'''

########################################################
################ 3. svc ######################
########################################################
'''
from sklearn.svm import SVC
svc = SVC()

steps = [('scaler',Min_Max_scaler),
        ('feature_selection',select),
        ('svc',svc)]

pipeline = Pipeline(steps)
parameters = dict(feature_selection__k=[5,10], 
              svc__kernel=['linear','poly','rbf','sigmoid'])
cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
'''

# gridsearchcv
gscv = GridSearchCV(pipeline,param_grid=parameters,cv=cv,scoring='f1')
gscv.fit(features,labels)
clf = gscv.best_estimator_    

# evaluation using tester.py
# import test_classifier from tester.py
from tester import test_classifier
print "Tester Classification report" 
test_classifier(clf,my_dataset,features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)