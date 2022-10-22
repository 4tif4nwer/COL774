import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack,vstack
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from nltk.corpus import stopwords 
import pickle
stop_words = set(stopwords.words('english'))
import re
import os
def splitdate(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
# os.listdir()
full_train_data = pd.read_csv(f'data/COL774_drug_review/DrugsComTrain.csv',parse_dates=['date'],na_filter=False)
full_val_data = pd.read_csv("data/COL774_drug_review/DrugsComVal.csv",parse_dates=['date'],na_filter=False)
full_test_data = pd.read_csv("data/COL774_drug_review/DrugsComTest.csv",parse_dates=['date'],na_filter=False)

splitdate(full_train_data)
splitdate(full_val_data)
splitdate(full_test_data)
def cleandata(x_train):
    for i in range(len(x_train)):
        s = x_train[i,1]
        s = s.lower()
        s = re.sub(r'[\r\n"]', '', s)
        s = s = re.sub(r"[,.]", ' ', s)
        s = s = re.sub(r"&#039;", "'", s)
        x_train[i,1] = s
x_train = full_train_data[["condition","review"]].copy().to_numpy()
x_val = full_val_data[["condition","review"]].copy().to_numpy()
x_test = full_test_data[["condition","review"]].copy().to_numpy()
cleandata(x_train)
cleandata(x_val)
cleandata(x_test)
y_train = full_train_data[["rating"]].copy().to_numpy()
y_val = full_val_data[["rating"]].copy().to_numpy()
y_test = full_test_data[["rating"]].copy().to_numpy()
condition_encoder = CountVectorizer(stop_words=stop_words)
reviews_encoder = CountVectorizer(stop_words=stop_words)
train_reviews = reviews_encoder.fit_transform(x_train[:,1])
train_conditions = condition_encoder.fit_transform(x_train[:,0])
val_reviews = reviews_encoder.transform(x_val[:,1])
test_reviews = reviews_encoder.transform(x_test[:,1])
val_conditions = condition_encoder.transform(x_val[:,0])
test_conditions = condition_encoder.transform(x_test[:,0])
training_data = hstack((train_reviews,train_conditions,full_train_data[['year', 'month', 'day','usefulCount']].values))
val_data = hstack((val_reviews,val_conditions,full_val_data[['year', 'month', 'day','usefulCount']].values))
test_data = hstack((test_reviews,test_conditions,full_test_data[['year', 'month', 'day','usefulCount']].values))

print("data ready")

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(training_data, y_train)
path = clf.cost_complexity_pruning_path(training_data, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

ccp_samples = [ccp_alphas[i] for i in range(0,len(ccp_alphas),50)]
ccp_alphas = ccp_samples

print("Alphas ready")

from multiprocessing import Pool
import tqdm
def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]
def trainmodel(ccp_alpha):
    clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    clf.fit(training_data, y_train)
    return clf
        
clfs = []
j = 1
with Pool() as p:
    for ccp_alpha in tqdm.tqdm(chunker(ccp_alphas,12)):
#         start = time.time()
        clfs.append(p.map(trainmodel,ccp_alpha))
#         end = time.time()
#         print(end-start)
        pickle.dump(clfs[-1], open(f"model{j}_clf_pickle", 'wb'))
        j = j+1

clf_flat = [item for sublist in clfs for item in sublist]
clfs = clf_flat
pickle.dump(clfs, open(f"allmodel_clf_pickle", 'wb'))
