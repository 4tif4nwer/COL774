from unittest.main import main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import sys
import os
from scipy.sparse import hstack,vstack
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import PredefinedSplit,GridSearchCV
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
import tqdm
import re

def data_loader(train_loc,val_loc,test_loc,imputation = None):
    full_train_data = pd.read_csv(f"{train_loc}/DrugsComTrain.csv",parse_dates=['date'],na_filter=False)
    full_val_data = pd.read_csv(f"{val_loc}/DrugsComVal.csv",parse_dates=['date'],na_filter=False)
    full_test_data = pd.read_csv(f"{test_loc}/DrugsComTest.csv",parse_dates=['date'],na_filter=False)

    def splitdate(df):
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df = df.astype({'year': np.float32,'month': np.float32,'day': np.float32,'usefulCount':np.float32})

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
    y_train = full_train_data[["rating"]].copy().to_numpy(dtype=np.float32)
    y_val = full_val_data[["rating"]].copy().to_numpy(dtype=np.float32)
    y_test = full_test_data[["rating"]].copy().to_numpy(dtype=np.float32)
    condition_encoder = CountVectorizer(dtype=np.float32,stop_words=stop_words)
    reviews_encoder = CountVectorizer(dtype=np.float32,stop_words=stop_words)
    train_reviews = reviews_encoder.fit_transform(x_train[:,1])
    train_conditions = condition_encoder.fit_transform(x_train[:,0])
    val_reviews = reviews_encoder.transform(x_val[:,1])
    test_reviews = reviews_encoder.transform(x_test[:,1])
    val_conditions = condition_encoder.transform(x_val[:,0])
    test_conditions = condition_encoder.transform(x_test[:,0])
    training_data = hstack((train_reviews,train_conditions,full_train_data[['year', 'month', 'day','usefulCount']].values))
    val_data = hstack((val_reviews,val_conditions,full_val_data[['year', 'month', 'day','usefulCount']].values))
    test_data = hstack((test_reviews,test_conditions,full_test_data[['year', 'month', 'day','usefulCount']].values))

    return training_data,y_train,val_data,y_val,test_data,y_test

def trainmodel_ccp(ccp_alpha):
    clf = tree.DecisionTreeClassifier(criterion = 'entropy',ccp_alpha=ccp_alpha)
    clf.fit(training_data, y_train)
    return clf

def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def main():
    
    train_loc = sys.argv[1]
    val_loc = sys.argv[2]
    test_loc = sys.argv[3]
    part = sys.argv[5]
    output_folder = sys.argv[4]
    output_file_path = f'{output_folder}/1_{part}.txt'
    output_file = open(output_file_path,"w")

    global training_data,y_train,val_data,y_val,test_data,y_test
    
    if part == 'a':
        
        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0)
        clf = clf.fit(training_data, y_train)
        output_file.write("Decision Tree Parameters :\n")
        output_file.write(f"max_depth = {clf.tree_.max_depth}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")

        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val)}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test)}\n")
    
    elif part == 'b':

        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        # param_grid = [
        # {'max_depth' : [i for i in range(150,201,10)], 'min_samples_split' : [i for i in range(2,20,4)], 'min_samples_leaf' : [i for i in range(1,10,1)]}  
        # ]
        param_grid = [
        {'max_depth' : [i for i in range(180,261,20)], 'min_samples_split' : [i for i in range(2,40,8)], 'min_samples_leaf' : [i for i in range(1,20,5)]}  
        ]

        clfs = GridSearchCV(estimator = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0),verbose = 2,param_grid=param_grid,n_jobs=-1)
        clfs = clfs.fit(training_data, y_train.reshape(-1,))
        clf = clfs.best_estimator_

        output_file.write("Best estimator parameters :\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val)}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test)}\n")

    elif part == 'c':
        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0)
        clf = clf.fit(training_data, y_train)
        path = clf.cost_complexity_pruning_path(training_data, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        os.chdir(output_folder)
        plt.rcParams['figure.dpi'] = 70
        plt.rcParams['figure.figsize']=[18,6] 
        fig, ax = plt.subplots()   
        ax.plot(ccp_alphas[:-1],impurities[:-1])
        plt.savefig(f"2_{part}_Impurities_vs_alpha")
        plt.close()
        
        ccp_samples = [ccp_alphas[i] for i in range(0,len(ccp_alphas),50)]
        ccp_alphas = ccp_samples

        clfs = []
        with Pool() as p:
            for ccp_alpha in tqdm.tqdm(chunker(ccp_alphas,os.cpu_count())):
                clfs.append(p.map(trainmodel_ccp,ccp_alpha))
        clf_flat = [item for sublist in clfs for item in sublist]
        clfs = clf_flat

        plt.rcParams['figure.figsize']=[18,12]
        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas[:-1], node_counts[:-1], marker="o", drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(ccp_alphas[:-1], depth[:-1], marker="o", drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()
        plt.savefig(f"2_{part}_Numberofnodes_depth_vs_alpha")
        plt.close()

        train_scores = [clf.score(training_data, y_train) for clf in clfs[:-1]]
        val_scores = [clf.score(val_data, y_val) for clf in clfs[:-1]]
        test_scores = [clf.score(test_data, y_test) for clf in clfs[:-1]]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas[:-1], train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas[:-1], val_scores, marker="o", label="validation", drawstyle="steps-post")
        ax.plot(ccp_alphas[:-1], test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(f"2_{part}_Accuracy_vs_alpha")
        plt.close()
        os.chdir('..')
        os.chdir('..')
        
        best_clf = (clfs[np.argsort(val_scores)[-1]])

        output_file.write("Best estimator parameters :\n")
        output_file.write(f"ccp_alpha = {ccp_alphas[np.argsort(val_scores)[-1]]}\n")
        output_file.write(f"max_depth = {best_clf.tree_.max_depth}\n")
        output_file.write(f"min_samples_split = {best_clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {best_clf.min_samples_leaf}\n\n")

        output_file.write(f"Training Accuracy : {train_scores[np.argsort(val_scores)[-1]]}\n")
        output_file.write(f"Validation Accuracy : {val_scores[np.argsort(val_scores)[-1]]}\n")
        output_file.write(f"Test Accuracy : {test_scores[np.argsort(val_scores)[-1]]}\n")

    elif part == 'd':

        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        param_grid = [
        {'n_estimators': [i for i in range(50,201,50)], 'max_features' : [float(i/10) for i in range(1,10,4)], 'min_samples_split' : [i for i in range(2,13,5)]}  
        ]

        clfs = GridSearchCV(estimator = RandomForestClassifier(criterion = 'entropy',oob_score=True,random_state=0),verbose = 2,param_grid=param_grid,n_jobs=-1)
        clfs = clfs.fit(training_data, y_train.reshape(-1,))
        clf = clfs.best_estimator_

        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_features = {clf.max_features}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train)}\n")
        output_file.write(f"Out-Of-Bag Accuracy : {clf.oob_score_}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val)}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test)}\n")
        
    elif part == 'e':

        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        y_train = le.fit_transform(y_train.reshape(-1,))
        y_val = le.fit_transform(y_val.reshape(-1,))
        y_test = le.fit_transform(y_test.reshape(-1,))
        
        param_grid = [
        {'n_estimators': range(50,451,50), 'max_depth' : [40,50,60,70], 'subsample' : [0.4,0.5,0.6,0.7,0.8]}  
        ]
        
        clfs = GridSearchCV(estimator = xgb.XGBClassifier(),param_grid=param_grid,verbose = 2)
        clfs = clfs.fit(training_data, y_train.reshape(-1,))
        clf = clfs.best_estimator_
        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"subsample = {clf.subsample}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val)}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test)}\n")
    
    elif part == 'f':
        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        param_grid = [
        {'n_estimators': range(500,2001,500), 'max_depth' : [40,50,60,70], 'subsample' : [0.4,0.5,0.6,0.7,0.8]}  
        ]
        clfs = GridSearchCV(estimator = lgb.LGBMClassifier(),param_grid=param_grid,verbose = 2)
        clfs = clfs.fit(training_data, y_train.reshape(-1,))
        clf = clfs.best_estimator_
        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"subsample = {clf.subsample}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val)}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test)}\n")

if __name__ == '__main__':
    main()