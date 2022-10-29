import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import sys
import os
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
import tqdm
import re

def data_loader(train_loc,val_loc,test_loc,sample = None):
    full_train_data = pd.read_csv(f"{train_loc}/DrugsComTrain.csv",parse_dates=['date'],na_filter=False)
    full_val_data = pd.read_csv(f"{val_loc}/DrugsComVal.csv",parse_dates=['date'],na_filter=False)
    full_test_data = pd.read_csv(f"{test_loc}/DrugsComTest.csv",parse_dates=['date'],na_filter=False)

    if sample:
        if sample <= full_train_data.shape[0]:    
            full_train_data = full_train_data.sample(n=sample,replace=False)
        else :
            full_train_data = full_train_data.sample(n=sample,replace=True)
            
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
    output_file_path = f'{output_folder}/2_{part}.txt'
    output_file = open(output_file_path,"w")

    global training_data,y_train,val_data,y_val,test_data,y_test
    
    if part == 'a':
        
        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)

        clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0)        
        start = time.time()
        clf = clf.fit(training_data, y_train)
        end = time.time()
        output_file.write("Decision Tree Parameters :\n")
        output_file.write(f"max_depth = {clf.tree_.max_depth}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")

        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val)}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test)}\n")
        output_file.write(f"Time taken to train : {end-start}\n")
        output_file.close()
    
    elif part == 'b':

        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        param_grid = [
        {'max_depth' : [i for i in range(40,121,40)], 'min_samples_split' : [i for i in range(2,11,4)], 'min_samples_leaf' : [i for i in range(1,20,5)]}  
        ]

        clfs = GridSearchCV(estimator = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0),verbose = 2,param_grid=param_grid,n_jobs=-1)
        clfs = clfs.fit(training_data, y_train.reshape(-1,))
        clf = clfs.best_estimator_

        test_clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth=clf.max_depth,min_samples_split=clf.min_samples_split,min_samples_leaf=clf.min_samples_leaf)
        start = time.time()
        test_clf.fit(training_data,y_train)
        end = time.time()

        output_file.write("Best estimator parameters :\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val)}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test)}\n")
        output_file.write(f"Time taken to train : {end-start}\n")
        output_file.close()

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
        
        # Taking every 100th value of ccp_alpha to reduce the number of models to be trained(I don't have time to train 32000 models :P)
        # ccp_samples = [ccp_alphas[i] for i in range(0,len(ccp_alphas),100)]
        # ccp_alphas = ccp_samples

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
        
        best_ccp = (ccp_alphas[np.argsort(val_scores)[-1]])
        clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0,ccp_alpha=best_ccp)

        start = time.time()
        best_clf = clf.fit(training_data, y_train)
        end = time.time()
        
        output_file.write("Best estimator parameters :\n")
        output_file.write(f"ccp_alpha = {ccp_alphas[np.argsort(val_scores)[-1]]}\n")
        output_file.write(f"max_depth = {best_clf.tree_.max_depth}\n")
        output_file.write(f"min_samples_split = {best_clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {best_clf.min_samples_leaf}\n\n")

        output_file.write(f"Training Accuracy : {train_scores[np.argsort(val_scores)[-1]]}\n")
        output_file.write(f"Validation Accuracy : {val_scores[np.argsort(val_scores)[-1]]}\n")
        output_file.write(f"Test Accuracy : {test_scores[np.argsort(val_scores)[-1]]}\n")
        output_file.write(f"Training Time : {end-start}\n")
        output_file.close()

    elif part == 'd':

        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        param_grid = [
        {'n_estimators': [i for i in range(50,451,50)], 'max_features' : [float(i/10) for i in range(4,9,1)], 'min_samples_split' : [i for i in range(2,11,2)]}  
        ]

        clfs = GridSearchCV(estimator = RandomForestClassifier(criterion = 'entropy',oob_score=True,random_state=0),verbose = 2,param_grid=param_grid,n_jobs=-1)
        clfs = clfs.fit(training_data, y_train.reshape(-1,))
        clf = clfs.best_estimator_

        test_clf = RandomForestClassifier(criterion = 'entropy',oob_score=True,random_state=0,n_estimators=clf.n_estimators,max_features=clf.max_features,min_samples_split=clf.min_samples_split)
        start = time.time()
        test_clf.fit(training_data,y_train.reshape(-1,))
        end = time.time()

        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_features = {clf.max_features}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train.reshape(-1,))}\n")
        output_file.write(f"Out-Of-Bag Accuracy : {clf.oob_score_}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val.reshape(-1,))}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test.reshape(-1,))}\n")
        output_file.write(f"Training Time : {end-start}\n")

        output_file.close()
        
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

        test_clf = xgb.XGBClassifier(n_estimators=clf.n_estimators,max_depth=clf.max_depth,subsample=clf.subsample)
        start = time.time()
        test_clf.fit(training_data,y_train.reshape(-1,))
        end = time.time()

        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"subsample = {clf.subsample}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train.reshape(-1,))}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val.reshape(-1,))}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test.reshape(-1,))}\n")
        output_file.write(f"Training Time : {end-start}\n")
        output_file.close()
    
    elif part == 'f':
        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        
        param_grid = [
        {'n_estimators': range(500,2001,500), 'max_depth' : [40,50,60,70], 'subsample' : [0.4,0.5,0.6,0.7,0.8]}  
        ]
        clfs = GridSearchCV(estimator = lgb.LGBMClassifier(),param_grid=param_grid,verbose = 2)
        clfs = clfs.fit(training_data, y_train.reshape(-1,))
        clf = clfs.best_estimator_

        test_clf = lgb.LGBMClassifier(n_estimators=clf.n_estimators,max_depth=clf.max_depth,subsample=clf.subsample)
        start = time.time()
        test_clf.fit(training_data,y_train.reshape(-1,))
        end = time.time()

        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"subsample = {clf.subsample}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(training_data,y_train.reshape(-1,))}\n")
        output_file.write(f"Validation Accuracy : {clf.score(val_data,y_val.reshape(-1,))}\n")
        output_file.write(f"Test Accuracy : {clf.score(test_data,y_test.reshape(-1,))}\n")
        output_file.write(f"Training Time : {end-start}\n")
        output_file.close()
    
    elif part == 'g':
        training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc)
        train_scores = []
        val_scores = []
        test_scores = []
        train_time = []
        
        train_data_sizes = [i*1000 for i in range(20,141,20)]
        output_file.write("Decision Tree Classifier\n\n")
        for train_data_size in train_data_sizes:
            training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc,sample = train_data_size)
            
            clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0)
            start = time.time()
            clf.fit(training_data,y_train)
            end = time.time()
            train_time.append(end-start)
            train_scores.append(clf.score(training_data,y_train))
            val_scores.append(clf.score(val_data,y_val))
            test_scores.append(clf.score(test_data,y_test))
           
            output_file.write(f"Training data size : {train_data_size}\n")
            output_file.write(f"max_depth = {clf.tree_.max_depth}\n")
            output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
            output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")
            output_file.write(f"Training Accuracy : {train_scores[-1]}\n")
            output_file.write(f"Validation Accuracy : {val_scores[-1]}\n")
            output_file.write(f"Test Accuracy : {test_scores[-1]}\n\n")
            output_file.write(f"Training time : {train_time[-1]}\n\n")
        
        
        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_scores,label = 'Training Accuracy')
        ax.plot(train_data_sizes,val_scores,label = 'Validation Accuracy')
        ax.plot(train_data_sizes,test_scores,label = 'Test Accuracy')
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Training Data Size')
        ax.legend()
        fig.savefig(f'{output_folder}/g_a_Accuracy_vs_Training_Data_Size.png')
        plt.close()

        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_time)
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Training Time')
        ax.set_title('Training Time vs Training Data Size')
        fig.savefig(f'{output_folder}/g_a_Training_Time_vs_Training_Data_Size.png')
        plt.close()

        output_file.write(f"Decision Tree with Grid Search results : max_depth = 120 | min_samples_split = 2 | min_samples_leaf = 1 \n\n")
        train_scores.clear()
        val_scores.clear()
        test_scores.clear()
        train_time.clear()
        for train_data_size in train_data_sizes:
            training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc,sample = train_data_size)
            
            clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0,max_depth=120,min_samples_split=2,min_samples_leaf=1)
            start = time.time()
            clf.fit(training_data,y_train)
            end = time.time()
            train_time.append(end-start)
            train_scores.append(clf.score(training_data,y_train))
            val_scores.append(clf.score(val_data,y_val))
            test_scores.append(clf.score(test_data,y_test))

            output_file.write(f"Training data size : {train_data_size}\n")
            output_file.write(f"Training Accuracy : {train_scores[-1]}\n")
            output_file.write(f"Validation Accuracy : {val_scores[-1]}\n")
            output_file.write(f"Test Accuracy : {test_scores[-1]}\n\n")
            output_file.write(f"Training time : {train_time[-1]}\n\n")
        
        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_scores,label = 'Training Accuracy')
        ax.plot(train_data_sizes,val_scores,label = 'Validation Accuracy')
        ax.plot(train_data_sizes,test_scores,label = 'Test Accuracy')
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Training Data Size')
        ax.legend()
        fig.savefig(f'{output_folder}/g_b_Accuracy_vs_Training_Data_Size.png')
        plt.close()

        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_time)
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Training Time')
        ax.set_title('Training Time vs Training Data Size')
        fig.savefig(f'{output_folder}/g_b_Training_Time_vs_Training_Data_Size.png')
        plt.close()

        output_file.write(f"Decision Tree with Optimal ccp_alpha = 0.0 \n\n")
        train_scores.clear()
        val_scores.clear()
        test_scores.clear()
        train_time.clear()
        for train_data_size in train_data_sizes:
            training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc,sample = train_data_size)
            
            clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state=0,ccp_alpha=0.0)
            start = time.time()
            clf.fit(training_data,y_train)
            end = time.time()
            train_time.append(end-start)
            train_scores.append(clf.score(training_data,y_train))
            val_scores.append(clf.score(val_data,y_val))
            test_scores.append(clf.score(test_data,y_test))

            output_file.write(f"Training data size : {train_data_size}\n")
            output_file.write(f"Training Accuracy : {train_scores[-1]}\n")
            output_file.write(f"Validation Accuracy : {val_scores[-1]}\n")
            output_file.write(f"Test Accuracy : {test_scores[-1]}\n\n")
            output_file.write(f"Training time : {train_time[-1]}\n\n")
    
        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_scores,label = 'Training Accuracy')
        ax.plot(train_data_sizes,val_scores,label = 'Validation Accuracy')
        ax.plot(train_data_sizes,test_scores,label = 'Test Accuracy')
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Training Data Size')
        ax.legend()
        fig.savefig(f'{output_folder}/g_c_Accuracy_vs_Training_Data_Size.png')
        plt.close()

        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_time)
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Training Time')
        ax.set_title('Training Time vs Training Data Size')
        fig.savefig(f'{output_folder}/g_c_Training_Time_vs_Training_Data_Size.png')
        plt.close()

        output_file.write(f"Random Forest with Grid Search results :  n_estimators = 450 | max_features = 0.8 | nim_sample_split = 2 \n\n")
        train_scores.clear()
        val_scores.clear()
        test_scores.clear()
        train_time.clear()
        for train_data_size in train_data_sizes:
            training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc,sample = train_data_size)
            
            clf = RandomForestClassifier(n_estimators=450,max_features= 0.8, min_samples_split= 2, criterion='entropy', random_state=0,n_jobs=-1)
            start = time.time() 
            clf.fit(training_data,y_train.reshape(-1,))
            end = time.time()
            train_time.append(end-start)
            train_scores.append(clf.score(training_data,y_train.reshape(-1,)))
            val_scores.append(clf.score(val_data,y_val.reshape(-1,)))
            test_scores.append(clf.score(test_data,y_test.reshape(-1,)))

            output_file.write(f"Training data size : {train_data_size}\n")
            output_file.write(f"Training Accuracy : {train_scores[-1]}\n")
            output_file.write(f"Validation Accuracy : {val_scores[-1]}\n")
            output_file.write(f"Test Accuracy : {test_scores[-1]}\n\n")
            output_file.write(f"Training time : {train_time[-1]}\n\n")
        
        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_scores,label = 'Training Accuracy')
        ax.plot(train_data_sizes,val_scores,label = 'Validation Accuracy')
        ax.plot(train_data_sizes,test_scores,label = 'Test Accuracy')
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Training Data Size')
        ax.legend()
        fig.savefig(f'{output_folder}/g_d_Accuracy_vs_Training_Data_Size.png')
        plt.close()
        
        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_time)
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Training Time')
        ax.set_title('Training Time vs Training Data Size')
        fig.savefig(f'{output_folder}/g_d_Training_Time_vs_Training_Data_Size.png')
        plt.close()

        output_file.write(f"XGBoost with Grid Search results : n_estimators = 450 | max_depth = 40 | subsample = 0.4\n\n")
        train_scores.clear()
        val_scores.clear()
        test_scores.clear()
        train_time.clear()
        for train_data_size in train_data_sizes:
            training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc,sample = train_data_size)
            xgb_y_train = le.fit_transform(y_train.reshape(-1,))
            xgb_y_val = le.fit_transform(y_val.reshape(-1,))
            xgb_y_test = le.fit_transform(y_test.reshape(-1,))
            clf = xgb.XGBClassifier(random_state=0, n_estimators = 450, max_depth = 40, subsample = 0.4)
            start = time.time()
            clf.fit(training_data,xgb_y_train)
            end = time.time()
            train_time.append(end-start)
            train_scores.append(clf.score(training_data,xgb_y_train))
            val_scores.append(clf.score(val_data,xgb_y_val))
            test_scores.append(clf.score(test_data,xgb_y_test))

            output_file.write(f"Training data size : {train_data_size}\n")
            output_file.write(f"Training Accuracy : {train_scores[-1]}\n")
            output_file.write(f"Validation Accuracy : {val_scores[-1]}\n")
            output_file.write(f"Test Accuracy : {test_scores[-1]}\n\n")
            output_file.write(f"Training time : {train_time[-1]}\n\n")
        
        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_scores,label = 'Training Accuracy')
        ax.plot(train_data_sizes,val_scores,label = 'Validation Accuracy')
        ax.plot(train_data_sizes,test_scores,label = 'Test Accuracy')
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Training Data Size')
        ax.legend()
        fig.savefig(f'{output_folder}/g_e_Accuracy_vs_Training_Data_Size.png')
        plt.close()

        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_time)
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Training Time')
        ax.set_title('Training Time vs Training Data Size')
        fig.savefig(f'{output_folder}/g_e_Training_Time_vs_Training_Data_Size.png')
        plt.close()

        output_file.write(f"LightGBM with Grid Search results : n_estimators = 2000 | max_depth = 40 | subsample = 0.4\n\n")    
        train_scores.clear()
        val_scores.clear()
        test_scores.clear()
        train_time.clear()
        for train_data_size in train_data_sizes:
            training_data,y_train,val_data,y_val,test_data,y_test = data_loader(train_loc,val_loc,test_loc,sample = train_data_size)
            
            clf = lgb.LGBMClassifier(random_state=0, n_estimators = 2000,max_depth = 40,subsample = 0.4)
            start = time.time()
            clf.fit(training_data,y_train.reshape(-1,))
            end = time.time()
            train_time.append(end-start)
            train_scores.append(clf.score(training_data,y_train.reshape(-1,)))
            val_scores.append(clf.score(val_data,y_val.reshape(-1,)))
            test_scores.append(clf.score(test_data,y_test.reshape(-1,)))

            output_file.write(f"Training data size : {train_data_size}\n")
            output_file.write(f"Training Accuracy : {train_scores[-1]}\n")
            output_file.write(f"Validation Accuracy : {val_scores[-1]}\n")
            output_file.write(f"Test Accuracy : {test_scores[-1]}\n\n")
            output_file.write(f"Training time : {train_time[-1]}\n\n")

        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_scores,label = 'Training Accuracy')
        ax.plot(train_data_sizes,val_scores,label = 'Validation Accuracy')
        ax.plot(train_data_sizes,test_scores,label = 'Test Accuracy')
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Training Data Size')
        ax.legend()
        fig.savefig(f'{output_folder}/g_f_Accuracy_vs_Training_Data_Size.png')
        plt.close()

        fig,ax = plt.subplots()
        ax.plot(train_data_sizes,train_time)
        ax.set_xlabel('Training Data Size')
        ax.set_ylabel('Training Time')
        ax.set_title('Training Time vs Training Data Size')
        fig.savefig(f'{output_folder}/g_f_Training_Time_vs_Training_Data_Size.png')
        plt.close()

        output_file.close()
        
if __name__ == '__main__':
    main()