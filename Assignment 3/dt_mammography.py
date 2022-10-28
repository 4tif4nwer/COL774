from unittest.main import main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
import sys
import os
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit,GridSearchCV
from multiprocessing import Pool
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import tqdm

def data_loader(train_loc,val_loc,test_loc,imputation = None):
    full_train_data = pd.read_csv(f"{train_loc}/train.csv",na_values='?')
    full_val_data = pd.read_csv(f"{val_loc}/val.csv",na_values='?')
    full_test_data = pd.read_csv(f"{test_loc}/test.csv",na_values='?')

    filtered_val_data = full_val_data.dropna(axis = 0)
    filtered_test_data = full_test_data.dropna(axis = 0)
        
    x_val_data = filtered_val_data[["Age","Shape","Margin","Density"]].copy().to_numpy()
    y_val_data = filtered_val_data[["Severity"]].copy().to_numpy().reshape((-1,))
    x_test_data = filtered_test_data[["Age","Shape","Margin","Density"]].copy().to_numpy()
    y_test_data = filtered_test_data[["Severity"]].copy().to_numpy().reshape((-1,))


    if imputation == None:
        filtered_train_data = full_train_data.dropna(axis = 0)
        x_train_data = filtered_train_data[["Age","Shape","Margin","Density"]].copy().to_numpy()
        y_train_data = filtered_train_data[["Severity"]].copy().to_numpy().reshape((-1,))

    elif imputation == 'mode':
        imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        full_train_data = imp_mode.fit_transform(full_train_data)
        x_train_data = full_train_data[:,1:5]
        y_train_data = full_train_data[:,5]
    
    elif imputation == 'median':
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        full_train_data = imp_median.fit_transform(full_train_data)
        x_train_data = full_train_data[:,1:5]
        y_train_data = full_train_data[:,5]

    elif imputation == 'xgb':
        x_train_data = full_train_data[["Age","Shape","Margin","Density"]].copy().to_numpy()
        y_train_data = full_train_data[["Severity"]].copy().to_numpy().reshape((-1,))        
        y_train_data = le.fit_transform(y_train_data.reshape(-1,))
        y_val_data = le.fit_transform(y_val_data.reshape(-1,))
        y_test_data = le.fit_transform(y_test_data.reshape(-1,))
       
    
    return x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data

def plot_tree(clf,save_loc,part):
    plt.rcParams['figure.dpi'] = 500
    plt.rcParams['figure.figsize']=[18,18]    
    fig,ax = plt.subplots()
    tree.plot_tree(clf, feature_names=["Age","Shape","Margin","Density"],ax = ax)
    os.chdir(save_loc)
    plt.savefig(f"1_{part}_dtree.png")
    os.chdir('..')
    os.chdir('..')
    plt.close()

def trainmodel_ccp(ccp_alpha):
    clf = tree.DecisionTreeClassifier(criterion='entropy',ccp_alpha=ccp_alpha)
    clf.fit(x_train_data, y_train_data)
    return clf

def chunker(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def main():
    
# python3 dt_mammography.py <train_data_path> <validation_data_path> <test_data_path> <output_folder_path> <question_part>

    train_loc = sys.argv[1]
    val_loc = sys.argv[2]
    test_loc = sys.argv[3]
    part = sys.argv[5]
    output_folder = sys.argv[4]
    output_file_path = f'{output_folder}/1_{part}.txt'
    output_file = open(output_file_path,"w")

    global x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data
    
    if part == 'a':
        
        x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data = data_loader(train_loc,val_loc,test_loc)
        clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
        clf = clf.fit(x_train_data, y_train_data)
        plot_tree(clf,output_folder,part)
        output_file.write("Decision Tree Parameters :\n")
        output_file.write(f"max_depth = {clf.tree_.max_depth}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")

        output_file.write(f"Training Accuracy : {clf.score(x_train_data,y_train_data)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(x_val_data,y_val_data)}\n")
        output_file.write(f"Test Accuracy : {clf.score(x_test_data,y_test_data)}\n")
    
    elif part == 'b':

        x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data = data_loader(train_loc,val_loc,test_loc)

        param_grid = [
        {'max_depth' : [i for i in range(2,20)], 'min_samples_split' : [i for i in range(2,20)], 'min_samples_leaf' : [i for i in range(1,20)]}  
        ]

        clfs = GridSearchCV(estimator = tree.DecisionTreeClassifier(criterion='entropy',random_state=0),param_grid=param_grid,n_jobs=-1)
        clfs = clfs.fit(x_train_data, y_train_data)
        clf = clfs.best_estimator_
        plot_tree(clf,output_folder,part)
        output_file.write("Best estimator parameters :\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
        output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(x_train_data,y_train_data)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(x_val_data,y_val_data)}\n")
        output_file.write(f"Test Accuracy : {clf.score(x_test_data,y_test_data)}\n")

    elif part == 'c':
        x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data = data_loader(train_loc,val_loc,test_loc)

        clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
        clf = clf.fit(x_train_data, y_train_data)
        path = clf.cost_complexity_pruning_path(x_train_data, y_train_data)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        clfs = []
        with Pool() as p:
            for ccp_alpha in tqdm.tqdm(chunker(ccp_alphas,12)):
                clfs.append(p.map(trainmodel_ccp,ccp_alpha))
        clf_flat = [item for sublist in clfs for item in sublist]
        clfs = clf_flat

        os.chdir(output_folder)
        plt.rcParams['figure.dpi'] = 70
        plt.rcParams['figure.figsize']=[18,6] 
        fig, ax = plt.subplots()   
        ax.plot(ccp_alphas[:-1],impurities[:-1])
        plt.savefig(f"1_{part}_Impurities_vs_alpha")
        plt.close()
        
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
        plt.savefig(f"1_{part}_Numberofnodes_depth_vs_alpha")
        plt.close()

        train_scores = [clf.score(x_train_data, y_train_data) for clf in clfs[:-1]]
        val_scores = [clf.score(x_val_data, y_val_data) for clf in clfs[:-1]]
        test_scores = [clf.score(x_test_data, y_test_data) for clf in clfs[:-1]]

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas[:-1], train_scores, marker="o", label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas[:-1], val_scores, marker="o", label="validation", drawstyle="steps-post")
        ax.plot(ccp_alphas[:-1], test_scores, marker="o", label="test", drawstyle="steps-post")
        ax.legend()
        plt.savefig(f"1_{part}_Accuracy_vs_alpha")
        plt.close()
        os.chdir('..')
        os.chdir('..')
        
        best_clf = (clfs[np.argsort(val_scores)[-1]])
        plot_tree(best_clf,output_folder,part)
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

        x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data = data_loader(train_loc,val_loc,test_loc)

        param_grid = [
        {'n_estimators': [100,500,1000], 'max_features' : [1,2,3,4], 'min_samples_split' : [i for i in range(2,30)]}  
        ]

        clfs = GridSearchCV(estimator = RandomForestClassifier(criterion='entropy',oob_score=True,random_state=0),param_grid=param_grid,n_jobs=-1)
        clfs = clfs.fit(x_train_data, y_train_data)
        clf = clfs.best_estimator_
        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_features = {clf.max_features}\n")
        output_file.write(f"min_samples_split = {clf.min_samples_split}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(x_train_data,y_train_data)}\n")
        output_file.write(f"Out-Of-Bag Accuracy : {clf.oob_score_}\n")
        output_file.write(f"Validation Accuracy : {clf.score(x_val_data,y_val_data)}\n")
        output_file.write(f"Test Accuracy : {clf.score(x_test_data,y_test_data)}\n")

    elif part == 'e':
        for imputation in ['median','mode']:
            output_file.write(f"Imputation by {imputation}\n")
            x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data = data_loader(train_loc,val_loc,test_loc,imputation)
            
            output_file.write("Decision Tree\n")
            clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
            clf = clf.fit(x_train_data, y_train_data)
            plot_tree(clf,output_folder,f"{part}_{imputation}")
            output_file.write("Decision Tree Parameters :\n")
            output_file.write(f"max_depth = {clf.tree_.max_depth}\n")
            output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
            output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")

            output_file.write(f"Training Accuracy : {clf.score(x_train_data,y_train_data)}\n")
            output_file.write(f"Validation Accuracy : {clf.score(x_val_data,y_val_data)}\n")
            output_file.write(f"Test Accuracy : {clf.score(x_test_data,y_test_data)}\n\n")
        
            output_file.write("Decision Tree Grid Search\n")
            param_grid = [
            {'max_depth' : [i for i in range(2,20)], 'min_samples_split' : [i for i in range(2,20)], 'min_samples_leaf' : [i for i in range(1,20)]}  
            ]

            clfs = GridSearchCV(estimator = tree.DecisionTreeClassifier(criterion='entropy',random_state=0),param_grid=param_grid,n_jobs=-1)
            clfs = clfs.fit(x_train_data, y_train_data)
            clf = clfs.best_estimator_
            plot_tree(clf,output_folder,f"{part}_{imputation}_grid")

            output_file.write("Best estimator parameters :\n")
            output_file.write(f"max_depth = {clf.max_depth}\n")
            output_file.write(f"min_samples_split = {clf.min_samples_split}\n")
            output_file.write(f"min_samples_leaf = {clf.min_samples_leaf}\n\n")
            
            output_file.write(f"Training Accuracy : {clf.score(x_train_data,y_train_data)}\n")
            output_file.write(f"Validation Accuracy : {clf.score(x_val_data,y_val_data)}\n")
            output_file.write(f"Test Accuracy : {clf.score(x_test_data,y_test_data)}\n\n")

            output_file.write("Decision Tree - Cost Complexity Pruning\n")

            clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
            clf = clf.fit(x_train_data, y_train_data)
            path = clf.cost_complexity_pruning_path(x_train_data, y_train_data)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            
            clfs = []
            with Pool() as p:
                for ccp_alpha in tqdm.tqdm(chunker(ccp_alphas,12)):
                    clfs.append(p.map(trainmodel_ccp,ccp_alpha))
            clf_flat = [item for sublist in clfs for item in sublist]
            clfs = clf_flat

            os.chdir(output_folder)
            plt.rcParams['figure.dpi'] = 70
            plt.rcParams['figure.figsize']=[18,6] 
            fig, ax = plt.subplots()   
            ax.plot(ccp_alphas[:-1],impurities[:-1])
            plt.savefig(f"1_{part}_{imputation}_Impurities_vs_alpha")
            plt.close()
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
            plt.savefig(f"1_{part}_{imputation}_Numberofnodes_depth_vs_alpha")
            plt.close()

            train_scores = [clf.score(x_train_data, y_train_data) for clf in clfs[:-1]]
            val_scores = [clf.score(x_val_data, y_val_data) for clf in clfs[:-1]]
            test_scores = [clf.score(x_test_data, y_test_data) for clf in clfs[:-1]]

            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("accuracy")
            ax.set_title("Accuracy vs alpha for training and testing sets")
            ax.plot(ccp_alphas[:-1], train_scores, marker="o", label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas[:-1], val_scores, marker="o", label="validation", drawstyle="steps-post")
            ax.plot(ccp_alphas[:-1], test_scores, marker="o", label="test", drawstyle="steps-post")
            ax.legend()
            plt.savefig(f"1_{part}_{imputation}_Accuracy_vs_alpha")
            plt.close()
            os.chdir('..')
            os.chdir('..')
            
            best_clf = (clfs[np.argsort(val_scores)[-1]])
            plot_tree(best_clf,output_folder,f"{part}_{imputation}_ccp")
            
            output_file.write("Best estimator parameters :\n")
            output_file.write(f"ccp_alpha = {ccp_alphas[np.argsort(val_scores)[-1]]}\n")
            output_file.write(f"max_depth = {best_clf.tree_.max_depth}\n")
            output_file.write(f"min_samples_split = {best_clf.min_samples_split}\n")
            output_file.write(f"min_samples_leaf = {best_clf.min_samples_leaf}\n\n")
            output_file.write(f"Training Accuracy : {best_clf.score(x_train_data,y_train_data)}\n")
            output_file.write(f"Validation Accuracy : {best_clf.score(x_val_data,y_val_data)}\n")
            output_file.write(f"Test Accuracy : {best_clf.score(x_test_data,y_test_data)}\n\n")

            output_file.write("Random Forest Grid Search\n")
            
            param_grid = [
            {'n_estimators': [100,500,1000], 'max_features' : [1,2,3,4], 'min_samples_split' : [i for i in range(2,30)]}  
            ]

            clfs = GridSearchCV(estimator = RandomForestClassifier(criterion='entropy',oob_score=True,random_state=0),param_grid=param_grid,n_jobs=-1)
            clfs = clfs.fit(x_train_data, y_train_data)
            clf = clfs.best_estimator_
            output_file.write("Best estimator parameters :\n")
            output_file.write(f"n_estimators = {clf.n_estimators}\n")
            output_file.write(f"max_features = {clf.max_features}\n")
            output_file.write(f"min_samples_split = {clf.min_samples_split}\n\n")
                
            output_file.write(f"Training Accuracy : {clf.score(x_train_data,y_train_data)}\n")
            output_file.write(f"Out-Of-Bag Accuracy : {clf.oob_score_}\n")
            output_file.write(f"Validation Accuracy : {clf.score(x_val_data,y_val_data)}\n")
            output_file.write(f"Test Accuracy : {clf.score(x_test_data,y_test_data)}\n\n")
        
    elif part == 'f':

        x_train_data,y_train_data,x_val_data,y_val_data,x_test_data,y_test_data = data_loader(train_loc,val_loc,test_loc,'xgb')
        param_grid = [
        {'n_estimators': [10,20,30,40,50], 'max_depth' : [4,5,6,7,8,9,10], 'subsample' : [0.1,0.2,0.3,0.4,0.5,0.6]}  
        ]

        clfs = GridSearchCV(estimator = xgb.XGBClassifier(),param_grid=param_grid)
        clfs = clfs.fit(x_train_data, y_train_data)
        clf = clfs.best_estimator_
        output_file.write("Best estimator parameters :\n")
        output_file.write(f"n_estimators = {clf.n_estimators}\n")
        output_file.write(f"max_depth = {clf.max_depth}\n")
        output_file.write(f"subsample = {clf.subsample}\n\n")
        
        output_file.write(f"Training Accuracy : {clf.score(x_train_data,y_train_data)}\n")
        output_file.write(f"Validation Accuracy : {clf.score(x_val_data,y_val_data)}\n")
        output_file.write(f"Test Accuracy : {clf.score(x_test_data,y_test_data)}\n")

if __name__ == '__main__':
    main()