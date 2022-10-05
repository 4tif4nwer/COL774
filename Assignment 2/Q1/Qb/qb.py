import numpy as np
import sys
import os
import random

def confusion_matrix(y_pred,y_act) :
    cmat = np.zeros((2,2))
    
    cmat[0,0] = len(np.where((y_pred==y_act) & (y_pred == 1))[0])
    cmat[1,1] = len(np.where((y_pred==y_act) & (y_pred == 0))[0])
    cmat[0,1] = len(np.where((y_pred!=y_act) & (y_pred == 1))[0])
    cmat[1,0] = len(np.where((y_pred!=y_act) & (y_pred == 0))[0])
    
    return cmat

def main():
    train_data = sys.argv[1]
    test_data = sys.argv[2]

    file_location_pos = test_data + "/pos/"
    file_location_neg = test_data + "/neg/"
    y_act = np.zeros(len(os.listdir(file_location_neg))+len(os.listdir(file_location_pos)))
    y_act[0:len(os.listdir(file_location_pos))] = 1
    y_pred = np.zeros_like(y_act)

    # Test Accuracy
    
    # Random Guessing

    y_pred = np.zeros_like(y_act)

    for iter in range(len(os.listdir(file_location_neg))+len(os.listdir(file_location_pos))):
        y_pred[iter] = random.randint(0,1)
    cmat_random = confusion_matrix(y_pred,y_act)
    # print(cmat_random)
    np.save("cmat_random.npy",cmat_random)
    print(f"Test Accuracy by random guessing : {100*(cmat_random[0,0]+cmat_random[1,1])/np.sum(cmat_random)}%")
    precision = cmat_random[0,0]/(cmat_random[0,0]+cmat_random[0,1])
    recall = cmat_random[0,0]/(cmat_random[0,0]+cmat_random[1,0])
    F1_score = (2*precision*recall)/(precision+recall)
    print(f"F1-Score : {F1_score}")

    # All positive
    
    y_pred = np.ones_like(y_act)
    cmat_positive = confusion_matrix(y_pred,y_act)
    # print(cmat_positive)
    np.save("cmat_positive.npy",cmat_positive)
    print(f"Test Accuracy by constant positive prediction : {100*(cmat_positive[0,0]+cmat_positive[1,1])/np.sum(cmat_positive)}%")
    precision = cmat_positive[0,0]/(cmat_positive[0,0]+cmat_positive[0,1])
    recall = cmat_positive[0,0]/(cmat_positive[0,0]+cmat_positive[1,0])
    F1_score = (2*precision*recall)/(precision+recall)
    print(f"F1-Score : {F1_score}")

if __name__ == '__main__':
    main()
    