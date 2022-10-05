import numpy as np
import os

def main():
    os.chdir("..")
    cmat_bigramsplit = np.load(f"Qe/cmat_bigramsplit.npy")
    
    print(f"Confusion Matrix with Naive Bayes with stemming and Bigram Features\n{cmat_bigramsplit}")
    
    precision = cmat_bigramsplit[0,0]/(cmat_bigramsplit[0,0]+cmat_bigramsplit[0,1])
    recall = cmat_bigramsplit[0,0]/(cmat_bigramsplit[0,0]+cmat_bigramsplit[1,0])

    F1_score = (2*precision*recall)/(precision+recall)
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1-Score = {F1_score}")
if __name__ == '__main__':
    main()
    