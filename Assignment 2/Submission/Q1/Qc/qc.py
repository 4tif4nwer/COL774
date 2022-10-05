import numpy as np
import os

def main():
    try:
        os.chdir("..")
        cmat_defaultsplit = np.load(f"Qa/cmat_defaultsplit.npy")
        cmat_positive = np.load(f"Qb/cmat_positive.npy")
        cmat_random = np.load(f"Qb/cmat_random.npy")
        os.chdir("./Qc")

        print(f"Confusion Matrix with Default Split Naive Bayes\n{cmat_defaultsplit}")
        print(f"Confusion Matrix with Random Guessing\n{cmat_random}")
        print(f"Confusion Matrix with constant Positive Prediction\n{cmat_positive}")
    except:
        print("Please run the previous parts(a,b) first")
if __name__ == '__main__':
    main()
    