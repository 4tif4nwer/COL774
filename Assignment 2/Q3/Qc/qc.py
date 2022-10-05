import numpy as np
import shutil
import os

def main():
    try:
        os.chdir("..")
        cmat_cvxopt = np.load(f"Qa/cmat_cvxopt.npy")
        cmat_sklearn = np.load(f"Qb/cmat_sklearn.npy")
        for i in range(1,11):
            shutil.copy(f"Qa/cvxopt_misclf{i}.png","Qc/")
        for i in range(1,11):
            shutil.copy(f"Qb/sklearn_misclf{i}.png","Qc/")

        os.chdir("./Qc")

        print(f"Confusion Matrix for SVM(ovo) with CVXOPT\n{cmat_cvxopt}")
        print(f"Confusion Matrix for SVM(ovo) with sklearn\n{cmat_sklearn}")
    except:
        print("Please run previous parts(a,b) first")
if __name__ == '__main__':
    main()
    