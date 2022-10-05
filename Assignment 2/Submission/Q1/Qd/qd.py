import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import os
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud

from nltk.stem import PorterStemmer,WordNetLemmatizer
Stemmer = PorterStemmer()
Lemmatizer = WordNetLemmatizer()
Stopwords = set(stopwords.words('english'))

StemMap = defaultdict()

def Stem(word):
    if word in StemMap:
        return StemMap[word]
    StemMap[word] = Stemmer.stem(word)
    return StemMap[word]

def stem_split(directory, stemmed_vocab = defaultdict(int)):
    
    x = []
    vocab_enum = len(stemmed_vocab)
    Stemmer = PorterStemmer()
    file_names = os.listdir(directory)
    for file_name in file_names:
        file = open(directory+file_name,'r')
        s = file.read()
        s = re.sub('<.*?>',repl = " ", string = s)
        s = re.sub(r"[^\w\s']", '', s)
#         s = re.sub(r"[/:,-]", ' ', s)
        s = s.lower()
        s = s.split()
        counter = defaultdict(int)
        for word in s:
            stemmed_word = Stem(word)
            if word not in Stopwords:
                counter[stemmed_word]+=1
                if stemmed_word not in stemmed_vocab:
                    stemmed_vocab[stemmed_word] = vocab_enum
                    vocab_enum += 1
#         print(np.array(counter))
        x.append(counter)
    return x

def train_naive_bayes(x,vocab):
    phi = np.ones(len(vocab))
    n = len(vocab)
    
    for review in x:
        n+=len(review)
        for word in review.keys():
            phi[vocab[word]] += review[word]  
    
    phi/=n
    return phi
    
def judge(review,phi,phi_y,vocab):
    prob_1 = np.log(phi_y)
    prob_0 = np.log(1-phi_y)
    for word in review:
        if word not in vocab:
            continue
        prob_0 += review[word] * np.log(phi[0][vocab[word]])
        prob_1 += review[word] * np.log(phi[1][vocab[word]])
    if prob_1 >= prob_0 :
        return 1
    
    return 0

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
    stemmed_vocab = defaultdict(int)

    # Training Naive Bayes

    file_location_pos = train_data + "/pos/"
    pos_x = stem_split(file_location_pos,stemmed_vocab)
    file_location_neg = train_data + "/neg/"
    neg_x = stem_split(file_location_neg,stemmed_vocab)

    plt.rcParams['figure.dpi'] = 70
    plt.rcParams['figure.figsize']=[10,10]
    
    Allwords = ""
    file_names = os.listdir(file_location_pos)
    for file_name in file_names:
        file = open(file_location_pos+file_name,'r')
        s = file.read()
        s = re.sub('<.*?>',repl = " ", string = s)
        s = re.sub(r"[^\w\s']", '', s)
        s = s.lower()
        s = s.split()
        for word in s:
            stemmed_word = Stem(word)
            if word not in Stopwords:
                Allwords += (stemmed_word+" ")

    Mycloud = WordCloud(width = 400,height = 400,background_color='white',max_words=2000,stopwords=set())
    
    Cloud = Mycloud.generate(Allwords)
    plt.imshow(Cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("StemWordCloudPos.png")
    plt.close()

    Allwords = ""
    file_names = os.listdir(file_location_neg)
    for file_name in file_names:
        file = open(file_location_neg+file_name,'r')
        s = file.read()
        s = re.sub('<.*?>',repl = " ", string = s)
        s = re.sub(r"[^\w\s']", '', s)
        s = s.lower()
        s = s.split()
        for word in s:
            stemmed_word = Stem(word)
            if word not in Stopwords:
                Allwords += (stemmed_word+" ")
    
    Cloud = Mycloud.generate(Allwords)
    plt.imshow(Cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("StemWordCloudNeg.png")
    plt.close()


    phi_pos = train_naive_bayes(pos_x,stemmed_vocab)
    phi_neg = train_naive_bayes(neg_x,stemmed_vocab)
    phi = [phi_neg,phi_pos]
    phi_y = len(os.listdir(file_location_pos))/(len(os.listdir(file_location_pos))+len(os.listdir(file_location_neg)))
    # print(len(stemmed_vocab))

    # Training Accuracy

    y_act = np.zeros(len(os.listdir(file_location_neg))+len(os.listdir(file_location_pos)))
    y_act[0:len(os.listdir(file_location_pos))] = 1
    y_pred = np.zeros_like(y_act)

    iter = 0

    pos_x = stem_split(file_location_pos)
    for review in pos_x:
        y_pred[iter] = judge(review,phi,phi_y,stemmed_vocab)
        iter += 1

    neg_x = stem_split(file_location_neg)
    for review in neg_x:
        y_pred[iter] = judge(review,phi,phi_y,stemmed_vocab)
        iter += 1
    cmat_train = confusion_matrix(y_pred,y_act)
    
    # print(cmat_train)
    
    print(f"Training Accuracy : {100*(cmat_train[0,0]+cmat_train[1,1])/np.sum(cmat_train)}%")

    # Test Accuracy

    file_location_pos = test_data + "/pos/"
    file_location_neg = test_data + "/neg/"
    y_act = np.zeros(len(os.listdir(file_location_neg))+len(os.listdir(file_location_pos)))
    y_act[0:len(os.listdir(file_location_pos))] = 1
    y_pred = np.zeros_like(y_act)

    iter = 0

    pos_x = stem_split(file_location_pos)
    for review in pos_x:
        y_pred[iter] = judge(review,phi,phi_y,stemmed_vocab)
        iter += 1

    neg_x = stem_split(file_location_neg)
    for review in neg_x:
        y_pred[iter] = judge(review,phi,phi_y,stemmed_vocab)
        iter += 1
    cmat_test = confusion_matrix(y_pred,y_act)
    # print(cmat_test)
    np.save("cmat_stemsplit.npy",cmat_test)
    print(f"Test Accuracy : {100*(cmat_test[0,0]+cmat_test[1,1])/np.sum(cmat_test)}%")
    precision = cmat_test[0,0]/(cmat_test[0,0]+cmat_test[0,1])
    recall = cmat_test[0,0]/(cmat_test[0,0]+cmat_test[1,0])
    F1_score = (2*precision*recall)/(precision+recall)
    print(f"F1-Score : {F1_score}")

if __name__ == '__main__':
    main()
    