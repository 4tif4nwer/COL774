import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import os
import re
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer,WordNetLemmatizer
Stemmer = PorterStemmer()
Lemmatizer = WordNetLemmatizer()

StemMap = defaultdict()
def Stem(word):
    if word in StemMap:
        return StemMap[word]
    StemMap[word] = Stemmer.stem(word)
    return StemMap[word]

def bigram_split(directory, bigram_vocab = defaultdict(int)):
    x = []
    vocab_enum = len(bigram_vocab)
    Stemmer = PorterStemmer()
    file_names = os.listdir(directory)
    Stopwords = set(stopwords.words('english'))
    for file_name in file_names:
        file = open(directory+file_name,'r')
        s = file.read()
        s = re.sub('<.*?>',repl = " ", string = s)
        s = re.sub(r"[^\w\s']", '', s)
#         s = re.sub(r"[/:,-]", ' ', s)
        s = s.lower()
        s = s.split()
        counter = defaultdict(int)
        wordlist = []
        for word in s:
            stemmed_word = Stem(word)#Stemmer.stem(word)
            if word not in Stopwords:
                wordlist.append(stemmed_word)
                counter[stemmed_word]+=1
                if stemmed_word not in bigram_vocab:
                    bigram_vocab[stemmed_word] = vocab_enum
                    vocab_enum += 1
            
        
        for i in range(len(wordlist)-1):
            word = wordlist[i]+" "+wordlist[i+1]
            
            counter[word]+=1
            if word not in bigram_vocab:
                    bigram_vocab[word] = vocab_enum
                    vocab_enum += 1
#         print(np.array(counter))
        x.append(counter)
    return x

def trigram_split(directory, trigram_vocab = defaultdict(int)):
    x = []
    vocab_enum = len(trigram_vocab)
    Stemmer = PorterStemmer()
    file_names = os.listdir(directory)
    Stopwords = set(stopwords.words('english'))
    for file_name in file_names:
        file = open(directory+file_name,'r')
        s = file.read()
        s = re.sub('<.*?>',repl = " ", string = s)
        s = re.sub(r"[^\w\s']", '', s)
#         s = re.sub(r"[/:,-]", ' ', s)
        s = s.lower()
        s = s.split()
        counter = defaultdict(int)
        wordlist = []
        for word in s:
            stemmed_word = Stem(word)#Stemmer.stem(word)
            if word not in Stopwords:
                wordlist.append(stemmed_word)
                counter[stemmed_word]+=1
                if stemmed_word not in trigram_vocab:
                    trigram_vocab[stemmed_word] = vocab_enum
                    vocab_enum += 1
            
        
        for i in range(len(wordlist)-2):
            word = wordlist[i]+" "+wordlist[i+1] + " " + wordlist[i+2]
            
            counter[word]+=1
            if word not in trigram_vocab:
                    trigram_vocab[word] = vocab_enum
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
    bigram_vocab = defaultdict(int)

    # Training Naive Bayes with bigrams and stemming

    file_location_pos = train_data + "/pos/"
    pos_x = bigram_split(file_location_pos,bigram_vocab)
    file_location_neg = train_data + "/neg/"
    neg_x = bigram_split(file_location_neg,bigram_vocab)

    phi_pos = train_naive_bayes(pos_x,bigram_vocab)
    phi_neg = train_naive_bayes(neg_x,bigram_vocab)
    phi = [phi_neg,phi_pos]
    phi_y = len(os.listdir(file_location_pos))/(len(os.listdir(file_location_pos))+len(os.listdir(file_location_neg)))
    # print(len(bigram_vocab))

    print("Naive Bayes on Bigrams with Stemming")
    # Training Accuracy

    y_act = np.zeros(len(os.listdir(file_location_neg))+len(os.listdir(file_location_pos)))
    y_act[0:len(os.listdir(file_location_pos))] = 1
    y_pred = np.zeros_like(y_act)

    iter = 0

    pos_x = bigram_split(file_location_pos)
    for review in pos_x:
        y_pred[iter] = judge(review,phi,phi_y,bigram_vocab)
        iter += 1

    neg_x = bigram_split(file_location_neg)
    for review in neg_x:
        y_pred[iter] = judge(review,phi,phi_y,bigram_vocab)
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

    pos_x = bigram_split(file_location_pos)
    for review in pos_x:
        y_pred[iter] = judge(review,phi,phi_y,bigram_vocab)
        iter += 1

    neg_x = bigram_split(file_location_neg)
    for review in neg_x:
        y_pred[iter] = judge(review,phi,phi_y,bigram_vocab)
        iter += 1
    cmat_test = confusion_matrix(y_pred,y_act)
    # print(cmat_test)
    np.save("cmat_bigramsplit.npy",cmat_test)
    print(f"Test Accuracy : {100*(cmat_test[0,0]+cmat_test[1,1])/np.sum(cmat_test)}%")
    precision = cmat_test[0,0]/(cmat_test[0,0]+cmat_test[0,1])
    recall = cmat_test[0,0]/(cmat_test[0,0]+cmat_test[1,0])
    F1_score = (2*precision*recall)/(precision+recall)
    print(f"F1-Score : {F1_score}")
    
    trigram_vocab = defaultdict(int)

    # Training Naive Bayes with bigrams and stemming

    file_location_pos = train_data + "/pos/"
    pos_x = trigram_split(file_location_pos,trigram_vocab)
    file_location_neg = train_data + "/neg/"
    neg_x = trigram_split(file_location_neg,trigram_vocab)

    phi_pos = train_naive_bayes(pos_x,trigram_vocab)
    phi_neg = train_naive_bayes(neg_x,trigram_vocab)
    phi = [phi_neg,phi_pos]
    phi_y = len(os.listdir(file_location_pos))/(len(os.listdir(file_location_pos))+len(os.listdir(file_location_neg)))
    # print(len(trigram_vocab))

    print("Naive Bayes on Trigrams with Stemming")
    # Training Accuracy

    y_act = np.zeros(len(os.listdir(file_location_neg))+len(os.listdir(file_location_pos)))
    y_act[0:len(os.listdir(file_location_pos))] = 1
    y_pred = np.zeros_like(y_act)

    iter = 0

    pos_x = trigram_split(file_location_pos)
    for review in pos_x:
        y_pred[iter] = judge(review,phi,phi_y,trigram_vocab)
        iter += 1

    neg_x = trigram_split(file_location_neg)
    for review in neg_x:
        y_pred[iter] = judge(review,phi,phi_y,trigram_vocab)
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

    pos_x = trigram_split(file_location_pos)
    for review in pos_x:
        y_pred[iter] = judge(review,phi,phi_y,trigram_vocab)
        iter += 1

    neg_x = trigram_split(file_location_neg)
    for review in neg_x:
        y_pred[iter] = judge(review,phi,phi_y,trigram_vocab)
        iter += 1
    cmat_test = confusion_matrix(y_pred,y_act)
    # print(cmat_test)
    np.save("cmat_trigramsplit.npy",cmat_test)
    print(f"Test Accuracy : {100*(cmat_test[0,0]+cmat_test[1,1])/np.sum(cmat_test)}%")
    precision = cmat_test[0,0]/(cmat_test[0,0]+cmat_test[0,1])
    recall = cmat_test[0,0]/(cmat_test[0,0]+cmat_test[1,0])
    F1_score = (2*precision*recall)/(precision+recall)
    print(f"F1-Score : {F1_score}")

if __name__ == '__main__':
    main()
    