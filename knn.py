# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 19:38:49 2017

@author: Hp
"""
import pandas as pd
import numpy as np
from string import digits
import re
from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer
import math
from collections import Counter

def chars_unigrams(text):
    chars = re.findall(r'\w{,1}', text)
    return chars

def remove_numbers(text):
    remove_digits = str.maketrans('', '', digits)
    res = text.translate(remove_digits)
    return res

def remove_xa(text):
    pattern = re.compile('\xada[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    s = text
    return (pattern.sub('', s))
 

#reading the train set files
xy_occurances = {}
train_set_y = pd.read_csv('train_set_y.csv')
train_set_x = pd.read_csv('train_set_x.csv')
train_set_xy = train_set_x.set_index('Id').join(train_set_y.set_index('Id'))

##Pre-processing for KNN
#remove rows with blank fields
train_set_xy.replace('', np.nan, inplace=True)
train_set_xy.dropna( inplace = True)    
#remove numbers from texts
full_data = train_set_xy.values.tolist()

text=[]
for i in range(len(full_data)):
    text.append(full_data[i][0])

for i in range(len(text)):
    text[i]=remove_numbers(text[i])
    
for i in range(len(text)):
    seeds = ["http", "url"]
    text[i] = (' '.join([i for i in text[i].split() if not any(w in i.lower() for w in seeds)]))
    text[i] = (re.sub(r'(.)\1+', r'\1\1', text[i]))

for i in range(len(text)):
    original = text[i]
    removed = original.replace("\xad", "")
    text[i] = removed


df1 = pd.DataFrame({'text': text})
train_set_xy.drop(['Text'],1,inplace=True)
df1['Category']= train_set_xy['Category']
df1.replace('', np.nan, inplace=True)
df1.dropna( inplace = True)
#df1.to_csv("Processed.csv")

#Term frequency record
count_vect = CountVectorizer(analyzer='char', ngram_range=(1,1), min_df=0.0,max_df=1.0)
X_train_counts = count_vect.fit_transform(text)

#TF-IDF record
tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

vec = (pd.DataFrame(X_train_tf.toarray()))
#Organizing the TF-IDF matrix in a dataframe along with the Category column for each document
names = count_vect.get_feature_names()
vec.columns = [names[i] for i in range(len(names))]
vec['Category'] = train_set_xy['Category']

#Calculating the mean of each column 
avg = {}
for i in range(len(names)):
   avg[names[i]] = vec[names[i]].mean() 

#Cross-validation of train set, ie, taking a portion of the data
vec = vec.sample(n = 100)

#read the test file
test_set_x = pd.read_csv('test_set_x.csv')
test_set_x = test_set_x.sample(n=10)

result = []
for index, row in test_set_x.iterrows():                        #for each row in test set
    c_occurrances = {}
    val={}
    chars = list(str(row['Text']).lower().replace(" ", ""))     #split the characters
    for c in chars:
            c_occurrances[c] = True
    for i in names:
        if i in c_occurrances:                                  #if character already seen in train set,
            val[i] = avg[i]                                     # assign the mean value of that character;s tf-idf here 
        else:
            val[i] = 0                                          #if not, assign zero

    A = {} 
    for j in range(len(vec)):                                               #calculate the euclidean distance of each test set value (coordinates)
        d = []                                                              #with that of the train set
        for i in range(len(vec.iloc[j])-1):
            d.append((val[names[i]] - (vec.iloc[j][names[i]]))**2)
        A[j] = math.sqrt(np.sum(d))
        

    m = min(A, key=A.get)                                       #Find the K(-=5) minimum values(distances)
    k_min = sorted(A, key=A.get, reverse=False)[:5]
    categories = []
    for i in k_min:                                             #For the K values found, extract the Category that was given to each in the train set
        categories.append(vec.iloc[i]['Category'])

        
    result.append((Counter(categories).most_common(1)[0][0]))           #The category that appears to have been occurred the most is the category assigned to the test data



