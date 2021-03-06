# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:37:52 2017

@author: Hp
"""

import pandas as pd
import codecs
import numpy as np
from sklearn import preprocessing, cross_validation
import nltk


types_of_encoding = ["utf8", "cp1252"]
for encoding_type in types_of_encoding:
    with codecs.open('train_set_x.csv', encoding = encoding_type, errors ='replace'):
        df1 = pd.read_csv('train_set_x.csv')
#        df1.drop(['Id'], 1, inplace=True)
        full_data1 = df1.values.tolist()
        
    with codecs.open('train_set_y.csv', encoding = encoding_type, errors ='replace'):
        df2 = pd.read_csv('train_set_y.csv')
#        df2.drop(['Id'], 1, inplace=True)
        full_data2 = df2.values.tolist()
        
df3 = pd.merge(df1,df2, on=['Id'])
df3.drop(['Id'], 1, inplace=True)
df3['Text'].replace('', np.nan, inplace=True)
df3.dropna(subset = ['Text'], inplace = True)

df4 = df3.sample(n = 1000)
full_data = df4.values.tolist()

text=[]
for i in range(len(full_data)):
    text.append(full_data[i][0])



from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df4['Text'])

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

vec = (pd.DataFrame(X_train_counts.toarray()))

names = count_vect.get_feature_names()
vec.columns = [names[i] for i in range(len(names))]

m = pd.merge(vec, df4, on =df4['Text'])
m.drop(['Text'],1,inplace=True)

def column_count(df):
    column_count = []
    for i in range(len(names)):
        column_count.append((df[names[i]] == 1).sum())
    return column_count

p_0 = (m['Category'] == 0).sum()/len(m)
p_1 = (m['Category'] == 1).sum()/len(m)
p_2 = (m['Category'] == 2).sum()/len(m)
p_3 = (m['Category'] == 3).sum()/len(m)
p_4 = (m['Category'] == 4).sum()/len(m)


df_0 = m.loc[m['Category'] == 0]
df_1 = m.loc[m['Category'] == 1]
df_2 = m.loc[m['Category'] == 2]
df_3 = m.loc[m['Category'] == 3]
df_4 = m.loc[m['Category'] == 4]


words_count_for_category_0 = (column_count(m.loc[m['Category'] == 0]))
words_count_for_category_1 = (column_count(m.loc[m['Category'] == 1]))
words_count_for_category_2 = (column_count(m.loc[m['Category'] == 2]))
words_count_for_category_3 = (column_count(m.loc[m['Category'] == 3]))
words_count_for_category_4 = (column_count(m.loc[m['Category'] == 4]))


prob_X_given_0 = {}
prob_X_given_1 = {}
prob_X_given_2 = {}
prob_X_given_3 = {}
prob_X_given_4 = {}
for i in range(len(names)):
    prob_X_given_0[names[i]]=((words_count_for_category_0[i]+1)/(len(df_0)+2))
    prob_X_given_1[names[i]]=((words_count_for_category_1[i]+1)/(len(df_1)+2))
    prob_X_given_2[names[i]]=((words_count_for_category_2[i]+1)/(len(df_2)+2))
    prob_X_given_3[names[i]]=((words_count_for_category_3[i]+1)/(len(df_3)+2))
    prob_X_given_4[names[i]]=((words_count_for_category_4[i]+1)/(len(df_4)+2))

#Need to calculate P(y|x)
#Calculate pf P(y=1|x)


test = "Je suis desole"
t = nltk.word_tokenize(test)

prob_test_in_0 = []
prob_test_in_1 = []
prob_test_in_2 = []
prob_test_in_3 = []
prob_test_in_4 = []
for k in t:
    if k in prob_X_given_0:    
        prob_test_in_0.append((prob_X_given_0[k]))
    if k in prob_X_given_1:    
        prob_test_in_1.append((prob_X_given_1[k]))
    if k in prob_X_given_2:    
        prob_test_in_2.append((prob_X_given_2[k]))
    if k in prob_X_given_3:    
        prob_test_in_3.append((prob_X_given_3[k]))
    if k in prob_X_given_4:    
        prob_test_in_4.append((prob_X_given_4[k]))


final_prob_0 =p_0
final_prob_1 =p_1
final_prob_2 =p_2
final_prob_3 =p_3
final_prob_4 =p_4

for i in range(len(prob_test_in_0)):
    final_prob_0 *= prob_test_in_0[i]
for i in range(len(prob_test_in_1)):
    final_prob_1 *= prob_test_in_1[i]
for i in range(len(prob_test_in_2)):
    final_prob_2 *= prob_test_in_2[i]
for i in range(len(prob_test_in_3)):
    final_prob_3 *= prob_test_in_3[i]
for i in range(len(prob_test_in_4)):
    final_prob_4 *= prob_test_in_4[i]


result = max(final_prob_0,final_prob_1,final_prob_2,final_prob_3,final_prob_4)
    
    
if(result==final_prob_0): 
    print("Slovak")
if(result==final_prob_1):
    print("French")
if(result==final_prob_2):
    print("Spanish")
if(result==final_prob_3):
    print("German")
if(result==final_prob_4):
    print("Polish")




    
    


