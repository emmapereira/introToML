#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:31:54 2022

@author: emma
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

 
#import the dataset into a dataframe
df = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data?fbclid=IwAR0bnadUy7l7_jwPgJzAW1Dg5RM_JyAKv_doOWxuP2Fx2XpkTAliWHRl73U') 

def newdf(df):

    #convert the famhist values into binary
    df['famhist'] = (df['famhist'] == 'Present').astype(int)
    
    #remove the row.names column
    newdf = df.drop(columns = 'row.names')
    
    #convert the dataframe to numpy arrays  
    #raw_data = newdf.values  
    
    return newdf


#newdf has the dataframe without row.names and binary values in famhist 
newdf = newdf(df)

#start making the data matrix X by indexing into data (ignore matrix stuff for now!!)
#get the values from the columns (we are not taking the chd column)
cols = range(1, 9) 
#X = raw_data[:, cols]

#extract attribute names
attributeNames = np.asarray(df.columns[cols])
#M = len(attributeNames)



#dataframe without the chd column
df_noCHD = newdf.drop(columns = 'chd')


#print some statistics on each column
def datastatistics(df):


    mean_df = df.mean()
    std_df = df.std(ddof=1)
    min_df = df.min()
    max_df = df.max()
    
    print( "the mean of each attribute is: ", mean_df)
    print( "the standard deviation of each attribute is: ", std_df)
    print( "the min of each attribute is: ", min_df)
    print( "the max of each attribute is: ", max_df)
        
        
datastatistics(df_noCHD)


#boxplots for each attribute
for column in df_noCHD:
    plt.figure()
    df_noCHD.boxplot([column], patch_artist=True, boxprops=dict(facecolor="lightcyan", color="darkcyan"), medianprops=dict(color="darkcyan"))
    plt.gca().set(ylabel='Frequency')
  

#correlation plot 
M = len(df_noCHD.columns)
C = len(df["chd"].unique())

fig, ax = plt.subplots(M,M,figsize=(20,20))
i = 0
j= 0
for col1 in df_noCHD.columns: #numerical_col:
    for col2 in df_noCHD.columns: #numerical_col:
        for c in range(C):
            class_mask = (df["chd"]==c)

            ax[i,j].scatter(df_noCHD.loc[class_mask,col2], df_noCHD.loc[class_mask,col1],alpha=0.3)
            
        if j == 0:
            ax[i,j].set_ylabel(col1)
        if i == M-1:
            ax[i,j].set_xlabel(col2)
            
        if i == M-1 and  j == M-1:
            ax[i,j].legend(range(C))
            
        j += 1
    j = 0
    i += 1
    

plt.show()






