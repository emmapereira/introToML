#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:31:54 2022

@author: emma
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.linalg as linalg
from sklearn.decomposition import PCA


# emma
# import the dataset into a dataframe
df = pd.read_csv('https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data?fbclid=IwAR0bnadUy7l7_jwPgJzAW1Dg5RM_JyAKv_doOWxuP2Fx2XpkTAliWHRl73U') 


# arrange dataset by converting the famhist values and removing the column with row names
def newdf(df):

    # convert the famhist values into binary
    df['famhist'] = (df['famhist'] == 'Present').astype(int)
    
    # remove the row.names column
    newdf = df.drop(columns = 'row.names')
    
    return newdf


# newdf has the dataframe without row.names and binary values in famhist 
newdf = newdf(df)

# get the values from the columns (we are not taking the chd column)
cols = range(1, 9) 

# extract attribute names
attributeNames = np.asarray(df.columns[cols])

# keep column of chd
classific = newdf[['chd']]

# dataframe without the chd column
df_noCHD = newdf.drop(columns = 'chd')



# print some statistics on each column
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


# boxplots for each attribute
for column in df_noCHD:
    plt.figure()
    df_noCHD.boxplot([column], patch_artist=True, boxprops=dict(facecolor="lightcyan", color="darkcyan"), medianprops=dict(color="darkcyan"))
    plt.gca().set(ylabel='Frequency')
  

# berta
# histogram plots 
newdfp=newdf[newdf['chd']==1] #observations of chp-positive patients
newdfn=newdf[newdf['chd']==0] #observations of chp-negative patients
for attribute in attributeNames:
    plt.figure()
    newdfn[attribute].plot.hist(grid=True, bins=20, rwidth=0.9, color='#82E0AA')
    newdfp[attribute].plot.hist(grid=True, bins=20, rwidth=0.6, color='#F1948A')
    plt.title(attribute)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    
# saba
# correlation plot 
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


# emma
# PCA computations
def pca_computations(cleandf):
    # standarize the data
    scale= StandardScaler()
    stanData = scale.fit_transform(cleandf)
    
    # PCA by computing SVD of Y
    U,S,V = linalg.svd(stanData,full_matrices=False)
    V = V.T

    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum() 

    # Project data onto principal component space
    threshold = 0.9

    # Plot variance explained
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-', color='#82E0AA')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-', color='#F1948A')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()
    
    
    
    # Plot two PC coefficients  
    # choose legends and colors we will use
    legendStrs = ['PC1', 'PC2']
    colors = ['#82E0AA','#F1948A']
    colors = colors*9
    
    pcs = [0,1]
    bw = .2
    r = np.arange(1,M+1)
    for i in pcs:    
        plt.bar(r+i*bw, V[:,i], width=bw, color = colors[i])
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8])
    plt.xlabel('Attributes', fontstyle = 'oblique')
    plt.ylabel('Component coefficients', fontstyle = 'oblique')
    plt.legend(legendStrs)
    plt.title('South African Heart Disease: PCA Component Coefficients', fontweight = 'normal',fontstyle = 'italic')
    plt.grid()
    plt.show()
    
    
    
    # Plot 2D with the first two PCs 
    pca = PCA(n_components=2)
    components = pca.fit_transform(stanData)
    
    # choose name of the classes, and colors we will use
    classes = ['noCHD', 'CHD']
    n_label = int(len(classes))
    colors = ['#82E0AA','#F1948A']
    
    #assign each category with a color
    cdict = {i: colors[i] for i in range(n_label)}
    label_dict = {i: classes[i] for i in range(n_label)}
    components = components * (-1) 
    
    # go through the labels and establish their classification 
    for i in range(n_label):
        indices = np.where(classific == i)
        plt.scatter(components[indices, 0], components[indices, 1], color=cdict[i], label=label_dict[i])
    
    #Plot
    plt.legend(loc='best')
    plt.xlabel('PC1',fontstyle ='oblique')
    plt.ylabel('PC2',fontstyle = 'oblique')
    #sns.set()
    plt.title('PC1 and PC2 in 2 dimensions', fontweight = 'normal',fontstyle = 'italic')
    plt.grid()
    plt.show()
 

pca_computations(df_noCHD)
