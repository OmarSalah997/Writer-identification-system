# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:50:35 2021

@author: somar
"""

from preprocessing import *
from Feature_Extraction import *
import numpy as np
from sklearn.svm import *
from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv
from os import path
import timeit
import os
TEST_DIRECTORY='F:/cmp/fourth year-classroom material/1st term/pattern recognition - cmp 450/project/sample/data/'
testset=[]
test_sets_number=100
segmentation_mode=3 # 1= paragraph  2= lines  3= words
LBP_patterns=16
LBP_raduis=3
def most_frequent(List): 
    list2=[0,List.count('1'),List.count('2'),List.count('3')]
    return str(list2.index(max(list2)))    
def read_test_set(N):
    for i in range(1,N+1):
        dirct= TEST_DIRECTORY
        d ={"1":[],"2":[],"3":[],"test":""}
        if i<10:
            dirct+="0"+str(i)
        else:
            dirct+=str(i)
        d["1"].append(dirct+'/1/1.png')
        d["1"].append(dirct+'/1/2.png')
        d["2"].append(dirct+'/2/1.png')
        d["2"].append(dirct+'/2/2.png')
        d["3"].append(dirct+'/3/1.png')
        d["3"].append(dirct+'/3/2.png')
        d["test"]=dirct+'/test.png'
        testset.append(d)
    return
read_test_set(test_sets_number)  
for tst in range(test_sets_number):
    time=0
    train_data=[]
    labels=[]
    for w in range (1,4):
        for i in range(2):
            img1=cv.imread(testset[tst][str(w)][i])
            start = timeit.default_timer()
            preProcessing1=preprocessing(img1)
            if segmentation_mode== 1 :
                img_list=[]
                img_list.append(preProcessing1.get_paragraph())
            if segmentation_mode== 2 :
                img_list=preProcessing1.get_lines()
            if segmentation_mode== 3 :
                img_list=preProcessing1.get_words()
            result=Feature_Extraction(img_list)
            result.LBP_uniform(LBP_patterns,LBP_raduis)
            train_data.extend(result.get_features())
            labels+=[str(w) for z in range(len(result.get_features()))]
            stop = timeit.default_timer()
            time+=(stop-start)
    start = timeit.default_timer()       
    model = LinearSVC(C=100, random_state=42,max_iter=2000)
    model.fit(train_data, labels)
    tstimg=cv.imread(testset[tst]["test"])
    preProcessing1=preprocessing(tstimg)
    if segmentation_mode== 1 :
        img_list=preProcessing1.get_paragraph()
    if segmentation_mode== 2 :
        img_list=preProcessing1.get_lines()
    if segmentation_mode== 3 :
        img_list=preProcessing1.get_words()
    result=Feature_Extraction(img_list)
    result.LBP_uniform(LBP_patterns,LBP_raduis)
    prediction=[]
    for f in result.get_features():
        prediction.append(model.predict(f.reshape(1,-1))[0])
    stop = timeit.default_timer()
    time+=(stop-start)
    os.chdir(TEST_DIRECTORY[:-5])
    results = open("results-linearSVM.txt", "a")
    results.write(str(most_frequent(prediction))+"\n")
    results.close() 
    timetxt = open("time-linearSVM.txt", "a")
    timetxt.write(str(round(time,2))+"\n")
    timetxt.close()

