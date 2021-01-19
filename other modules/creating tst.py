# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 22:42:27 2021

@author: somar
"""

from os import path
import os
import cv2 as cv
import sys
import random
TRAINING_DIRECTORY='F:/cmp/fourth year-classroom material/1st term/pattern recognition - cmp 450/project/formsA-D/'
TEST_DIRECTORY='tests/'
SampleDirectory='F:/cmp/fourth year-classroom material/1st term/pattern recognition - cmp 450/project/sample/data/'
Trainset={}
def read_training_set():
    myfile = open('forms.txt','r')
    for line in myfile:
        if line[0]=='#':
            continue
        else:
            line = line.split()
            key = str(int(line[1]))
            value=line[0]+'.png'
            if Trainset.get(key):
                Trainset[key].append(value)
            else:
                Trainset[key]=[value]
                
    myfile.close()
    return 
read_training_set()
validation=open("C:/Users/somar/OneDrive/Desktop/validation.txt",'a')
for i in list(Trainset):
    if len(Trainset[i])<3 or (not path.exists(TRAINING_DIRECTORY+Trainset[str(i)][0]))  or (not path.exists(TRAINING_DIRECTORY+Trainset[str(i)][1]))  or (not path.exists(TRAINING_DIRECTORY+Trainset[str(i)][2])):
        Trainset.pop(i)
for i in list(Trainset):
    print(i,"===>",Trainset[i])       
for tstnum in range(1,301):
    try:
        os.mkdir(SampleDirectory+str(tstnum))
    except OSError:
        print ("Creation of the directory failed")
        sys.exit()
    os.chdir(SampleDirectory+str(tstnum))
    count=0
    randomlist =set()
    while len(randomlist) < 3 :
        key_iterable = Trainset.keys()
        key_list = list(key_iterable)
        randomlist.add(int(random.choice(key_list)))
    randomlist=list(randomlist)    
    for w in randomlist:
            count+=1
            try:
                os.mkdir(str(count))
            except OSError:
                print ("Creation of the directory failed")
                sys.exit()
            os.chdir(SampleDirectory+str(tstnum)+"/"+str(count)) 
            if path.exists(TRAINING_DIRECTORY+Trainset[str(w)][0]):
                img= cv.imread(TRAINING_DIRECTORY+Trainset[str(w)][0])
                cv.imwrite(os.path.join(SampleDirectory+str(tstnum)+"/"+str(count) , "1.png"), img)
            if path.exists(TRAINING_DIRECTORY+Trainset[str(w)][1]):
                img= cv.imread(TRAINING_DIRECTORY+Trainset[str(w)][1])
                cv.imwrite(os.path.join(SampleDirectory+str(tstnum)+"/"+str(count) , "2.png"), img)
            if count==3:
                tstid=random.choice(randomlist)
                if path.exists(TRAINING_DIRECTORY+Trainset[str(tstid)][2]):
                    img= cv.imread(TRAINING_DIRECTORY+Trainset[str(tstid)][2])
                    tstid=randomlist.index(tstid)+1
                    validation.write(str(tstid)+"\n")
                    print(tstid)
                    cv.imwrite(os.path.join(SampleDirectory+str(tstnum) , "test.png"), img)
            #Trainset.pop(w)
            os.chdir(SampleDirectory+str(tstnum))
            if(count==3):
                break
validation.close()            
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)            
