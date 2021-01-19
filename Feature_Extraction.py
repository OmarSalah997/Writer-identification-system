# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 22:30:12 2021

@author: somar
"""
import numpy as np
from skimage import feature as feature

class Feature_Extraction:
    def __init__(self,imgesList):
        self.imgesList = imgesList
        self.img=None
        self.center = None
        self.features =[]
    def get_features(self):
        return self.features
    def _get_diff(self,x, y):
        new_value = 0
        try:# if out of range  -1 boarders added
            if self.img[x][y] >= self.center:
                new_value = 1
        except:
            pass
        return new_value

    def lbp_calculated(self,x, y):
        '''R = 2 and window 5x5'''

        self.center = self.img[x][y]
        v = np.empty(8,dtype=int)
     
        v[0]=self._get_diff( x-2, y-2)     
        v[1]=self._get_diff(x-2, y) 
        v[2]=self._get_diff(x-2, y+2)
        v[3]=self._get_diff(x, y-2)
        v[4]=self._get_diff(x, y+2)
        v[5]=self._get_diff(x+2, y+2)   
        v[6]=self._get_diff(x+2, y)      
        v[7]=self._get_diff(x+2, y-2)
        return sum(np.multiply(np.array([1, 2, 4, 8, 16, 32, 64, 128]),v))
    
    
    def extract(self):
        for i in self.imgesList:
            self.img=i.copy()
            result = np.empty(i.shape , dtype=int)
            print('shape', i.shape )
            for k in range(i.shape[0]):
                for j in range(i.shape[1]):
                    result[k][j]=self.lbp_calculated(k,j)
            self.features.append(np.histogram(result.ravel(),256,[0,256])[0])
        return 
    def LBP_uniform(self,num_patterns,radius):
        for i in self.imgesList:
            lbp = feature.local_binary_pattern(i, num_patterns,radius, method="uniform")
            hist=np.histogram(lbp.ravel(),num_patterns+2,[0,num_patterns+2])[0].astype("float")
            hist /= (hist.sum())
            self.features.append(hist)
def most_frequent(List): 
    counter = 0
    num = List[0] 
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num   
