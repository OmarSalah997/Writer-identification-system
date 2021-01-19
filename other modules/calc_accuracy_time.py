# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 15:07:56 2021

@author: somar
"""
f1 = open("validation.txt" , 'r')
f2 = open("results.txt" , 'r')
f3 = open("time.txt" , 'r')
l1=[]
l2=[]
l3=[]
for i in f1:
	l1.append(i.split()[0])
for i in f2:
	l2.append(i.split()[0])
c=0
for i in range(len(l1)):
	if l1[i]==l2[i]:
		c+=1
for i in f3:
	l3.append(float(i.split()[0]))
     
print("AVG time = ",round(sum(l3)/len(l3),2)," s")
print("The accuracy = ",round(c/len(l1)*100,2),"%")
f1.close()
f2.close()
f3.close()
