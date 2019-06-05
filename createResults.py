# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:40:31 2019

@author: Niclas
"""

#result plots

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


annCsvPath = 'annotations/annotations_3f.csv'
intCsvPath = 'intersections/intersections_3f_comb.csv'
contCsvPath = 'contours/contours_3f.csv'
    
annDf = pd.read_csv(annCsvPath)
annDf = annDf.values.tolist()
intDf = pd.read_csv(intCsvPath)
intDf = intDf.values.tolist()
contDf = pd.read_csv(contCsvPath)
contDf = contDf.values.tolist()

annotations = [tuple(l) for l in annDf]
intersections = [tuple(m) for m in intDf]
contours = [tuple(n) for n in contDf]

print(np.count_nonzero(annotations)/2)
print(np.count_nonzero(intersections)/2)

zip(*contours)
fig_cont = plt.figure()
plt.axis(xmax=1920,ymin=0,ymax=805)
plt.scatter(*zip(*contours),1)
plt.title('Contour Detections')

plt.show(fig_cont)
fig_cont.savefig('3fcontours.pdf', bbox_inches = 'tight')

i = 0
j = 0
anLength = len(annotations)
inLength = len(intersections)
while (i<anLength):
    if(annotations[i] == (0,0)):
        annotations.remove(annotations[i])
        anLength = anLength - 1
        continue
    i += 1

while (j<inLength):
    if(intersections[j] == (0, 0)):
        intersections.remove(intersections[j])
        inLength = inLength - 1
        continue
    j += 1        

zip(*annotations)
zip(*intersections)
fig_int = plt.figure()
plt.subplot(2,1,1)
plt.axis(xmax=1920)
plt.scatter(*zip(*annotations),1)
plt.title('Annotations')

plt.subplots_adjust(hspace=0.4)

plt.subplot(2,1,2)
plt.axis(xmax=1920)
plt.scatter(*zip(*intersections),1)
plt.title('Intersections')


plt.show(fig_int)

fig_int.savefig('3fIntersections.pdf', bbox_inches = 'tight')

'''
combinations = intersections+contours
zip(*combinations)
fig_comb = plt.figure()
plt.axis(xmax=1920,ymin=0,ymax=805)
plt.scatter(*zip(*combinations),1)
plt.title('Combinated Detections')

plt.show(fig_comb)

fig_comb.savefig('3fcombination.pdf',bbox_inches = 'tight')
'''

'''
def Diff(li1, li2): 
    return (list(set(li1) - set(li2)))

diffs = (Diff(annotations,intersections))
diffs2 = np.isclose(annotations,intersections,atol=3)

print(diffs2[34][0])


fails = np.where(diffs2 == [False])


for i in diffs2:
    if diffs2[i][0] == False:
        fails.append(i)
'''