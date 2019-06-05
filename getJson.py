# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:14:39 2019

@author: Niclas
"""

#get label info from json files

import json
import os
import csv

directory = 'imgs/skel3'
points = []
csvPath = 'annotations/annotations_3f.csv'
posCount = 0

with open('imgs/skel3/frame90.json', 'r') as fp:
    obj = json.load(fp)
    
for i in range(3000): points.append((0,0))

for filename in os.listdir(directory):
    if filename.endswith('.json'): 
         # print(os.path.join(directory, filename))
         with open(os.path.join(directory, filename), 'r') as fp:
             obj = json.load(fp)
         string = obj['imagePath']
         frameNr = int(''.join(filter(str.isdigit, string)))
         points[frameNr-1] = (obj['shapes'][0]['points'][0][0],obj['shapes'][0]['points'][0][1])
         continue
    else:
        continue

with open(csvPath, 'w', newline="") as csvfile:
    #fieldnames = ['x_position', 'y_position']
    writer = csv.writer(csvfile)
    #writer.writeheader()
    for position in points:
        posCount += 1
        writer.writerow(position)