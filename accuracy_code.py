import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def read (mainFolder):
    xx = os.listdir(mainFolder)

    folder_list =[]
    file_list = []

    for name in xx:
        if ".ppm" in name:
            file_list.append(name)
        elif len(name)==2:
            folder_list.append(name)
        
    dataset= []

    for folder in folder_list:
        strTemp = mainFolder + folder
        xx=os.listdir(strTemp)
        
        count = 0
        for name in xx:
            img = plt.imread(strTemp + "//" + name)
            dataset.append([strTemp + "//" +  name, img])
            count = count+1
            if count > 10:
                break
        annotation = {}

    with open(mainFolder+"gt.txt","r") as inst:
        for line in inst:
            filename,x1,y1,x2,y2,t = line.split(";")
            
            if filename in annotation:
                annotation[filename].append([int(x1),int(y1),int(x2),int(y2)])
            else:
                annotation[filename] = [int(x1),int(y1),int(x2),int(y2)]
                
    return dataset, file_list, annotation
def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[2], b[2]) - x
    h = max(a[3], b[3]) - y
    return (x, y, w, h)

#def intersection(a,b):
#    x = max(a[0], b[0])
#    y = max(a[1], b[1])
#    w = min(a[2], b[2]) 
#    h = min(a[3], b[3]) 
#    if w<0 or h<0: 
#        return () 
#    return ( w, h)

#def area(a):
#  x = abs(a[0]-a[2])
#  y = abs(a[1]-a[3])
#  return x*y

def findMatches(listGT, lista):
    matchCount = 0
    missMatchCount = 0
    for x in lista:
        isMatched = False
        for gt in listGT:
#            intersect = intersection(x,gt)
#            if len(intersect)>0:
    #            r = area(intersect)
                isMatched = True
        
        if isMatched:
            matchCount = matchCount + 1
        else:
            missMatchCount = missMatchCount + 1
    
    return matchCount, missMatchCount

def draw_rects(img, listRect, color):
    
     for x1,y1,x2,y2 in listRect:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2) 
    
        return img

def read_results(filename):

    results = {}
    with open(filename, "r") as ins:
        for line in ins:
            filename,x1,y1,x2,y2 = line.split(';')
            
            if filename in results:
                results[filename].append([int(x1),int(y1),int(x2),int(y2)])
            else:
                results[filename] = [[int(x1),int(y1),int(x2),int(y2)]]
                
    return results

mainFolder = "C:/Users/ngowt/OneDrive/Documents/FALL 2019 coursework/Computer vision and image analysis/Project/FullIJCNN2013//"

dataset, file_list, annotation = read(mainFolder)

results = read_results("template-results1.txt")

total = 0
totalMatched = 0
totalMissMatched = 0
totalMissed = 0


for key in annotation:
    
    listGT = annotation[key]
    
    if key not in results:
            #total = total + len(listGT)
            continue
    
    lista = results[key]
    
    matchCount, missMatchCount = findMatches(listGT, lista)
    missed = len(listGT)-matchCount
    print(key, " total:", len(listGT) ," matched: ", matchCount, " miss-matched: ", missMatchCount, " missed: ",missed)

    img = cv2.imread(mainFolder + key)
    #img = draw_rects(img, listGT, (0,0,255))
    #img = draw_rects(img, lista, (255,0,0))
    #cv2.imshow("img",img)
    
    total = total + len(listGT)
    totalMatched = totalMatched + matchCount
    totalMissMatched = totalMissMatched + missMatchCount
    totalMissed = totalMissed + missed
    
    cv2.waitKey(0)
            


print ("total: ", total)
print("matched: ", totalMatched)
miss_ed = (totalMissed/total)*100
accuracy = (totalMatched / total)*100
print("accuracy (%): ", accuracy)
print("miss_ed (%): ", miss_ed)