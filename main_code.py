import numpy as np
from matplotlib import pyplot as plt
import os
import cv2
from imutils.object_detection import non_max_suppression
from timeit import default_timer as timer
############################ Function for Training DATA ########################
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


############################## FUNCTION FOR TEMPLATE MATCHING #################
def detect(img,template,method):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

 
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,method)
    threshold = 0.9

  
    loc = np.where(res>=threshold)

    results=[]
    for pt in zip(*loc[::-1]):
        results.append([pt[0] ,pt[1] ,pt[0]+w,pt[1]+ h])
        
    return results

############################ READ DATA    ###################################
mainFolder = "C:/Users/ngowt/OneDrive/Documents/FALL 2019 coursework/Computer vision and image analysis/Project/FullIJCNN2013//"
dataset, file_list, annotation = read(mainFolder)

############################ tottal DATASET #################################
print("length of dataset: ",len(dataset))
############################ Display COordinates and results ################

f=open("template-results.txt","w+")

############################ Different Methods #############################
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

method = eval(methods[1])

########################### START TIMER /// DETECT For every 5 IMAGES ##############
for i in range(1,2): #len(file_list)):
    start=timer()
    print(file_list[i])
    img = cv2.imread(mainFolder + file_list[i])

    results = []
    for name, template in dataset:
############################   TEMPLATE MATCHING    ###############################
        results_temp = detect(img.copy(),template,method)
        
        if(len(results_temp)>0):
            results.extend(results_temp)

############################   REMOVE OVERLAP   ####################################
        rects = np.array([[x1,y1,x2,y2]for (x1,y1,x2,y2) in results])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.2)
 

############################### Detect OBJECT #########################################   
        for x1,y1,x2,y2 in pick:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3)
############################### Inserting data in Saving File ################        
            f.write(file_list[i]+";"+str(x1)+";"+str(y1)+";"+str(x2)+";"+str(y2)+"\n")
    
        cv2.imshow("results", cv2.resize(img,(448,448)))
################################ END TIMER ###################################    
    end = timer()
    print ("elapsed time: ", end-start)
    print("results: ", pick)

############################### SHOW IMAGE ##################################
    k = cv2.waitKey(30) & 0xff
    if k == 27:
       break
     
    if(len(results)>0):
        cv2.waitKey(0)
f.close()








