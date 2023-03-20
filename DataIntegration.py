#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np
import glob
import os
from tqdm import tqdm


# In[ ]:


def loadData(oriPath, label):
    allImagePath = glob.glob(os.path.join(oriPath, '*.jpg'))

    x = np.empty([len(allImagePath), 224,224,3])
    y = np.empty([0])

    for idx in tqdm(range(len(allImagePath))):
        imagePath = allImagePath[idx]
        image = cv.imread(imagePath)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image = cv.resize(image, (224, 224))

        x[idx,:] = image

    y = np.linspace(label,label,x.shape[0])

    return x, y


# In[ ]:


trainCatX, trainCatY = loadData('data/training_set/cats', 0)
trainDogX, trainDogY = loadData('data/training_set/dogs', 1)
evalCatX, evalCatY = loadData('data/test_set/cats', 0)
evalDogX, evalDogY = loadData('data/test_set/dogs', 1)


# In[ ]:


trainData = np.concatenate((trainCatX, trainDogX), axis = 0)
trainLabel = np.concatenate((trainCatY, trainDogY), axis = 0)
evalData = np.concatenate((evalCatX, evalDogX), axis = 0)
evalLabel = np.concatenate((evalCatY, evalDogY), axis = 0)


# In[ ]:


trainImage = trainData[233]
testImage = evalData[233]


# In[ ]:


trainImage = cv.cvtColor(trainImage.astype(np.uint8), cv.COLOR_RGB2BGR)
testImage = cv.cvtColor(testImage.astype(np.uint8), cv.COLOR_RGB2BGR)

cv.imwrite('test1.jpg', trainImage)
cv.imwrite('test2.jpg', testImage)


# In[ ]:


print(trainLabel[233])
print(evalLabel[233])


# In[ ]:


np.save('data/trainData.npy', trainData)
np.save('data/trainLabel.npy', trainLabel)
np.save('data/evalData.npy', evalData)
np.save('data/evalLabel.npy', evalLabel)