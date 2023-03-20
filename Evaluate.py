#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import cv2 as cv
import numpy as np


# In[ ]:


model = keras.models.load_model('model.h5')


# In[ ]:


evalData = np.load('data/evalDataNormalized.npy')
evalLabel = np.load('data/evalLabelOneHot.npy')


# In[ ]:


result = model.evaluate(evalData, evalLabel)


# In[ ]:


def preparePredict(path):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image = cv.resize(image, (224, 224))
    image = (image - 128.0) / 128.0

    image = np.expand_dims(image, axis = 0)

    return image


# In[ ]:


test1 = preparePredict('test1.jpg')
test2 = preparePredict('test2.jpg')


# In[ ]:


x = np.concatenate((test1, test2), axis = 0)


# In[ ]:


yPredict = model.predict(x)


# In[ ]:


print(yPredict)