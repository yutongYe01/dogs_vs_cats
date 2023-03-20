#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow.keras as keras


# In[ ]:


trainData = np.load('data/trainData.npy')
trainLabel = np.load('data/trainLabel.npy')
evalData = np.load('data/evalData.npy')
evalLabel = np.load('data/evalLabel.npy')


# In[ ]:


print(trainData[0])


# In[ ]:


trainData = (trainData - 128.0) / 128.0
evalData = (evalData - 128.0) / 128.0


# In[ ]:


print(trainData[0])


# In[ ]:


trainLabelOneHot = keras.utils.to_categorical(trainLabel, 2)
evalLabelOneHot = keras.utils.to_categorical(evalLabel, 2)


# In[ ]:


print(trainLabel[233])
print('---')
print(trainLabelOneHot[233])

print('------')

print(trainLabel[233])
print('---')
print(trainLabelOneHot[233])


# In[ ]:


permutation = np.random.permutation(trainData.shape[0])
trainData = trainData[permutation, :]
trainLabelOneHot = trainLabelOneHot[permutation]

permutation = np.random.permutation(evalData.shape[0])
evalData = evalData[permutation, :]
evalLabelOneHot = evalLabelOneHot[permutation]


# In[ ]:


np.save('data/trainDataNormalized.npy', trainData)
np.save('data/trainLabelOneHot.npy', trainLabelOneHot)
np.save('data/evalDataNormalized.npy', evalData)
np.save('data/evalLabelOneHot.npy', evalLabelOneHot)