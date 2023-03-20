#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import datetime


# In[ ]:


mobilenet = tf.keras.applications.MobileNet(weights='imagenet',input_shape=(224,224,3))


# In[ ]:


keras.utils.plot_model(mobilenet, show_shapes=True)


# In[ ]:


x = mobilenet.get_layer('conv_pw_13_relu').output


# In[ ]:


x = layers.Flatten()(x)


# In[ ]:


x = layers.Dense(100, activation='relu')(x)
predictLayer = layers.Dense(2, activation='softmax')(x)


# In[ ]:


newModel = keras.Model(inputs=mobilenet.input, outputs=predictLayer)


# In[ ]:


keras.utils.plot_model(newModel, show_shapes=True)


# In[ ]:


for layer in newModel.layers:
    layer.trainable = False


# In[ ]:


newModel.layers[-2].trainable = True
newModel.layers[-1].trainable = True


# In[ ]:


adam = keras.optimizers.Adam(0.0001)


# In[ ]:


newModel.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['categorical_accuracy'])


# In[ ]:


tensorBoard = keras.callbacks.TensorBoard(log_dir='kerasLog', write_images=1, histogram_freq=1)


# In[ ]:


trainData = np.load('data/trainDataNormalized.npy')
trainLabel = np.load('data/trainLabelOneHot.npy')


# In[ ]:


newModel.fit(trainData, trainLabel, epochs=10, batch_size=64, validation_split=0.2, callbacks=[tensorBoard], verbose=1)


# In[ ]:


newModel.save('model.h5')